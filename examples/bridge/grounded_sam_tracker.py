import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np

from PIL import Image
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection,
    Sam2Processor, Sam2Model,
    Sam2VideoProcessor, Sam2VideoModel, infer_device
)

class GroundedSam2Tracker:
    def __init__(self,
                 dino_id="IDEA-Research/grounding-dino-tiny",
                 sam2_id="facebook/sam2.1-hiera-small",
                 box_thresh=0.35, text_thresh=0.25):
        """
        Initialize models once. Use reset(init_frame, text) to start a new tracking session.
        """
        self.device = infer_device()

        # Cache config
        self.dino_id = dino_id
        self.sam2_id = sam2_id
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh

        # ---- Load models once ----
        self.dino_proc = AutoProcessor.from_pretrained(self.dino_id)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_id).to(self.device)

        self.s2_proc = Sam2Processor.from_pretrained(self.sam2_id)
        self.s2 = Sam2Model.from_pretrained(self.sam2_id).to(self.device)

        self.vproc = Sam2VideoProcessor.from_pretrained(self.sam2_id)
        self.vmodel = Sam2VideoModel.from_pretrained(self.sam2_id).to(
            self.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        self.box_thresh = box_thresh
        self.text_thresh = text_thresh

        # Unset state before first reset
        self.session = None
        self.obj_ids = None
        self.init_masks = None
        self.last_masks = None
        self.frame_idx = 0

    @torch.no_grad()
    def reset(self,
              init_frame: Image.Image,
              text: str):
        """
        Recompute detections and initial masks on a new video frame and reset SAM2 video history.
        After this call, frame_idx is 1 and last_masks are the masks for the init frame.
        """

        # ---- GroundingDINO: language → boxes on init_frame ----
        din = self.dino_proc(images=init_frame, text=text, return_tensors="pt").to(self.device)
        dout = self.dino(**din)
        det = self.dino_proc.post_process_grounded_object_detection(
            dout, din.input_ids, threshold=self.box_thresh, text_threshold=self.text_thresh,
            target_sizes=[init_frame.size[::-1]]
        )[0]
        if det["boxes"].numel() == 0:
            raise ValueError("No detections above thresholds on the init frame.")

        # Stash detections
        self.boxes_xyxy = det["boxes"].detach().cpu()   # (N,4)
        self.scores = det["scores"].detach().cpu()      # (N,)
        self.labels = det["labels"]                     # list[str]

        # ---- SAM2 image: boxes → masks on init_frame ----
        input_boxes = [self.boxes_xyxy.tolist()]  # [image][objects][4]
        sin = self.s2_proc(images=init_frame, input_boxes=input_boxes, return_tensors="pt").to(self.device)
        sout = self.s2(**sin, multimask_output=False)
        masks = self.s2_proc.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"])[0]
        if masks.dim() == 4 and masks.size(0) == 1:
            masks = masks[0]
        self.init_masks = (masks > 0.5).to(torch.bool)  # (N,H,W)

        # ---- SAM2 Video: create fresh streaming session and seed with init masks ----
        self.session = self.vproc.init_video_session(
            inference_device=self.device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        self.obj_ids = list(range(1, self.init_masks.shape[0] + 1))
        vin0 = self.vproc(images=init_frame, return_tensors="pt")
        self.vproc.add_inputs_to_inference_session(
            inference_session=self.session,
            frame_idx=0,
            obj_ids=self.obj_ids,
            input_masks=[m.numpy() for m in self.init_masks.cpu()],  # list of (H,W)
            original_size=tuple(vin0.original_sizes[0])
        )
        out0 = self.vmodel(inference_session=self.session,
                           frame=vin0.pixel_values[0].to(self.vmodel.device))
        masks0 = self.vproc.post_process_masks([out0.pred_masks],
                                               original_sizes=vin0.original_sizes,
                                               binarize=True)[0]
        self.frame_idx = 1
        self.last_masks = masks0.to(torch.bool)

    def apply_masks_to_frames(self, frames, masks_list):
        """
        frames: list of PIL.Image or np.ndarray [H,W,3]
        masks_list: list of torch.BoolTensor or np.ndarray [N,H,W]
        returns: list of np.ndarray frames with masks applied
        """
        out_frames = []
        for frame, masks in zip(frames, masks_list):
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            if hasattr(masks, "numpy"):  # torch.Tensor
                masks = masks.cpu().numpy()
            masks = masks.squeeze(1)
            if masks.ndim == 3:  # multiple objects -> combine
                mask = np.any(masks, axis=0)
            else:
                mask = masks

            mask3 = np.repeat(mask[..., None], 3, axis=-1)
            out = frame * mask3

            out_frames.append(out.astype(np.uint8))
        return out_frames

    @torch.no_grad()
    def step(self, frame: Image.Image):
        """
        Push one new frame; returns (frame_idx, masks) where masks is (N,H,W) bool tensor.
        """
        vin = self.vproc(images=frame, return_tensors="pt")
        out = self.vmodel(inference_session=self.session,
                          frame=vin.pixel_values[0].to(self.vmodel.device))
        masks_t = self.vproc.post_process_masks([out.pred_masks],
                                                original_sizes=vin.original_sizes,
                                                binarize=True)[0].to(torch.bool)
        idx = self.frame_idx
        self.frame_idx += 1
        self.last_masks = masks_t
        return idx, masks_t