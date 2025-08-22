import asyncio
import datetime
import base64
import logging
import traceback
import os
import numpy as np
from PIL import Image
import time
import cv2
import concurrent.futures
from functools import partial

from vila_utils.utils.decode import add_mask_2d_to_img, add_path_2d_to_img_alt_fast, get_path_from_answer
from vila_utils.utils.encode import scale_path
from vila_utils.utils.prompts import get_prompt

from openai import OpenAI, APIConnectionError
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.frames

VLM_DOWNSAMPLE_RESOLUTION = 256
POLICY_INPUT_RESOLUTION = 224
PATH_MODEL_NAME_MASK = PATH_MODEL_NAME = "vila_3b_path_mask_fast"
OLD_PROMPT = False

def send_request(
    image,
    quest,
    prompt_type,
    crop_type,
    server_ip,
    max_tokens=512,
    temperature=0.0,
    top_p=0.95,
    max_retries=5,
    verbose=False,
):
    """Send image and quest to HAMSTER model and get response."""
    # Ensure image is in BGR format for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    # preprocess the image
    image_bgr = preprocess_image(image_bgr, crop_type)

    if prompt_type == "path":
        model_name = PATH_MODEL_NAME
    elif prompt_type == "path_mask":
        model_name = PATH_MODEL_NAME_MASK
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    # Encode image to base64
    _, encoded_image_array = cv2.imencode(".jpg", image_bgr)
    encoded_image = base64.b64encode(encoded_image_array.tobytes()).decode("utf-8")

    if verbose:
        print(f"Sending request with quest: {quest}")

    retry_count = 0
    while retry_count < max_retries:
        try:
            start_time = time.time()  # Record start time
            client = OpenAI(base_url=server_ip, api_key="fake-key")
            prompt = get_prompt(quest, prompt_type, prompt_eval=OLD_PROMPT)
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=int(max_tokens),
                model=model_name,
                extra_body={
                    "num_beams": 1,
                    "use_cache": True,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                },
            )
            end_time = time.time()  # Record end time
            response_text = response.choices[0].message.content[0]["text"]
            duration = end_time - start_time
            if verbose:
                print(f"Server response received in {duration:.2f} seconds.")
            return response_text
        except APIConnectionError as e:
            print(f"Error connecting to server: {e}")
            wait_time = 2**retry_count  # Exponential backoff
            retry_count += 1 # this doesn't count as a retry
            max_retries += 1 
            print(f"Retrying in {wait_time} seconds... (Attempt {retry_count} of {max_retries})")
            time.sleep(wait_time)
            continue
        except Exception as e:
            retry_count += 1
            wait_time = 2**retry_count  # Exponential backoff
            if retry_count < max_retries:
                print(f"Error connecting to server: {e}")
                print(f"Retrying in {wait_time} seconds... (Attempt {retry_count} of {max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
    return None


def draw_onto_image(vlm_path_mask_output, prompt_type, img, mask_ratio=0.15, verbose=False):
    # default inference code which is a bit different from the original data processing code because of legacy code reasons.
    h, w, c = img.shape
    scaled_mask = None
    if "mask" in prompt_type:
        min_in, max_in = np.zeros(2), np.array([w, h])
        min_out, max_out = np.zeros(2), np.ones(2)
        mask = vlm_path_mask_output[1] if len(vlm_path_mask_output) == 2 else vlm_path_mask_output
        scaled_mask = scale_path(mask, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)

    scaled_path = None
    if "path" in prompt_type:
        min_in, max_in = np.zeros(2), np.array([w, h])
        min_out, max_out = np.zeros(2), np.ones(2)
        path = vlm_path_mask_output[0] if len(vlm_path_mask_output) == 2 else vlm_path_mask_output
        scaled_path = scale_path(path, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)

        # check if there's any very close points in the path, get rid of duplicates
        new_path = []
        for i, point in enumerate(scaled_path):
            if i == 0:
                new_path.append(point)
            else:
                if not np.allclose(point, new_path[-1]):
                    new_path.append(point)
        scaled_path = np.array(new_path)

    if "mask" in prompt_type and scaled_mask is not None:
        if verbose:
            print("adding mask")
        img = add_mask_2d_to_img(img, scaled_mask, mask_pixels=int(h * mask_ratio))

    if "path" in prompt_type and scaled_path is not None:
        if verbose:
            print("adding path")
        img = add_path_2d_to_img_alt_fast(img, scaled_path, line_size=2)
    return img


def preprocess_image(image, crop_type):
    """Process the image by either stretching or center cropping."""
    height, width, _ = image.shape
    if crop_type == "Center Crop":
        crop_size = min(height, width)
        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2
        image = image[start_y : start_y + crop_size, start_x : start_x + crop_size]
    # then, resize the image to DOWNSAMPLE_RESOLUTION x DOWNSAMPLE_RESOLUTION
    return cv2.resize(image, (VLM_DOWNSAMPLE_RESOLUTION, VLM_DOWNSAMPLE_RESOLUTION))

def get_path_mask_from_vlm(
    image: np.ndarray,
    crop_type: str,
    task_instr: str,
    draw_path=True,
    draw_mask=True,
    verbose=False,
    vlm_server_ip: str | None = None,
    path=None,
    mask=None,
    mask_ratio=0.15,
):
    # used for VLM inference during eval
    assert draw_path or draw_mask
    # try up to 5 times
    temperature = 0.0
    for _ in range(5):
        try:
            if path is None and draw_path or mask is None and draw_mask:
                prompt_type = "path_mask"
                response_text = send_request(
                    image,
                    task_instr,
                    prompt_type,
                    crop_type,
                    server_ip=vlm_server_ip,
                    verbose=verbose,
                    temperature=temperature,
                )
                path, mask = get_path_from_answer(response_text, prompt_type)
            if draw_path:
                drawn_rgb = draw_onto_image((path, mask), "path", image.copy(), mask_ratio=mask_ratio)
                image = drawn_rgb
            if draw_mask:
                masked_rgb = draw_onto_image((path, mask), "mask", image.copy(), mask_ratio=mask_ratio)
                image = masked_rgb

            return image, path, mask
        except Exception as e:
            print(f"Error: {e}")
            temperature += 0.1  # increase temperature for next attempt
            continue
    raise Exception("Failed to get path and mask from VLM")


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        obs_remap_key: str | None = None,
        vlm_img_key: str | None = None,
        vlm_server_ip: str | None = None,
        vlm_query_frequency: int = 10,
        vlm_draw_path: bool = True,
        vlm_draw_mask: bool = True,
        vlm_mask_ratio: float = 0.08,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

        self._obs_remap_key = obs_remap_key
        # VLM integration parameters
        self._vlm_img_key = vlm_img_key
        self._vlm_server_ip = vlm_server_ip
        self._vlm_query_frequency = int(vlm_query_frequency)
        self._vlm_draw_path = bool(vlm_draw_path)
        self._vlm_draw_mask = bool(vlm_draw_mask)
        self._vlm_mask_ratio = float(vlm_mask_ratio)
        self._vlm_current_path = None
        self._vlm_current_mask = None
        self._vlm_step = 0
        
        # VLM save directory setup
        self._vlm_save_dir = None
        if self._vlm_img_key is not None:
            self._vlm_save_dir = os.path.join(os.getcwd(), "vlm_tmp")
            os.makedirs(self._vlm_save_dir, exist_ok=True)
            logging.info(f"VLM images will be saved to: {self._vlm_save_dir}")
            print(f"🖼️ VLM images will be saved to: {self._vlm_save_dir}")
        
        # Initialize thread pool executor for background image saving
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,  # Limit to 2 workers to avoid overwhelming the system
            thread_name_prefix="VLMImageSaver"
        )

    def __del__(self):
        """Cleanup method to properly shut down the thread pool executor."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)

    def cleanup(self):
        """Explicit cleanup method to shut down the thread pool executor."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
            logging.info("Thread pool executor shut down successfully")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def _save_vlm_images(self, obs, original_img, img, step):
        """Save both original and processed VLM images to separate subfolders in a background thread."""
        if self._vlm_save_dir is not None:
            # Submit the image saving task to the thread pool executor
            future = self._executor.submit(self._save_vlm_images_sync, obs, original_img, img, step)
            # Add a callback to log any errors that occur in the thread
            future.add_done_callback(self._log_save_result)

    def _save_vlm_images_sync(self, obs, original_img, img, step):
        """Synchronous version of image saving that runs in a separate thread."""
        try:
            # Save original image
            original_dir = os.path.join(self._vlm_save_dir, obs.get('prompt', ''), 'original')
            os.makedirs(original_dir, exist_ok=True)
            original_save_name = f"original/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{step:06d}.png"
            original_save_path = os.path.join(self._vlm_save_dir, obs.get('prompt', ''), original_save_name)
            Image.fromarray(original_img).save(original_save_path)
            logging.info(f"Saved original image to {original_save_path}")
            print(f"🖼️ Saved original image to {original_save_path}")
            
            # Save processed image
            processed_dir = os.path.join(self._vlm_save_dir, obs.get('prompt', ''), 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            processed_save_name = f"processed/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{step:06d}.png"
            processed_save_path = os.path.join(self._vlm_save_dir, obs.get('prompt', ''), processed_save_name)
            Image.fromarray(img).save(processed_save_path)
            logging.info(f"Saved processed VLM image to {processed_save_path}")
            print(f"🖼️ Saved processed VLM image to {processed_save_path}")
        except Exception as save_err:
            logging.warning(f"Failed to save VLM images: {save_err}")
            print(f"❌ Failed to save VLM images: {save_err}")
            raise  # Re-raise to be caught by the callback

    def _log_save_result(self, future):
        """Callback to log the result of the image saving operation."""
        try:
            future.result()  # This will raise any exception that occurred
        except Exception as e:
            logging.error(f"Error in background image saving thread: {e}")
            print(f"❌ Error in background image saving thread: {e}")

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                
                # Handle reset
                if obs.get("reset", False):
                    logging.info(f"Resetting policy and VLM step")
                    self._policy.reset()
                    self._vlm_step = 0
                
                # VLM image processing
                if self._vlm_img_key is not None and self._vlm_img_key in obs:
                    
                    try:
                        original_img = obs[self._vlm_img_key]
                        
                        if self._vlm_draw_path or self._vlm_draw_mask:
                            if self._vlm_step % self._vlm_query_frequency == 0:
                                try:
                                    img, self._vlm_current_path, self._vlm_current_mask = get_path_mask_from_vlm(
                                        image=original_img,
                                        crop_type=None,
                                        task_instr=obs.get("prompt", ""),
                                        draw_path=self._vlm_draw_path,
                                        draw_mask=self._vlm_draw_mask,
                                        verbose=False,
                                        vlm_server_ip=self._vlm_server_ip,
                                        mask_ratio=self._vlm_mask_ratio,
                                    )
                                    success = True
                                except Exception as e:
                                    logging.warning(f"VLM overlay error on query: {e}")
                                    self._vlm_current_path = None
                                    self._vlm_current_mask = None
                                    success = False
                                if success:
                                    # Save both the original and overlaid images for this fresh query
                                    self._save_vlm_images(obs, original_img, img, self._vlm_step)
                                else:
                                    img = original_img
                            elif self._vlm_current_path is not None or self._vlm_current_mask is not None:
                                try:
                                    img, _, _ = get_path_mask_from_vlm(
                                        image=original_img,
                                        crop_type=None,
                                        task_instr=obs.get("prompt", ""),
                                        draw_path=self._vlm_draw_path,
                                        draw_mask=self._vlm_draw_mask,
                                        verbose=False,
                                        vlm_server_ip=None,
                                        path=self._vlm_current_path,
                                        mask=self._vlm_current_mask,
                                        mask_ratio=self._vlm_mask_ratio,
                                    )
                                    success = True
                                except Exception as e:
                                    logging.warning(f"VLM overlay error on reuse: {e}")
                                    self._vlm_current_path = None
                                    self._vlm_current_mask = None
                                    success = False
                                if success:
                                    # Save both the original and overlaid images for this fresh query
                                    self._save_vlm_images(obs, original_img, img, self._vlm_step)
                                else:
                                    img = original_img
                        # Update the image in the observation
                        obs[self._vlm_img_key] = img
                        # downsample
                        obs[self._vlm_img_key] = cv2.resize(obs[self._vlm_img_key], (POLICY_INPUT_RESOLUTION, POLICY_INPUT_RESOLUTION))
                        
                    finally:
                        self._vlm_step += 1

                # rename keys in observation
                if self._obs_remap_key is not None:
                    obs[self._obs_remap_key] = obs[self._vlm_img_key]
                    del obs[self._vlm_img_key]

                action = self._policy.infer(obs)
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
