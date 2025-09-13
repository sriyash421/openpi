import asyncio
import websockets.asyncio.server
import websockets.frames
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
from collections import deque
from typing import Dict, Any
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from openpi.serving.arro_client import GroundedSam2TrackerClient


import spacy
import cv2
nlp = spacy.load("en_core_web_sm")

def instruction_to_dino_instr(instruction):
    # find all the nouns in the image and use gripper via spacy
    doc = nlp(instruction)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    nouns = nouns + ["robot gripper"]

    # add "a " prefix to each object
    objects = ["a " + o for o in nouns]
    # separate objects with ". "
    dino_instr = ". ".join(objects)
    return dino_instr

def _apply_masks_to_frames(frames, masks_list):
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


POLICY_INPUT_RESOLUTION = 224

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol with VLM integration and temporal ensembling.
    
    Provides temporal ensembling of action predictions similar to the HTTP server for improved
    prediction stability and reduced noise in action outputs.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        obs_remap_key: str | None = None,
        arro_img_key: str | None = None,
        arro_server_ip: str | None = None,
        action_chunk_history_size: int = 10,
        ensemble_window_size: int = 5,
        temporal_weight_decay: float = 0.5,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

        self._obs_remap_key = obs_remap_key
        # VLM integration parameters
        self._arro_img_key = arro_img_key
        self._arro_server_ip = arro_server_ip

        # ARRO client
        print(f"🔄 Initializing ARRO client with server IP: {self._arro_server_ip}")
        self._arro_client = GroundedSam2TrackerClient(self._arro_server_ip)
        print(f"✅ ARRO client initialized")
        
        # Temporal ensembling parameters
        self._action_chunk_history_size = action_chunk_history_size
        self._ensemble_window_size = ensemble_window_size
        self._temporal_weight_decay = temporal_weight_decay
        
        # Rolling buffer for action chunks and observations
        self._action_chunk_history = deque(maxlen=action_chunk_history_size)
        self._observation_history = deque(maxlen=action_chunk_history_size)
        
        # ARRO save directory setup
        self._vlm_save_dir = None
        if self._arro_server_ip is not None:
            self._arro_save_dir = os.path.join(os.getcwd(), "arro_tmp")
            os.makedirs(self._arro_save_dir, exist_ok=True)
            logging.info(f"ARRO images will be saved to: {self._arro_save_dir}")
            print(f"🖼️ ARRO images will be saved to: {self._arro_save_dir}")
        
        # Initialize thread pool executor for background image saving
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,  # Limit to 2 workers to avoid overwhelming the system
            thread_name_prefix="ARROImageSaver"
        )
        
        logging.info(f"Initialized ARRO websocket policy server with action chunk history size: {action_chunk_history_size}, ensemble window: {ensemble_window_size}")

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
            original_dir = os.path.join(self._arro_save_dir, obs.get('prompt', ''), 'original')
            os.makedirs(original_dir, exist_ok=True)
            original_save_name = f"original/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{step:06d}.png"
            original_save_path = os.path.join(self._arro_save_dir, obs.get('prompt', ''), original_save_name)
            Image.fromarray(original_img).save(original_save_path)
            logging.info(f"Saved original image to {original_save_path}")
            print(f"🖼️ Saved original image to {original_save_path}")
            
            # Save processed image
            processed_dir = os.path.join(self._arro_save_dir, obs.get('prompt', ''), 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            processed_save_name = f"processed/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{step:06d}.png"
            processed_save_path = os.path.join(self._arro_save_dir, obs.get('prompt', ''), processed_save_name)
            Image.fromarray(img).save(processed_save_path)
            logging.info(f"Saved processed ARRO image to {processed_save_path}")
            print(f"🖼️ Saved processed ARRO image to {processed_save_path}")
        except Exception as save_err:
            logging.warning(f"Failed to save ARRO images: {save_err}")
            print(f"❌ Failed to save ARRO images: {save_err}")
            raise  # Re-raise to be caught by the callback

    def _log_save_result(self, future):
        """Callback to log the result of the image saving operation."""
        try:
            future.result()  # This will raise any exception that occurred
        except Exception as e:
            logging.error(f"Error in background ARRO image saving thread: {e}")
            print(f"❌ Error in background image saving thread: {e}")

    def _extract_action_chunk(self, action: Dict[str, Any]) -> np.ndarray:
        """Extract action chunk from policy response."""
        if "actions" in action:
            return np.array(action["actions"])
        elif "action" in action:
            return np.array(action["action"])
        else:
            # If no clear action chunk, use the entire action dict
            return np.array(list(action.values()))

    def _update_history(self, observation: Dict[str, Any], action_chunk: np.ndarray):
        """Update action chunk and observation history."""
        self._action_chunk_history.append(action_chunk.copy())
        self._observation_history.append(observation.copy())
        logging.debug(f"Updated history. Current size: {len(self._action_chunk_history)}")

    def _temporal_ensemble(self, current_action: Dict[str, Any], current_action_chunk: np.ndarray) -> Dict[str, Any]:
        """Perform temporal ensembling of action predictions."""
        if len(self._action_chunk_history) < self._ensemble_window_size:
            # Not enough history for ensembling, return current action
            return current_action
        
        # Get recent action chunks for ensemble
        recent_chunks = list(self._action_chunk_history)[-self._ensemble_window_size:]
        
        # Apply temporal weighting with decay
        weights = np.array([self._temporal_weight_decay ** i for i in range(len(recent_chunks))])
        weights = weights / weights.sum()  # Normalize weights
        
        # Weighted ensemble of action chunks
        ensemble_chunk = np.zeros_like(current_action_chunk)
        for i, chunk in enumerate(recent_chunks):
            if chunk.shape == current_action_chunk.shape:
                ensemble_chunk += weights[i] * chunk
            else:
                # Handle shape mismatches by using current chunk
                ensemble_chunk += weights[i] * current_action_chunk
        
        # Create ensemble action response
        ensemble_action = current_action.copy()
        if "actions" in ensemble_action:
            ensemble_action["actions"] = ensemble_chunk
        elif "action" in ensemble_action:
            ensemble_action["action"] = ensemble_chunk
        else:
            # Update all numeric values with ensemble
            for key, value in ensemble_action.items():
                if isinstance(value, (int, float, np.number)):
                    ensemble_action[key] = float(ensemble_chunk[0] if len(ensemble_chunk) > 0 else value)
        
        logging.info(f"Applied temporal ensemble with {len(recent_chunks)} chunks, weights: {weights}")
        return ensemble_action

    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about temporal ensembling state."""
        return {
            "action_chunk_history_size": len(self._action_chunk_history),
            "observation_history_size": len(self._observation_history),
            "max_history_size": self._action_chunk_history_size,
            "ensemble_window_size": self._ensemble_window_size,
            "temporal_weight_decay": self._temporal_weight_decay,
            "recent_action_chunks": list(self._action_chunk_history)[-5:] if self._action_chunk_history else []
        }

    def reset_ensemble_history(self) -> Dict[str, Any]:
        """Reset temporal ensembling history programmatically."""
        self._action_chunk_history.clear()
        self._observation_history.clear()
        logging.info("Temporal ensembling history has been reset programmatically")
        return {
            "action_chunk_history_size": len(self._action_chunk_history),
            "observation_history_size": len(self._observation_history)
        }

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
                original_img = obs[self._arro_img_key]
                
                # Handle reset
                if obs.get("reset", False):
                    logging.info(f"Resetting policy and VLM step")
                    self._policy.reset()
                    self._arro_step = 0
                    print(f"🔄 Resetting ARRO client")
                    print(f"Original image shape: {original_img.shape}")
                    dino_instruction = instruction_to_dino_instr(obs["prompt"])
                    print(f"Instruction: {dino_instruction}")
                    self._arro_client.reset(init_frame=Image.fromarray(original_img), text=dino_instruction)
                    print(f"✅ ARRO client reset")
                    # Also reset temporal ensembling history
                    self._action_chunk_history.clear()
                    self._observation_history.clear()
                    logging.info("Temporal ensembling history has been reset")
                
                # arro image processing
                try:
                    idx, masks = self._arro_client.step(Image.fromarray(original_img))
                    img = _apply_masks_to_frames([original_img], [masks])[0]
                    success = True
                except Exception as e:
                    logging.warning(f"arro overlay error on query: {e}")
                    success = False
                
                if success:
                    # Save both the original and overlaid images for this fresh query
                    self._save_vlm_images(obs, original_img, img, self._arro_step)
                else:
                    img = original_img

                # Update the image in the observation
                obs[self._arro_img_key] = img
                # downsample
                obs[self._arro_img_key] = cv2.resize(obs[self._arro_img_key], (POLICY_INPUT_RESOLUTION, POLICY_INPUT_RESOLUTION))

                self._arro_step += 1
                    

                # rename keys in observation
                if self._obs_remap_key is not None:
                    obs[self._obs_remap_key] = obs[self._arro_img_key]
                    del obs[self._arro_img_key]

                action = self._policy.infer(obs)
                
                # Extract action chunk for history
                action_chunk = self._extract_action_chunk(action)

                if self._temporal_weight_decay != 0: 
                    # Update history
                    self._update_history(obs, action_chunk)
                    
                    # Perform temporal ensembling if we have enough history
                    ensemble_action = self._temporal_ensemble(action, action_chunk)
                else:
                    ensemble_action = action
                
                await websocket.send(packer.pack(ensemble_action))
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
