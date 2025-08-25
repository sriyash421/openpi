import json_numpy
import cv2
import os
import datetime
import concurrent.futures
import logging
import traceback
import base64
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np
from PIL import Image

json_numpy.patch()

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import time

# Import the base HTTP policy server
from .http_policy_server import HTTPPolicyServer, ObservationRequest, ActionResponse

# VLM utilities for path and mask drawing
try:
    from vila_utils.utils.decode import add_mask_2d_to_img, add_path_2d_to_img_alt_fast, get_path_from_answer
    from vila_utils.utils.encode import scale_path
    from vila_utils.utils.prompts import get_prompt
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    logging.warning("VLM utilities not available. Path and mask drawing will be disabled.")


class HTTPPolicyServerVLM(HTTPPolicyServer):
    """Serves a policy using HTTP endpoints with FastAPI and VLM path/mask drawing capabilities.
    
    Extends HTTPPolicyServer with VLM path and mask drawing functionality while maintaining
    all the existing temporal ensembling and action chunk history capabilities.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        action_chunk_history_size: int = 10,
        ensemble_window_size: int = 5,
        temporal_weight_decay: float = -0.8,
        # VLM parameters
        vlm_img_key: str | None = None,
        vlm_server_ip: str | None = None,
        vlm_query_frequency: int = 10,
        vlm_draw_path: bool = True,
        vlm_draw_mask: bool = True,
        vlm_mask_ratio: float = 0.08,
    ) -> None:
        # Call parent constructor first
        super().__init__(
            policy=policy,
            host=host,
            port=port,
            metadata=metadata,
            action_chunk_history_size=action_chunk_history_size,
            ensemble_window_size=ensemble_window_size,
            temporal_weight_decay=temporal_weight_decay,
            setup_act_route=False, # avoid setting up the act route in the parent class so we can override it
        )
        
        # VLM integration parameters
        self._vlm_img_key = vlm_img_key
        self._vlm_server_ip = vlm_server_ip
        self._vlm_query_frequency = int(vlm_query_frequency)
        self._vlm_draw_path = bool(vlm_draw_path) if VLM_AVAILABLE else False
        self._vlm_draw_mask = bool(vlm_draw_mask) if VLM_AVAILABLE else False
        self._vlm_mask_ratio = float(vlm_mask_ratio)
        self._vlm_current_path = None
        self._vlm_current_mask = None
        self._vlm_step = 0
        
        # VLM save directory setup
        self._vlm_save_dir = None
        if self._vlm_img_key is not None and (self._vlm_draw_path or self._vlm_draw_mask):
            self._vlm_save_dir = os.path.join(os.getcwd(), "vlm_tmp")
            os.makedirs(self._vlm_save_dir, exist_ok=True)
            logging.info(f"VLM images will be saved to: {self._vlm_save_dir}")
            print(f"🖼️ VLM images will be saved to: {self._vlm_save_dir}")
        
        # Initialize thread pool executor for background image saving
        if self._vlm_save_dir is not None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2,  # Limit to 2 workers to avoid overwhelming the system
                thread_name_prefix="VLMImageSaver"
            )
        else:
            self._executor = None
        
        # Update FastAPI app title and description
        self._app.title = "OpenPI Policy Server with VLM"
        self._app.description = "HTTP server for OpenPI policy inference with temporal ensembling and VLM path/mask drawing"
        
        # Override the act endpoint to include VLM processing
        self._setup_vlm_routes()
        
        # Override the root endpoint to include VLM information
        self._setup_vlm_root_endpoint()
        
        # Override the history endpoint to include VLM step information
        self._setup_vlm_history_endpoint()
        
        logging.info(f"Initialized HTTP policy server with VLM capabilities. Action chunk history size: {action_chunk_history_size}, ensemble window: {ensemble_window_size}")
        if self._vlm_img_key:
            logging.info(f"VLM enabled with image key: {vlm_img_key}, draw_path: {self._vlm_draw_path}, draw_mask: {self._vlm_draw_mask}")

    def __del__(self):
        """Cleanup method to properly shut down the thread pool executor."""
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=True)

    def cleanup(self):
        """Explicit cleanup method to shut down the thread pool executor."""
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=True)
            logging.info("Thread pool executor shut down successfully")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def _setup_vlm_routes(self):
        """Override the act endpoint to include VLM processing."""
        
        @self._app.post("/act", response_model=ActionResponse)
        async def act(request: ObservationRequest):
            """Main endpoint for policy inference with temporal ensembling and VLM processing."""
            try:
                # Convert observation data to the format expected by the policy
                obs = request.dict() # Convert Pydantic model to a dict
                # Handle numpy arrays if they're serialized as lists
                obs = self._deserialize_observation(obs)
                policy_obs = {}
                
                # Transform to policy format
                policy_obs["observation.images.image_0"] = cv2.resize(obs.pop("image"), (256, 256))
                policy_obs["state"] = obs.pop("proprio")
                policy_obs["prompt"] = obs.pop("instruction")
                policy_obs["camera_present"] = np.array([1])
                instruction = policy_obs["prompt"]
                
                # VLM image processing if enabled
                if self._vlm_img_key is not None:
                    try:
                        original_img = policy_obs["observation.images.image_0"]
                        
                        if self._vlm_draw_path or self._vlm_draw_mask:
                            if self._vlm_step % self._vlm_query_frequency == 0:
                                try:
                                    img, self._vlm_current_path, self._vlm_current_mask = self._get_path_mask_from_vlm(
                                        image=original_img,
                                        task_instr=instruction,
                                        draw_path=self._vlm_draw_path,
                                        draw_mask=self._vlm_draw_mask,
                                        verbose=False,
                                        mask_ratio=self._vlm_mask_ratio,
                                    )
                                    logging.info(f"Successfully queried VLM")
                                    success = True
                                except Exception as e:
                                    logging.warning(f"VLM overlay error on query: {e}")
                                    self._vlm_current_path = None
                                    self._vlm_current_mask = None
                                    success = False
                                if success:
                                    # Save both the original and overlaid images for this fresh query
                                    self._save_vlm_images(obs, original_img, img, self._vlm_step, instruction)
                                else:
                                    img = original_img
                            elif self._vlm_current_path is not None or self._vlm_current_mask is not None:
                                try:
                                    img, _, _ = self._get_path_mask_from_vlm(
                                        image=original_img,
                                        task_instr=instruction,
                                        draw_path=self._vlm_draw_path,
                                        draw_mask=self._vlm_draw_mask,
                                        verbose=False,
                                        path=self._vlm_current_path,
                                        mask=self._vlm_current_mask,
                                        mask_ratio=self._vlm_mask_ratio,
                                    )
                                    logging.info(f"Successfully drew path and mask from VLM")
                                    success = True
                                except Exception as e:
                                    logging.warning(f"VLM overlay error on reuse: {e}")
                                    self._vlm_current_path = None
                                    self._vlm_current_mask = None
                                    success = False
                                if success:
                                    # Save both the original and overlaid images for this fresh query
                                    self._save_vlm_images(obs, original_img, img, self._vlm_step, instruction)
                                else:
                                    img = original_img
                            else:
                                img = original_img
                        else:
                            img = original_img
                        
                        # Update the image in the observation
                        policy_obs["observation.images.image_0"] = img
                        
                    finally:
                        self._vlm_step += 1
                
                policy_obs["observation.images.image_0"] = cv2.resize(policy_obs["observation.images.image_0"], (224, 224))
                # Get action from policy
                action = self._policy.infer(policy_obs)
                
                # Extract action chunk for history
                action_chunk = self._extract_action_chunk(action)
                
                # Update history
                self._update_history(obs, action_chunk)
                
                # Perform temporal ensembling if we have enough history
                ensemble_action = self._temporal_ensemble(action, action_chunk)

                if "actions" in ensemble_action:    
                    ensemble_action_array = ensemble_action["actions"]
                elif "action" in ensemble_action:
                    ensemble_action_array = ensemble_action["action"]

                # Return the first action of the ensemble
                return JSONResponse(ensemble_action_array[0])
                
            except HTTPException:
                # Re-raise HTTP exceptions
                raise
            except Exception as e:
                logging.error(f"Error in policy inference: {e}")
                logging.error(traceback.format_exc())
                raise HTTPException(
                    status_code=500,
                    detail=f"Policy inference failed: {str(e)}"
                )

    def _setup_vlm_root_endpoint(self):
        """Override the root endpoint to include VLM information."""
        
        @self._app.get("/")
        async def root():
            """Root endpoint returning server info with VLM capabilities."""
            return {
                "message": "OpenPI Policy Server with VLM and Temporal Ensembling",
                "metadata": self._metadata,
                "endpoints": {
                    "/act": "POST - Submit observation for policy inference",
                    "/health": "GET - Health check endpoint",
                    "/history": "GET - Get action chunk history info",
                    "/reset": "POST - Reset action chunk history"
                },
                "config": {
                    "action_chunk_history_size": self._action_chunk_history_size,
                    "ensemble_window_size": self._ensemble_window_size,
                    "temporal_weight_decay": self._temporal_weight_decay,
                    "vlm_enabled": self._vlm_img_key is not None,
                    "vlm_draw_path": self._vlm_draw_path,
                    "vlm_draw_mask": self._vlm_draw_mask,
                    "vlm_query_frequency": self._vlm_query_frequency
                }
            }

    def _setup_vlm_history_endpoint(self):
        """Override the history endpoint to include VLM step information."""
        
        @self._app.get("/history")
        async def get_history_info():
            """Get information about action chunk history, ensemble predictions, and VLM step."""
            return {
                "action_chunk_history_size": len(self._action_chunk_history),
                "observation_history_size": len(self._observation_history),
                "max_history_size": self._action_chunk_history_size,
                "ensemble_window_size": self._ensemble_window_size,
                "recent_action_chunks": list(self._action_chunk_history)[-5:] if self._action_chunk_history else [],
                "vlm_step": self._vlm_step
            }

    def _get_path_mask_from_vlm(
        self,
        image: np.ndarray,
        task_instr: str,
        draw_path=True,
        draw_mask=True,
        verbose=False,
        vlm_server_ip: str | None = None,
        path=None,
        mask=None,
        mask_ratio=0.15,
    ):
        """Get path and mask from VLM model."""
        if not VLM_AVAILABLE:
            raise Exception("VLM utilities not available")
            
        assert draw_path or draw_mask
        
        # Use provided server IP or fall back to instance variable
        server_ip = vlm_server_ip or self._vlm_server_ip
        
        # try up to 5 times
        temperature = 0.0
        for _ in range(5):
            try:
                if path is None and draw_path or mask is None and draw_mask:
                    prompt_type = "path_mask"
                    response_text = self._send_vlm_request(
                        image,
                        task_instr,
                        prompt_type,
                        server_ip=server_ip,
                        verbose=verbose,
                        temperature=temperature,
                    )
                    path, mask = get_path_from_answer(response_text, prompt_type)
                if draw_path:
                    drawn_rgb = self._draw_onto_image((path, mask), "path", image.copy(), mask_ratio=mask_ratio)
                    image = drawn_rgb
                if draw_mask:
                    masked_rgb = self._draw_onto_image((path, mask), "mask", image.copy(), mask_ratio=mask_ratio)
                    image = masked_rgb

                return image, path, mask
            except Exception as e:
                print(f"Error: {e}")
                temperature += 0.1  # increase temperature for next attempt
                continue
        raise Exception("Failed to get path and mask from VLM")

    def _send_vlm_request(
        self,
        image,
        quest,
        prompt_type,
        server_ip,
        max_tokens=512,
        temperature=0.0,
        top_p=0.95,
        max_retries=5,
        verbose=False,
    ):
        """Send image and quest to VLM model and get response."""
        if not VLM_AVAILABLE:
            raise Exception("VLM utilities not available")
            
        # Ensure image is in BGR format for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Encode image to base64
        _, encoded_image_array = cv2.imencode(".jpg", image_bgr)
        encoded_image = base64.b64encode(encoded_image_array.tobytes()).decode("utf-8")

        if verbose:
            print(f"Sending request with quest: {quest}")

        retry_count = 0
        while retry_count < max_retries:
            try:
                start_time = time.time()  # Record start time
                from openai import OpenAI, APIConnectionError
                client = OpenAI(base_url=server_ip, api_key="fake-key")
                prompt = get_prompt(quest, prompt_type, prompt_eval=False)
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
                    model="vila_3b_path_mask_fast",
                    extra_body={
                        "num_beams": 1,
                        "use_cache": True,
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                    },
                )
                end_time = time.time()  # Record start time
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

    def _draw_onto_image(self, vlm_path_mask_output, prompt_type, img, mask_ratio=0.15, verbose=False):
        """Draw paths and masks onto images."""
        if not VLM_AVAILABLE:
            return img
            
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
            scaled_path = scale_path(path, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_out)

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

    def _save_vlm_images(self, obs, original_img, img, step, instruction):
        """Save both original and processed VLM images to separate subfolders in a background thread."""
        if self._vlm_save_dir is not None and self._executor is not None:
            # Submit the image saving task to the thread pool executor
            future = self._executor.submit(self._save_vlm_images_sync, obs, original_img, img, step, instruction)
            # Add a callback to log any errors that occur in the thread
            future.add_done_callback(self._log_save_result)

    def _save_vlm_images_sync(self, obs, original_img, img, step, instruction):
        """Synchronous version of image saving that runs in a separate thread."""
        try:
            # Save original image
            original_dir = os.path.join(self._vlm_save_dir, instruction, 'original')
            os.makedirs(original_dir, exist_ok=True)
            original_save_name = f"original/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{step:06d}.png"
            original_save_path = os.path.join(self._vlm_save_dir, instruction, original_save_name)
            Image.fromarray(original_img).save(original_save_path)
            logging.info(f"Saved original image to {original_save_path}")
            print(f"🖼️ Saved original image to {original_save_path}")
            
            # Save processed image
            processed_dir = os.path.join(self._vlm_save_dir, instruction, 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            processed_save_name = f"processed/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{step:06d}.png"
            processed_save_path = os.path.join(self._vlm_save_dir, instruction, processed_save_name)
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

    def reset_history(self):
        """Reset action chunk history, observation history, and VLM step."""
        # Call parent method first
        result = super().reset_history()
        # Reset VLM-specific state
        self._vlm_step = 0
        logging.info("Action chunk history, observation history, and VLM step have been reset programmatically")
        result["vlm_step"] = self._vlm_step
        return result
