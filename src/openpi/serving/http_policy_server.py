import json_numpy
import cv2

json_numpy.patch()
import asyncio
import logging
import traceback
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import time


class ObservationRequest(BaseModel):
    """Request model for policy inference."""
    # Allow flexible observation data - for autoeval format, data comes directly
    # without nesting under 'observation' key
    # Make fields flexible to handle different naming conventions
    image: Any = None           # Image data
    instruction: str = None      # Instruction/prompt
    proprio: Any = None         # Proprioceptive data (optional)


class ActionResponse(BaseModel):
    """Response model for policy inference."""
    action: Dict[str, Any]
    action_chunk: List[List[float]] = None  # Current action chunk
    ensemble_info: Dict[str, Any] = None    # Ensemble prediction info


class HTTPPolicyServer:
    """Serves a policy using HTTP endpoints with FastAPI.
    
    Provides an /act endpoint for policy inference while maintaining compatibility
    with the existing policy infrastructure. Includes action chunk history and
    temporal ensembling for improved prediction stability.
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
        setup_act_route: bool = True,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        
        # Action chunk history and temporal ensembling parameters
        self._action_chunk_history_size = action_chunk_history_size
        self._ensemble_window_size = ensemble_window_size
        self._temporal_weight_decay = temporal_weight_decay
        
        # Rolling buffer for action chunks
        self._action_chunk_history = deque(maxlen=action_chunk_history_size)
        self._observation_history = deque(maxlen=action_chunk_history_size)
        
        # Create FastAPI app
        self._app = FastAPI(
            title="OpenPI Policy Server",
            description="HTTP server for OpenPI policy inference with temporal ensembling",
            version="1.0.0"
        )
        
        # Setup routes
        self._setup_routes(setup_act_route)
        
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.info(f"Initialized HTTP policy server with action chunk history size: {action_chunk_history_size}, ensemble window: {ensemble_window_size}")

    def _setup_routes(self, setup_act_route: bool = True):
        """Setup FastAPI routes."""
        
        @self._app.get("/")
        async def root():
            """Root endpoint returning server info."""
            return {
                "message": "OpenPI Policy Server with Temporal Ensembling",
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
                    "temporal_weight_decay": self._temporal_weight_decay
                }
            }
        
        @self._app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "policy_loaded": True}
        
        @self._app.get("/history")
        async def get_history_info():
            """Get information about action chunk history and ensemble predictions."""
            return {
                "action_chunk_history_size": len(self._action_chunk_history),
                "observation_history_size": len(self._observation_history),
                "max_history_size": self._action_chunk_history_size,
                "ensemble_window_size": self._ensemble_window_size,
                "recent_action_chunks": list(self._action_chunk_history)[-5:] if self._action_chunk_history else []
            }
        
        @self._app.post("/reset")
        async def reset_history():
            """Reset action chunk history and observation history."""
            try:
                # Clear both history buffers
                self._action_chunk_history.clear()
                self._observation_history.clear()
                
                logging.info("Action chunk history and observation history have been reset")
                
                return {"status": "reset successful"}
                
            except Exception as e:
                logging.error(f"Error resetting history: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to reset history: {str(e)}"
                )
        
        if setup_act_route:
            @self._app.post("/act", response_model=ActionResponse)
            async def act(request: ObservationRequest):
                """Main endpoint for policy inference with temporal ensembling."""
                try:
                    # Convert observation data to the format expected by the policy
                    obs = request.dict() # Convert Pydantic model to a dict
                    # Handle numpy arrays if they're serialized as lists
                    obs = self._deserialize_observation(obs)
                    policy_obs = {}
                    
                    # Extract and validate required fields with flexible naming
                    
                    policy_obs["instruction"] = obs.pop("instruction") 
                    
                    # Transform to policy format
                    policy_obs["observation.images.image_0"] = cv2.resize(obs.pop("image"), (224, 224))
                    policy_obs["state"] = obs.pop("proprio")
                    policy_obs["prompt"] = policy_obs.pop("instruction")
                    # TODO: resize
                    policy_obs["camera_present"] = np.array([1])

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

                    # ret the first action of the ensemble
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

    def _deserialize_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert observation data to the format expected by the policy."""
        deserialized = {}

        for key, value in obs.items():
            if isinstance(value, list):
                # Convert lists to numpy arrays if they look like numeric data
                try:
                    deserialized[key] = np.array(value, dtype=np.float32)
                except (ValueError, TypeError):
                    deserialized[key] = value
            else:
                deserialized[key] = value
                
        return deserialized
    
    def _serialize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert action data to JSON-serializable format."""
        serialized = {}
        
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = value.item()
            else:
                serialized[key] = value
                
        return serialized

    def reset_history(self):
        """Reset action chunk history and observation history."""
        self._action_chunk_history.clear()
        self._observation_history.clear()
        logging.info("Action chunk history and observation history have been reset programmatically")
        return {
            "action_chunk_history_size": len(self._action_chunk_history),
            "observation_history_size": len(self._observation_history)
        }

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

    def serve_forever(self) -> None:
        """Start the HTTP server."""
        uvicorn.run(
            self._app,
            host=self._host,
            port=self._port,
            log_level="info"
        )

    async def run(self):
        """Async version of serve_forever for compatibility."""
        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
