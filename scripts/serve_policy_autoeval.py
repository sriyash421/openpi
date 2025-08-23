import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import http_policy_server
from openpi.serving import http_policy_server_vlm
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    
    # Temporal ensembling parameters
    action_chunk_history_size: int = 10
    ensemble_window_size: int = 5
    temporal_weight_decay: float = 0.0
    
    # VLM path and mask drawing parameters
    draw_path: bool = False
    draw_mask: bool = False
    vlm_img_key: str | None = None  # Image key for VLM processing (e.g., "image")
    vlm_server_ip: str | None = None  # VLM server address
    vlm_query_frequency: int = 10  # How often to query VLM for new paths/masks
    vlm_mask_ratio: float = 0.08  # Mask ratio for VLM mask drawing


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # Choose server type based on VLM capabilities
    if args.draw_path or args.draw_mask:
        if args.vlm_img_key is None:
            logging.warning("VLM drawing enabled but no vlm_img_key specified. Using regular HTTP server.")
            server = http_policy_server.HTTPPolicyServer(
                policy=policy,
                host="0.0.0.0",
                port=args.port,
                metadata=policy_metadata,
                action_chunk_history_size=args.action_chunk_history_size,
                ensemble_window_size=args.ensemble_window_size,
                temporal_weight_decay=args.temporal_weight_decay,
            )
        else:
            logging.info("Creating VLM-enabled HTTP policy server")
            server = http_policy_server_vlm.HTTPPolicyServerVLM(
                policy=policy,
                host="0.0.0.0",
                port=args.port,
                metadata=policy_metadata,
                action_chunk_history_size=args.action_chunk_history_size,
                ensemble_window_size=args.ensemble_window_size,
                temporal_weight_decay=args.temporal_weight_decay,
                vlm_img_key=args.vlm_img_key,
                vlm_server_ip=args.vlm_server_ip,
                vlm_query_frequency=args.vlm_query_frequency,
                vlm_draw_path=args.draw_path,
                vlm_draw_mask=args.draw_mask,
                vlm_mask_ratio=args.vlm_mask_ratio,
            )
    else:
        logging.info("Creating regular HTTP policy server")
        server = http_policy_server.HTTPPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
            action_chunk_history_size=args.action_chunk_history_size,
            ensemble_window_size=args.ensemble_window_size,
            temporal_weight_decay=args.temporal_weight_decay,
        )
    
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
