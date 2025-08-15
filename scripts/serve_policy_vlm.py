import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server_vlm
from openpi.training import config as _config
from scripts.serve_policy import EnvMode, Checkpoint, Default, create_policy


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

    # VLM image key
    vlm_img_key: str | None = None
    # VLM server IP
    vlm_server_ip: str = "localhost:8000" # default to local vlm server
    # VLM query frequency
    vlm_query_frequency: int = 5
    # VLM draw path
    vlm_draw_path: bool = True
    # VLM draw mask
    vlm_draw_mask: bool = True
    # VLM mask ratio
    vlm_mask_ratio: float = 0.08


# Default checkpoints that should be used for each environment.
def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server_vlm.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        vlm_img_key=args.vlm_img_key,
        vlm_server_ip=args.vlm_server_ip,
        vlm_query_frequency=args.vlm_query_frequency,
        vlm_draw_path=args.vlm_draw_path,
        vlm_draw_mask=args.vlm_draw_mask,
        vlm_mask_ratio=args.vlm_mask_ratio,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
