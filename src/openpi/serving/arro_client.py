import io
import json
from typing import Any, Tuple

import numpy as np
import zmq
from PIL import Image
import torch

class GroundedSam2TrackerClient:
    """
    Thin ZMQ client that mirrors the GroundedSam2Tracker API (reset, step).

    Usage:
        client = GroundedSam2TrackerClient("tcp://127.0.0.1:5555")
        client.reset(init_frame, text)
        idx, masks = client.step(frame)
    """

    def __init__(self, address: str = "tcp://127.0.0.1:5555", request_timeout_ms: int = 600000, **_: Any):
        self.address = address
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(self.address)
        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)
        self._request_timeout_ms = request_timeout_ms

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        finally:
            pass

    # --- Public API matching GroundedSam2Tracker ---
    def reset(self, init_frame: Image.Image, text: str) -> None:
        header = {"cmd": "reset", "text": text}
        img_bytes = _encode_image_to_png_bytes(init_frame)
        self._send_and_expect_ok(header, img_bytes)

    def step(self, frame: Image.Image) -> Tuple[int, torch.BoolTensor]:
        header = {"cmd": "step"}
        img_bytes = _encode_image_to_png_bytes(frame)
        reply_header, extra_frames = self._send_and_recv(header, img_bytes)

        if reply_header.get("status") != "ok":
            raise RuntimeError(reply_header.get("error", "Unknown server error"))

        idx = int(reply_header["idx"])
        if len(extra_frames) != 1:
            raise RuntimeError("Malformed reply: expected 1 extra frame with masks.")
        masks_np = _load_npy_from_bytes(extra_frames[0])  # (N,H,W) bool
        masks_t = torch.from_numpy(masks_np).to(torch.bool)
        return idx, masks_t

    # --- Utilities ---
    def _send_and_expect_ok(self, header: dict, img_bytes: bytes) -> None:
        reply_header, _ = self._send_and_recv(header, img_bytes)
        if reply_header.get("status") != "ok":
            raise RuntimeError(reply_header.get("error", "Unknown server error"))

    def _send_and_recv(self, header: dict, img_bytes: bytes):
        header_bytes = json.dumps(header).encode("utf-8")
        self._sock.send_multipart([header_bytes, img_bytes])

        socks = dict(self._poller.poll(self._request_timeout_ms))
        if socks.get(self._sock) != zmq.POLLIN:
            raise TimeoutError("Timed out waiting for server reply.")

        parts = self._sock.recv_multipart()
        if not parts:
            raise RuntimeError("Empty reply from server.")
        reply_header = json.loads(parts[0].decode("utf-8"))
        extra_frames = parts[1:]
        return reply_header, extra_frames


def _encode_image_to_png_bytes(img_like: Any) -> bytes:
    if isinstance(img_like, Image.Image):
        img = img_like
    else:
        img = Image.fromarray(np.array(img_like))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _load_npy_from_bytes(buf_bytes: bytes) -> np.ndarray:
    with io.BytesIO(buf_bytes) as bio:
        arr = np.load(bio, allow_pickle=False)
    return arr