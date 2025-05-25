import logging
import time
from typing import Dict, Tuple, Union
from urllib.parse import urlparse

import websockets.sync.client
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "localhost", port: int = 8000, address: Union[str, None] = None) -> None:
        if address is None:
            address = f"ws://{host}:{port}"
        else:
            if not address.startswith(("ws://", "wss://", "http://", "https://")):
                address = f"ws://{address}"

        parsed_url = urlparse(address)

        scheme = parsed_url.scheme
        hostname = parsed_url.hostname
        port = parsed_url.port

        if hostname is None:
            raise ValueError(f"Could not extract hostname from address: {address}")

        ws_scheme = "ws"
        if scheme in ["https", "wss"]:
            ws_scheme = "wss"
            if port is None:
                port = 443
        elif scheme in ["http", "ws"]:
            ws_scheme = "ws"
            if port is None:
                port = 80
        else:
            if port is None:
                print(f"Warning: Unknown scheme '{scheme}' or no scheme, defaulting to port 8000 for ws://")
                port = 8000

        self._uri = f"{ws_scheme}://{hostname}:{port}{parsed_url.path or ''}"

        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
