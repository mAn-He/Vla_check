"""Websocket policy client (openpi protocol).

Prefers the official `openpi-client` package when it is installed (it is, in
the venv built by setup/10_libero.sh); otherwise falls back to a bundled
implementation that mirrors openpi's WebsocketClientPolicy wire behavior
byte-for-byte (verified against
packages/openpi-client/src/openpi_client/websocket_client_policy.py):
connect with compression=None/max_size=None, receive the server metadata dict
first, then send one msgpack_numpy-packed observation per infer() and receive
one packed action dict back; a str frame from the server is an error.
"""

import logging
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

try:  # official client first
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    logger.debug("using openpi_client.WebsocketClientPolicy")
except ImportError:  # bundled fallback, same wire format
    import websockets.sync.client

    from common import msgpack_numpy

    class WebsocketClientPolicy:  # noqa: D101 - mirrors the openpi class
        def __init__(self, host: str = "0.0.0.0", port: int | None = None, api_key: str | None = None) -> None:
            if host.startswith("ws"):
                self._uri = host
            else:
                self._uri = f"ws://{host}"
            if port is not None:
                self._uri += f":{port}"
            self._packer = msgpack_numpy.Packer()
            self._api_key = api_key
            self._ws, self._server_metadata = self._wait_for_server()

        def get_server_metadata(self) -> dict:
            return self._server_metadata

        def _wait_for_server(self):
            logger.info("Waiting for server at %s...", self._uri)
            while True:
                try:
                    headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                    conn = websockets.sync.client.connect(
                        self._uri, compression=None, max_size=None, additional_headers=headers
                    )
                    metadata = msgpack_numpy.unpackb(conn.recv())
                    return conn, metadata
                except ConnectionRefusedError:
                    logger.info("Still waiting for server...")
                    time.sleep(5)

        def infer(self, obs: dict) -> dict:
            data = self._packer.pack(obs)
            self._ws.send(data)
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"Error in inference server:\n{response}")
            return msgpack_numpy.unpackb(response)

        def reset(self) -> None:
            pass
