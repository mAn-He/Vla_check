"""Common policy-server skeleton speaking the openpi websocket protocol.

Protocol (verified against openpi sources, see DOC_TRACE.md):
  - Transport: WebSocket, `compression=None`, `max_size=None`.
  - Encoding: msgpack + numpy extension (common/msgpack_numpy.py).
  - Handshake: on connect the SERVER sends a metadata dict first.
  - Request:  one msgpack message per inference call. For this harness the
    observation dict is (client/run_eval.py builds it):
        {
          "observation/image":       uint8 (H, W, 3)  agentview, rotated 180°,
          "observation/wrist_image": uint8 (H, W, 3)  eye-in-hand, rotated 180°,
          "observation/state":       float (8,)  [eef_pos(3), eef axis-angle(3),
                                                  gripper_qpos(2)],
          "prompt":                  str,
        }
  - Response: {"actions": float (T, 7)} — an ENV-READY action chunk (the client
    steps the LIBERO env with each row as-is; any gripper-sign convention fixes
    happen server-side). The server adds a "server_timing" dict like openpi's.
  - Errors: the server sends a plain-text traceback (a str frame) and closes
    with an internal-error code; a str frame therefore signals a server error
    to the client.
  - Health: HTTP GET /healthz on the same port returns 200 "OK" (same as
    openpi's WebsocketPolicyServer).

Model-specific servers subclass PolicyAdapter and implement exactly two
methods: load_model() and predict(obs) (see serve_random.py for the smallest
example).

Structure mirrors openpi's server implementation (Apache-2.0):
https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/serving/websocket_policy_server.py
"""

import abc
import asyncio
import http
import logging
import pathlib
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import websockets.asyncio.server as _server
import websockets.frames

from common import msgpack_numpy

logger = logging.getLogger(__name__)


class PolicyAdapter(abc.ABC):
    """The only two things that differ between model servers."""

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load the model / connect to the backend. Called once before serving."""

    @abc.abstractmethod
    def predict(self, obs: dict) -> dict:
        """Map one observation dict to {"actions": (T, 7) env-ready chunk}."""

    def metadata(self) -> dict:
        """Sent to the client on connect (openpi handshake)."""
        return {"model": type(self).__name__}


class WebsocketPolicyServer:
    """Mirror of openpi's WebsocketPolicyServer around a PolicyAdapter."""

    def __init__(self, adapter: PolicyAdapter, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._adapter = adapter
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        logger.info("Serving %s on ws://%s:%s", type(self._adapter).__name__, self._host, self._port)
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        # openpi handshake: metadata first.
        await websocket.send(packer.pack(self._adapter.metadata()))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._adapter.predict(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                # Same error contract as openpi: text traceback, then close.
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def run(adapter: PolicyAdapter, host: str = "0.0.0.0", port: int = 8000) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    adapter.load_model()
    WebsocketPolicyServer(adapter, host=host, port=port).serve_forever()
