"""stdio MCP server exposing pyroffi's motion primitives.

Compute and simulate only. No tool here commands hardware, and the boundary is
structural rather than a convention: this adapter has no transport to a robot,
and adding execution would mean a separate, separately-gated adapter.

The server is one long-lived process holding a warm session, because that is the
only shape that works: pyroffi's first trajopt call pays tens of seconds of XLA
compilation and its steady state is milliseconds, so a process-per-request design
would pay the compile every time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from typing import Any

from loguru import logger

from ._tools import dispatch, list_tool_payloads


class PyroffiServer:
    """Holds the warm session and routes tool calls to it.

    Delegates unknown attributes to the current :class:`Toolbox`, so the tool
    table can name toolbox methods directly while this class adds only what is
    genuinely server-level (session lifecycle).
    """

    def __init__(
        self,
        *,
        process_config: dict[str, Any] | None = None,
        warmup: bool = False,
        warmup_trajopt: bool = True,
        **session_kwargs: Any,
    ) -> None:
        self._session_kwargs = dict(session_kwargs)
        self._process_config = dict(process_config or {})
        self._warmup = warmup
        self._warmup_trajopt = warmup_trajopt
        self._ready = False
        self._lock = threading.Lock()
        self._session = None
        self._toolbox = None

    # Construction is deferred to the first tool call. Importing JAX, building
    # the session and compiling take far longer than the MCP client's
    # initialisation timeout, so the process must reach the stdio loop first and
    # pay that cost inside a call, where the client is willing to wait.
    def ensure_ready(self) -> None:
        with self._lock:
            if self._ready:
                return
            if self._process_config:
                from pyroffi.toolbox import configure_process

                applied = configure_process(**self._process_config)
                logger.info(f"process configuration: {applied}")
            self._build()
            self._ready = True
            if self._warmup:
                logger.info("warming up (this is the compile cost, paid on purpose) ...")
                result = self._toolbox.warmup(include_trajopt=self._warmup_trajopt)
                logger.info(f"warmup finished in {result['total_seconds']}s")

    @property
    def session(self):
        self.ensure_ready()
        return self._session

    @property
    def toolbox(self):
        self.ensure_ready()
        return self._toolbox

    def _build(self) -> None:
        from pyroffi.toolbox import Session, Toolbox

        self._session = Session(**self._session_kwargs)
        self._toolbox = Toolbox(self._session)

    def recreate_session(
        self,
        robot: str | None = None,
        max_objects: int | None = None,
        n_timesteps: int | None = None,
        collision_model: str | None = None,
    ) -> dict[str, Any]:
        """Rebuild the session, invalidating all handles.

        Device and precision are process-level (they must be pinned before JAX
        initialises its backend) and are deliberately not settable here.
        """
        self.ensure_ready()
        old = self._session
        for key, value in (
            ("robot", robot),
            ("max_objects", max_objects),
            ("n_timesteps", n_timesteps),
            ("collision_model", collision_model),
        ):
            if value is not None:
                self._session_kwargs[key] = value
        if old is not None:
            old.close()
        self._build()
        result = self._toolbox.create_scene_info()
        result["handles_invalidated"] = True
        result["note"] = (
            "all previous config/path handles are gone; gpu and x64 are fixed for "
            "the process and unchanged"
        )
        return result

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this class does not define itself.
        if name.startswith("_"):
            raise AttributeError(name)
        self.ensure_ready()
        toolbox = self.__dict__.get("_toolbox")
        if toolbox is None:
            raise AttributeError(name)
        return getattr(toolbox, name)

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute one tool call, returning a JSON-ready dict.

        Failures become structured error payloads rather than exceptions: an
        agent can act on ``{"error": ...}`` but a transport-level crash just
        loses the session.
        """
        try:
            self.ensure_ready()
            return dispatch(self, name, arguments or {})
        except Exception as exc:
            logger.exception(f"tool {name} failed")
            return {
                "success": False,
                "error": type(exc).__name__,
                "message": str(exc),
                "tool": name,
            }


def _default_session_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "robot": args.robot,
        "max_objects": args.max_objects,
        "n_timesteps": args.n_timesteps,
        "collision_model": args.collision_model,
        "ground_plane": not args.no_ground_plane,
    }


def build_server(args: argparse.Namespace) -> PyroffiServer:
    """Describe the session; nothing heavy happens until the first tool call."""
    return PyroffiServer(
        process_config={"gpu": args.gpu, "x64": not args.float32},
        warmup=args.warmup,
        warmup_trajopt=not args.no_trajopt_warmup,
        **_default_session_kwargs(args),
    )


async def serve_stdio(server: PyroffiServer) -> None:
    """Run the MCP stdio loop."""
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    app: Server = Server("pyroffi")

    @app.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [types.Tool(**payload) for payload in list_tool_payloads()]

    @app.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None):
        # JAX dispatch and the scene are shared mutable state, and a session is
        # logically single-threaded, so calls are serialised onto a worker thread
        # rather than run concurrently.
        result = await asyncio.to_thread(server.call, name, arguments or {})
        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream, write_stream, app.create_initialization_options()
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pyroffi-mcp",
        description="MCP server exposing pyroffi motion primitives (compute only).",
    )
    parser.add_argument(
        "--robot",
        default="panda_spherized",
        help="URDF path, robot_descriptions name, or alias. Default panda_spherized, "
             "whose primitive collision geometry gives usable self-collision results.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Physical GPU index. On a shared box, pick a free one: a long-lived "
             "server pins its device memory for its whole lifetime.",
    )
    parser.add_argument("--max-objects", type=int, default=16, dest="max_objects")
    parser.add_argument("--n-timesteps", type=int, default=64, dest="n_timesteps")
    parser.add_argument(
        "--collision-model",
        default="auto",
        choices=("auto", "capsule", "spherized"),
        dest="collision_model",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Disable float64. Not recommended: several solver paths are "
             "precision-sensitive and quietly give worse results.",
    )
    parser.add_argument("--no-ground-plane", action="store_true")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Compile everything as part of the first tool call, so later calls "
             "do not pay it piecemeal. Cannot happen before serving: the client's "
             "initialisation timeout is far shorter than the compile.",
    )
    parser.add_argument("--no-trajopt-warmup", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # stdout is the MCP transport: anything written there that is not a protocol
    # message corrupts the stream, so logs go to stderr.
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("PYROFFI_MCP_LOG", "INFO"))

    server = build_server(args)
    logger.info(
        f"pyroffi-mcp serving: {len(list_tool_payloads())} tools, robot={args.robot}; "
        f"session builds on the first tool call"
        + (" (with warmup)" if args.warmup else "")
    )
    asyncio.run(serve_stdio(server))


if __name__ == "__main__":  # pragma: no cover
    main()
