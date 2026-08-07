"""stdio MCP server exposing the execution sandbox.

**This adapter commands things.** That is exactly why it is a separate package
with a separate entry point from :mod:`pyroffi.mcp`, which computes and
simulates but never acts: the scope for the planning server states that if
execution is ever added it goes behind a separately-gated tool in a different
adapter, and that the boundary lives in the code. It does — this file is the
boundary. What it commands is a MuJoCo simulation and nothing else; there is no
transport to hardware here, and adding one would be another adapter again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from loguru import logger

from ._tools import dispatch, list_tool_payloads


class SandboxServer:
    """Routes tool calls to one :class:`~pyroffi.sandbox.Sandbox`."""

    def __init__(self, sandbox) -> None:
        self.sandbox = sandbox

    def get_task(self) -> dict[str, Any]:
        task = self.sandbox.task
        variant = self.sandbox.variant
        return {
            "success": True,
            "task_id": task["task_id"],
            "variant": variant,
            "description": task["description"],
            "conventions": task["conventions"],
            "robot": task["robot"],
            "robot_setup": task["robot_setup"],
            "block_size_m": task["block_size_m"],
            "blocks": task["blocks"],
            "obstacles": task["variants"][variant]["obstacles"],
            "variant_note": task["variants"][variant]["note"],
            "goal": task["goal"],
            "viewer_url": (
                self.sandbox.viewer.url if self.sandbox.viewer is not None else None
            ),
            "planning_server": (
                "Plan with the separate pyroffi MCP server, configured for this "
                "same robot and scene. This server only executes and observes."
            ),
        }

    def observe(self) -> dict[str, Any]:
        return {"success": True, **self.sandbox.observe()}

    def render(self, **kwargs: Any) -> dict[str, Any]:
        from ..viewer import NoViewerClient

        try:
            image = self.sandbox.render(**kwargs)
        except NoViewerClient as exc:
            return {
                "success": False,
                "error": "no_viewer_client",
                "message": str(exc),
                "viewer_url": self.sandbox.viewer.url,
            }
        return {
            "success": True,
            "image_base64": image,
            "mime_type": "image/png",
            "viewpoint": kwargs.get("viewpoint", "current camera"),
        }

    def execute_path(self, **kwargs: Any) -> dict[str, Any]:
        return self.sandbox.execute_path(**kwargs)

    def set_gripper(self, **kwargs: Any) -> dict[str, Any]:
        return self.sandbox.set_gripper(**kwargs)

    def reset(self) -> dict[str, Any]:
        return {"success": True, "observation": self.sandbox.reset()}

    def report(self) -> dict[str, Any]:
        return {"success": True, **self.sandbox.report()}

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            return dispatch(self, name, arguments or {})
        except Exception as exc:
            logger.exception(f"sandbox tool {name} failed")
            return {
                "success": False,
                "error": type(exc).__name__,
                "message": str(exc),
                "tool": name,
            }


async def serve_stdio(server: SandboxServer) -> None:
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    app: Server = Server("pyroffi-sandbox")

    @app.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [types.Tool(**payload) for payload in list_tool_payloads()]

    @app.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None):
        # The simulation is single mutable state and motions take real time, so
        # calls are serialised onto a worker thread rather than run concurrently.
        result = await asyncio.to_thread(server.call, name, arguments or {})
        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pyroffi-sandbox",
        description="MCP server exposing a MuJoCo execution sandbox (simulation only).",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Path to the task specification JSON (see examples/tasks/).",
    )
    parser.add_argument("--variant", default="wall")
    parser.add_argument(
        "--viewer-port", type=int, default=8080, dest="viewer_port",
        help="Port for the viser viewer. Open it in a browser: render() captures "
             "through a connected client and fails without one.",
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--fast", action="store_true",
        help="Drop realtime pacing and step as fast as possible. Useful for "
             "scripted runs; makes the viewer unwatchable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # stdout is the MCP transport; anything else there corrupts the stream.
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("PYROFFI_SANDBOX_LOG", "INFO"))

    from ._sandbox import Sandbox

    with open(args.task) as fh:
        task = json.load(fh)

    sandbox = Sandbox(
        task,
        variant=args.variant,
        viewer_port=args.viewer_port,
        start_viewer=not args.no_viewer,
        realtime=not args.fast,
    )
    server = SandboxServer(sandbox)
    logger.info(
        f"pyroffi-sandbox ready: {len(list_tool_payloads())} tools, "
        f"task={task['task_id']}:{args.variant}"
        + (f", viewer at {sandbox.viewer.url}" if sandbox.viewer else "")
    )
    try:
        asyncio.run(serve_stdio(server))
    finally:
        sandbox.close()


if __name__ == "__main__":  # pragma: no cover
    main()
