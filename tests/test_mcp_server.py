"""End-to-end test of the MCP server over a real stdio client session.

Everything else tests the toolbox directly; this checks the adapter itself —
that the process starts, advertises its tools, and answers a realistic call
sequence through the actual MCP transport with JSON-serialisable results.

Kept to cheap tools (no trajopt) so it stays a transport test rather than a
second copy of the integration suite.

Run:
    CUDA_VISIBLE_DEVICES=<free gpu> pytest tests/test_mcp_server.py -q
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

mcp = pytest.importorskip("mcp", reason="pyroffi[mcp] not installed")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

DOWN = [0.0, 0.0, 1.0, 0.0]


def _server_params() -> StdioServerParameters:
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return StdioServerParameters(
        command=sys.executable,
        args=[
            "-m", "pyroffi.mcp",
            "--robot", "panda_spherized",
            "--max-objects", "4",
            "--n-timesteps", "32",
        ],
        env=env,
    )


def _payload(result) -> dict:
    """Unwrap the JSON body of a tool result."""
    assert result.content, "tool returned no content"
    return json.loads(result.content[0].text)


def test_server_session_handles_a_realistic_call_sequence():
    """Driven through asyncio.run so the suite needs no pytest-asyncio plugin."""
    asyncio.run(_realistic_call_sequence())


async def _realistic_call_sequence():
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ── the tool surface ─────────────────────────────────────────
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert {
                "get_capabilities", "add_object", "solve_ik", "solve_ik_batch",
                "check_collision", "check_edge", "validate_path", "import_path",
                "export_path", "optimize_path", "retime", "simulate",
                "explain_failure", "warmup",
            } <= names
            for tool in tools.tools:
                assert tool.description
                assert tool.inputSchema["type"] == "object"

            # ── capabilities ─────────────────────────────────────────────
            caps = _payload(await session.call_tool("get_capabilities", {}))
            assert caps["success"]
            dof = caps["capabilities"]["dof"]
            joint_names = caps["capabilities"]["joint_names"]
            assert dof == len(joint_names)
            assert caps["capabilities"]["quaternion_convention"] == "wxyz"

            # ── scene ────────────────────────────────────────────────────
            added = _payload(
                await session.call_tool(
                    "add_object",
                    {
                        "name": "wall",
                        "shape": "box",
                        "position": [0.5, 0.0, 0.2],
                        "params": {"length": 0.1, "width": 0.4, "height": 0.4},
                    },
                )
            )
            assert added["success"]

            listing = _payload(await session.call_tool("list_objects", {}))
            assert "wall" in [o["name"] for o in listing["objects"]]

            # ── IK, then validation of the straight line between endpoints ─
            ik_a = _payload(
                await session.call_tool(
                    "solve_ik",
                    {"pose": {"position": [0.5, -0.35, 0.35], "wxyz": DOWN},
                     "num_seeds": 64, "collision_free": True},
                )
            )
            assert ik_a["success"], ik_a
            ik_b = _payload(
                await session.call_tool(
                    "solve_ik",
                    {"pose": {"position": [0.5, 0.35, 0.35], "wxyz": DOWN},
                     "num_seeds": 64, "collision_free": True, "seed": 1},
                )
            )
            assert ik_b["success"], ik_b

            edge = _payload(
                await session.call_tool(
                    "check_edge",
                    {"config_a": ik_a["config_id"], "config_b": ik_b["config_id"]},
                )
            )
            assert edge["success"] and not edge["valid"]  # the wall is in the way
            assert edge["first_failure"]["world_collisions"][0]["object"] == "wall"

            # ── round-trip a path through the machine-facing register ────
            cfg_a = _payload(
                await session.call_tool("export_config", {"config_id": ik_a["config_id"]})
            )["joint_values"]
            cfg_b = _payload(
                await session.call_tool("export_config", {"config_id": ik_b["config_id"]})
            )["joint_values"]
            waypoints = [
                {
                    name: cfg_a[name] + (cfg_b[name] - cfg_a[name]) * i / 9.0
                    for name in joint_names
                }
                for i in range(10)
            ]
            imported = _payload(
                await session.call_tool("import_path", {"waypoints": waypoints})
            )
            assert imported["success"]

            validated = _payload(
                await session.call_tool("validate_path", {"path": imported["path_id"]})
            )
            assert validated["success"] and not validated["valid"]

            explained = _payload(
                await session.call_tool(
                    "explain_failure", {"request_id": validated["request_id"]}
                )
            )
            assert explained["cause"] == "path_invalid"

            # ── retime and simulate ──────────────────────────────────────
            timed = _payload(
                await session.call_tool("retime", {"path": imported["path_id"]})
            )
            assert timed["success"] and timed["duration_s"] > 0

            rolled = _payload(
                await session.call_tool(
                    "simulate", {"trajectory": timed["trajectory_id"]}
                )
            )
            assert rolled["success"] and not rolled["diverged"], rolled

            exported = _payload(
                await session.call_tool(
                    "export_path", {"path_id": timed["trajectory_id"]}
                )
            )
            assert len(exported["waypoints"]) == 10
            assert set(exported["waypoints"][0]) == set(joint_names)
            assert "times_s" in exported


def test_tool_errors_come_back_as_structured_payloads():
    """A bad call must not take the warm session down with it."""
    asyncio.run(_tool_errors())


async def _tool_errors():
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            bad = _payload(
                await session.call_tool("check_collision", {"config": "cfg_9999"})
            )
            assert bad["success"] is False
            assert bad["error"] == "KeyError"

            # An xyzw quaternion must be refused, not silently reinterpreted.
            # The schema's enum catches this before the handler runs, so the
            # rejection comes back as a protocol-level error rather than a JSON
            # payload — either way it must not be accepted.
            wrong_quat = await session.call_tool(
                "solve_ik",
                {"pose": {"position": [0.4, 0.0, 0.4], "wxyz": [0, 0, 0, 1],
                          "quaternion_convention": "xyzw"}},
            )
            text = wrong_quat.content[0].text if wrong_quat.content else ""
            if wrong_quat.isError:
                assert "quaternion_convention" in text or "xyzw" in text, text
            else:  # reached the handler; then it must have refused there
                assert json.loads(text)["success"] is False

            # A pose with no position is likewise rejected rather than defaulted.
            no_position = await session.call_tool("solve_ik", {"pose": {"wxyz": DOWN}})
            assert no_position.isError or not json.loads(
                no_position.content[0].text
            )["success"]

            # The session is still alive and serving afterwards.
            caps = _payload(await session.call_tool("get_capabilities", {}))
            assert caps["success"]
