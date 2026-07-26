"""The pyroffi planning endpoint: one persistent server, many problems.

This is the launcher and the client config for ``pyroffi-mcp``, kept in its own
file because the endpoint is meant to outlive any particular problem. It knows
about a robot, a capacity and a device. It does not know about blocks, towers,
tasks, or the sandbox -- a problem script pours a problem in and takes it back
out (``reset_scene``), and this file is where that contract is written down.

    python pyroffi_endpoint.py serve --gpu 2
    python pyroffi_endpoint.py config --gpu 2 > .mcp.json

Lifetime. Start it once and leave it up. Its cost is entirely front-loaded --
the first tool call pays tens of seconds of XLA compilation and every later one
is milliseconds -- so restarting it per problem throws away the only thing that
makes it fast. Between problems the scene is wiped with ``reset_scene``, which
keeps the compiled functions; ``create_scene`` rebuilds the session and does not.

Hygiene, which is the part that bites. A scene left behind by a dead problem is
invisible to the next one and silently makes its paths invalid, so the rule is
to reset at both ends: on connect, before adding anything, and on finish. A
problem script that runs in-process does this in a ``finally``; an agent driving
the endpoint over MCP has to call the tool itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_DEFAULTS = {
    "robot": "panda_spherized",
    "max_objects": 8,
    "n_timesteps": 32,
}


def open_endpoint(
    gpu: int | None = None,
    robot: str = _DEFAULTS["robot"],
    max_objects: int = _DEFAULTS["max_objects"],
    n_timesteps: int = _DEFAULTS["n_timesteps"],
):
    """Bring the endpoint up in-process: the same object ``pyroffi-mcp`` wraps.

    Takes no task and reads no task file on purpose. Its arguments are
    capacities, not contents -- the moment this function knows about blocks,
    "one endpoint, many problems" stops being true.

    For a script that talks to a *separate* ``pyroffi-mcp`` process this is not
    the entry point; that script gets its scene through MCP tools and this one
    is only for planning inside the same interpreter.
    """
    from pyroffi.toolbox import Session, Toolbox, configure_process

    configure_process(gpu=gpu, x64=True)
    session = Session(robot=robot, max_objects=max_objects, n_timesteps=n_timesteps)
    return Toolbox(session)


def server_argv(
    robot: str = _DEFAULTS["robot"],
    max_objects: int = _DEFAULTS["max_objects"],
    n_timesteps: int = _DEFAULTS["n_timesteps"],
    warmup: bool = True,
) -> list[str]:
    """The arguments ``pyroffi-mcp`` is launched with, in one place."""
    argv = [
        "--robot", robot,
        "--max-objects", str(max_objects),
        "--n-timesteps", str(n_timesteps),
    ]
    if warmup:
        argv.append("--warmup")
    return argv


def server_env(gpu: int | None) -> dict[str, str]:
    """Device and allocator settings, which must be set before JAX is imported.

    ``CUDA_VISIBLE_DEVICES`` rather than the server's ``--gpu``: importing
    pyroffi initialises the JAX CUDA backend, so a flag parsed after that import
    is already too late to choose a device. Preallocation is off because the
    sandbox's MuJoCo EGL context usually shares the card, and the planner will
    otherwise take most of it.
    """
    env = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _bindir() -> str:
    """Where the console scripts live -- checked while the env is still on screen.

    An MCP client launches these itself and does not inherit the conda
    environment, so bare names off ``PATH`` either are not found or resolve to a
    different interpreter. Which makes the *generating* interpreter load-bearing:
    running this from the wrong env writes a well-formed config pointing at
    binaries that do not exist, and since the config is normally redirected into
    a file, the first symptom is "failed to connect" in a client, an hour later.
    """
    bindir = os.path.dirname(sys.executable)
    missing = [
        exe for exe in ("pyroffi-mcp", "pyroffi-sandbox")
        if not os.path.exists(os.path.join(bindir, exe))
    ]
    if missing:
        raise SystemExit(
            f"no pyroffi entry points in {bindir}: {missing}\n"
            f"This is {sys.executable}, which is probably not the env pyroffi is\n"
            "installed into. Activate it and re-run:\n"
            "    conda activate pyroffi\n"
            "If the env is right, the console scripts were never generated "
            "(they postdate the install):\n"
            "    pip install -e . --no-deps"
        )
    return bindir


def mcp_config(
    gpu: int | None = None,
    robot: str = _DEFAULTS["robot"],
    sandbox_task: str | None = None,
    variant: str = "wall",
    viewer_port: int = 8080,
) -> dict:
    """The MCP client entry for the endpoint, plus a sandbox if one is named.

    The two servers are listed together for convenience but do not share a
    lifetime: ``pyroffi`` is persistent and problem-agnostic, ``pyroffi-sandbox``
    is one world for one problem. Omit ``sandbox_task`` and you get the endpoint
    alone, which is the right config for an agent that will meet several
    problems.
    """
    bindir = _bindir()
    servers: dict = {
        "pyroffi": {
            "command": os.path.join(bindir, "pyroffi-mcp"),
            "args": server_argv(robot=robot),
            "env": server_env(gpu),
        }
    }
    if sandbox_task is not None:
        servers["pyroffi-sandbox"] = {
            "command": os.path.join(bindir, "pyroffi-sandbox"),
            "args": ["--task", os.path.abspath(sandbox_task),
                     "--variant", variant, "--viewer-port", str(viewer_port)],
            "env": {"MUJOCO_GL": "egl"},
        }
    return {"mcpServers": servers}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, help_text in (
        ("serve", "Run pyroffi-mcp in the foreground (stdio; for debugging)."),
        ("config", "Print the MCP client entry for the endpoint."),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--gpu", type=int, default=None,
                       help="Physical GPU index. Check nvidia-smi first: a persistent "
                            "endpoint pins its device memory for its whole lifetime.")
        p.add_argument("--robot", default=_DEFAULTS["robot"])
        p.add_argument("--max-objects", type=int, default=_DEFAULTS["max_objects"],
                       dest="max_objects")
        p.add_argument("--n-timesteps", type=int, default=_DEFAULTS["n_timesteps"],
                       dest="n_timesteps")

    sub.choices["config"].add_argument(
        "--sandbox-task", default=None,
        help="Also emit a pyroffi-sandbox entry for this task file. Per-problem; "
             "leave it off for an agent that will see several problems.")
    sub.choices["config"].add_argument("--variant", default="wall")
    sub.choices["config"].add_argument("--viewer-port", type=int, default=8080,
                                       dest="viewer_port")

    args = parser.parse_args(argv)

    if args.cmd == "config":
        print(json.dumps(
            mcp_config(gpu=args.gpu, robot=args.robot, sandbox_task=args.sandbox_task,
                       variant=args.variant, viewer_port=args.viewer_port),
            indent=2,
        ))
        return 0

    # exec rather than import-and-call: the device has to be chosen in the
    # environment before JAX is imported, and by the time this process could
    # import pyroffi.mcp it would already have initialised a backend on the
    # wrong card.
    exe = os.path.join(_bindir(), "pyroffi-mcp")
    os.execve(
        exe,
        [exe, *server_argv(robot=args.robot, max_objects=args.max_objects,
                           n_timesteps=args.n_timesteps)],
        {**os.environ, **server_env(args.gpu)},
    )


if __name__ == "__main__":
    raise SystemExit(main())
