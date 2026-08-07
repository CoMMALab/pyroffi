"""MCP adapter for pyroffi — a thin shell over :mod:`pyroffi.toolbox`.

Everything of substance lives in the toolbox layer, which knows nothing about
MCP. This package owns only tool schemas, the description text the model reads,
and handle plumbing.

Run it as ``pyroffi-mcp`` (see ``pyproject.toml``) or::

    python -m pyroffi.mcp --gpu 1 --robot panda_spherized --warmup

Nothing here commands hardware: the tools compute and simulate only.
"""

from ._server import PyroffiServer as PyroffiServer
from ._server import main as main
from ._server import parse_args as parse_args
from ._tools import TOOLS as TOOLS
from ._tools import TOOLS_BY_NAME as TOOLS_BY_NAME
from ._tools import ToolSpec as ToolSpec
from ._tools import dispatch as dispatch
from ._tools import list_tool_payloads as list_tool_payloads
