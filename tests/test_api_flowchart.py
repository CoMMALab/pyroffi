"""tools/api_flowchart.py: a package diffed against itself has no breaks and nothing new."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("api_flowchart", ROOT / "tools/api_flowchart.py")
af = sys.modules["api_flowchart"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(af)


def test_self_diff_is_identity():
    pkg = ROOT / "src/pyroffi"
    api = af.Walker(pkg.parent).module(pkg)
    rows = af.diff(api, af.Walker(pkg.parent).module(pkg))
    # `missing` can still appear: a module __init__ imports that is absent on disk.
    assert rows and {r.status for r in rows} <= {"same", "missing"}


def test_factory_signature_drops_vals():
    api = af.Walker((ROOT / "src/pyroffi").parent).module(ROOT / "src/pyroffi")
    params = api.members["costs"].members["pose_cost"].params
    assert params and "vals" not in params and "robot" in params
