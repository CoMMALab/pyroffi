"""Measure the default tier table for ContractDynamics on this GPU.

    CUDA_VISIBLE_DEVICES=<idle gpu> python tests/calibrate_contract_tiers.py

Calibrates panda (7 DOF), fetch and G1 (29 DOF) and writes the merged winners to
resources/tier_tables/contract_dynamics_sm86.json, which ``tier="auto"`` falls back to for
robots without their own calibration. Run on an otherwise idle GPU.
"""

import json
from pathlib import Path

import yourdfpy

from pyroffi.dynamics._contract_dynamics import _DEFAULT_TABLE, ContractDynamics

ROBOTS = {
    "panda": "resources/panda/panda_spherized__dynbench.urdf",
    "fetch": "resources/fetch/fetch_grid__dynbench.urdf",
    "g1": "resources/g1_description/g1_29dof__dynbench.urdf",
}

merged: dict = {}
for name, path in ROBOTS.items():
    contract = ContractDynamics(yourdfpy.URDF.load(path, load_meshes=False))
    for op, rows in contract.calibrate().items():
        for row in rows:
            row["robot"] = name
            print(f"{name:6} n={row['n']:2} {op:6} B={row['batch']:6} -> {row['winner']:12} {row['ms']}")
        merged.setdefault(op, []).extend(rows)

Path(_DEFAULT_TABLE).parent.mkdir(parents=True, exist_ok=True)
Path(_DEFAULT_TABLE).write_text(json.dumps(merged, indent=1) + "\n")
print(f"wrote {_DEFAULT_TABLE}")
