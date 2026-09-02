"""Saving and loading policy parameters.

Uses ``flax.serialization`` msgpack rather than orbax: the thing being saved is
one parameter pytree of plain arrays, orbax's directory-per-checkpoint layout
buys nothing here, and a single self-contained ``.msgpack`` next to the run's
``config.json`` is what makes a run reproducible months later.

A run writes three kinds of checkpoint into its own directory:

    params_best.msgpack    lowest validation reconstruction seen so far
    params_step_*.msgpack  periodic, so a killed run is not a total loss
    params_final.msgpack   whatever the last step produced

``params_best`` is the one to evaluate with.  The final step of a cosine
schedule is not automatically the best model, and on this task validation
reconstruction is still moving late in training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flax import serialization

__all__ = ["save_params", "load_params", "latest_checkpoint"]


def save_params(path: str | Path, params: Any) -> Path:
    """Write a parameter pytree; returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file and move it into place: a checkpoint half-written
    # when a run is killed is worse than no checkpoint, because it looks fine
    # until you try to load it.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(serialization.to_bytes(params))
    tmp.replace(path)
    return path


def load_params(path: str | Path, template: Any) -> Any:
    """Load into the structure of ``template`` (e.g. freshly ``init``-ed params)."""
    return serialization.from_bytes(template, Path(path).read_bytes())


def latest_checkpoint(run_dir: str | Path, prefer_best: bool = True) -> Path | None:
    """``params_best`` if present, else the highest-numbered periodic, else final."""
    run_dir = Path(run_dir)
    if prefer_best and (run_dir / "params_best.msgpack").exists():
        return run_dir / "params_best.msgpack"
    periodic = sorted(run_dir.glob("params_step_*.msgpack"),
                      key=lambda p: int(p.stem.split("_")[-1]))
    if periodic:
        return periodic[-1]
    final = run_dir / "params_final.msgpack"
    return final if final.exists() else None
