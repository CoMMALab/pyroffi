"""TensorBoard logging for DiffTORI runs.

Named ``tblog`` rather than ``logging`` so it cannot shadow the stdlib module.
Uses ``tensorboardX``, which is already in the ``pyroffi`` env and needs neither
TensorFlow nor PyTorch to *write* event files.  (Reading them back needs the
``tensorboard`` package; see the README.)

A run is a directory under ``runs/``:

    runs/<name>-<YYYYmmdd-HHMMSS>/
        events.out.tfevents.*   scalars
        config.json             the full resolved config + git commit

The config dump is the point of the directory layout: a scalar curve you cannot
tie back to the exact hyperparameters and commit that produced it is not a
result.  ``Logger`` writes it before the first training step.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

__all__ = ["Logger", "flatten_config"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def flatten_config(cfg: Any, prefix: str = "") -> dict[str, Any]:
    """Dataclass (possibly nested) -> flat ``{"solver.n_iters": 100, ...}``."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        key = f"{prefix}{f.name}"
        if dataclasses.is_dataclass(v):
            out.update(flatten_config(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


class Logger:
    """Scalar logger with a run directory; a no-op when ``enabled=False``.

    Metrics are accumulated and flushed on ``log()``, so the caller can log
    every step cheaply and still keep the event file small by passing
    ``flush_every``.
    """

    def __init__(
        self,
        name: str = "difftori_il",
        root: str | Path = "runs",
        config: Any = None,
        enabled: bool = True,
        flush_every: int = 50,
    ):
        self.enabled = enabled
        self.flush_every = flush_every
        self.start = time.time()
        self.writer = None
        self.dir = None
        if not enabled:
            return

        from tensorboardX import SummaryWriter

        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / f"{name}-{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(logdir=str(self.dir))

        payload: dict[str, Any] = {"run": name, "git_commit": _git_commit(),
                                   "started": stamp}
        if config is not None:
            payload["config"] = {k: _jsonable(v)
                                 for k, v in flatten_config(config).items()}
        (self.dir / "config.json").write_text(json.dumps(payload, indent=2))
        # hparams also go into the event file so runs are comparable in the UI.
        if config is not None:
            try:
                self.writer.add_hparams(
                    {k: v for k, v in payload["config"].items()
                     if isinstance(v, (int, float, str, bool))},
                    {"hparam/placeholder": 0.0})
            except Exception:
                pass   # hparams are a convenience; never fail a run over them

    def log(self, step: int, metrics: Mapping[str, Any],
            prefix: str = "train") -> None:
        if self.writer is None:
            return
        for k, v in metrics.items():
            self.writer.add_scalar(f"{prefix}/{k}", float(v), step)
        self.writer.add_scalar("time/elapsed_s", time.time() - self.start, step)
        if step % self.flush_every == 0:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _jsonable(v: Any) -> Any:
    return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
