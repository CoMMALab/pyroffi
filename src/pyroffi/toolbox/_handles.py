"""Handle table: server-local opaque ids for configs, paths and trajectories.

Handles exist so agent-facing responses can refer to a joint array without
carrying it. They are deliberately *not* portable — a different MCP server
cannot dereference ``path_7f3a``.  Crossing a server boundary goes through
:mod:`pyroffi.toolbox._exchange` instead.
"""

from __future__ import annotations

import dataclasses
import itertools
import threading
from typing import Any, Iterator, Literal

import numpy as np

HandleKind = Literal["config", "path", "trajectory"]

_PREFIX: dict[str, str] = {
    "config": "cfg",
    "path": "path",
    "trajectory": "traj",
}


@dataclasses.dataclass
class Entry:
    """One stored array plus the metadata needed to report on it later."""

    handle: str
    kind: HandleKind
    values: np.ndarray
    """``(dof,)`` for a config, ``(n_waypoints, dof)`` for a path/trajectory."""
    joint_names: tuple[str, ...]
    """Actuated joint names, positionally aligned with the last axis of ``values``."""
    scene_version: int
    """Scene version at creation time. Lets a stale validation be detected."""
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Free-form provenance: solver used, residual, cost, source request id, ..."""
    times: np.ndarray | None = None
    """Per-waypoint times (s), set only once the path has been retimed."""

    @property
    def dof(self) -> int:
        return int(self.values.shape[-1])

    @property
    def n_waypoints(self) -> int:
        return 1 if self.values.ndim == 1 else int(self.values.shape[0])

    @property
    def is_retimed(self) -> bool:
        return self.times is not None


class HandleTable:
    """Thread-safe insert/lookup table for handles.

    Ids are short and human-readable (``cfg_0001``) rather than uuids: they end
    up in an LLM's context, where a long opaque token is pure cost.
    """

    def __init__(self) -> None:
        self._entries: dict[str, Entry] = {}
        self._counters: dict[str, itertools.count] = {
            kind: itertools.count(1) for kind in _PREFIX
        }
        self._lock = threading.Lock()

    def insert(
        self,
        kind: HandleKind,
        values: np.ndarray,
        joint_names: tuple[str, ...],
        scene_version: int,
        meta: dict[str, Any] | None = None,
        times: np.ndarray | None = None,
    ) -> Entry:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim not in (1, 2):
            raise ValueError(
                f"handle values must be 1-D (config) or 2-D (path), got shape {values.shape}"
            )
        if values.shape[-1] != len(joint_names):
            raise ValueError(
                f"values last axis is {values.shape[-1]} but {len(joint_names)} joint "
                "names were given; joint arrays are always name-keyed at the boundary"
            )
        with self._lock:
            handle = f"{_PREFIX[kind]}_{next(self._counters[kind]):04d}"
            entry = Entry(
                handle=handle,
                kind=kind,
                values=values,
                joint_names=tuple(joint_names),
                scene_version=scene_version,
                meta=dict(meta or {}),
                times=None if times is None else np.asarray(times, dtype=np.float64),
            )
            self._entries[handle] = entry
        return entry

    def get(self, handle: str, kind: HandleKind | None = None) -> Entry:
        try:
            entry = self._entries[handle]
        except KeyError:
            raise KeyError(
                f"unknown handle {handle!r}; handles are server-local and are "
                f"invalidated when the session is recreated. Known: "
                f"{sorted(self._entries)[:8]}{'...' if len(self._entries) > 8 else ''}"
            ) from None
        if kind is not None and entry.kind != kind:
            raise KeyError(f"handle {handle!r} is a {entry.kind}, expected a {kind}")
        return entry

    def drop(self, handle: str) -> None:
        with self._lock:
            self._entries.pop(handle, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __contains__(self, handle: object) -> bool:
        return handle in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Entry]:
        return iter(list(self._entries.values()))
