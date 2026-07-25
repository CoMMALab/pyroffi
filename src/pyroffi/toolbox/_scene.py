"""Named, mutable obstacle scene over fixed-capacity padded arrays.

XLA specialises on shape, so object *count* cannot be allowed to reach the
kernels: a request with 41 obstacles after one with 40 would recompile
everything.  Instead each shape family gets a pool of fixed capacity, sized
once at scene creation, with unused slots *parked* far outside the workspace.

Parking rather than masking is deliberate: a parked slot is a real geometry at
``(1e4, 1e4, 1e4)`` with a sub-millimetre size, so every signed distance to it
is enormous and positive.  Min-distance and collision-margin reductions are
therefore correct with no mask arithmetic in the kernel; the active mask is
needed only on the host, to turn a pool index back into an object name.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import numpy as np

from ..collision import Box, Capsule, CollGeom, HalfSpace, Sphere

Shape = Literal["sphere", "box", "capsule", "halfspace"]
SHAPES: tuple[Shape, ...] = ("sphere", "box", "capsule", "halfspace")

_PARK_POSITION = (1.0e4, 1.0e4, 1.0e4)
"""Where inactive pool slots live: far enough that every distance to them
dominates any real clearance, near enough to stay well inside float32 range."""
_PARK_SIZE = 1.0e-4
_IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)

_REQUIRED_PARAMS: dict[Shape, tuple[str, ...]] = {
    "sphere": ("radius",),
    "box": ("length", "width", "height"),
    "capsule": ("radius", "height"),
    "halfspace": ("normal",),
}


@dataclasses.dataclass
class SceneObject:
    """One named obstacle. Poses are world-frame, metres, quaternion ``wxyz``."""

    name: str
    shape: Shape
    position: np.ndarray
    wxyz: np.ndarray
    params: dict[str, Any]
    slot: int
    """Index into this shape's pool. Stable for the object's lifetime."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "position": [float(v) for v in self.position],
            "wxyz": [float(v) for v in self.wxyz],
            "params": {
                k: ([float(x) for x in v] if isinstance(v, (list, tuple, np.ndarray))
                    else float(v))
                for k, v in self.params.items()
            },
        }


class Scene:
    """A mutable named obstacle set that presents a shape-static face to XLA.

    ``world_geoms()`` returns the same tuple of pytree *shapes* for the whole
    life of the scene, so adding or moving an object changes array values only
    and never triggers a retrace.
    """

    def __init__(self, max_objects: int = 16, ground_plane: bool = True) -> None:
        if max_objects < 1:
            raise ValueError("max_objects must be >= 1")
        self.max_objects = int(max_objects)
        self._objects: dict[str, SceneObject] = {}
        self._free: dict[Shape, list[int]] = {
            shape: list(range(self.max_objects)) for shape in SHAPES
        }
        self.version = 0
        self._geoms_cache: tuple[CollGeom, ...] | None = None
        self._cache_version = -1

        if ground_plane:
            self.add_object(
                "ground",
                "halfspace",
                position=(0.0, 0.0, 0.0),
                params={"normal": (0.0, 0.0, 1.0)},
            )

    # ── mutation ──────────────────────────────────────────────────────────

    def add_object(
        self,
        name: str,
        shape: str,
        position: Any = (0.0, 0.0, 0.0),
        wxyz: Any = _IDENTITY_WXYZ,
        params: dict[str, Any] | None = None,
    ) -> SceneObject:
        """Add (or replace) a named obstacle.

        Replacing an existing name reuses its slot, so an object can be moved
        without churning the pool layout.
        """
        if shape not in SHAPES:
            raise ValueError(f"unknown shape {shape!r}; expected one of {SHAPES}")
        shape_t: Shape = shape  # type: ignore[assignment]
        params = dict(params or {})
        missing = [p for p in _REQUIRED_PARAMS[shape_t] if p not in params]
        if missing:
            raise ValueError(
                f"shape {shape!r} requires params {_REQUIRED_PARAMS[shape_t]}; "
                f"missing {missing}"
            )

        existing = self._objects.get(name)
        if existing is not None:
            if existing.shape != shape_t:
                # A different family means a different pool; give the old slot back.
                self._free[existing.shape].append(existing.slot)
                self._free[existing.shape].sort()
                slot = self._take_slot(shape_t)
            else:
                slot = existing.slot
        else:
            slot = self._take_slot(shape_t)

        obj = SceneObject(
            name=name,
            shape=shape_t,
            position=np.asarray(position, dtype=np.float64).reshape(3),
            wxyz=np.asarray(wxyz, dtype=np.float64).reshape(4),
            params=params,
            slot=slot,
        )
        self._objects[name] = obj
        self.version += 1
        return obj

    def remove_object(self, name: str) -> None:
        try:
            obj = self._objects.pop(name)
        except KeyError:
            raise KeyError(
                f"no object named {name!r} in the scene; have {sorted(self._objects)}"
            ) from None
        self._free[obj.shape].append(obj.slot)
        self._free[obj.shape].sort()
        self.version += 1

    def _take_slot(self, shape: Shape) -> int:
        if not self._free[shape]:
            raise ValueError(
                f"{shape} pool is full ({self.max_objects} slots). Capacity is fixed "
                "at scene creation to keep array shapes static; create a scene with a "
                "larger max_objects."
            )
        return self._free[shape].pop(0)

    # ── inspection ────────────────────────────────────────────────────────

    def objects(self) -> list[SceneObject]:
        return sorted(self._objects.values(), key=lambda o: (o.shape, o.slot))

    def names(self) -> list[str]:
        return sorted(self._objects)

    def __contains__(self, name: object) -> bool:
        return name in self._objects

    def slot_names(self, shape: Shape) -> list[str | None]:
        """Pool-index → object name (``None`` for a parked slot).

        This is the table that turns a kernel's ``(link, obstacle)`` index pair
        back into the named pair an agent can act on.
        """
        out: list[str | None] = [None] * self.max_objects
        for obj in self._objects.values():
            if obj.shape == shape:
                out[obj.slot] = obj.name
        return out

    def geom_names(self) -> list[tuple[Shape, list[str | None]]]:
        """Per-pool slot name tables, in the same order as ``world_geoms()``."""
        return [(shape, self.slot_names(shape)) for shape in SHAPES]

    # ── the shape-static face ─────────────────────────────────────────────

    def world_geoms(self) -> tuple[CollGeom, ...]:
        """One padded geometry per shape family, in ``SHAPES`` order.

        Cached on ``version``: repeated queries against an unchanged scene reuse
        the same pytree, which also keeps jit cache keys stable.
        """
        if self._geoms_cache is not None and self._cache_version == self.version:
            return self._geoms_cache

        n = self.max_objects
        pools: list[CollGeom] = []
        for shape in SHAPES:
            objs = [o for o in self._objects.values() if o.shape == shape]
            position = np.tile(np.asarray(_PARK_POSITION, dtype=np.float64), (n, 1))
            wxyz = np.tile(np.asarray(_IDENTITY_WXYZ, dtype=np.float64), (n, 1))
            for obj in objs:
                position[obj.slot] = obj.position
                wxyz[obj.slot] = obj.wxyz

            if shape == "sphere":
                radius = np.full(n, _PARK_SIZE)
                for obj in objs:
                    radius[obj.slot] = float(obj.params["radius"])
                pools.append(Sphere.from_center_and_radius(position, radius))
            elif shape == "box":
                dims = np.full((n, 3), _PARK_SIZE)
                for obj in objs:
                    dims[obj.slot] = [
                        float(obj.params["length"]),
                        float(obj.params["width"]),
                        float(obj.params["height"]),
                    ]
                pools.append(
                    Box.from_center_and_dimensions(
                        center=position,
                        length=dims[:, 0],
                        width=dims[:, 1],
                        height=dims[:, 2],
                        wxyz=wxyz,
                    )
                )
            elif shape == "capsule":
                radius = np.full(n, _PARK_SIZE)
                height = np.full(n, _PARK_SIZE)
                for obj in objs:
                    radius[obj.slot] = float(obj.params["radius"])
                    height[obj.slot] = float(obj.params["height"])
                pools.append(
                    Capsule.from_radius_height(
                        radius=radius, height=height, position=position, wxyz=wxyz
                    )
                )
            elif shape == "halfspace":
                # A parked half-space sits 1e4 m below the floor facing up, so
                # its signed distance to anything in the workspace is ~ +1e4.
                point = np.tile(
                    np.array([0.0, 0.0, -_PARK_POSITION[2]], dtype=np.float64), (n, 1)
                )
                normal = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (n, 1))
                for obj in objs:
                    point[obj.slot] = obj.position
                    normal[obj.slot] = np.asarray(
                        obj.params["normal"], dtype=np.float64
                    ).reshape(3)
                pools.append(HalfSpace.from_point_and_normal(point, normal))

        self._geoms_cache = tuple(pools)
        self._cache_version = self.version
        return self._geoms_cache

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_version": self.version,
            "max_objects": self.max_objects,
            "n_objects": len(self._objects),
            "frame": "world",
            "units": {"length": "m", "angle": "rad"},
            "quaternion_convention": "wxyz",
            "objects": [o.to_dict() for o in self.objects()],
        }
