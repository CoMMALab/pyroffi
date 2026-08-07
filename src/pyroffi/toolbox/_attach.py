"""Session-level attach / detach: moving a scene object onto the robot's body.

This is the toolbox's pick-and-place primitive.  "Attached" and "in the world"
are the *same* object seen from two frames, so the two operations are exact
inverses:

* :func:`attach` reads the object's world pose out of the scene, converts it to
  a grasp transform in the grip link's frame, removes it from the world pool and
  adds it to the session's :class:`~pyroffi.attachments.AttachmentSet`;
* :func:`detach` reads the object's world pose back out of FK at the current
  robot state and returns it to the world pool at that pose.

Doing it this way — rather than keeping a world copy around and masking it —
means an object is never both an obstacle and part of the robot, which is the
bug that makes a grasped object collide with itself.

Geometry conversion
-------------------

The scene stores boxes / spheres / capsules / half-spaces, but a robot collision
model carries exactly one primitive type (capsules per link, or spheres per link
for the spherized model), and an attachment has to concatenate onto that array.
So the scene shape is converted to the model's primitive as a **conservative
bound** — a shape that fully contains the original.  Over-approximating is the
right direction: it can refuse a plan that was actually feasible, but it never
lets the robot drive a carried object through an obstacle.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _bounding_radius(shape: str, params: dict[str, Any]) -> float:
    """Radius of a sphere centred on the object that fully contains it."""
    if shape == "sphere":
        return float(params["radius"])
    if shape == "box":
        half = 0.5 * np.array(
            [float(params["length"]), float(params["width"]), float(params["height"])]
        )
        return float(np.linalg.norm(half))
    if shape == "capsule":
        return float(params["radius"]) + 0.5 * float(params["height"])
    raise ValueError(
        f"a {shape!r} cannot be attached to the robot; attachable shapes are "
        "sphere, box and capsule (a half-space is unbounded)."
    )


def attachment_geom(shape: str, params: dict[str, Any], model: str):
    """Convert a scene shape into the collision model's own primitive type.

    ``model`` is ``"spherized"`` (spheres) or ``"capsule"``.  Both cases use the
    bounding sphere: it is the conservative choice, it is orientation-free (so
    the attachment's own rotation cannot make it wrong), and for the spherized
    model it is the only primitive the array can hold.
    """
    import jax.numpy as jnp

    from ..collision import Capsule, Sphere

    r = _bounding_radius(shape, params)
    if model == "spherized":
        return Sphere.from_center_and_radius(jnp.zeros((1, 3)), jnp.full((1,), r))
    # A zero-height capsule *is* a sphere, so the capsule model gets the same
    # conservative bound without a second code path.
    return Capsule.from_radius_height(
        jnp.full((1,), r), jnp.zeros((1,)), jnp.zeros((1, 3))
    )


def attach(
    session,
    name: str,
    link: str | None = None,
    ignore_links: tuple[str, ...] = (),
    ignore_objects: tuple[str, ...] = (),
    mass: float | None = None,
) -> dict[str, Any]:
    """Move scene object ``name`` onto the robot, grasped at the current state.

    ``ignore_objects`` names world obstacles this carried body is allowed to
    overlap — the surface it was resting on, typically. The bounding sphere is
    a strict over-approximation, so a freshly grasped block penetrates the
    table it is sitting on by construction; without this, the lift-off that
    resolves it validates as invalid and the agent has no way to tell that
    apart from a real fault.

    Returns a report describing the resulting grasp.  Changing what is attached
    is a *topology* change, so it invalidates the jit cache for anything that
    reduces over the collision array — a handful of recompiles across a plan,
    not one per state, which is exactly the trade the attachments design makes.
    """
    import jax.numpy as jnp
    import jaxlie

    from ..attachments import Attachment

    # Check attached-ness first: after a successful attach the object has left
    # the scene pool, so a second call would otherwise report a confusing
    # "no such object" instead of the actual problem.
    if name in session.attachments.names():
        raise ValueError(f"{name!r} is already attached to the robot.")
    obj = session.scene.get_object(name)

    link = link or session.ee_link
    link_index = session.link_index(link)
    ignore_indices = tuple(session.link_index(n) for n in ignore_links)
    # Validated here rather than silently ignored: a typo'd obstacle name would
    # otherwise present as the collision report the caller was trying to mute.
    known = set(session.scene.names())
    unknown = [n for n in ignore_objects if n not in known]
    if unknown:
        raise KeyError(
            f"ignore_objects names no such scene object(s): {unknown}; "
            f"the scene holds {sorted(known)}"
        )

    # T_LB = T_WL(q)^-1 . T_WB, at the current robot state.
    Ts = np.asarray(session.robot.forward_kinematics(session.as_array(session.robot_state)))
    T_WL = jaxlie.SE3(jnp.asarray(Ts[link_index]))
    # Unannotated dtype so x64, when enabled, survives the grasp transform: this
    # pose is what detach() inverts, so any precision lost here shows up as the
    # object landing somewhere other than where the robot was holding it.
    T_WB = jaxlie.SE3(
        jnp.concatenate([jnp.asarray(obj.wxyz), jnp.asarray(obj.position)])
    )
    T_LB = (T_WL.inverse() @ T_WB).wxyz_xyz

    geom = attachment_geom(obj.shape, obj.params, session.collision_model)
    # Scene objects are pure geometry -- the world model carries no masses -- so
    # an attachment is collision-only unless the caller supplies one. Passing a
    # mass additionally folds the object into the robot's dynamics, which is
    # what makes the transport torques account for what is being transported.
    kw: dict[str, Any] = {}
    if mass is not None:
        if mass < 0:
            raise ValueError(f"mass must be non-negative, got {mass}.")
        kw["mass"] = float(mass)
    attachment = Attachment.from_geom(
        geom,
        link_index,
        T_LB,
        name=name,
        ignored_links=ignore_indices,
        **kw,
    )
    session.scene.remove_object(name)
    session.attachments = session.attachments.attach(attachment)
    session._attachment_meta[name] = {
        "shape": obj.shape,
        "params": dict(obj.params),
        "mass": mass,
        "ignore_objects": tuple(ignore_objects),
    }
    session._rebuild_attached_models()

    return {
        "attached": name,
        "link": link,
        "T_link_body": [float(v) for v in np.asarray(T_LB)],
        "bounding_radius": _bounding_radius(obj.shape, obj.params),
        "mass_kg": mass,
        "in_dynamics": mass is not None,
        "ignored_links": list(ignore_links),
        "ignored_objects": list(ignore_objects),
        "scene_version": session.scene.version,
        "note": (
            f"{name!r} is no longer a world obstacle; it now moves with "
            f"{link!r} and is checked against the rest of the robot and the "
            "world. Collision geometry is its bounding sphere (conservative)."
            + (
                f" Its {mass} kg is also folded into the robot's dynamics."
                if mass is not None
                else " Collision only -- pass mass to also load the dynamics."
            )
        ),
    }


def detach(session, name: str) -> dict[str, Any]:
    """Return attached object ``name`` to the world at its current pose."""
    import jax.numpy as jnp
    import jaxlie

    if name not in session.attachments.names():
        raise KeyError(
            f"{name!r} is not attached; attached objects are "
            f"{list(session.attachments.names())}"
        )
    a = session.attachments.attachments[session.attachments.index_of(name)]
    meta = session._attachment_meta.pop(name)

    Ts = np.asarray(session.robot.forward_kinematics(session.as_array(session.robot_state)))
    T_WB = jaxlie.SE3(jnp.asarray(Ts[a.parent_link_index])) @ jaxlie.SE3(a.T_parent_body)
    position = np.asarray(T_WB.translation(), dtype=np.float64)
    wxyz = np.asarray(T_WB.rotation().wxyz, dtype=np.float64)

    session.attachments = session.attachments.detach(name)
    session.scene.add_object(
        name, meta["shape"], position=position, wxyz=wxyz, params=meta["params"]
    )
    session._rebuild_attached_models()

    return {
        "detached": name,
        "shape": meta["shape"],
        "position": [float(v) for v in position],
        "wxyz": [float(v) for v in wxyz],
        "scene_version": session.scene.version,
        "note": (
            f"{name!r} is a world obstacle again, placed where the robot was "
            "holding it."
        ),
    }
