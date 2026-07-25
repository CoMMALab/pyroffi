# The pyroffi render layer

`pyroffi.viewer` is the official way to look at a scene. It exists so that
"render the world" is **one code path** whether the world is a MuJoCo rollout,
a planned configuration, or a physical cell watched by a perception stack.

```
   WorldSource          .describe() -> WorldDescription   (static: URDF, objects)
        │               .read()     -> WorldState         (dynamic: joints, poses)
        ▼
   SceneView            the viser scene graph pyroffi owns
        ▼
   RenderViewer         server, named viewpoints, capture()
```

The split is the whole point. Moving from simulation to hardware changes the
**source** and nothing else, and two renders taken from different sources are
directly comparable because the same code drew them.

## Sources

| Source | Reads | Use |
|---|---|---|
| `ToolboxSource(session, config)` | a `pyroffi.toolbox.Session` | inspect an IK solution or a planned path — purely kinematic |
| `MuJoCoSource(model, data, ...)` | `MjData` in place | what the physics actually did, not what the plan asked for |
| `CallableSource(description, read)` | anything you supply | the seam for a perception stack |

`WorldState` is name-keyed (`{joint_name: radians}`), following the same interop
contract as the rest of pyroffi's boundaries — a source that silently reorders
joints would draw a picture that is wrong in a way nobody notices.

`MuJoCoSource` takes `mujoco_joint_names` because the two models routinely
disagree on spelling: the Menagerie Franka calls its joints `joint1..7` where
pyroffi's URDF says `panda_joint1..7`. The mapping is an explicit table rather
than a prefix rule, because those names are close enough that a prefix rule
would look right and be wrong.

## Capture is viser, or it is nothing

```python
viewer = RenderViewer(source, port=8080).start()
print(viewer.url)                    # open this in a browser
viewer.wait_for_client(timeout=120)
png = viewer.capture_png("top")
```

viser renders through a connected browser client — `ClientHandle.get_render` is
the only capture API, and there is no server-side renderer. So `capture()`
raises `NoViewerClient` when no tab is open.

That is deliberate. MuJoCo can rasterise offscreen and trimesh can too, and
falling back to either was rejected: a render that came from somewhere else is
not the thing the viewer shows, and an image that silently changes renderer is
worse than no image. If you need a picture, open the URL.

## Viewpoints

`DEFAULT_VIEWPOINTS` gives `front`, `side`, `top`, `iso`, aimed at a tabletop
workspace in front of a base at the origin. They are named rather than ad-hoc so
an agent's second look at a scene is comparable to its first. Pass `None` to
render through whatever camera a human is currently looking through.

## Annotations

```python
viewer.draw_path("plan", ee_path_positions(robot, path, "panda_hand"))
viewer.draw_frame("grasp", Pose.of([0.45, 0.0, 0.12], [0, 0, 1, 0]))
viewer.clear_annotations()
```

A joint-space path is not something anyone can look at; `ee_path_positions` is
the projection that makes it one.
