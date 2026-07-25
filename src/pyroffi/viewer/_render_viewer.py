"""The viewer: a viser server bound to a :class:`WorldSource`, with capture.

This is pyroffi's official rendering entry point. It exists so that "look at
the scene" is one code path regardless of where the scene came from — a
MuJoCo rollout, a planned configuration, or a perception stack watching a real
cell. Swapping the source swaps the world; nothing else moves, and two renders
taken from different sources are directly comparable because the same code
drew them.

**Capture is viser, or it is nothing.** :meth:`RenderViewer.capture` renders
through a connected browser client and raises :class:`NoViewerClient` when
there is none. There is a tempting fallback here — MuJoCo can rasterise
offscreen, trimesh can too — and it is deliberately not taken: a render that
came from somewhere else is not the thing the viewer shows, and an image that
silently changes renderer is worse than no image. Open the URL.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from loguru import logger

from ._scene_view import SceneView
from ._world import Pose, WorldSource, WorldState


class NoViewerClient(RuntimeError):
    """Raised by :meth:`RenderViewer.capture` when no browser client is connected.

    Not a failure of the thing being rendered — the scene is fine, nobody is
    looking at it. Open the viewer URL (or call
    :meth:`RenderViewer.wait_for_client`) and capture again.
    """


@dataclasses.dataclass(frozen=True)
class Viewpoint:
    """A named camera pose, so 'show me the top view' is repeatable."""

    position: tuple[float, float, float]
    look_at: tuple[float, float, float] = (0.4, 0.0, 0.15)
    fov: float = 0.9
    """Vertical field of view, radians."""

    def wxyz(self) -> np.ndarray:
        """Orientation looking from ``position`` toward ``look_at``, +z up."""
        forward = np.asarray(self.look_at, dtype=np.float64) - np.asarray(
            self.position, dtype=np.float64
        )
        n = np.linalg.norm(forward)
        forward = forward / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(float(forward @ world_up)) > 0.999:      # looking straight down
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        # viser cameras look down -z with +y up, in the usual OpenGL convention.
        R = np.stack([right, up, -forward], axis=1)
        return _matrix_to_wxyz(R)


DEFAULT_VIEWPOINTS: dict[str, Viewpoint] = {
    "front": Viewpoint(position=(1.5, 0.0, 0.7)),
    "side": Viewpoint(position=(0.4, -1.4, 0.7)),
    "top": Viewpoint(position=(0.45, 0.0, 1.6), look_at=(0.45, 0.0, 0.0)),
    "iso": Viewpoint(position=(1.1, -1.0, 0.9)),
}
"""Standard viewpoints, aimed at a tabletop workspace in front of a base at the
origin. Named rather than ad-hoc so an agent's second look at a scene is
comparable to its first."""


class RenderViewer:
    """A viser server rendering whatever a :class:`WorldSource` reports.

    Typical use::

        viewer = RenderViewer(MuJoCoSource(model, data, joint_names, ...))
        viewer.start()                        # background thread, non-blocking
        print(viewer.url)                     # open this in a browser
        viewer.wait_for_client(timeout=120)   # capture needs one
        png = viewer.capture_png("top")

    The polling loop reads the source at ``rate_hz`` and pushes into the scene,
    so a simulator being stepped on another thread is followed with no explicit
    plumbing. Reads happen on the viewer's thread: a source's ``read`` must be
    cheap and must not block.
    """

    def __init__(
        self,
        source: WorldSource,
        port: int = 8080,
        rate_hz: float = 30.0,
        show_collision_meshes: bool = False,
        viewpoints: Mapping[str, Viewpoint] | None = None,
        server: Any = None,
    ) -> None:
        import viser

        self.source = source
        self.rate_hz = float(rate_hz)
        self.viewpoints = dict(viewpoints or DEFAULT_VIEWPOINTS)
        self._server = server or viser.ViserServer(port=port)
        self._port = port
        self.view = SceneView(
            self._server, source.describe(), show_collision_meshes=show_collision_meshes
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_state: WorldState | None = None
        self._observers: list[Callable[[WorldState], None]] = []

        self.refresh()

    # ── lifecycle ────────────────────────────────────────────────────────

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}"

    @property
    def server(self):
        return self._server

    def start(self) -> "RenderViewer":
        """Begin polling the source on a background thread."""
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="pyroffi-render", daemon=True
        )
        self._thread.start()
        logger.info(f"pyroffi viewer at {self.url} (open it; capture needs a client)")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "RenderViewer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def _loop(self) -> None:
        period = 1.0 / max(self.rate_hz, 1e-3)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self.refresh()
            except Exception as exc:  # a bad frame must not kill the viewer
                logger.warning(f"render frame failed: {exc}")
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def refresh(self) -> WorldState:
        """Read the source once and push it into the scene."""
        state = self.source.read()
        with self._lock:
            self.view.update(state)
            self._last_state = state
        for observer in list(self._observers):
            try:
                observer(state)
            except Exception as exc:
                logger.warning(f"render observer failed: {exc}")
        return state

    def on_state(self, fn: Callable[[WorldState], None]) -> None:
        """Register a callback run on every refreshed state."""
        self._observers.append(fn)

    @property
    def last_state(self) -> WorldState | None:
        return self._last_state

    # ── capture ──────────────────────────────────────────────────────────

    def clients(self) -> list[Any]:
        return list(self._server.get_clients().values())

    def wait_for_client(self, timeout: float = 60.0, poll: float = 0.25) -> bool:
        """Block until a browser connects. Returns whether one did."""
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            if self.clients():
                time.sleep(0.5)      # let the scene finish loading before capture
                return True
            time.sleep(poll)
        return False

    def capture(
        self,
        viewpoint: str | Viewpoint | None = None,
        width: int = 960,
        height: int = 720,
        client_index: int = 0,
    ) -> np.ndarray:
        """Render the current scene through a connected viser client.

        Args:
            viewpoint: a name from :attr:`viewpoints`, an explicit
                :class:`Viewpoint`, or ``None`` to use the client's own camera —
                i.e. whatever a human is currently looking at.
            width, height: image size in pixels.
            client_index: which connected client to render through, when more
                than one browser is open.

        Returns:
            ``(H, W, 3)`` uint8 RGB.

        Raises:
            NoViewerClient: nothing is connected. This is the only failure mode
                that is about the *viewer* rather than the scene, and it is
                raised rather than papered over with a different renderer.
        """
        clients = self.clients()
        if not clients:
            raise NoViewerClient(
                f"no viser client is connected to {self.url}, and capture renders "
                "through the browser — there is no server-side renderer. Open the "
                "URL (or call wait_for_client) and try again."
            )
        if client_index >= len(clients):
            raise IndexError(
                f"client_index {client_index} but only {len(clients)} connected"
            )
        client = clients[client_index]

        vp = self._resolve_viewpoint(viewpoint)
        kwargs: dict[str, Any] = {}
        if vp is not None:
            kwargs = {
                "position": np.asarray(vp.position, dtype=np.float64),
                "wxyz": vp.wxyz(),
                "fov": float(vp.fov),
            }
        return client.get_render(height=int(height), width=int(width), **kwargs)

    def capture_png(self, *args: Any, **kwargs: Any) -> bytes:
        """:meth:`capture`, encoded as PNG bytes."""
        import io

        import imageio.v3 as iio

        image = self.capture(*args, **kwargs)
        buf = io.BytesIO()
        iio.imwrite(buf, np.asarray(image, dtype=np.uint8), extension=".png")
        return buf.getvalue()

    def capture_base64(self, *args: Any, **kwargs: Any) -> str:
        """:meth:`capture_png`, base64-encoded — what an MCP tool returns."""
        import base64

        return base64.b64encode(self.capture_png(*args, **kwargs)).decode("ascii")

    def _resolve_viewpoint(
        self, viewpoint: str | Viewpoint | None
    ) -> Viewpoint | None:
        if viewpoint is None or isinstance(viewpoint, Viewpoint):
            return viewpoint
        try:
            return self.viewpoints[viewpoint]
        except KeyError:
            raise ValueError(
                f"unknown viewpoint {viewpoint!r}; have {sorted(self.viewpoints)}"
            ) from None

    # ── annotations ──────────────────────────────────────────────────────

    def draw_path(self, name: str, positions: np.ndarray, **kwargs: Any) -> None:
        with self._lock:
            self.view.draw_path(name, positions, **kwargs)

    def draw_frame(self, name: str, pose: Pose, **kwargs: Any) -> None:
        with self._lock:
            self.view.draw_frame(name, pose, **kwargs)

    def clear_annotations(self) -> None:
        with self._lock:
            self.view.clear_annotations()


def ee_path_positions(robot, configs: Sequence[np.ndarray], link: str) -> np.ndarray:
    """Cartesian positions of *link* along a joint-space path.

    A convenience for :meth:`RenderViewer.draw_path`: a joint-space path is not
    something anyone can look at, and this is the projection that makes it one.
    """
    import numpy as _np

    idx = robot.links.names.index(link)
    out = []
    for cfg in _np.asarray(configs, dtype=_np.float64):
        fk = _np.asarray(robot.forward_kinematics(cfg), dtype=_np.float64)
        out.append(fk[idx, 4:7])
    return _np.stack(out, axis=0)


def _matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → scalar-first quaternion."""
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            ]
        )
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)
