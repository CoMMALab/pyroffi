"""Loading the zarr replay buffer into DiffTORI training batches.

Mirrors the authors' ``MetaworldPointcloudDataset`` + ``SequenceSampler``:
a sample is a window of ``n_obs_steps`` observations and the ``horizon``-step
action chunk that follows, drawn from within a single episode.  Their
``pad_before = n_obs_steps - 1`` / ``pad_after = n_action_steps - 1`` edge
padding is reproduced by clamping indices to the episode, which is what their
sampler does.

Normalisation follows their ``LinearNormalizer(mode='limits')``: each dimension
is mapped to ``[-1, 1]`` by the dataset's own min/max.  For actions that matters
twice over -- the inner problem's barrier assumes a unit box.
"""

from __future__ import annotations

import json
from typing import Iterator

import numpy as np

__all__ = ["ReplayBuffer", "LimitsNormalizer", "SequenceDataset", "batches"]


class ReplayBuffer:
    """The arrays written by ``panda_reach.write_zarr`` (or their generator)."""

    def __init__(self, state, action, episode_ends, point_cloud=None, meta=None):
        self.state = np.asarray(state)
        self.action = np.asarray(action)
        self.episode_ends = np.asarray(episode_ends)
        self.point_cloud = point_cloud
        self.meta = meta or {}

    @classmethod
    def load(cls, path: str, with_point_cloud: bool = False) -> "ReplayBuffer":
        import zarr

        root = zarr.open(path, mode="r")
        pc = np.asarray(root["data/point_cloud"]) if with_point_cloud else None
        meta = json.loads(root.attrs.get("difftori", "{}"))
        return cls(np.asarray(root["data/state"]),
                   np.asarray(root["data/action"]),
                   np.asarray(root["meta/episode_ends"]), pc, meta)

    @property
    def n_episodes(self) -> int:
        return len(self.episode_ends)

    def episode_bounds(self, i: int) -> tuple[int, int]:
        start = 0 if i == 0 else int(self.episode_ends[i - 1])
        return start, int(self.episode_ends[i])


class LimitsNormalizer:
    """Per-dimension affine map to ``[-1, 1]`` from the data's own min/max."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray, eps: float = 1e-8):
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)
        self.scale = np.maximum(self.hi - self.lo, eps)

    @classmethod
    def fit(cls, x: np.ndarray) -> "LimitsNormalizer":
        flat = x.reshape(-1, x.shape[-1])
        return cls(flat.min(axis=0), flat.max(axis=0))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.lo) / self.scale - 1.0

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x + 1.0) / 2.0 * self.scale + self.lo


class SequenceDataset:
    """``(obs_window, action_chunk)`` samples, normalised to ``[-1, 1]``.

    Yields ``obs (n_obs_steps, obs_dim)`` and ``action (horizon, action_dim)``.
    Indices are clamped to the episode at both ends, reproducing their padding.
    """

    def __init__(self, buffer: ReplayBuffer, n_obs_steps: int = 2,
                 horizon: int = 4, val_ratio: float = 0.02, seed: int = 42,
                 validation: bool = False):
        self.buf = buffer
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon

        rng = np.random.default_rng(seed)
        is_val = np.zeros(buffer.n_episodes, dtype=bool)
        n_val = max(1, int(round(buffer.n_episodes * val_ratio))) if val_ratio else 0
        if n_val:
            is_val[rng.permutation(buffer.n_episodes)[:n_val]] = True
        keep = np.flatnonzero(is_val if validation else ~is_val)

        self.index = [(ep, t) for ep in keep
                      for t in range(*buffer.episode_bounds(int(ep)))]
        # Normalisers are fit on the *training* split only, and reused for
        # validation via `share_normalizers`.
        self.obs_norm = LimitsNormalizer.fit(buffer.state)
        self.act_norm = LimitsNormalizer.fit(buffer.action)

    def share_normalizers(self, other: "SequenceDataset") -> "SequenceDataset":
        self.obs_norm, self.act_norm = other.obs_norm, other.act_norm
        return self

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        ep, t = self.index[i]
        lo, hi = self.buf.episode_bounds(int(ep))
        obs_idx = np.clip(np.arange(t - self.n_obs_steps + 1, t + 1), lo, hi - 1)
        act_idx = np.clip(np.arange(t, t + self.horizon), lo, hi - 1)
        return (self.obs_norm(self.buf.state[obs_idx]),
                self.act_norm(self.buf.action[act_idx]))


def batches(dataset: SequenceDataset, batch_size: int, seed: int = 0
            ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Infinite shuffled batch iterator; drops the short tail of each epoch."""
    rng = np.random.default_rng(seed)
    n = len(dataset)
    while True:
        order = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            items = [dataset[int(j)] for j in order[i:i + batch_size]]
            yield (np.stack([o for o, _ in items]).astype(np.float32),
                   np.stack([a for _, a in items]).astype(np.float32))
