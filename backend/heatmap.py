"""
heatmap.py
----------
Accumulates a footfall density grid over a shared floor-plan coordinate
system (0..1 x, 0..1 y), independent of which camera saw the person. This
is the payoff of doing cross-camera re-ID at all: without it, a person
walking from Camera A's view into Camera B's view would count as two
separate footfall events at two disjoint locations. With identity
resolution, their path is one continuous trace across the merged floor
plan.
"""

from __future__ import annotations

import numpy as np


class FootfallHeatmap:
    def __init__(self, grid_size: int = 40, decay: float = 0.0):
        """
        grid_size: resolution of the accumulation grid (grid_size x grid_size)
        decay: per-update multiplicative decay (0.0 = no decay, permanent
               accumulation; set e.g. 0.995 for a "recent activity" view
               that fades old traffic instead of an all-time total)
        """
        self.grid_size = grid_size
        self.decay = decay
        self.grid = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.total_events = 0

    def record(self, x: float, y: float, weight: float = 1.0):
        """x, y are normalized floor-plan coordinates in [0, 1]."""
        if self.decay:
            self.grid *= self.decay
        gx = min(self.grid_size - 1, max(0, int(x * self.grid_size)))
        gy = min(self.grid_size - 1, max(0, int(y * self.grid_size)))
        # a small Gaussian-ish splat instead of a single hard pixel, so the
        # rendered heatmap reads as continuous density rather than noise
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    falloff = 1.0 if (dx == 0 and dy == 0) else 0.35
                    self.grid[ny, nx] += weight * falloff
        self.total_events += 1

    def normalized(self) -> np.ndarray:
        peak = self.grid.max()
        if peak <= 0:
            return self.grid
        return self.grid / peak

    def as_list(self) -> list:
        """JSON-serializable normalized grid for the frontend canvas."""
        return self.normalized().round(4).tolist()
