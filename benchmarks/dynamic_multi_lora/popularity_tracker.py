# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Online popularity tracker for the dynamic multi-LoRA system.

Goal
----
Decide *when* the dynamic router should switch the currently fused
("hot") adapter, based on a sliding window of recent request adapter
IDs, while remaining robust against thrashing.

Design
------
The tracker is intentionally pure-Python and side-effect free so that
it is easy to unit-test and reason about. It has no dependency on
torch, vLLM, or any IO. Wall-clock time is injected via ``now`` to
make tests deterministic.

A switch from ``current_hot`` to a candidate ``c`` is allowed only if
*all* of the following hold:

* The window has at least ``min_window_size`` samples.
* No switch is currently in progress (``building_hot`` is ``None``).
* ``c != current_hot``.
* ``share(c) >= merge_threshold``.
* ``share(c) - share(current_hot) >= switch_margin``.
* ``now - last_switch_time >= cooldown_sec``.

The first three guard against premature/concurrent decisions. The next
two enforce that the candidate is truly dominant and meaningfully
ahead of the incumbent (not a 51% / 49% jitter). The cooldown adds
hysteresis to prevent oscillation when a workload flips back and
forth.

Anti-thrashing rationale
------------------------
Pure ``argmax`` over a sliding window is unstable: under a 50/50
oscillating workload it switches every other request. The combination
of (margin, cooldown) gives the tracker the same stability that LRU
caches get from "must be evicted N times before re-admit" policies,
without needing any global coordination.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PopularityTrackerConfig:
    """Tunable knobs for the popularity-aware switch policy."""

    window_size: int = 200
    min_window_size: int = 100
    merge_threshold: float = 0.70
    switch_margin: float = 0.15
    cooldown_sec: float = 120.0

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if not (0 < self.min_window_size <= self.window_size):
            raise ValueError(
                "min_window_size must be in (0, window_size]")
        if not (0.0 < self.merge_threshold <= 1.0):
            raise ValueError("merge_threshold must be in (0, 1]")
        if not (0.0 <= self.switch_margin < 1.0):
            raise ValueError("switch_margin must be in [0, 1)")
        if self.cooldown_sec < 0:
            raise ValueError("cooldown_sec must be non-negative")


@dataclass
class TrackerStats:
    window_size: int
    counts: dict[str, int]
    shares: dict[str, float]
    current_hot: Optional[str]
    building_hot: Optional[str]
    last_switch_time: float
    seconds_since_switch: float = field(default=float("inf"))


class PopularityTracker:
    """Sliding-window popularity tracker with anti-thrashing guards."""

    def __init__(
        self,
        config: Optional[PopularityTrackerConfig] = None,
        *,
        initial_hot: Optional[str] = None,
        initial_time: Optional[float] = None,
    ) -> None:
        self.config = config or PopularityTrackerConfig()
        self._window: deque[str] = deque(maxlen=self.config.window_size)
        self._counts: Counter[str] = Counter()
        self._current_hot: Optional[str] = initial_hot
        self._building_hot: Optional[str] = None
        # Treat startup as if we just switched: the cooldown protects
        # against immediately switching off the bootstrap profile.
        self._last_switch_time: float = (initial_time
                                         if initial_time is not None else
                                         time.time())

    @property
    def current_hot(self) -> Optional[str]:
        return self._current_hot

    @property
    def building_hot(self) -> Optional[str]:
        return self._building_hot

    @property
    def last_switch_time(self) -> float:
        return self._last_switch_time

    def record(self, adapter_id: str) -> None:
        """Record an adapter request in the sliding window."""
        if len(self._window) == self._window.maxlen:
            evicted = self._window[0]
            self._counts[evicted] -= 1
            if self._counts[evicted] <= 0:
                del self._counts[evicted]
        self._window.append(adapter_id)
        self._counts[adapter_id] += 1

    def share(self, adapter_id: str) -> float:
        if not self._window:
            return 0.0
        return self._counts.get(adapter_id, 0) / len(self._window)

    def stats(self, *, now: Optional[float] = None) -> TrackerStats:
        now = now if now is not None else time.time()
        size = len(self._window)
        counts = dict(self._counts)
        shares = ({k: c / size
                   for k, c in counts.items()} if size else {})
        return TrackerStats(
            window_size=size,
            counts=counts,
            shares=shares,
            current_hot=self._current_hot,
            building_hot=self._building_hot,
            last_switch_time=self._last_switch_time,
            seconds_since_switch=max(0.0, now - self._last_switch_time),
        )

    def decide_switch(self, *, now: Optional[float] = None) -> Optional[str]:
        """Return the candidate to switch to, or None if no switch is due."""
        now = now if now is not None else time.time()
        cfg = self.config
        if self._building_hot is not None:
            return None
        if len(self._window) < cfg.min_window_size:
            return None
        if (now - self._last_switch_time) < cfg.cooldown_sec:
            return None

        candidate, count = self._counts.most_common(1)[0]
        candidate_share = count / len(self._window)
        if candidate == self._current_hot:
            return None
        if candidate_share < cfg.merge_threshold:
            return None
        current_share = self.share(self._current_hot) if self._current_hot \
            else 0.0
        if (candidate_share - current_share) < cfg.switch_margin:
            return None
        return candidate

    def begin_switch(self, target: str) -> None:
        """Mark that we have started preparing/serving a switch to ``target``."""
        if self._building_hot is not None and self._building_hot != target:
            raise RuntimeError(
                f"Cannot begin switch to {target!r}; already preparing "
                f"{self._building_hot!r}")
        self._building_hot = target

    def commit_switch(self, target: str, *,
                      now: Optional[float] = None) -> None:
        """Atomically promote ``target`` to be the current hot."""
        now = now if now is not None else time.time()
        self._current_hot = target
        self._building_hot = None
        self._last_switch_time = now

    def cancel_switch(self) -> None:
        """Drop a pending switch (e.g. on health-check failure)."""
        self._building_hot = None
