#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sensitivity sweep for the guarded popularity-aware switch policy.

Drives ``PopularityTracker`` over a synthetic thrashing trace and
reports how many spurious commits each (tau_merge, tau_margin,
T_cooldown) combination makes.

Usage (from repo root)::

    python benchmarks/dynamic_multi_lora/sensitivity_sweep.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter, deque
from typing import Optional

# ---- path setup (mirrors tests/test_popularity_tracker.py) ----
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from popularity_tracker import (  # noqa: E402
    PopularityTracker,
    PopularityTrackerConfig,
)

# Replaced by PopularityTracker/PopularityTrackerConfig from
# benchmarks.dynamic_multi_lora.popularity_tracker.

# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------

REQUESTS_PER_SEC = 5
TOTAL_SECONDS = 300
FLIP_EVERY_SEC = 30
DOMINANT_SHARE = 0.80

ADAPTER_A = "hot_general"   # initial current_hot
ADAPTER_B = "hot_other"


# ----------------------------------------------------------------
# Trace builder
# ----------------------------------------------------------------

def build_thrashing_trace() -> list[tuple[float, str]]:
    """Synthetic 2-adapter thrashing trace.

    Dominance alternates every FLIP_EVERY_SEC seconds; the dominant
    adapter accounts for DOMINANT_SHARE of requests in each epoch.
    Timestamps are synthetic seconds since t=0.
    """
    reqs_per_epoch = REQUESTS_PER_SEC * FLIP_EVERY_SEC   # 150
    n_epochs = TOTAL_SECONDS // FLIP_EVERY_SEC            # 10

    trace: list[tuple[float, str]] = []
    for epoch in range(n_epochs):
        dominant = ADAPTER_A if epoch % 2 == 0 else ADAPTER_B
        minority = ADAPTER_B if epoch % 2 == 0 else ADAPTER_A
        base_t = float(epoch * FLIP_EVERY_SEC)
        # 4 dominant + 1 minority per group of 5 → 80 % dominant
        for i in range(reqs_per_epoch):
            t = base_t + i / REQUESTS_PER_SEC
            aid = dominant if (i % 5) != 4 else minority
            trace.append((t, aid))
    return trace


# ----------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------

def run_sweep(trace: list[tuple[float, str]]) -> list[dict]:
    grid = [
        # (merge_threshold, switch_margin, cooldown_sec)
        (0.50, 0.05,   5.0),    # very loose
        (0.60, 0.10,  60.0),    # loose
        (0.70, 0.15, 120.0),    # main paper config (defaults)
        (0.70, 0.10, 120.0),    # smaller margin
        (0.70, 0.15,  60.0),    # shorter cooldown
        (0.80, 0.20, 120.0),    # very strict
    ]

    results = []
    for tau_m, tau_g, cd in grid:
        cfg = PopularityTrackerConfig(
            merge_threshold=tau_m,
            switch_margin=tau_g,
            cooldown_sec=cd,
            min_window_size=100,    # match the production default
            window_size=200,        # match the production default
        )
        tracker = PopularityTracker(
            config=cfg,
            initial_hot=ADAPTER_A,
            initial_time=0.0,       # we use synthetic timestamps
        )
        commit_log: list[tuple[float, str, str]] = []
        for t, aid in trace:
            tracker.record(aid)
            target = tracker.decide_switch(now=t)
            if target is not None and target != tracker.current_hot:
                old = tracker.current_hot
                tracker.begin_switch(target)
                # Simulate an instantaneous successful infra switch.
                # Real code waits for a vLLM server to come up; here
                # we are testing only the policy.
                tracker.commit_switch(target, now=t)
                commit_log.append((t, old, target))
        results.append({
            "tau_m": tau_m,
            "tau_g": tau_g,
            "cd": cd,
            "commits": len(commit_log),
            "commit_times": [round(t, 1) for t, _, _ in commit_log],
            "main": (tau_m == 0.70 and tau_g == 0.15 and cd == 120.0),
        })
    return results


# ----------------------------------------------------------------
# Naive baseline (no PopularityTracker; for comparison)
# ----------------------------------------------------------------

class _NaiveBaseline:
    """Most-frequent adapter in the last window_size requests.

    No margin, no cooldown — pure argmax.
    """

    def __init__(self, window_size: int = 200) -> None:
        self._window: deque[str] = deque(maxlen=window_size)
        self._counts: Counter[str] = Counter()

    def record_and_decide(self, adapter_id: str) -> str:
        if len(self._window) == self._window.maxlen:
            evicted = self._window[0]
            self._counts[evicted] -= 1
            if self._counts[evicted] <= 0:
                del self._counts[evicted]
        self._window.append(adapter_id)
        self._counts[adapter_id] += 1
        top, _ = self._counts.most_common(1)[0]
        return top


def run_naive(trace: list[tuple[float, str]]) -> int:
    baseline = _NaiveBaseline(window_size=200)
    switches = 0
    current: Optional[str] = ADAPTER_A
    for _t, aid in trace:
        top = baseline.record_and_decide(aid)
        if top != current:
            current = top
            switches += 1
    return switches


# ----------------------------------------------------------------
# Output
# ----------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    print("| tau_merge | tau_margin | T_cooldown (s) | #switches | commit times (s) |")
    print("|-----------|------------|----------------|-----------|------------------|")
    for r in results:
        label = " (main)" if r["main"] else ""
        commit_str = (
            ", ".join(str(t) for t in r["commit_times"])
            if r["commit_times"] else "—"
        )
        switches_field = f"{r['commits']}{label}"
        print(
            f"| {r['tau_m']:.2f}      "
            f"| {r['tau_g']:.2f}       "
            f"| {r['cd']:>14.1f} "
            f"| {switches_field:<9} "
            f"| {commit_str} |"
        )


def main() -> None:
    trace = build_thrashing_trace()
    n_req = len(trace)

    results = run_sweep(trace)
    print_table(results)

    print()
    naive_switches = run_naive(trace)
    print(f"Naive policy (most-frequent in window, no guards): {naive_switches} switches")

    print()
    print(
        f"Trace: {n_req} requests over {TOTAL_SECONDS}s, two candidates"
        f" oscillating dominance every {FLIP_EVERY_SEC}s."
    )


if __name__ == "__main__":
    main()
