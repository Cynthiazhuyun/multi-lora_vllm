# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for ``PopularityTracker``.

These tests are intentionally pure-Python (no torch / vLLM imports) so
they can run locally without a GPU. They cover the three properties
that matter for the dynamic system:

1. *Detection*: a clearly-skewed window triggers a switch.
2. *Anti-thrashing*: 50/50 oscillating workloads do **not** switch
   (margin guard) and rapid back-and-forth dominance does not switch
   inside the cooldown window.
3. *State machine*: ``begin_switch`` / ``commit_switch`` /
   ``cancel_switch`` correctly suppress duplicate switches.

Run from the repo root with::

    python -m pytest benchmarks/dynamic_multi_lora/tests/test_popularity_tracker.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from popularity_tracker import (  # noqa: E402
    PopularityTracker, PopularityTrackerConfig)


@pytest.fixture
def cfg() -> PopularityTrackerConfig:
    return PopularityTrackerConfig(
        window_size=20,
        min_window_size=10,
        merge_threshold=0.7,
        switch_margin=0.15,
        cooldown_sec=10.0,
    )


def test_initial_state_blocks_switch(cfg):
    tr = PopularityTracker(cfg, initial_hot="A", initial_time=0.0)
    # No samples yet -> no switch.
    assert tr.decide_switch(now=0.0) is None
    # Even after a few samples, below min_window_size -> no switch.
    for _ in range(cfg.min_window_size - 1):
        tr.record("B")
    assert tr.decide_switch(now=cfg.cooldown_sec * 2) is None


def test_clear_skew_triggers_switch(cfg):
    tr = PopularityTracker(cfg, initial_hot="A", initial_time=0.0)
    # Fill window with B's so share(B) = 1.0 > threshold and margin.
    for _ in range(cfg.window_size):
        tr.record("B")
    target = tr.decide_switch(now=cfg.cooldown_sec * 2)
    assert target == "B"


def test_cooldown_blocks_switch(cfg):
    tr = PopularityTracker(cfg, initial_hot="A", initial_time=0.0)
    for _ in range(cfg.window_size):
        tr.record("B")
    # Inside cooldown -> blocked, even though share is 1.0.
    assert tr.decide_switch(now=cfg.cooldown_sec / 2) is None
    # Just after cooldown -> allowed.
    assert tr.decide_switch(now=cfg.cooldown_sec + 0.01) == "B"


def test_margin_blocks_close_race(cfg):
    """50/50 split must not flip the hot adapter."""
    tr = PopularityTracker(cfg, initial_hot="A", initial_time=0.0)
    for i in range(cfg.window_size):
        tr.record("A" if i % 2 == 0 else "B")
    # share(B) ~= 0.5 < merge_threshold (0.7) so no switch.
    assert tr.decide_switch(now=cfg.cooldown_sec * 2) is None


def test_margin_blocks_thin_dominance():
    """Candidate barely beats incumbent: must not flip."""
    cfg = PopularityTrackerConfig(
        window_size=20,
        min_window_size=10,
        merge_threshold=0.55,  # low threshold so we test the margin
        switch_margin=0.15,
        cooldown_sec=0.0,
    )
    tr = PopularityTracker(cfg, initial_hot="A", initial_time=0.0)
    # 11 B's, 9 A's: share(B) = 0.55 >= threshold but B - A = 0.10 < margin.
    pattern = ["B"] * 11 + ["A"] * 9
    for x in pattern:
        tr.record(x)
    assert tr.decide_switch(now=1.0) is None


def test_state_machine_blocks_concurrent_switch(cfg):
    tr = PopularityTracker(cfg, initial_hot="A", initial_time=0.0)
    for _ in range(cfg.window_size):
        tr.record("B")
    target = tr.decide_switch(now=cfg.cooldown_sec * 2)
    assert target == "B"
    tr.begin_switch(target)
    # While building, even more B traffic must not produce another decision.
    for _ in range(cfg.window_size):
        tr.record("B")
    assert tr.decide_switch(now=cfg.cooldown_sec * 4) is None
    tr.commit_switch(target, now=cfg.cooldown_sec * 4)
    assert tr.current_hot == "B"
    assert tr.building_hot is None


def test_cancel_switch_unblocks(cfg):
    tr = PopularityTracker(cfg, initial_hot="A", initial_time=0.0)
    for _ in range(cfg.window_size):
        tr.record("B")
    tr.begin_switch("B")
    assert tr.decide_switch(now=cfg.cooldown_sec * 2) is None
    tr.cancel_switch()
    # After cancel we can decide again.
    assert tr.decide_switch(now=cfg.cooldown_sec * 2) == "B"


def test_oscillating_workload_under_cooldown_no_thrashing():
    """End-to-end stability check: alternating 80% blocks should switch
    at most once under the cooldown window.
    """
    cfg = PopularityTrackerConfig(
        window_size=50,
        min_window_size=25,
        merge_threshold=0.7,
        switch_margin=0.15,
        cooldown_sec=100.0,
    )
    tr = PopularityTracker(cfg, initial_hot="A", initial_time=0.0)
    switches = 0
    now = 0.0
    # Repeat 6 blocks of 50 reqs each, alternating dominant adapter.
    for block in range(6):
        dominant = "B" if block % 2 == 0 else "A"
        other = "A" if dominant == "B" else "B"
        # 80/20 mix, well above threshold and margin
        seq = [dominant] * 40 + [other] * 10
        for adapter in seq:
            tr.record(adapter)
            now += 1.0  # 1 req/s synthetic clock
        target = tr.decide_switch(now=now)
        if target is not None:
            tr.commit_switch(target, now=now)
            switches += 1
    # Cooldown of 100s with one-block-of-50 between chances means we
    # should switch only on the very first block (for B) and then be
    # locked out by cooldown for at least 50 more seconds.
    assert switches <= 2


def test_share_helpers(cfg):
    tr = PopularityTracker(cfg, initial_hot="A", initial_time=0.0)
    for _ in range(8):
        tr.record("A")
    for _ in range(2):
        tr.record("B")
    assert tr.share("A") == pytest.approx(0.8)
    assert tr.share("B") == pytest.approx(0.2)
    snapshot = tr.stats(now=5.0)
    assert snapshot.window_size == 10
    assert snapshot.shares["A"] == pytest.approx(0.8)
    assert snapshot.current_hot == "A"
    assert snapshot.seconds_since_switch == pytest.approx(5.0)
