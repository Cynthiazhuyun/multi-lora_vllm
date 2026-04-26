# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Workload generators for the multi-LoRA benchmarks.

The dynamic-multi-LoRA story rests on three workload patterns:

* :func:`make_stable_skew_schedule` -- the same hot-ratio sweep the
  existing baseline uses. This is the headline number for "static"
  comparisons.

* :func:`make_hot_change_schedule` -- the workload starts with one
  dominant adapter, then *flips* to a different dominant adapter
  partway through. This is the experiment that demonstrates the
  online popularity-aware system actually adapts.

* :func:`make_thrashing_schedule` -- alternating short blocks where
  the dominant adapter switches every ``block_size`` requests. This
  stresses the cooldown/margin guards against unnecessary switching.

All schedules return a list of (adapter_name, prompt_index) pairs
(matching the existing baseline interface), and prompts are produced
by :func:`generate_prompts_with_styles`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

# ----------------------------------------------------------------------------
# Prompt templates
# ----------------------------------------------------------------------------

_BASE_TOPICS = [
    "machine learning",
    "climate change",
    "quantum computing",
    "artificial intelligence",
    "renewable energy",
    "deep learning",
    "neural networks",
    "data science",
    "graph algorithms",
    "operating systems",
    "distributed systems",
    "compiler design",
    "database internals",
    "computer networks",
    "computer security",
    "robotics",
    "computer vision",
    "natural language processing",
    "reinforcement learning",
    "cloud computing",
]


def generate_prompts_with_styles(
    num_prompts: int,
    styles: list[str],
    *,
    base_topics: Optional[list[str]] = None,
) -> dict[str, list[str]]:
    """Build per-style prompt lists, identical interface to the baseline."""
    topics = base_topics or _BASE_TOPICS
    prompts_by_style: dict[str, list[str]] = {}
    for style in styles:
        prompts_by_style[style] = []
        for i in range(num_prompts):
            topic = topics[i % len(topics)]
            if style == "formal":
                prompt = (
                    f"Provide a formal academic explanation of {topic}. "
                    "Include definitions and theoretical foundations.")
            elif style == "casual":
                prompt = (
                    f"Explain {topic} in simple, casual language like you're "
                    "talking to a friend.")
            elif style == "technical":
                prompt = (
                    f"Give a detailed technical explanation of {topic}. "
                    "Include implementation details and mathematical "
                    "foundations.")
            elif style == "story":
                prompt = (f"Tell a short story that explains {topic} through "
                          "narrative and examples.")
            else:
                prompt = f"Explain {topic} in 3 short paragraphs."
            prompts_by_style[style].append(prompt)
    return prompts_by_style


def style_for_adapter(adapter_name: str) -> str:
    """Pick a style that matches adapter naming conventions."""
    if adapter_name.startswith("hot"):
        return "formal"
    if adapter_name.startswith("cold_instruct"):
        return "technical"
    if adapter_name.startswith("cold"):
        return "casual"
    return "casual"


# ----------------------------------------------------------------------------
# Schedule entries
# ----------------------------------------------------------------------------


@dataclass
class ScheduleEntry:
    """One scheduled request: which adapter, which prompt slot, which segment."""

    index: int
    adapter_name: str
    prompt_index: int
    segment: str = "default"
    metadata: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Schedule generators
# ----------------------------------------------------------------------------


def make_stable_skew_schedule(
    num_requests: int,
    *,
    hot_adapter: str,
    cold_adapters: list[str],
    hot_ratio: float,
    seed: int = 0,
    num_prompts: int = 20,
) -> list[ScheduleEntry]:
    """Static skew matching the existing baseline.

    ``hot_ratio`` of the requests use ``hot_adapter``; the rest are
    distributed evenly (round-robin) across ``cold_adapters``. Order
    is shuffled.
    """
    if not (0.0 <= hot_ratio <= 1.0):
        raise ValueError("hot_ratio must be in [0, 1]")
    if hot_ratio < 1.0 and not cold_adapters:
        raise ValueError("cold_adapters required when hot_ratio < 1.0")
    rng = random.Random(seed)

    num_hot = int(round(num_requests * hot_ratio))
    num_cold = num_requests - num_hot

    entries: list[ScheduleEntry] = []
    for i in range(num_hot):
        entries.append(
            ScheduleEntry(index=i,
                          adapter_name=hot_adapter,
                          prompt_index=i % num_prompts,
                          segment="stable",
                          metadata={"role": "hot"}))
    for i in range(num_cold):
        adapter = cold_adapters[i % len(cold_adapters)]
        entries.append(
            ScheduleEntry(index=num_hot + i,
                          adapter_name=adapter,
                          prompt_index=(num_hot + i) % num_prompts,
                          segment="stable",
                          metadata={"role": "cold"}))

    rng.shuffle(entries)
    for new_idx, entry in enumerate(entries):
        entry.index = new_idx
    return entries


def make_hot_change_schedule(
    *,
    segment_size: int,
    segments: list[tuple[str, list[str], float]],
    seed: int = 0,
    num_prompts: int = 20,
) -> list[ScheduleEntry]:
    """Concatenated stable-skew segments, each with its own hot adapter.

    ``segments`` is a list of ``(hot_adapter, cold_adapters, hot_ratio)``
    triples. The total number of requests is
    ``segment_size * len(segments)``. Each segment is internally
    shuffled.

    Use this for Experiment 2: the workload's dominant adapter changes
    over time, so a static system cannot keep up; the dynamic system
    should detect the change and switch profiles.
    """
    rng = random.Random(seed)
    all_entries: list[ScheduleEntry] = []
    global_idx = 0
    for seg_idx, (hot_adapter, cold_adapters, hot_ratio) in enumerate(
            segments):
        seg_entries = make_stable_skew_schedule(
            segment_size,
            hot_adapter=hot_adapter,
            cold_adapters=cold_adapters,
            hot_ratio=hot_ratio,
            seed=seed + seg_idx,
            num_prompts=num_prompts,
        )
        for entry in seg_entries:
            entry.segment = f"segment_{seg_idx}_hot_{hot_adapter}"
            entry.metadata["segment_index"] = seg_idx
            entry.metadata["segment_hot"] = hot_adapter
            entry.metadata["segment_hot_ratio"] = hot_ratio
            entry.index = global_idx
            global_idx += 1
            all_entries.append(entry)
    # Note: we DO NOT shuffle across segments, because the experiment
    # depends on the temporal ordering.
    return all_entries


def make_thrashing_schedule(
    *,
    block_size: int,
    num_blocks: int,
    candidates: list[str],
    block_hot_ratio: float = 0.8,
    seed: int = 0,
    num_prompts: int = 20,
) -> list[ScheduleEntry]:
    """Alternate the dominant adapter every ``block_size`` requests.

    Use this for Experiment 3 (anti-thrashing): a *naive* policy
    would switch ``num_blocks - 1`` times, paying repeated rebuild
    cost. With cooldown / margin guards the dynamic system should
    switch at most a handful of times.
    """
    if len(candidates) < 2:
        raise ValueError("Need at least 2 candidates for thrashing")
    entries: list[ScheduleEntry] = []
    global_idx = 0
    for block in range(num_blocks):
        hot = candidates[block % len(candidates)]
        cold = [c for c in candidates if c != hot]
        block_entries = make_stable_skew_schedule(
            block_size,
            hot_adapter=hot,
            cold_adapters=cold,
            hot_ratio=block_hot_ratio,
            seed=seed + block,
            num_prompts=num_prompts,
        )
        for entry in block_entries:
            entry.segment = f"block_{block}_hot_{hot}"
            entry.metadata["block_index"] = block
            entry.metadata["block_hot"] = hot
            entry.index = global_idx
            global_idx += 1
            entries.append(entry)
    return entries


# ----------------------------------------------------------------------------
# Self-test (sanity for shape/lengths)
# ----------------------------------------------------------------------------


def _self_test() -> None:
    s = make_stable_skew_schedule(
        100,
        hot_adapter="hot_general",
        cold_adapters=["cold_dummy", "cold_instruct"],
        hot_ratio=0.8,
    )
    assert len(s) == 100
    assert sum(1 for e in s if e.adapter_name == "hot_general") == 80

    h = make_hot_change_schedule(
        segment_size=50,
        segments=[
            ("hot_general", ["cold_dummy", "cold_instruct"], 0.8),
            ("cold_dummy", ["hot_general", "cold_instruct"], 0.8),
        ],
    )
    assert len(h) == 100
    seg_hots = {e.metadata["segment_hot"] for e in h}
    assert seg_hots == {"hot_general", "cold_dummy"}

    t = make_thrashing_schedule(
        block_size=30,
        num_blocks=4,
        candidates=["hot_general", "cold_dummy"],
    )
    assert len(t) == 120
    assert {e.metadata["block_hot"] for e in t} == {"hot_general", "cold_dummy"}


if __name__ == "__main__":
    _self_test()
    print("workload self-test OK")
