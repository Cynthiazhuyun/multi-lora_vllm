# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Modal benchmark: ONLINE popularity-aware dynamic multi-LoRA serving.

This is the headline experiment of the project. It exercises the
dynamic system end-to-end on Modal A100:

* Builds one fused-base profile per candidate hot adapter offline.
* Starts a vLLM server for the initial profile.
* Drives requests through :class:`DynamicLoRARouter`, which records
  popularity in a sliding window and triggers blue-green profile
  switches when a different adapter becomes dominant.
* Runs three experiments back-to-back:

  - Experiment 1 (``stable_skew``): the same hot-ratio sweep used by
    the LRU baseline. Demonstrates that the dynamic system does not
    *regress* on the stable case the static system already wins.
  - Experiment 2 (``hot_change``): two segments with different
    dominant adapters, concatenated in time. Demonstrates online
    detection + profile switching. Produces the latency-vs-time
    series and the moment of switch.
  - Experiment 3 (``thrashing``): rapidly oscillating dominance.
    Demonstrates that cooldown + margin guards prevent excessive
    switching and keep latency stable.

Run from the repo root::

    modal run benchmarks/dynamic_multi_lora/run_dynamic_multi_lora_on_modal.py \
        2>&1 | tee benchmarks/dynamic_multi_lora/dynamic_multi_lora_run.log

Outputs land in the ``vllm-benchmark-results`` Modal volume with the
prefix ``dynamic_multi_lora_*``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

# This module is imported BOTH locally (to define the Modal app) and
# inside the Modal container. The baseline LRU runner lives in the
# sibling ``benchmarks/multi_lora/`` directory so we can share LoRA
# bootstrap + ``wait_for_vllm_server`` without duplicating code.
SOURCE_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SOURCE_DIR.parent / "multi_lora"
CONTAINER_LORA_DIR = "/root/loras"
CONTAINER_SOURCE_DIR = "/root/dynamic_lora_code"
CONTAINER_BASELINE_DIR = "/root/baseline_code"

if "MODAL_TASK_ID" not in os.environ:
    sys.path.insert(0, str(BASELINE_DIR))
    from run_multi_lora_baseline_on_modal import (  # noqa: E402
        LOCAL_LORA_DIR, ensure_local_loras)

    ensure_local_loras()
else:
    LOCAL_LORA_DIR = Path(CONTAINER_LORA_DIR)

app = modal.App(name="vllm-dynamic-multi-lora")

image = (
    # Pin Python <3.14: numba (a vLLM transitive dep) does not support
    # 3.14 yet, and Modal's default debian_slim() now ships 3.14.
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub[hf_transfer]>=0.24.0",
        "vllm>=0.6.0",
        "torch==2.9.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "safetensors>=0.4.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "aiohttp>=3.8.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
    )
    .add_local_dir(LOCAL_LORA_DIR, remote_path=CONTAINER_LORA_DIR)
    .add_local_dir(
        str(SOURCE_DIR),
        remote_path=CONTAINER_SOURCE_DIR,
        ignore=["*.pyc", "__pycache__", "*.log", "*.json"],
    )
    .add_local_dir(
        str(BASELINE_DIR),
        remote_path=CONTAINER_BASELINE_DIR,
        ignore=["*.pyc", "__pycache__", "*.log", "*.md"],
    )
)

results_volume = modal.Volume.from_name("vllm-benchmark-results",
                                        create_if_missing=True)
RESULTS_DIR = "/results"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTERS = {
    "hot_general": f"{CONTAINER_LORA_DIR}/hot_general",
    "cold_dummy": f"{CONTAINER_LORA_DIR}/cold_dummy",
    "cold_instruct": f"{CONTAINER_LORA_DIR}/cold_instruct",
}


# ----------------------------------------------------------------------------
# Helpers (run inside the Modal container)
# ----------------------------------------------------------------------------


def _setup_logging():
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _build_all_profiles_and_log(adapters, output_root):
    """Wrapper: build profiles and print a small summary."""
    from lora_profile_builder import build_all_profiles  # noqa: E402

    print(f"Building profiles under {output_root} for {list(adapters)} ...")
    t0 = time.time()
    profiles = build_all_profiles(
        base_model=BASE_MODEL,
        adapters=adapters,
        output_root=output_root,
    )
    dt = time.time() - t0
    print(f"Built {len(profiles)} profile(s) in {dt:.1f}s.")
    for name, p in profiles.items():
        print(
            f"  profile={name} fused={p.fused_model_dir} "
            f"deltas={list(p.delta_adapter_dirs)}")
    return profiles


def _run_workload(
    *,
    router,
    schedule,
    prompts_by_style,
    style_for_adapter,
    request_rate_per_s: float | None = None,
    label: str = "",
):
    """Drive ``schedule`` through ``router`` and collect per-request data."""
    from tqdm import tqdm  # noqa: E402

    workload_start_wall = time.time()
    workload_start_perf = time.perf_counter()

    per_request: list[dict] = []

    for entry in tqdm(schedule, desc=label):
        if request_rate_per_s is not None and request_rate_per_s > 0:
            target_offset = entry.index / request_rate_per_s
            now_offset = time.perf_counter() - workload_start_perf
            sleep_for = target_offset - now_offset
            if sleep_for > 0:
                time.sleep(sleep_for)

        style = style_for_adapter(entry.adapter_name)
        prompt = prompts_by_style[style][entry.prompt_index]

        t_send = time.time() - workload_start_wall
        result = router.send(entry.adapter_name, prompt, max_tokens=128,
                             temperature=0.7)
        t_done = time.time() - workload_start_wall

        per_request.append({
            "request_index": entry.index,
            "segment": entry.segment,
            "adapter_name": entry.adapter_name,
            "metadata": entry.metadata,
            "send_offset_s": t_send,
            "done_offset_s": t_done,
            "latency_s": result.latency_s,
            "latency_ms": result.latency_s * 1000.0,
            "status_code": result.status_code,
            "routed_to_hot": result.routed_to_hot,
            "profile_used": result.profile_used,
            "served_model": result.served_model,
            "error": result.error,
            "text_preview": (result.text or "")[:60],
        })

    return {
        "started_at_wall": workload_start_wall,
        "duration_s": time.time() - workload_start_wall,
        "per_request": per_request,
    }


def _summarize(workload_data, *, router):
    """Reduce per-request data to summary stats + event/router state."""
    import numpy as np  # noqa: E402

    per_req = workload_data["per_request"]
    successful = [r for r in per_req if r["status_code"] == 200]
    failed = [r for r in per_req if r["status_code"] != 200]

    latencies = np.asarray([r["latency_ms"] for r in successful],
                           dtype=np.float64)
    summary = {
        "duration_s": workload_data["duration_s"],
        "num_requests": len(per_req),
        "num_successful": len(successful),
        "num_failed": len(failed),
        "request_throughput": (len(successful) / workload_data["duration_s"]
                               if workload_data["duration_s"] > 0 else 0.0),
        "mean_latency_ms": float(latencies.mean()) if latencies.size else 0.0,
        "median_latency_ms": float(np.median(latencies))
        if latencies.size else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95))
        if latencies.size else 0.0,
        "p99_latency_ms": float(np.percentile(latencies, 99))
        if latencies.size else 0.0,
        "std_latency_ms": float(latencies.std()) if latencies.size else 0.0,
    }
    # Per-adapter slice.
    adapters = sorted({r["adapter_name"] for r in successful})
    summary["per_adapter"] = {}
    for a in adapters:
        a_lats = np.asarray(
            [r["latency_ms"] for r in successful if r["adapter_name"] == a],
            dtype=np.float64)
        summary["per_adapter"][a] = {
            "count": int(a_lats.size),
            "mean_latency_ms": float(a_lats.mean()) if a_lats.size else 0.0,
            "p95_latency_ms": float(np.percentile(a_lats, 95))
            if a_lats.size else 0.0,
        }
    # Routed-to-hot share and per-profile slice.
    summary["fast_path_share"] = (
        sum(1 for r in successful if r["routed_to_hot"]) / len(successful)
        if successful else 0.0)
    profiles_used = sorted({r["profile_used"] for r in successful})
    summary["per_profile"] = {}
    for p in profiles_used:
        p_lats = np.asarray(
            [r["latency_ms"] for r in successful if r["profile_used"] == p],
            dtype=np.float64)
        summary["per_profile"][p] = {
            "count": int(p_lats.size),
            "mean_latency_ms": float(p_lats.mean()) if p_lats.size else 0.0,
        }
    # Switch counts from router events.
    events = [dataclasses.asdict(e) for e in router.events()]
    switch_decided = [e for e in events if e["kind"] == "switch_decided"]
    switch_committed = [e for e in events if e["kind"] == "switch_committed"]
    summary["num_switch_decisions"] = len(switch_decided)
    summary["num_switch_commits"] = len(switch_committed)
    summary["events"] = events
    summary["final_router_state"] = {
        "current_hot": router.tracker.current_hot,
        "building_hot": router.tracker.building_hot,
        "stats": dataclasses.asdict(router.tracker.stats()),
    }
    return summary


def _save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ----------------------------------------------------------------------------
# Modal entrypoints
# ----------------------------------------------------------------------------


@app.function(
    gpu="A100",
    image=image,
    volumes={RESULTS_DIR: results_volume},
    timeout=10800,
)
def run_dynamic_experiments(
    initial_hot: str = "hot_general",
    candidates: tuple[str, ...] = ("hot_general", "cold_dummy", "cold_instruct"),
    stable_skew_hot_ratios: tuple[float, ...] = (0.0, 0.4, 0.8, 1.0),
    stable_skew_num_requests: int = 100,
    # E2 segments default to 600 reqs @ 2 req/s ≈ 5 minutes each, ~10
    # minutes total. The blue-green profile switch costs ~2 minutes on
    # A100 (cold-start vLLM + warmup), so a 5-minute second segment
    # leaves >=2 minutes of post-switch fast-path observation, which is
    # what lets us see latency actually drop after the switch.
    hot_change_segment_size: int = 600,
    hot_change_request_rate_per_s: float | None = 2.0,
    thrashing_block_size: int = 30,
    thrashing_num_blocks: int = 8,
    request_rate_per_s: float | None = None,
):
    """Run all three dynamic experiments back-to-back.

    Parameters mirror the schedule generators in ``workload.py``.

    ``request_rate_per_s`` controls E1/E3 pacing. ``hot_change_request_rate_per_s``
    is a separate knob for E2 because E2 needs to be long enough in
    *wall-clock* for the (~2 min) blue-green switch to complete with
    headroom; we therefore pace E2 to a fixed rate by default.
    """
    sys.path.insert(0, CONTAINER_SOURCE_DIR)
    sys.path.insert(0, CONTAINER_BASELINE_DIR)
    _setup_logging()

    import numpy as np  # noqa: F401
    import requests  # noqa: F401

    from dynamic_router import build_router  # noqa: E402
    from popularity_tracker import PopularityTrackerConfig  # noqa: E402
    from workload import (generate_prompts_with_styles,  # noqa: E402
                          make_hot_change_schedule,
                          make_stable_skew_schedule,
                          make_thrashing_schedule, style_for_adapter)

    print("=" * 80)
    print("  DYNAMIC Multi-LoRA Benchmark on Modal A100")
    print("=" * 80)
    print(f"Base model:      {BASE_MODEL}")
    print(f"Initial hot:     {initial_hot}")
    print(f"Candidate hots:  {list(candidates)}")
    print(f"Timestamp:       {datetime.now().isoformat()}")
    print()

    if initial_hot not in candidates:
        raise ValueError(
            f"initial_hot {initial_hot!r} must be in candidates {candidates}")

    # -----------------------------------------------------------------
    # 1. Build all candidate profiles offline. This is the one-time
    #    cost the dynamic system pays before serving begins. We do NOT
    #    count this toward online metrics.
    # -----------------------------------------------------------------
    profile_root = Path(RESULTS_DIR) / "profiles_dynamic"
    adapters_subset = {n: LORA_ADAPTERS[n] for n in candidates}
    profiles = _build_all_profiles_and_log(adapters_subset, profile_root)

    # -----------------------------------------------------------------
    # 2. Bring up the initial profile.
    # -----------------------------------------------------------------
    log_dir = Path(RESULTS_DIR) / "vllm_logs_dynamic"
    print(f"Starting initial vLLM with hot={initial_hot} ...")

    # Tracker tuned for short experiments (~100-300 reqs). Lower the
    # window and cooldown so dynamics happen within one run.
    tracker_cfg_default = PopularityTrackerConfig(
        window_size=80,
        min_window_size=40,
        merge_threshold=0.65,
        switch_margin=0.15,
        cooldown_sec=20.0,
    )

    router = build_router(
        profiles=profiles,
        base_model_id=BASE_MODEL,
        log_dir=log_dir,
        initial_hot=initial_hot,
        tracker_config=tracker_cfg_default,
        gpu_memory_utilization=0.4,
        max_loras=4,
        max_lora_rank=256,
    )
    print("Initial vLLM ready; starting experiments.")

    prompts_by_style = generate_prompts_with_styles(
        num_prompts=20, styles=["formal", "casual", "technical", "story"])

    summary_all: dict = {
        "base_model": BASE_MODEL,
        "initial_hot": initial_hot,
        "candidates": list(candidates),
        "tracker_config": dataclasses.asdict(tracker_cfg_default),
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
    }

    try:
        # ---------------------------------------------------------
        # Experiment 1: stable_skew (sweep hot_ratio)
        # ---------------------------------------------------------
        exp1: dict = {}
        cold_for_initial = [n for n in candidates if n != initial_hot]
        for hot_ratio in stable_skew_hot_ratios:
            label = f"E1 stable_skew hr={hot_ratio:.1f}"
            print(f"\n[Experiment 1] {label}")
            schedule = make_stable_skew_schedule(
                num_requests=stable_skew_num_requests,
                hot_adapter=initial_hot,
                cold_adapters=cold_for_initial,
                hot_ratio=hot_ratio,
                seed=int(hot_ratio * 100),
            )
            workload = _run_workload(
                router=router,
                schedule=schedule,
                prompts_by_style=prompts_by_style,
                style_for_adapter=style_for_adapter,
                request_rate_per_s=request_rate_per_s,
                label=label,
            )
            summary = _summarize(workload, router=router)
            exp1[f"hot_ratio_{hot_ratio}"] = {
                "summary": summary,
                "per_request": workload["per_request"],
            }
            _save_json(
                {"summary": summary, "per_request": workload["per_request"]},
                Path(RESULTS_DIR) /
                f"dynamic_multi_lora_E1_stable_skew_hr{hot_ratio}.json")
        summary_all["experiments"]["stable_skew"] = exp1

        # ---------------------------------------------------------
        # Experiment 2: hot_change (the headline)
        # We construct two segments with different dominant
        # adapters. The dynamic system should detect the change in
        # the second segment and switch profiles.
        # ---------------------------------------------------------
        # Choose the second-segment hot to be a different adapter.
        # Pick deterministically: first non-initial candidate.
        second_segment_hot = next(c for c in candidates if c != initial_hot)
        cold_for_initial = [c for c in candidates if c != initial_hot]
        cold_for_second = [c for c in candidates if c != second_segment_hot]
        print(
            f"\n[Experiment 2] hot_change: "
            f"{initial_hot} -> {second_segment_hot} "
            f"({hot_change_segment_size} reqs each)")
        schedule_hc = make_hot_change_schedule(
            segment_size=hot_change_segment_size,
            segments=[
                (initial_hot, cold_for_initial, 0.85),
                (second_segment_hot, cold_for_second, 0.85),
            ],
        )
        workload_hc = _run_workload(
            router=router,
            schedule=schedule_hc,
            prompts_by_style=prompts_by_style,
            style_for_adapter=style_for_adapter,
            request_rate_per_s=hot_change_request_rate_per_s,
            label="E2 hot_change",
        )
        # Wait for any pending switch to finish so we have a clean
        # state going into Experiment 3.
        router.wait_for_pending_switch(timeout=300.0)
        summary_hc = _summarize(workload_hc, router=router)
        summary_all["experiments"]["hot_change"] = {
            "segments": [
                {"hot": initial_hot, "size": hot_change_segment_size},
                {"hot": second_segment_hot, "size": hot_change_segment_size},
            ],
            "started_at_wall": workload_hc["started_at_wall"],
            "summary": summary_hc,
            "per_request": workload_hc["per_request"],
        }
        _save_json(
            {
                "summary": summary_hc,
                "started_at_wall": workload_hc["started_at_wall"],
                "per_request": workload_hc["per_request"],
            },
            Path(RESULTS_DIR) /
            f"dynamic_multi_lora_E2_hot_change_{initial_hot}_to_"
            f"{second_segment_hot}.json")

        # ---------------------------------------------------------
        # Experiment 3: anti-thrashing.
        # We run the same oscillating workload twice: once with the
        # default cooldown/margin, and once with both relaxed (so we
        # can show that the guards actually do their job).
        # ---------------------------------------------------------
        thrashing_candidates = list(candidates[:2])
        for guard_label, guard_cfg in [
            ("guarded",
             PopularityTrackerConfig(
                 window_size=60,
                 min_window_size=30,
                 merge_threshold=0.6,
                 switch_margin=0.15,
                 cooldown_sec=60.0,
             )),
            ("naive",
             PopularityTrackerConfig(
                 window_size=20,
                 min_window_size=10,
                 merge_threshold=0.5,
                 switch_margin=0.0,
                 cooldown_sec=0.0,
             )),
        ]:
            print(
                f"\n[Experiment 3] thrashing ({guard_label}) "
                f"between {thrashing_candidates} "
                f"({thrashing_num_blocks} blocks * "
                f"{thrashing_block_size} reqs)")
            # Re-create a router with the new tracker config but the
            # same switch manager, so we don't restart vLLM between
            # the two thrashing variants.
            from dynamic_router import DynamicLoRARouter  # noqa: E402

            sm = router._switch_manager  # noqa: SLF001
            current_active, _ = sm.get_active()
            assert current_active is not None
            router_t = DynamicLoRARouter(
                profiles=profiles,
                switch_manager=sm,
                tracker_config=guard_cfg,
                initial_hot=current_active,
            )
            schedule_t = make_thrashing_schedule(
                block_size=thrashing_block_size,
                num_blocks=thrashing_num_blocks,
                candidates=thrashing_candidates,
                block_hot_ratio=0.85,
            )
            workload_t = _run_workload(
                router=router_t,
                schedule=schedule_t,
                prompts_by_style=prompts_by_style,
                style_for_adapter=style_for_adapter,
                request_rate_per_s=request_rate_per_s,
                label=f"E3 thrashing/{guard_label}",
            )
            router_t.wait_for_pending_switch(timeout=180.0)
            summary_t = _summarize(workload_t, router=router_t)
            key = f"thrashing_{guard_label}"
            summary_all["experiments"][key] = {
                "guard_config": dataclasses.asdict(guard_cfg),
                "candidates": thrashing_candidates,
                "summary": summary_t,
                "per_request": workload_t["per_request"],
            }
            _save_json(
                {
                    "summary": summary_t,
                    "per_request": workload_t["per_request"],
                },
                Path(RESULTS_DIR) /
                f"dynamic_multi_lora_E3_thrashing_{guard_label}.json")

        # -----------------------------------------------------------------
        # Final aggregated summary.
        # -----------------------------------------------------------------
        agg_path = Path(RESULTS_DIR) / "dynamic_multi_lora_summary.json"
        _save_json(summary_all, agg_path)
        print(f"\nWrote summary: {agg_path}")
    finally:
        print("Shutting down all vLLM servers...")
        try:
            router._switch_manager.shutdown()  # noqa: SLF001
        except Exception as exc:
            print(f"Shutdown error (continuing): {exc!r}")


@app.local_entrypoint()
def main(initial_hot: str = "hot_general",
         stable_skew_num_requests: int = 100,
         hot_change_segment_size: int = 600,
         hot_change_request_rate_per_s: float = 2.0,
         thrashing_block_size: int = 30,
         thrashing_num_blocks: int = 8):
    """Entrypoint::

        modal run benchmarks/dynamic_multi_lora/run_dynamic_multi_lora_on_modal.py

    The E2 (hot_change) defaults are tuned so each segment takes ~5
    minutes wall-clock (600 reqs at 2 req/s); this leaves enough room
    after the blue-green switch (~2 min on A100) to actually observe
    the post-switch fast-path latency drop.
    """
    print(f"Launching dynamic multi-LoRA experiments (initial_hot={initial_hot})")
    run_dynamic_experiments.remote(
        initial_hot=initial_hot,
        stable_skew_num_requests=stable_skew_num_requests,
        hot_change_segment_size=hot_change_segment_size,
        hot_change_request_rate_per_s=hot_change_request_rate_per_s,
        thrashing_block_size=thrashing_block_size,
        thrashing_num_blocks=thrashing_num_blocks,
    )
    print("Done. Results in Modal volume vllm-benchmark-results.")
