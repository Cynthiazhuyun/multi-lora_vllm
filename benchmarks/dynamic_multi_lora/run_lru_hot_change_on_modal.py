# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Modal benchmark: vanilla vLLM Multi-LoRA (LRU) on the *same* hot_change
workload the dynamic system runs.

The existing ``run_multi_lora_baseline_on_modal.py`` only sweeps the
stable-skew workload, which means we currently have *no* head-to-head
data point for the headline experiment (E2: hot_change). This runner
fills that gap. It:

1. Bakes the LoRAs into the same image used by the LRU stable_skew
   baseline.
2. Starts a single vLLM server with all three LoRAs registered (no
   eviction, ``--max-loras 3``), exactly as the LRU baseline does.
3. Drives the **same** :func:`make_hot_change_schedule` workload as
   ``run_dynamic_multi_lora_on_modal.py`` -- same ``segment_size``,
   same ``hot_ratio``, same seed, same pacing -- so per-request
   timing is directly comparable to the dynamic E2 run.
4. Writes ``lru_E2_hot_change_<initial>_to_<second>.json`` into the
   shared Modal volume in the same shape as the dynamic E2 file:
   ``{"started_at_wall", "summary", "per_request": [...]}``.

The analyzer (``analyze_dynamic_results.py``) picks this up
automatically and overlays the LRU rolling-mean curve on the
hot_change plot, so the picture answers "how much better is the
dynamic system than vanilla vLLM Multi-LoRA on the *same* workload?"
rather than just "the dynamic system is able to switch".

Run from the repo root::

    modal run benchmarks/dynamic_multi_lora/run_lru_hot_change_on_modal.py \\
        2>&1 | tee benchmarks/dynamic_multi_lora/lru_hot_change_run.log

Override the segments / pacing if needed::

    modal run benchmarks/dynamic_multi_lora/run_lru_hot_change_on_modal.py \\
        --initial-hot hot_general --second-hot cold_dummy \\
        --segment-size 600 --request-rate-per-s 2.0
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

# Module is imported BOTH locally (to define the Modal app) and
# inside the Modal container. The baseline LRU runner lives in the
# sibling ``benchmarks/multi_lora/`` directory, and the workload
# generators live next to this file. We mount both into the
# container.
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

app = modal.App(name="vllm-lru-hot-change")

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


@app.function(
    gpu="A100",
    image=image,
    volumes={RESULTS_DIR: results_volume},
    timeout=7200,
)
def run_lru_hot_change_benchmark(
    initial_hot: str = "hot_general",
    second_hot: str = "cold_dummy",
    segment_size: int = 600,
    hot_ratio_per_segment: float = 0.85,
    request_rate_per_s: float | None = 2.0,
    seed: int = 0,
):
    """Run the same ``hot_change`` workload as the dynamic E2 against
    a vanilla vLLM Multi-LoRA (LRU) server.

    Parameters mirror the dynamic runner so that result files line
    up apples-to-apples.
    """
    sys.path.insert(0, CONTAINER_SOURCE_DIR)
    sys.path.insert(0, CONTAINER_BASELINE_DIR)

    import numpy as np
    import requests
    from tqdm import tqdm

    # Reuse helpers from the existing LRU baseline so we don't rot
    # two copies of vLLM startup logic.
    from run_multi_lora_baseline_on_modal import (  # noqa: E402
        wait_for_vllm_server)
    # Schedule generator is shared with the dynamic runner so the
    # two systems see *the same* requests in the same order.
    from workload import (generate_prompts_with_styles,  # noqa: E402
                          make_hot_change_schedule, style_for_adapter)

    if initial_hot not in LORA_ADAPTERS:
        raise ValueError(f"initial_hot {initial_hot!r} not in {list(LORA_ADAPTERS)}")
    if second_hot not in LORA_ADAPTERS:
        raise ValueError(f"second_hot {second_hot!r} not in {list(LORA_ADAPTERS)}")
    if initial_hot == second_hot:
        raise ValueError("initial_hot and second_hot must differ")

    print("=" * 80)
    print("  LRU vLLM Multi-LoRA: hot_change workload")
    print("=" * 80)
    print(f"Base model:           {BASE_MODEL}")
    print(f"Initial hot:          {initial_hot}")
    print(f"Second hot:           {second_hot}")
    print(f"Segment size:         {segment_size} requests")
    print(f"Hot ratio per seg:    {hot_ratio_per_segment}")
    print(f"Request rate:         {request_rate_per_s} req/s")
    print(f"Seed:                 {seed}")
    print(f"Timestamp:            {datetime.now().isoformat()}")
    print("=" * 80)
    print()

    # -----------------------------------------------------------------
    # 1. Boot vLLM with all three LoRAs registered (LRU mode).
    # -----------------------------------------------------------------
    lora_modules = [
        f"{name}={path}" for name, path in LORA_ADAPTERS.items()
    ]
    server_argv = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--port", "8000",
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.9",
        "--enable-lora",
        # Match the existing LRU stable_skew baseline so the comparison
        # to that experiment is consistent: 3 slots, no eviction with
        # only 3 adapters.
        "--max-loras", "3",
        "--max-lora-rank", "256",
        "--lora-modules",
    ] + lora_modules
    print("Starting vLLM with command:")
    for tok in server_argv:
        print(f"  {tok}")
    print()

    server_process = subprocess.Popen(
        server_argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        model_ids, _ = wait_for_vllm_server(
            server_process,
            min_wait_s=180,
            max_wait_s=420,
            idle_grace_s=90,
        )
        print(f"vLLM ready, models={model_ids}")
        for required in [BASE_MODEL, *LORA_ADAPTERS]:
            if required not in model_ids:
                raise RuntimeError(
                    f"vLLM is missing expected model: {required}; got {model_ids}")

        # -----------------------------------------------------------------
        # 2. Build the same hot_change schedule the dynamic runner uses.
        # -----------------------------------------------------------------
        cold_for_initial = [n for n in LORA_ADAPTERS if n != initial_hot]
        cold_for_second = [n for n in LORA_ADAPTERS if n != second_hot]
        schedule = make_hot_change_schedule(
            segment_size=segment_size,
            segments=[
                (initial_hot, cold_for_initial, hot_ratio_per_segment),
                (second_hot, cold_for_second, hot_ratio_per_segment),
            ],
            seed=seed,
        )
        print(f"Generated schedule of {len(schedule)} requests "
              f"({segment_size} per segment)")

        prompts_by_style = generate_prompts_with_styles(
            num_prompts=20, styles=["formal", "casual", "technical", "story"])

        session = requests.Session()

        # -----------------------------------------------------------------
        # 3. Drive the workload against the LRU server with the SAME
        #    pacing the dynamic runner uses for E2.
        # -----------------------------------------------------------------
        per_request: list[dict] = []
        workload_start_wall = time.time()
        workload_start_perf = time.perf_counter()

        for entry in tqdm(schedule, desc="LRU hot_change"):
            if request_rate_per_s is not None and request_rate_per_s > 0:
                target_offset = entry.index / request_rate_per_s
                now_offset = time.perf_counter() - workload_start_perf
                sleep_for = target_offset - now_offset
                if sleep_for > 0:
                    time.sleep(sleep_for)

            style = style_for_adapter(entry.adapter_name)
            prompt = prompts_by_style[style][entry.prompt_index]

            t_send = time.time() - workload_start_wall
            t_req = time.perf_counter()
            status_code = 0
            error: str | None = None
            text_preview = ""
            try:
                r = session.post(
                    "http://localhost:8000/v1/completions",
                    json={
                        "model": entry.adapter_name,
                        "prompt": prompt,
                        "max_tokens": 128,
                        "temperature": 0.7,
                    },
                    timeout=60,
                )
                latency = time.perf_counter() - t_req
                status_code = r.status_code
                if r.status_code == 200:
                    try:
                        body = r.json()
                        text_preview = ((body.get("choices") or [{}])[0]
                                        .get("text") or "")[:60]
                    except Exception as exc:
                        error = f"parse: {exc!r}"
                else:
                    error = f"http {r.status_code}: {r.text[:120]}"
            except Exception as exc:
                latency = time.perf_counter() - t_req
                error = repr(exc)
            t_done = time.time() - workload_start_wall

            per_request.append({
                "request_index": entry.index,
                "segment": entry.segment,
                "adapter_name": entry.adapter_name,
                "metadata": entry.metadata,
                "send_offset_s": t_send,
                "done_offset_s": t_done,
                "latency_s": latency,
                "latency_ms": latency * 1000.0,
                "status_code": status_code,
                # LRU has no "fast path" -- every request goes through
                # the LoRA adapter. We record this explicitly so the
                # analyzer doesn't have to special-case the schema.
                "routed_to_hot": False,
                "profile_used": "lru_baseline",
                "served_model": entry.adapter_name,
                "error": error,
                "text_preview": text_preview,
            })

        duration_s = time.time() - workload_start_wall
        successful = [r for r in per_request if r["status_code"] == 200]
        failed = [r for r in per_request if r["status_code"] != 200]

        latencies = np.asarray([r["latency_ms"] for r in successful],
                               dtype=np.float64)
        summary = {
            "system": "vllm_lru_baseline",
            "experiment": "hot_change",
            "initial_hot": initial_hot,
            "second_hot": second_hot,
            "segment_size": segment_size,
            "hot_ratio_per_segment": hot_ratio_per_segment,
            "request_rate_per_s": request_rate_per_s,
            "seed": seed,
            "duration_s": duration_s,
            "num_requests": len(per_request),
            "num_successful": len(successful),
            "num_failed": len(failed),
            "request_throughput": (len(successful) / duration_s
                                   if duration_s > 0 else 0.0),
            "mean_latency_ms": float(latencies.mean()) if latencies.size else 0.0,
            "median_latency_ms": float(np.median(latencies))
            if latencies.size else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95))
            if latencies.size else 0.0,
            "p99_latency_ms": float(np.percentile(latencies, 99))
            if latencies.size else 0.0,
            "std_latency_ms": float(latencies.std()) if latencies.size else 0.0,
        }
        # Per-segment slice, mirroring the dynamic file's structure.
        segments = sorted({r["segment"] for r in successful})
        summary["per_segment"] = {}
        for seg in segments:
            seg_lats = np.asarray(
                [r["latency_ms"] for r in successful if r["segment"] == seg],
                dtype=np.float64)
            summary["per_segment"][seg] = {
                "count": int(seg_lats.size),
                "mean_latency_ms": float(seg_lats.mean()) if seg_lats.size
                else 0.0,
                "p99_latency_ms": float(np.percentile(seg_lats, 99))
                if seg_lats.size else 0.0,
            }
        # Per-adapter slice (useful when post-mortem reading the json).
        adapters = sorted({r["adapter_name"] for r in successful})
        summary["per_adapter"] = {}
        for a in adapters:
            a_lats = np.asarray(
                [r["latency_ms"] for r in successful if r["adapter_name"] == a],
                dtype=np.float64)
            summary["per_adapter"][a] = {
                "count": int(a_lats.size),
                "mean_latency_ms": float(a_lats.mean()) if a_lats.size else 0.0,
            }

        out_path = Path(RESULTS_DIR) / (
            f"lru_E2_hot_change_{initial_hot}_to_{second_hot}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "summary": summary,
                "started_at_wall": workload_start_wall,
                "per_request": per_request,
            }, f, indent=2, default=str)
        print(f"\nWrote {out_path}")
        print(f"  mean_latency_ms = {summary['mean_latency_ms']:.1f}")
        print(f"  p99_latency_ms  = {summary['p99_latency_ms']:.1f}")
        print(f"  throughput      = {summary['request_throughput']:.2f} req/s")
        print(f"  failed          = {summary['num_failed']}")
        for seg, s in summary["per_segment"].items():
            print(f"  {seg}: mean={s['mean_latency_ms']:.1f}ms "
                  f"p99={s['p99_latency_ms']:.1f}ms count={s['count']}")
    finally:
        print("\nShutting down vLLM ...")
        server_process.terminate()
        try:
            server_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_process.kill()


@app.local_entrypoint()
def main(initial_hot: str = "hot_general",
         second_hot: str = "cold_dummy",
         segment_size: int = 600,
         hot_ratio_per_segment: float = 0.85,
         request_rate_per_s: float = 2.0,
         seed: int = 0):
    """Entrypoint::

        modal run benchmarks/dynamic_multi_lora/run_lru_hot_change_on_modal.py

    Defaults match the dynamic E2 (~5 minutes per segment, ~10 minutes
    total at 2 req/s). Use ``--segment-size`` and
    ``--request-rate-per-s`` to widen or narrow the window.
    """
    print(f"Launching LRU hot_change benchmark "
          f"({initial_hot} -> {second_hot}, "
          f"{segment_size} reqs/seg @ {request_rate_per_s} req/s)")
    run_lru_hot_change_benchmark.remote(
        initial_hot=initial_hot,
        second_hot=second_hot,
        segment_size=segment_size,
        hot_ratio_per_segment=hot_ratio_per_segment,
        request_rate_per_s=request_rate_per_s,
        seed=seed,
    )
    print("Done. Result in Modal volume vllm-benchmark-results "
          f"as lru_E2_hot_change_{initial_hot}_to_{second_hot}.json.")
