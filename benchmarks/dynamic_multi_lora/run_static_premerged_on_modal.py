# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Modal benchmark: STATIC pre-merged hot LoRA (oracle upper bound).

This script exists to give the dynamic system a target to chase. It
takes the *known* hot LoRA, builds the corresponding profile offline
(fused base + delta adapters for the colds), and runs the same
hot-ratio sweep that the LRU baseline runs. Because the hot LoRA is
"perfectly" predicted in advance, this is the best a popularity-aware
system can possibly do for a stable workload, and it is the upper
bound the dynamic system tries to approach.

Run from the repo root::

    modal run benchmarks/dynamic_multi_lora/run_static_premerged_on_modal.py \
        2>&1 | tee benchmarks/dynamic_multi_lora/static_premerged_run.log

Output JSON files land in the ``vllm-benchmark-results`` Modal Volume
(same as the existing baselines), with the prefix
``static_premerged_*``.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import modal

# Baseline LRU code (with ``wait_for_vllm_server``, ``ensure_local_loras``)
# lives in the sibling ``benchmarks/multi_lora`` directory.
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

app = modal.App(name="vllm-static-premerged-multi-lora")

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
    # Local source must come AFTER pip installs because Modal evaluates
    # build steps in order and we don't want add_local_dir invalidating
    # cached layers.
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
PROFILES_VOLUME_DIR = "/profiles"


@app.function(
    gpu="A100",
    image=image,
    volumes={
        RESULTS_DIR: results_volume,
    },
    timeout=7200,
)
def run_static_premerged_benchmark(
    hot_adapter_name: str = "hot_general",
    hot_ratios: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    num_requests: int = 200,
):
    """Run the hot-ratio sweep with a static pre-merged profile.

    Parameters
    ----------
    hot_adapter_name : str
        Which adapter to merge into the base model (the "oracle" hot).
    hot_ratios : sequence of float
        Hot/cold mix to test, identical to the LRU baseline.
    num_requests : int
        Requests per hot-ratio. The existing baseline uses 200.
    """
    # Make sure we can import the local source we mounted.
    sys.path.insert(0, CONTAINER_SOURCE_DIR)
    sys.path.insert(0, CONTAINER_BASELINE_DIR)

    import numpy as np  # noqa
    import requests
    from tqdm import tqdm

    from lora_profile_builder import build_profile  # noqa: E402
    from workload import (generate_prompts_with_styles,  # noqa: E402
                          make_stable_skew_schedule, style_for_adapter)

    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LORA_ADAPTERS = {
        "hot_general": f"{CONTAINER_LORA_DIR}/hot_general",
        "cold_dummy": f"{CONTAINER_LORA_DIR}/cold_dummy",
        "cold_instruct": f"{CONTAINER_LORA_DIR}/cold_instruct",
    }

    if hot_adapter_name not in LORA_ADAPTERS:
        raise ValueError(
            f"hot_adapter_name {hot_adapter_name!r} not in {list(LORA_ADAPTERS)}"
        )

    print("=" * 80)
    print("  Static Pre-Merged Multi-LoRA Benchmark on Modal A100 GPU")
    print(f"  Oracle Hot Adapter: {hot_adapter_name}")
    print("=" * 80)
    print(f"Base model:    {BASE_MODEL}")
    print(f"Hot ratios:    {list(hot_ratios)}")
    print(f"Num requests:  {num_requests}")
    print(f"Timestamp:     {datetime.now().isoformat()}")
    print()

    # -----------------------------------------------------------------
    # 1. Build static pre-merged profile.
    # -----------------------------------------------------------------
    profile_dir = Path(RESULTS_DIR) / "profiles_static" / f"hot_{hot_adapter_name}"
    profile_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building static profile under {profile_dir} ...")
    profile = build_profile(
        base_model=BASE_MODEL,
        hot_name=hot_adapter_name,
        hot_adapter_path=LORA_ADAPTERS[hot_adapter_name],
        cold_adapters=LORA_ADAPTERS,
        output_root=profile_dir.parent,
    )
    print(
        f"Built profile: fused={profile.fused_model_dir} "
        f"deltas={list(profile.delta_adapter_dirs)}")

    # -----------------------------------------------------------------
    # 2. Start a single vLLM server fronting the fused base + delta
    #    adapters. Reuse the existing baseline's wait-for-ready logic
    #    by inlining a tighter version that knows about our delta names.
    # -----------------------------------------------------------------
    from run_multi_lora_baseline_on_modal import wait_for_vllm_server  # noqa: E402

    lora_modules = [
        f"{cold_name}={delta_path}"
        for cold_name, delta_path in profile.delta_adapter_dirs.items()
    ]
    server_argv = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(profile.fused_model_dir),
        "--served-model-name", BASE_MODEL,
        "--port", "8000",
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.9",
        "--enable-lora",
        "--max-loras", "4",
        "--max-lora-rank", "256",
        "--lora-modules",
    ] + lora_modules
    print("Starting vLLM with command:")
    for tok in server_argv:
        print(f"  {tok}")

    server_process = subprocess.Popen(
        server_argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        model_ids, startup_logs = wait_for_vllm_server(server_process)
        print(f"vLLM ready, models={model_ids}")
        # Sanity: BASE_MODEL + each delta is registered.
        for required in [BASE_MODEL, *profile.delta_adapter_dirs]:
            if required not in model_ids:
                raise RuntimeError(
                    f"vLLM is missing expected model: {required}; got {model_ids}"
                )

        # -----------------------------------------------------------------
        # 3. Run the hot-ratio sweep.
        # -----------------------------------------------------------------
        prompts_by_style = generate_prompts_with_styles(
            num_prompts=20, styles=["formal", "casual", "technical", "story"])

        session = requests.Session()
        cold_adapters = [
            n for n in LORA_ADAPTERS if n != hot_adapter_name
        ]

        summary = {
            "model": BASE_MODEL,
            "benchmark_type": "static_premerged",
            "hot_adapter_name": hot_adapter_name,
            "hot_ratios_tested": list(hot_ratios),
            "num_requests": num_requests,
            "timestamp": datetime.now().isoformat(),
            "results": {},
        }

        for hot_ratio in hot_ratios:
            print()
            print("=" * 60)
            print(f"hot_ratio = {hot_ratio}")
            print("=" * 60)
            schedule = make_stable_skew_schedule(
                num_requests=num_requests,
                hot_adapter=hot_adapter_name,
                cold_adapters=cold_adapters,
                hot_ratio=hot_ratio,
                seed=int(hot_ratio * 100),
            )

            latencies: list[float] = []
            per_adapter: dict[str, list[float]] = {}
            errors = 0
            empty_text_responses = 0
            sample = []
            t_start = time.time()

            for entry in tqdm(schedule, desc=f"hr={hot_ratio}"):
                # Hot path: ask for the *base* model id, no LoRA.
                # Cold path: ask for the cold adapter name (= delta name).
                if entry.adapter_name == hot_adapter_name:
                    served_model = BASE_MODEL
                else:
                    served_model = entry.adapter_name
                style = style_for_adapter(entry.adapter_name)
                prompt = prompts_by_style[style][entry.prompt_index]

                t_req = time.perf_counter()
                try:
                    r = session.post(
                        "http://localhost:8000/v1/completions",
                        json={
                            "model": served_model,
                            "prompt": prompt,
                            "max_tokens": 128,
                            "temperature": 0.7,
                        },
                        timeout=30,
                    )
                except Exception as exc:
                    errors += 1
                    if len(sample) < 5:
                        sample.append({"adapter": entry.adapter_name,
                                       "error": repr(exc)})
                    continue
                latency = (time.perf_counter() - t_req) * 1000.0

                if r.status_code != 200:
                    errors += 1
                    if len(sample) < 5:
                        sample.append({"adapter": entry.adapter_name,
                                       "status": r.status_code,
                                       "body": r.text[:120]})
                    continue
                try:
                    body = r.json()
                    text = ((body.get("choices") or [{}])[0].get("text") or "").strip()
                    finish_reason = (body.get("choices") or [{}])[0].get(
                        "finish_reason")
                except Exception as exc:
                    errors += 1
                    if len(sample) < 5:
                        sample.append({"adapter": entry.adapter_name,
                                       "parse_error": repr(exc)})
                    continue
                latencies.append(latency)
                per_adapter.setdefault(entry.adapter_name, []).append(latency)
                if not text:
                    empty_text_responses += 1
                if len(sample) < 5:
                    sample.append({
                        "adapter": entry.adapter_name,
                        "served_model": served_model,
                        "latency_ms": latency,
                        "text": text[:60],
                        "finish_reason": finish_reason,
                    })

            elapsed = time.time() - t_start
            arr = np.asarray(latencies, dtype=np.float64)
            result = {
                "hot_ratio": hot_ratio,
                "duration": elapsed,
                "completed": int(arr.size),
                "errors": errors,
                "empty_text_responses": empty_text_responses,
                "request_throughput": float(arr.size / elapsed) if elapsed > 0 else 0.0,
                "mean_latency_ms": float(arr.mean()) if arr.size else 0.0,
                "median_latency_ms": float(np.median(arr)) if arr.size else 0.0,
                "p99_latency_ms": float(np.percentile(arr, 99)) if arr.size else 0.0,
                "std_latency_ms": float(arr.std()) if arr.size else 0.0,
                "per_adapter_metrics": {
                    adapter_name: {
                        "count": len(values),
                        "mean_latency_ms": float(np.mean(values))
                        if values else 0.0,
                    }
                    for adapter_name, values in per_adapter.items()
                },
                "sample_responses": sample,
            }
            summary["results"][f"hot_ratio_{hot_ratio}"] = result
            out_path = Path(RESULTS_DIR) / (
                f"static_premerged_hot_{hot_adapter_name}_ratio{hot_ratio}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved {out_path}")
            print(f"  mean_latency = {result['mean_latency_ms']:.1f}ms, "
                  f"throughput = {result['request_throughput']:.2f} req/s, "
                  f"errors = {errors}")

        summary_path = Path(RESULTS_DIR) / (
            f"static_premerged_hot_{hot_adapter_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary: {summary_path}")

    finally:
        print("\nShutting down vLLM ...")
        server_process.terminate()
        try:
            server_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_process.kill()


@app.local_entrypoint()
def main(hot_adapter_name: str = "hot_general"):
    """Entrypoint::

        modal run benchmarks/dynamic_multi_lora/run_static_premerged_on_modal.py \
            --hot-adapter-name hot_general
    """
    print(f"Launching static pre-merged benchmark with hot={hot_adapter_name}")
    run_static_premerged_benchmark.remote(hot_adapter_name=hot_adapter_name)
    print("Done. Results in Modal volume vllm-benchmark-results.")
