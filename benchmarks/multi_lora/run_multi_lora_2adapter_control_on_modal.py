# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Control experiment: 2-adapter multi-LoRA at hot_ratio=0.0.

Identical to run_multi_lora_baseline_on_modal.py EXCEPT:
  - Only cold_dummy and cold_instruct are loaded into vLLM (NOT hot_general).
  - Only hot_ratio=0.0 is tested.
  - Results written to results/control_2adapter_multi_lora_hot_ratio0.0.json.

Everything else — vLLM engine args, num_requests, schedule, prompt style,
GPU — is unchanged so the only variable is the number of loaded LoRA slots.
"""

import json
import os
import queue
import random
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import modal
import numpy as np
import requests
from tqdm import tqdm

app = modal.App(name="vllm-multi-lora-2adapter-control")
CONTAINER_LORA_DIR = "/root/loras"

_THIS_FILE = Path(__file__).resolve()
_PARENTS = _THIS_FILE.parents
REPO_ROOT = _PARENTS[2] if len(_PARENTS) > 2 else Path("/root")
DEFAULT_LOCAL_LORA_DIR = REPO_ROOT / "loras"
LOCAL_LORA_DIR = Path(
    os.environ.get("MULTI_LORA_DIR", str(DEFAULT_LOCAL_LORA_DIR))).expanduser()

# Only require the two cold adapters for this experiment, but the full
# loras/ dir is mounted so hot_general is also present (unused).
if "MODAL_TASK_ID" not in os.environ:
    required = ["cold_dummy", "cold_instruct"]
    for name in required:
        cfg = LOCAL_LORA_DIR / name / "adapter_config.json"
        if not cfg.exists():
            raise FileNotFoundError(f"Required LoRA missing: {cfg}")
    print(f"LoRAs verified: {required}")

image = (
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
)

results_volume = modal.Volume.from_name("vllm-benchmark-results",
                                        create_if_missing=True)
RESULTS_DIR = "/results"


# ---------------------------------------------------------------------------
# Helpers (identical to baseline)
# ---------------------------------------------------------------------------

def _enqueue_stream(stream, queue_obj, stream_name):
    for raw_line in iter(stream.readline, b""):
        if not raw_line:
            break
        queue_obj.put(
            (stream_name, raw_line.decode("utf-8", errors="replace").rstrip()))
    stream.close()


def _drain_log_queue(log_queue, collected_logs, log_errors_only=False):
    drained = []
    while True:
        try:
            stream_name, line = log_queue.get_nowait()
        except queue.Empty:
            break
        collected_logs.append((stream_name, line))
        drained.append((stream_name, line))
        if not log_errors_only or stream_name == "STDERR":
            print(f"[vLLM {stream_name}] {line}")
    return drained


def _save_server_logs(collected_logs, filename):
    log_path = os.path.join(RESULTS_DIR, filename)
    with open(log_path, "w") as f:
        for stream_name, line in collected_logs:
            f.write(f"[{stream_name}] {line}\n")
    return log_path


def wait_for_vllm_server(server_process,
                         min_wait_s=180,
                         max_wait_s=420,
                         idle_grace_s=90):
    log_queue = queue.Queue()
    collected_logs = []
    for target, name in [
        (server_process.stdout, "STDOUT"),
        (server_process.stderr, "STDERR"),
    ]:
        threading.Thread(target=_enqueue_stream,
                         args=(target, log_queue, name),
                         daemon=True).start()

    print(f"Waiting for vLLM server (base: {min_wait_s}s, max: {max_wait_s}s)...")
    start_wait = time.time()
    last_status = 0.0
    last_log_time = start_wait
    extension_announced = False

    while True:
        drained = _drain_log_queue(log_queue, collected_logs)
        if drained:
            last_log_time = time.time()

        if server_process.poll() is not None:
            _drain_log_queue(log_queue, collected_logs)
            log_path = _save_server_logs(collected_logs,
                                         "control_server_startup_failure.log")
            recent = "\n".join(
                f"[{s}] {l}" for s, l in collected_logs[-20:])
            raise RuntimeError(
                f"vLLM exited (code {server_process.returncode}). "
                f"Logs: {log_path}\n{recent}")

        try:
            r = requests.get("http://localhost:8000/v1/models", timeout=2)
            if r.status_code == 200:
                _drain_log_queue(log_queue, collected_logs)
                _save_server_logs(collected_logs,
                                  "control_server_startup_success.log")
                model_ids = [m["id"]
                             for m in r.json().get("data", [])]
                print(f"✓ Server ready! models={model_ids}")
                return model_ids, collected_logs
        except requests.exceptions.RequestException:
            pass

        elapsed = time.time() - start_wait
        idle_time = time.time() - last_log_time

        if elapsed >= min_wait_s and idle_time >= idle_grace_s:
            break
        if elapsed >= min_wait_s and not extension_announced:
            print("  ...exceeded base wait; still active, continuing")
            extension_announced = True
        if elapsed >= max_wait_s:
            break
        if elapsed - last_status >= 15:
            print(f"  ...waiting ({int(elapsed)}s, idle {int(idle_time)}s)")
            last_status = elapsed
        time.sleep(2)

    _drain_log_queue(log_queue, collected_logs)
    log_path = _save_server_logs(collected_logs,
                                 "control_server_startup_timeout.log")
    raise TimeoutError(
        f"Timed out after {int(time.time()-start_wait)}s. Logs: {log_path}")


def extract_completion_choice(response):
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("No choices in response payload")
    choice = choices[0]
    return choice.get("text", ""), choice.get("finish_reason")


def generate_request_schedule(num_requests: int, hot_ratio: float):
    """Same logic as baseline. At hot_ratio=0.0 all requests are cold."""
    schedule = []
    num_hot = int(num_requests * hot_ratio)
    num_cold = num_requests - num_hot
    for i in range(num_hot):
        schedule.append(("cold_dummy", i % 20))   # placeholder hot slot
    for i in range(num_cold):
        adapter = "cold_dummy" if i % 2 == 0 else "cold_instruct"
        schedule.append((adapter, (num_hot + i) % 20))
    random.seed(42)
    random.shuffle(schedule)
    return schedule


def generate_prompts():
    base_topics = [
        "machine learning", "climate change", "quantum computing",
        "artificial intelligence", "renewable energy", "deep learning",
        "neural networks", "data science",
    ]
    prompts = []
    for i in range(20):
        topic = base_topics[i % len(base_topics)]
        prompts.append(
            f"Explain {topic} in simple, casual language like you're talking to a friend."
        )
    return prompts


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100",
    image=image,
    volumes={RESULTS_DIR: results_volume},
    timeout=3600,
)
def run_2adapter_control():
    """2-adapter control: cold_dummy + cold_instruct only, hot_ratio=0.0."""

    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # ONLY two adapters — this is the one variable we change.
    LORA_ADAPTERS = {
        "cold_dummy":    f"{CONTAINER_LORA_DIR}/cold_dummy",
        "cold_instruct": f"{CONTAINER_LORA_DIR}/cold_instruct",
    }

    NUM_REQUESTS = 200   # identical to baseline
    HOT_RATIO = 0.0

    print("=" * 80)
    print("  2-Adapter Control Benchmark (cold_dummy + cold_instruct only)")
    print("=" * 80)
    print(f"Base Model:    {BASE_MODEL}")
    print(f"Adapters:      {list(LORA_ADAPTERS)}")
    print(f"Num Requests:  {NUM_REQUESTS}")
    print(f"Hot Ratio:     {HOT_RATIO}")
    print(f"Timestamp:     {datetime.now().isoformat()}")
    print("=" * 80)

    for name, path in LORA_ADAPTERS.items():
        cfg = Path(path) / "adapter_config.json"
        if not cfg.exists():
            raise FileNotFoundError(f"Missing: {cfg}")
        print(f"  ✓ {name}: {path}")

    prompts = generate_prompts()

    # -----------------------------------------------------------------------
    # Start vLLM — same engine args as baseline, max_loras=3 unchanged.
    # -----------------------------------------------------------------------
    lora_modules = [f"{n}={p}" for n, p in LORA_ADAPTERS.items()]
    server_cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--port", "8000",
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.9",
        "--enable-lora",
        "--max-loras", "3",           # unchanged from baseline
        "--max-lora-rank", "256",     # unchanged from baseline
        "--lora-modules",
    ] + lora_modules

    print(f"\nLaunching vLLM with {len(lora_modules)} LoRA(s): {lora_modules}")
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    wait_for_vllm_server(server_process,
                         min_wait_s=180,
                         max_wait_s=420,
                         idle_grace_s=90)

    # Verify adapters
    r = requests.get("http://localhost:8000/v1/models", timeout=5)
    registered = [m["id"] for m in r.json().get("data", [])]
    for name in LORA_ADAPTERS:
        if name not in registered:
            raise RuntimeError(f"Adapter {name!r} not registered: {registered}")
        print(f"  ✅ {name} registered")

    # Small sample verification
    session = requests.Session()
    print("\nSample verification (2 requests per adapter)...")
    for name in LORA_ADAPTERS:
        for i in range(2):
            r = session.post(
                "http://localhost:8000/v1/completions",
                json={"model": name, "prompt": prompts[i],
                      "max_tokens": 128, "temperature": 0.7},
                timeout=30,
            )
            if r.status_code != 200:
                raise RuntimeError(
                    f"Sample request failed for {name}: HTTP {r.status_code}")
            text, _ = extract_completion_choice(r)
            print(f"  ✓ {name}[{i}]: {text.strip()[:50]!r}")

    # -----------------------------------------------------------------------
    # Full benchmark — hot_ratio=0.0 only
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"Full benchmark: {NUM_REQUESTS} requests, hot_ratio={HOT_RATIO}")
    print(f"{'='*80}")

    schedule = generate_request_schedule(NUM_REQUESTS, HOT_RATIO)

    latencies = []
    per_adapter: dict[str, list[float]] = {n: [] for n in LORA_ADAPTERS}
    completed = 0
    errors = 0
    empty_text = 0
    error_details: list[str] = []

    start_time = time.time()
    for idx, (adapter_name, prompt_idx) in enumerate(
            tqdm(schedule, desc="Requests")):
        try:
            t0 = time.perf_counter()
            resp = session.post(
                "http://localhost:8000/v1/completions",
                json={
                    "model": adapter_name,
                    "prompt": prompts[prompt_idx],
                    "max_tokens": 128,
                    "temperature": 0.8,
                },
                timeout=30,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            if resp.status_code == 200:
                text, _ = extract_completion_choice(resp)
                text = text.strip()
                completed += 1
                latencies.append(latency_ms)
                if adapter_name in per_adapter:
                    per_adapter[adapter_name].append(latency_ms)
                if not text:
                    empty_text += 1
            else:
                errors += 1
                if len(error_details) < 10:
                    error_details.append(
                        f"{adapter_name} HTTP {resp.status_code}: "
                        f"{resp.text[:120]}")
        except Exception as exc:
            errors += 1
            if len(error_details) < 10:
                error_details.append(
                    f"{adapter_name} {type(exc).__name__}: {str(exc)[:120]}")

    elapsed = time.time() - start_time

    result = {
        "benchmark_type": "control_2adapter_multi_lora",
        "adapters_loaded": list(LORA_ADAPTERS.keys()),
        "num_adapters_loaded": len(LORA_ADAPTERS),
        "model": BASE_MODEL,
        "timestamp": datetime.now().isoformat(),
        "hot_ratio": HOT_RATIO,
        "duration": elapsed,
        "completed": completed,
        "errors": errors,
        "empty_text_responses": empty_text,
        "error_rate": errors / max(1, completed + errors),
        "request_throughput": completed / elapsed,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "median_latency_ms": float(np.median(latencies)) if latencies else 0.0,
        "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
        "std_latency_ms": float(np.std(latencies)) if latencies else 0.0,
        "per_adapter_metrics": {
            n: {
                "count": len(v),
                "mean_latency_ms": float(np.mean(v)) if v else 0.0,
            }
            for n, v in per_adapter.items()
        },
        "error_details": error_details,
    }

    out_path = os.path.join(RESULTS_DIR,
                            "control_2adapter_multi_lora_hot_ratio0.0.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    results_volume.commit()

    print(f"\n{'='*80}")
    print("Control 2-LoRA result at hot_ratio=0.0:")
    print(f"  mean_latency_ms    = {result['mean_latency_ms']:.1f} ms")
    print(f"  request_throughput = {result['request_throughput']:.3f} req/s")
    print(f"  completed          = {completed}/{NUM_REQUESTS}")
    print(f"  errors             = {errors}")
    print(f"  result saved to    = {out_path}")
    print(f"{'='*80}")


@app.local_entrypoint()
def main():
    run_2adapter_control.remote()
