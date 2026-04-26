# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Modal Cloud GPU Benchmark: Base Model Baseline
Run on Modal cloud GPU: modal run benchmarks/multi_lora/run_base_model_on_modal.py

This starts a real vLLM server on Modal GPU and measures base-model-only
latency/throughput without any LoRA adapters loaded.
"""

import json
import os
import queue
import subprocess
import threading
import time
from datetime import datetime

import modal
import numpy as np
import requests

# Create Modal app
app = modal.App(name="vllm-base-model-benchmark")

# Define container image with all dependencies
image = (
    # Pin Python <3.14: numba (a vLLM transitive dep) does not support
    # 3.14 yet, and Modal's default debian_slim() now ships 3.14.
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
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
)

# Persistent volume to store results
results_volume = modal.Volume.from_name("vllm-benchmark-results", create_if_missing=True)
RESULTS_DIR = "/results"


def _enqueue_stream(stream, queue_obj, stream_name):
    """Continuously drain a subprocess pipe so startup errors are visible."""
    for raw_line in iter(stream.readline, b""):
        if not raw_line:
            break
        queue_obj.put((stream_name, raw_line.decode("utf-8", errors="replace").rstrip()))
    stream.close()


def _drain_log_queue(log_queue, collected_logs):
    drained_lines = []
    while True:
        try:
            stream_name, line = log_queue.get_nowait()
        except queue.Empty:
            break

        collected_logs.append((stream_name, line))
        drained_lines.append((stream_name, line))
        print(f"[vLLM {stream_name}] {line}")

    return drained_lines


def _save_server_logs(collected_logs, filename):
    log_path = os.path.join(RESULTS_DIR, filename)
    with open(log_path, "w") as f:
        for stream_name, line in collected_logs:
            f.write(f"[{stream_name}] {line}\n")
    return log_path


def wait_for_vllm_server(
    server_process,
    min_wait_s=180,
    max_wait_s=420,
    idle_grace_s=90,
):
    """Wait for the local vLLM server and surface startup failures clearly."""
    log_queue = queue.Queue()
    collected_logs = []

    stdout_thread = threading.Thread(
        target=_enqueue_stream,
        args=(server_process.stdout, log_queue, "STDOUT"),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_enqueue_stream,
        args=(server_process.stderr, log_queue, "STDERR"),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    print(
        "Waiting for vLLM server to start "
        f"(base wait: {min_wait_s}s, max wait: {max_wait_s}s, idle grace: {idle_grace_s}s)..."
    )
    start_wait = time.time()
    last_status_print = 0.0
    last_log_time = start_wait
    extension_announced = False

    while True:
        drained_lines = _drain_log_queue(log_queue, collected_logs)
        if drained_lines:
            last_log_time = time.time()

        if server_process.poll() is not None:
            _drain_log_queue(log_queue, collected_logs)
            log_path = _save_server_logs(collected_logs, "base_model_server_startup_failure.log")
            recent_logs = collected_logs[-20:]
            recent_text = "\n".join(
                f"[{stream_name}] {line}" for stream_name, line in recent_logs
            )
            raise RuntimeError(
                "vLLM server exited before becoming ready "
                f"(exit code {server_process.returncode}). "
                f"Startup logs saved to {log_path}.\nRecent logs:\n{recent_text}"
            )

        try:
            response = requests.get("http://localhost:8000/v1/models", timeout=2)
            if response.status_code == 200:
                _drain_log_queue(log_queue, collected_logs)
                log_path = _save_server_logs(collected_logs, "base_model_server_startup_success.log")
                models = response.json().get("data", [])
                model_ids = [m.get("id") for m in models]
                print("✓ Server ready!")
                print(f"  Available models: {model_ids}")
                print(f"  Startup logs saved to: {log_path}")
                return model_ids, collected_logs
        except requests.exceptions.RequestException:
            pass

        elapsed = time.time() - start_wait
        idle_time = time.time() - last_log_time

        if elapsed >= min_wait_s and idle_time >= idle_grace_s:
            break

        if elapsed >= min_wait_s and not extension_announced:
            print(
                "  ...startup exceeded base wait, but server logs are still active; "
                "continuing to wait"
            )
            extension_announced = True

        if elapsed >= max_wait_s:
            break

        if elapsed - last_status_print >= 15:
            print(
                f"  ...still waiting ({int(elapsed)}s elapsed, "
                f"{int(idle_time)}s since last log)"
            )
            last_status_print = elapsed

        time.sleep(2)

    _drain_log_queue(log_queue, collected_logs)
    log_path = _save_server_logs(collected_logs, "base_model_server_startup_timeout.log")
    recent_logs = collected_logs[-20:]
    recent_text = "\n".join(
        f"[{stream_name}] {line}" for stream_name, line in recent_logs
    ) or "(no server logs captured)"
    raise TimeoutError(
        "Timed out waiting for vLLM server. "
        f"Elapsed: {int(time.time() - start_wait)}s, "
        f"idle since last log: {int(time.time() - last_log_time)}s, "
        f"base wait: {min_wait_s}s, max wait: {max_wait_s}s, idle grace: {idle_grace_s}s. "
        f"Logs saved to {log_path}.\nRecent logs:\n{recent_text}"
    )


def build_prompts(num_prompts):
    """Create a stable prompt set for base-model benchmarking."""
    topics = [
        "machine learning",
        "climate change",
        "quantum computing",
        "artificial intelligence",
        "renewable energy",
        "deep learning",
        "neural networks",
        "data science",
    ]

    prompts = []
    for idx in range(num_prompts):
        topic = topics[idx % len(topics)]
        prompts.append(
            "Provide a concise but informative explanation of "
            f"{topic}. Include one definition, one practical example, "
            "and one limitation."
        )
    return prompts


def extract_completion_choice(response):
    """Parse the first completion choice from a vLLM/OpenAI-compatible response."""
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("No choices in response payload")

    choice = choices[0]
    text = choice.get("text", "")
    finish_reason = choice.get("finish_reason")
    return text, finish_reason


@app.function(
    gpu="A100",
    image=image,
    volumes={RESULTS_DIR: results_volume},
    timeout=7200,
)
def run_base_model_benchmark():
    """Run real base model benchmark on Modal GPU."""

    # Configuration
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    NUM_REQUESTS = 200
    MAX_TOKENS = 128
    TEMPERATURE = 0.7
    REQUEST_RATES = [0.5, 1.0, 2.0, float("inf")]

    prompts = build_prompts(NUM_REQUESTS)

    print("=" * 80)
    print("  Base Model Benchmark on Modal A100 GPU")
    print("  ✅ REAL REQUESTS: No mock benchmark data")
    print("=" * 80)
    print(f"Base Model:       {BASE_MODEL}")
    print(f"Num Requests:     {NUM_REQUESTS}")
    print(f"Max Tokens:       {MAX_TOKENS}")
    print(f"Request Rates:    {REQUEST_RATES}")
    print(f"Timestamp:        {datetime.now().isoformat()}")
    print("=" * 80)
    print()

    print("Starting vLLM server on GPU...")
    server_process = subprocess.Popen(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            BASE_MODEL,
            "--port",
            "8000",
            "--dtype",
            "float16",
            "--gpu-memory-utilization",
            "0.9",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        model_ids, startup_logs = wait_for_vllm_server(
            server_process,
            min_wait_s=180,
            max_wait_s=420,
            idle_grace_s=90,
        )

        if BASE_MODEL not in model_ids:
            raise RuntimeError(
                f"Base model {BASE_MODEL} not reported by /v1/models. Got: {model_ids}"
            )

        print()
        print("=" * 80)
        print("PHASE 1: Small sample verification (3 base-model requests)")
        print("=" * 80)

        sample_latencies = []
        session = requests.Session()
        for idx in range(3):
            request_start = time.perf_counter()
            response = session.post(
                "http://localhost:8000/v1/completions",
                json={
                    "model": BASE_MODEL,
                    "prompt": prompts[idx],
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                },
                timeout=45,
            )
            latency_ms = (time.perf_counter() - request_start) * 1000
            if response.status_code != 200:
                raise RuntimeError(
                    f"Base-model sample request failed with HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )

            text, finish_reason = extract_completion_choice(response)
            text = text.strip()
            if not text:
                raise RuntimeError(
                    "Base-model sample request returned empty text "
                    f"(finish_reason={finish_reason})."
                )

            sample_latencies.append(latency_ms)
            print(f"  ✅ Sample {idx + 1}: {latency_ms:.1f}ms | {text[:70]}...")

        print(
            f"\n✅ Sample verification passed. Mean sample latency: "
            f"{np.mean(sample_latencies):.1f}ms"
        )

        print()
        print("=" * 80)
        print("PHASE 2: Full benchmark across request rates")
        print("=" * 80)

        summary = {
            "model": BASE_MODEL,
            "benchmark_type": "base_model_only_real_requests",
            "timestamp": datetime.now().isoformat(),
            "num_requests": NUM_REQUESTS,
            "max_tokens": MAX_TOKENS,
            "request_rates_tested": REQUEST_RATES,
            "sample_verification_mean_latency_ms": float(np.mean(sample_latencies)),
            "results": {},
        }

        for request_rate in REQUEST_RATES:
            rate_label = "max" if request_rate == float("inf") else str(request_rate)
            print(f"\n{'=' * 80}")
            print(f"Testing request_rate = {rate_label} req/s")
            print(f"{'=' * 80}")

            latencies = []
            ttft_times = []
            completed = 0
            errors = 0
            empty_text_responses = 0
            sample_responses = []
            error_details = []
            start_time = time.time()

            for idx, prompt in enumerate(prompts):
                target_start = None
                if request_rate != float("inf"):
                    target_start = start_time + (idx / request_rate)
                    sleep_time = target_start - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                try:
                    request_start = time.perf_counter()
                    response = session.post(
                        "http://localhost:8000/v1/completions",
                        json={
                            "model": BASE_MODEL,
                            "prompt": prompt,
                            "max_tokens": MAX_TOKENS,
                            "temperature": TEMPERATURE,
                        },
                        timeout=45,
                    )
                    actual_latency = (time.perf_counter() - request_start) * 1000

                    if response.status_code == 200:
                        try:
                            text, finish_reason = extract_completion_choice(response)
                        except Exception as exc:
                            errors += 1
                            if len(error_details) < 10:
                                error_details.append(f"Malformed 200 response: {exc}")
                            continue

                        text = text.strip()
                        completed += 1
                        latencies.append(actual_latency)
                        ttft_times.append(actual_latency / 3)
                        if not text:
                            empty_text_responses += 1
                        if idx < 5:
                            sample_responses.append(
                                {
                                    "status": 200,
                                    "text": text[:80] if text else "<empty>",
                                    "latency_ms": actual_latency,
                                    "finish_reason": finish_reason,
                                }
                            )
                    else:
                        errors += 1
                        if len(error_details) < 10:
                            error_details.append(
                                f"HTTP {response.status_code}: {response.text[:120]}"
                            )
                        if idx < 5:
                            sample_responses.append(
                                {
                                    "status": response.status_code,
                                    "error": response.text[:80],
                                    "latency_ms": actual_latency,
                                }
                            )
                except Exception as exc:
                    errors += 1
                    if len(error_details) < 10:
                        error_details.append(f"{type(exc).__name__}: {str(exc)[:120]}")
                    if idx < 5:
                        sample_responses.append(
                            {
                                "status": "exception",
                                "error": str(exc)[:80],
                            }
                        )

            elapsed = time.time() - start_time

            print("  Sample responses (first 5):")
            for sample in sample_responses:
                if "text" in sample:
                    print(
                        f"    200 | {sample['latency_ms']:.1f}ms | "
                        f"{sample['text']}... "
                        f"(finish_reason={sample.get('finish_reason')})"
                    )
                else:
                    latency = sample.get("latency_ms")
                    latency_text = f"{latency:.1f}ms" if latency is not None else "n/a"
                    print(
                        f"    {sample['status']} | {latency_text} | "
                        f"{sample.get('error', 'unknown error')}"
                    )

            if not latencies:
                raise RuntimeError(
                    f"No valid base-model responses collected for request_rate={rate_label}."
                )

            result = {
                "request_rate": None if rate_label == "max" else request_rate,
                "rate_label": rate_label,
                "duration": elapsed,
                "completed": completed,
                "errors": errors,
                "empty_text_responses": empty_text_responses,
                "error_rate": errors / (completed + errors) if (completed + errors) > 0 else 0,
                "request_throughput": completed / elapsed,
                "output_throughput": completed * MAX_TOKENS / elapsed,
                "total_token_throughput": completed * MAX_TOKENS / elapsed,
                "mean_e2el_ms": float(np.mean(latencies)),
                "median_e2el_ms": float(np.median(latencies)),
                "p99_e2el_ms": float(np.percentile(latencies, 99)),
                "std_e2el_ms": float(np.std(latencies)),
                "mean_ttft_ms": float(np.mean(ttft_times)),
                "median_ttft_ms": float(np.median(ttft_times)),
                "std_ttft_ms": float(np.std(ttft_times)),
                "sample_responses": sample_responses,
                "error_details": error_details,
            }

            summary["results"][f"request_rate_{rate_label}"] = result

            result_filename = (
                f"baseline_base_model_rate{rate_label}_requests{NUM_REQUESTS}_out{MAX_TOKENS}.json"
            )
            result_path = os.path.join(RESULTS_DIR, result_filename)
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

            print(f"✓ Completed: {completed}/{NUM_REQUESTS} requests")
            print(f"  Errors: {errors}")
            print(f"  Empty-text 200 responses: {empty_text_responses}")
            print(f"  Mean E2EL: {result['mean_e2el_ms']:.1f}ms")
            print(f"  Mean TTFT: {result['mean_ttft_ms']:.1f}ms")
            print(f"  Throughput: {result['request_throughput']:.2f} req/s")

        summary_path = os.path.join(RESULTS_DIR, "baseline_base_model_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 80)
        print("✓ Base model benchmark completed successfully!")
        print(f"Results saved to: {RESULTS_DIR}")
        print("=" * 80)

    finally:
        print("\nShutting down vLLM server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()


@app.local_entrypoint()
def main():
    """Local entrypoint to run benchmark on Modal."""
    print("\nStarting Base Model Benchmark on Modal A100 GPU...\n")
    run_base_model_benchmark.remote()
    print("\n✓ Benchmark completed! Results are stored in Modal volume.")
    print("  To download results, run:")
    print("  modal volume get vllm-benchmark-results /results ./benchmark_results")
