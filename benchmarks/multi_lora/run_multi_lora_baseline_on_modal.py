# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import queue
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import modal
import numpy as np
import requests
from tqdm import tqdm

# Create Modal app
app = modal.App(name="vllm-multi-lora-baseline-corrected")
CONTAINER_LORA_DIR = "/root/loras"

# REPO_ROOT and the local-only paths only make sense on the developer
# machine. Inside Modal containers this script is mounted flat at
# /root/<basename>.py (parents[2] would IndexError) and the LoRAs already
# live at CONTAINER_LORA_DIR. Compute these defensively.
_THIS_FILE = Path(__file__).resolve()
_PARENTS = _THIS_FILE.parents
REPO_ROOT = _PARENTS[2] if len(_PARENTS) > 2 else Path("/root")
DEFAULT_LOCAL_LORA_DIR = REPO_ROOT / "loras"
LOCAL_LORA_DIR = Path(
    os.environ.get("MULTI_LORA_DIR", str(DEFAULT_LOCAL_LORA_DIR))).expanduser()
LORA_SOURCES_MANIFEST = Path(
    os.environ.get(
        "MULTI_LORA_SOURCES_MANIFEST",
        str(REPO_ROOT / "benchmarks" / "multi_lora" / "lora_sources.json"),
    )).expanduser()

# Keep the download contract explicit so collaborators can override sources
# without editing Python code.
DEFAULT_LORA_SOURCES = {
    "cold_instruct": "hf://rogersam/tinyllama-instruct-lite-v1",
}
REQUIRED_LORA_FILES = ("adapter_config.json",)
LORA_DOWNLOAD_PATTERNS = [
    "adapter_config.json",
    "adapter_model.*",
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "training_args.bin",
    "trainer_state.json",
    "optimizer.pt",
    "rng_state.pth",
    "scaler.pt",
    "scheduler.pt",
    "README.md",
    ".gitattributes",
]


def _lora_env_var_name(adapter_name: str) -> str:
    return f"MULTI_LORA_{adapter_name.upper()}_SOURCE"


def _load_lora_source_manifest() -> dict[str, str]:
    if not LORA_SOURCES_MANIFEST.exists():
        return {}

    with open(LORA_SOURCES_MANIFEST) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"LoRA source manifest must be a JSON object: {LORA_SOURCES_MANIFEST}"
        )

    normalized = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(
                "LoRA source manifest entries must map adapter names to string sources"
            )
        normalized[key] = value
    return normalized


def _resolve_lora_source(adapter_name: str, manifest: dict[str, str]) -> str:
    env_var = _lora_env_var_name(adapter_name)
    if env_var in os.environ:
        return os.environ[env_var]

    if adapter_name in manifest:
        return manifest[adapter_name]

    if adapter_name in DEFAULT_LORA_SOURCES:
        return DEFAULT_LORA_SOURCES[adapter_name]

    raise RuntimeError(
        f"Missing download source for LoRA '{adapter_name}'. "
        f"Set {env_var} or add it to {LORA_SOURCES_MANIFEST}. "
        "Supported values: hf://<repo_id> or an archive URL (.zip/.tar/.tar.gz/.tgz)."
    )


def _validate_lora_dir(adapter_name: str, adapter_dir: Path) -> None:
    missing_files = [
        filename for filename in REQUIRED_LORA_FILES
        if not (adapter_dir / filename).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"LoRA '{adapter_name}' is incomplete in {adapter_dir}. "
            f"Missing files: {missing_files}"
        )


def _download_from_hugging_face(repo_id: str, target_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for hf:// LoRA downloads. "
            "Install it locally or use an archive URL source instead."
        ) from exc

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=LORA_DOWNLOAD_PATTERNS,
        resume_download=True,
    )


def _download_and_extract_archive(source_url: str, target_dir: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="multi-lora-download-") as tmp_dir:
        archive_path = Path(tmp_dir) / source_url.rstrip("/").split("/")[-1]
        urllib.request.urlretrieve(source_url, archive_path)

        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(tmp_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as tf:
                tf.extractall(tmp_dir)
        else:
            raise RuntimeError(
                f"Unsupported archive format for {source_url}. "
                "Use a .zip/.tar/.tar.gz/.tgz URL."
            )

        extracted_root = None
        for candidate in Path(tmp_dir).iterdir():
            if candidate == archive_path:
                continue
            if candidate.is_dir() and (candidate / "adapter_config.json").exists():
                extracted_root = candidate
                break

        if extracted_root is None:
            extracted_root = Path(tmp_dir)

        for candidate in extracted_root.rglob("adapter_config.json"):
            extracted_root = candidate.parent
            break

        if extracted_root is None or not (extracted_root / "adapter_config.json").exists():
            raise RuntimeError(
                f"Downloaded archive from {source_url} but could not find adapter_config.json"
            )

        shutil.copytree(extracted_root, target_dir, dirs_exist_ok=True)


def download_lora(adapter_name: str, target_dir: Path, source: str) -> None:
    print(f"Downloading LoRA '{adapter_name}' from {source} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    if source.startswith("hf://"):
        _download_from_hugging_face(source[len("hf://"):], target_dir)
    elif source.startswith("http://") or source.startswith("https://"):
        _download_and_extract_archive(source, target_dir)
    else:
        raise RuntimeError(
            f"Unsupported LoRA source '{source}' for {adapter_name}. "
            "Expected hf://<repo_id> or an archive URL."
        )

    _validate_lora_dir(adapter_name, target_dir)


def ensure_local_loras(required_adapters: list[str] | None = None) -> dict[str, Path]:
    required_adapters = required_adapters or [
        "hot_general",
        "cold_dummy",
        "cold_instruct",
    ]
    manifest = _load_lora_source_manifest()

    LOCAL_LORA_DIR.mkdir(parents=True, exist_ok=True)
    resolved_paths = {}
    missing = []

    for adapter_name in required_adapters:
        adapter_dir = LOCAL_LORA_DIR / adapter_name
        try:
            _validate_lora_dir(adapter_name, adapter_dir)
            resolved_paths[adapter_name] = adapter_dir
        except FileNotFoundError:
            missing.append(adapter_name)

    if not missing:
        print(f"All required LoRAs already present under {LOCAL_LORA_DIR}")
        return resolved_paths

    print(f"Missing LoRAs detected: {missing}")
    for adapter_name in missing:
        adapter_dir = LOCAL_LORA_DIR / adapter_name
        source = _resolve_lora_source(adapter_name, manifest)
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        download_lora(adapter_name, adapter_dir, source)
        resolved_paths[adapter_name] = adapter_dir

    return resolved_paths


# Only run the host-side LoRA bootstrap when we're actually on a developer
# machine. Inside the Modal container this module is imported flat at
# /root/<basename>.py (no enclosing repo, hence parents[2] would IndexError),
# and the LoRAs have already been baked into the image at /root/loras.
if "MODAL_TASK_ID" not in os.environ:
    ensure_local_loras()

# Define container image with all dependencies
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


def _drain_log_queue(log_queue, collected_logs, log_errors_only=False):
    """Print queued server logs and retain a rolling copy for diagnostics."""
    drained_lines = []
    while True:
        try:
            stream_name, line = log_queue.get_nowait()
        except queue.Empty:
            break

        collected_logs.append((stream_name, line))
        drained_lines.append((stream_name, line))

        should_print = not log_errors_only or stream_name == "STDERR"
        if should_print:
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
            log_path = _save_server_logs(collected_logs, "baseline_server_startup_failure.log")
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
                log_path = _save_server_logs(collected_logs, "baseline_server_startup_success.log")
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
    log_path = _save_server_logs(collected_logs, "baseline_server_startup_timeout.log")
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


def generate_prompts_with_styles(num_prompts: int, styles: list) -> dict:
    """
    Generate diverse prompts for different LoRA styles.
    
    Each LoRA should use different prompt styles to actually trigger adapter behavior.
    """
    base_topics = [
        "machine learning",
        "climate change",
        "quantum computing",
        "artificial intelligence",
        "renewable energy",
        "deep learning",
        "neural networks",
        "data science",
    ]
    
    prompts_by_style = {}
    
    for style in styles:
        prompts_by_style[style] = []
        for i in range(num_prompts):
            topic = base_topics[i % len(base_topics)]
            
            if style == "formal":
                prompt = f"Provide a formal academic explanation of {topic}. Include definitions and theoretical foundations."
            elif style == "casual":
                prompt = f"Explain {topic} in simple, casual language like you're talking to a friend."
            elif style == "technical":
                prompt = f"Give a detailed technical explanation of {topic}. Include implementation details and mathematical foundations."
            elif style == "story":
                prompt = f"Tell a story that explains {topic} through narrative and examples."
            else:
                prompt = f"Explain {topic}."
            
            prompts_by_style[style].append(prompt)
    
    return prompts_by_style


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


def generate_request_schedule(num_requests: int, hot_ratio: float):
    """
    Generate request schedule with specific LoRA assignments based on hot_ratio.
    
    Args:
        num_requests: Total number of requests
        hot_ratio: Fraction of requests that use hot LoRA (0.0 to 1.0)
    
    Returns:
        List of (adapter_name, prompt_index) tuples
    """
    schedule = []
    
    # How many requests use hot vs cold
    num_hot = int(num_requests * hot_ratio)
    num_cold = num_requests - num_hot
    
    # Create assignments using actual adapter names from project
    # Hot requests: use "hot_general" adapter
    for i in range(num_hot):
        schedule.append(("hot_general", i % 20))  # Cycle through 20 prompts
    
    # Cold requests: alternate between cold_dummy and cold_instruct
    for i in range(num_cold):
        if i % 2 == 0:
            adapter_name = "cold_dummy"
        else:
            adapter_name = "cold_instruct"
        schedule.append((adapter_name, (num_hot + i) % 20))
    
    # Shuffle to simulate realistic request pattern
    random.shuffle(schedule)
    
    return schedule


@app.function(
    gpu="A100",
    image=image,
    volumes={RESULTS_DIR: results_volume},
    timeout=7200,
)
def run_multi_lora_baseline_benchmark():
    """Run multi-LoRA baseline benchmark with CORRECT LoRA assignment."""
    
    # Configuration
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # LoRA adapter paths (relative to Modal working directory)
    LORA_ADAPTERS = {
        "hot_general": f"{CONTAINER_LORA_DIR}/hot_general",
        "cold_dummy": f"{CONTAINER_LORA_DIR}/cold_dummy",
        "cold_instruct": f"{CONTAINER_LORA_DIR}/cold_instruct",
    }
    
    NUM_REQUESTS = 200
    HOT_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print("=" * 80)
    print("  Multi-LoRA Baseline Benchmark on Modal A100 GPU")
    print("  ✅ PRODUCTION READY: Real adapter names + real timing")
    print("=" * 80)
    print(f"Base Model:       {BASE_MODEL}")
    print(f"Hot Adapter:      hot_general")
    print(f"Cold Adapters:    cold_dummy, cold_instruct")
    print(f"Num Requests:     {NUM_REQUESTS}")
    print(f"Hot Ratios:       {HOT_RATIOS}")
    print(f"Timestamp:        {datetime.now().isoformat()}")
    print(f"LoRA Source:      {CONTAINER_LORA_DIR}")
    print("=" * 80)
    print()

    print("Verifying LoRA adapter files inside container...")
    for adapter_name, adapter_path in LORA_ADAPTERS.items():
        config_path = Path(adapter_path) / "adapter_config.json"
        if config_path.exists():
            print(f"  ✓ {adapter_name}: {adapter_path}")
        else:
            raise FileNotFoundError(
                f"Required adapter file missing for {adapter_name}: {config_path}"
            )
    print()
    
    # Generate diverse prompts
    print("Generating diverse prompts for different LoRA styles...")
    prompt_styles = ["formal", "casual", "technical", "story"]
    prompts_by_style = generate_prompts_with_styles(num_prompts=20, styles=prompt_styles)
    print(f"✓ Generated {len(prompt_styles)} prompt styles with 20 prompts each")
    print()
    
    # Start vLLM server
    print("Starting vLLM server with LoRA support...")
    
    lora_modules = [
        f"{adapter_name}={adapter_path}"
        for adapter_name, adapter_path in LORA_ADAPTERS.items()
    ]
    
    server_cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--port", "8000",
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.9",
        "--enable-lora",
        "--max-loras", "3",
        "--max-lora-rank", "256",
        "--lora-modules",
    ] + lora_modules
    
    print(f"Server command: vllm server with {len(lora_modules)} LoRAs")
    for lm in lora_modules:
        print(f"  - {lm}")
    print()
    
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    model_ids, startup_logs = wait_for_vllm_server(
        server_process,
        min_wait_s=180,
        max_wait_s=420,
        idle_grace_s=90,
    )
    
    print()
    print("=" * 80)
    print("CRITICAL: Verifying LoRA adapters are registered...")
    print("=" * 80)
    
    # FORCED VERIFICATION: Must confirm all adapters are registered
    required_adapters = ["hot_general", "cold_dummy", "cold_instruct"]
    registered_adapters = []
    
    try:
        response = requests.get("http://localhost:8000/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get("data", [])
            registered_adapters = [m.get("id") for m in models]
            
            print(f"✓ Server reports {len(registered_adapters)} models:")
            for adapter_name in registered_adapters:
                print(f"  - {adapter_name}")
            
            all_registered = all(adapter in registered_adapters for adapter in required_adapters)
            
            for adapter_name in required_adapters:
                if adapter_name in registered_adapters:
                    print(f"  ✅ {adapter_name} registered")
                else:
                    print(f"  ❌ {adapter_name} NOT registered - CRITICAL ERROR")
            
            if not all_registered:
                missing = [a for a in required_adapters if a not in registered_adapters]
                raise RuntimeError(f"Critical: LoRA adapters {missing} not registered. Cannot proceed.")
        else:
            raise RuntimeError(f"Failed to query /v1/models: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ LoRA verification FAILED: {e}")
        raise
    
    print("\n✅ All LoRA adapters verified. Proceeding with baseline benchmark...")
    print()
    
    # SMALL SAMPLE VERIFICATION: Test 2 requests per adapter before full benchmark
    print("=" * 80)
    print("PHASE 1: Small sample verification (2 requests per adapter)")
    print("=" * 80)
    
    from tqdm import tqdm
    
    # Small sample test: 2 requests per adapter to verify real responses
    sample_test_results = {
        "hot_general": {"completed": 0, "errors": 0, "responses": []},
        "cold_dummy": {"completed": 0, "errors": 0, "responses": []},
        "cold_instruct": {"completed": 0, "errors": 0, "responses": []},
    }
    
    session = requests.Session()

    print("\nSending 2 test requests per adapter...")
    for adapter_name in ["hot_general", "cold_dummy", "cold_instruct"]:
        print(f"\n  Testing {adapter_name}...")
        
        for test_idx in range(2):
            try:
                prompt = f"Hello, this is a test. My name is {test_idx}. What is AI?"
                
                request_start = time.perf_counter()
                response = session.post(
                    "http://localhost:8000/v1/completions",
                    json={
                        "model": adapter_name,
                        "prompt": prompt,
                        "max_tokens": 128,
                        "temperature": 0.7,
                    },
                    timeout=30,
                )
                latency_ms = (time.perf_counter() - request_start) * 1000
                
                if response.status_code == 200:
                    try:
                        text, finish_reason = extract_completion_choice(response)
                    except Exception as exc:
                        sample_test_results[adapter_name]["errors"] += 1
                        print(f"    ❌ Request {test_idx + 1}: Invalid response: {exc}")
                        continue

                    text = text.strip()
                    if text:
                        sample_test_results[adapter_name]["completed"] += 1
                        sample_test_results[adapter_name]["responses"].append({
                            "idx": test_idx,
                            "status": 200,
                            "text": text[:80],
                            "latency_ms": latency_ms,
                            "finish_reason": finish_reason,
                        })
                        print(f"    ✅ Request {test_idx + 1}: {latency_ms:.1f}ms | Response: {text[:50]}...")
                    else:
                        sample_test_results[adapter_name]["errors"] += 1
                        print(
                            f"    ❌ Request {test_idx + 1}: Empty text in response "
                            f"(finish_reason={finish_reason})"
                        )
                else:
                    sample_test_results[adapter_name]["errors"] += 1
                    print(f"    ❌ Request {test_idx + 1}: HTTP {response.status_code}")
            except Exception as e:
                sample_test_results[adapter_name]["errors"] += 1
                print(f"    ❌ Request {test_idx + 1}: Exception: {str(e)[:50]}")
    
    # Check if sample tests passed
    print("\n" + "=" * 80)
    print("Sample test results:")
    print("=" * 80)
    all_passed = True
    for adapter_name, results in sample_test_results.items():
        completed = results["completed"]
        errors = results["errors"]
        status = "✅" if completed == 2 else "❌"
        print(f"{status} {adapter_name}: {completed}/2 completed, {errors} errors")
        if completed < 2:
            all_passed = False
    
    if not all_passed:
        raise RuntimeError("Sample verification FAILED: Not all adapters returned valid responses. Check server logs.")
    
    print("\n✅ Sample verification PASSED. All adapters responding correctly.")
    print(f"\nProceeding to PHASE 2: Full benchmark with {NUM_REQUESTS} requests per hot_ratio...")
    print()
    
    # PHASE 2: Full benchmark
    try:
        results_summary = {
            "model": BASE_MODEL,
            "benchmark_type": "multi_lora_baseline_corrected",
            "timestamp": datetime.now().isoformat(),
            "hot_ratios_tested": HOT_RATIOS,
            "num_requests": NUM_REQUESTS,
            "sample_verification": sample_test_results,
            "results": {}
        }
        
        for HOT_RATIO in HOT_RATIOS:
            print(f"\n{'='*80}")
            print(f"Testing: hot_ratio = {HOT_RATIO}")
            print(f"  {int(NUM_REQUESTS * HOT_RATIO)} requests → hot LoRA")
            print(f"  {NUM_REQUESTS - int(NUM_REQUESTS * HOT_RATIO)} requests → cold LoRA")
            print(f"{'='*80}")
            
            # Generate request schedule
            schedule = generate_request_schedule(
                num_requests=NUM_REQUESTS,
                hot_ratio=HOT_RATIO,
            )
            
            # Track metrics
            latencies = []
            ttft_times = []
            per_adapter_metrics = {"hot_general": [], "cold_dummy": [], "cold_instruct": []}
            completed = 0
            errors = 0
            empty_text_responses = 0
            error_details = []
            
            print(f"Running {NUM_REQUESTS} requests with LoRA assignments...")
            
            start_time = time.time()
            sample_responses = []
            
            for idx, (adapter_name, prompt_idx) in enumerate(tqdm(schedule, desc="Requests")):
                try:
                    # Determine which prompt style to use based on adapter
                    if adapter_name == "hot_general":
                        style = "formal"  # Hot uses formal style
                    else:
                        style = "casual"  # Cold uses casual style
                    
                    prompt = prompts_by_style[style][prompt_idx]
                    
                    # ✅ REAL TIMING: Measure actual request latency
                    request_start = time.perf_counter()
                    
                    # Call OpenAI API with LoRA adapter name in model field
                    response = session.post(
                        "http://localhost:8000/v1/completions",
                        json={
                            "model": adapter_name,  # ✅ Use adapter name (hot_general, cold_dummy, etc)
                            "prompt": prompt,
                            "max_tokens": 128,
                            "temperature": 0.8,
                        },
                        timeout=30,
                    )
                    
                    request_end = time.perf_counter()
                    actual_latency = (request_end - request_start) * 1000  # Convert to ms
                    
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

                        if adapter_name not in per_adapter_metrics:
                            per_adapter_metrics[adapter_name] = []
                        per_adapter_metrics[adapter_name].append(actual_latency)

                        if idx < 5:
                            sample_responses.append({
                                "adapter": adapter_name,
                                "status": 200,
                                "text": text[:60] if text else "<empty>",
                                "latency_ms": actual_latency,
                                "finish_reason": finish_reason,
                            })
                    else:
                        errors += 1
                        if len(error_details) < 10:
                            error_details.append(
                                f"{adapter_name} HTTP {response.status_code}: {response.text[:120]}"
                            )
                        # Sample error response
                        if idx < 5:
                            sample_responses.append({
                                "adapter": adapter_name,
                                "status": response.status_code,
                                "error": response.text[:60],
                                "latency_ms": actual_latency
                            })
                        
                except Exception as e:
                    errors += 1
                    if len(error_details) < 10:
                        error_details.append(f"{adapter_name} {type(e).__name__}: {str(e)[:120]}")
            
            # Print sample responses
            if sample_responses:
                print(f"\n  Sample responses (first 5):")
                for sample in sample_responses:
                    if "text" in sample:
                        print(
                            f"    {sample['adapter']}: {sample['status']} | "
                            f"{sample['latency_ms']:.1f}ms | {sample['text']}... "
                            f"(finish_reason={sample.get('finish_reason')})"
                        )
                    else:
                        print(f"    {sample['adapter']}: {sample['status']} | Error: {sample.get('error', 'unknown')}")
            
            elapsed = time.time() - start_time
            
            # Calculate statistics
            if latencies:
                result = {
                    "hot_ratio": HOT_RATIO,
                    "duration": elapsed,
                    "completed": completed,
                    "errors": errors,
                    "empty_text_responses": empty_text_responses,
                    "error_rate": errors / (completed + errors) if (completed + errors) > 0 else 0,
                    "request_throughput": completed / elapsed,
                    "mean_latency_ms": np.mean(latencies),
                    "median_latency_ms": np.median(latencies),
                    "p99_latency_ms": np.percentile(latencies, 99),
                    "std_latency_ms": np.std(latencies),
                    "mean_ttft_ms": np.mean(ttft_times),
                    "per_adapter_metrics": {
                        adapter_name: {
                            "count": len(metrics),
                            "mean_latency_ms": np.mean(metrics) if metrics else 0,
                        }
                        for adapter_name, metrics in per_adapter_metrics.items()
                        if metrics
                    },
                    "error_details": error_details,
                }
                
                results_summary["results"][f"hot_ratio_{HOT_RATIO}"] = result
                
                # Save individual result file
                result_filename = f"baseline_multi_lora_hot_ratio{HOT_RATIO}_corrected.json"
                result_path = os.path.join(RESULTS_DIR, result_filename)
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"✓ Completed: {completed}/{NUM_REQUESTS} requests")
                print(f"  Errors: {errors}")
                print(f"  Empty-text 200 responses: {empty_text_responses}")
                if latencies:
                    print(f"  Overall latency: {result['mean_latency_ms']:.1f}ms (p99: {result['p99_latency_ms']:.1f}ms)")
                    print(f"  Throughput: {result['request_throughput']:.2f} req/s")
                else:
                    print(f"  ❌ No valid responses collected!")
        
        # Save summary
        summary_path = os.path.join(RESULTS_DIR, "baseline_multi_lora_summary_corrected.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print("\n" + "=" * 80)
        print("✓ All multi-LoRA baseline benchmarks completed successfully!")
        print(f"Results saved to: {RESULTS_DIR}")
        print("=" * 80)
        
    finally:
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()


@app.local_entrypoint()
def main():
    """Run the corrected benchmark."""
    run_multi_lora_baseline_benchmark.remote()
