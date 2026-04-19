# Modal Benchmarks: Base Model + Multi-LoRA

This directory contains the current working Modal benchmark flow for:

- base-model-only serving
- multi-LoRA serving

## `alpaca-lora/`Models In Use

### Base model

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

This is the base model used by both benchmark scripts.

### LoRAs used by the multi-LoRA benchmark

The current multi-LoRA benchmark uses local adapters from this repo:

- `loras/hot_general`
- `loras/cold_dummy`
- `loras/cold_instruct`

Inside the Modal container they are mounted to:

- `/root/loras/hot_general`
- `/root/loras/cold_dummy`
- `/root/loras/cold_instruct`

If one or more adapters are missing locally, the benchmark script now tries to
download them before packaging the Modal image.

Download source resolution order:

- `MULTI_LORA_<ADAPTER_NAME>_SOURCE`, for example
  `MULTI_LORA_HOT_GENERAL_SOURCE`
- `benchmarks/multi_lora/lora_sources.json`
- built-in defaults for adapters that already have a known source

Supported source formats:

- `hf://owner/repo`
- direct archive URL ending in `.zip`, `.tar`, `.tar.gz`, or `.tgz`

Manual prefetch command:

```bash
bash scripts/download_loras.sh
```

## Scripts

- `run_base_model_on_modal.py`
  - real base-model benchmark on Modal A100
- `run_multi_lora_baseline_on_modal.py`
  - real multi-LoRA benchmark on Modal A100
- `analyze_baseline_results.py`
  - analyzer for the multi-LoRA hot-ratio results

## How To Run

Run from the repository root.

### 1. Base model benchmark

```bash
modal run benchmarks/multi_lora/run_base_model_on_modal.py 2>&1 | tee benchmarks/multi_lora/base_model_run.log
```

What it does:

- starts a real vLLM server on Modal A100
- validates the server with 3 sample requests
- benchmarks 200 real requests at:
  - `0.5 req/s`
  - `1.0 req/s`
  - `2.0 req/s`
  - `max`

### 2. Multi-LoRA benchmark

```bash
modal run benchmarks/multi_lora/run_multi_lora_baseline_on_modal.py 2>&1 | tee benchmarks/multi_lora/multi_lora_model_run.log
```

What it does:

- mounts local LoRAs into `/root/loras`
- starts vLLM with static LoRA registration
- verifies `hot_general`, `cold_dummy`, and `cold_instruct`
- runs 2 sample requests per adapter
- benchmarks 200 real requests for each:
  - `hot_ratio=0.0`
  - `hot_ratio=0.2`
  - `hot_ratio=0.4`
  - `hot_ratio=0.6`
  - `hot_ratio=0.8`
  - `hot_ratio=1.0`

## Download Results

List files in the Volume root:

```bash
modal volume ls vllm-benchmark-results /
```

Download everything from the Volume root:

```bash
mkdir -p results
modal volume get vllm-benchmark-results / ./results
```

Important:

- the container writes to `/results/...`
- but in the Modal Volume CLI, those files live at the Volume root `/`

## Expected Output Files

### Base model

```text
baseline_base_model_rate0.5_requests200_out128.json
baseline_base_model_rate1.0_requests200_out128.json
baseline_base_model_rate2.0_requests200_out128.json
baseline_base_model_ratemax_requests200_out128.json
baseline_base_model_summary.json
```

### Multi-LoRA

```text
baseline_multi_lora_hot_ratio0.0_corrected.json
baseline_multi_lora_hot_ratio0.2_corrected.json
baseline_multi_lora_hot_ratio0.4_corrected.json
baseline_multi_lora_hot_ratio0.6_corrected.json
baseline_multi_lora_hot_ratio0.8_corrected.json
baseline_multi_lora_hot_ratio1.0_corrected.json
baseline_multi_lora_summary_corrected.json
```

## Current Takeaways

### Base model

- benchmark runs successfully end to end
- request-rate control is working
- `Errors: 0` in the current run
- there are still many `empty_text_responses`

### Multi-LoRA

- benchmark runs successfully end to end
- all three adapters register correctly
- hot ratio trends are visible:
  - higher `hot_ratio` gives lower latency
  - higher `hot_ratio` gives higher throughput
- `Errors: 0` in the current run
- there are still many `empty_text_responses`

## Analyze Multi-LoRA Results

```bash
python benchmarks/multi_lora/analyze_baseline_results.py --results-dir ./results --plot --csv
```
