# Dynamic Multi-LoRA Benchmarks on Modal

This directory hosts the **online popularity-aware dynamic multi-LoRA**
system and its experiments. The reference baselines (base-only and
vLLM Multi-LoRA / LRU) live in the sibling
[`benchmarks/multi_lora/`](../multi_lora/) directory because they are
not specific to this project; we share their code and `lora_sources.json`
without duplicating files.

Layout at a glance:

| Directory                          | Purpose                                                   |
| ---------------------------------- | --------------------------------------------------------- |
| `benchmarks/multi_lora/`           | Base-only baseline + vLLM Multi-LoRA (LRU) baseline.      |
| `benchmarks/dynamic_multi_lora/`   | **This directory** — dynamic system + static oracle + analyzer. |

This directory contains:

* `run_static_premerged_on_modal.py` — *oracle* upper bound: the hot
  LoRA is known in advance, merged into the base model, and cold
  LoRAs are served as pre-built delta adapters (the static
  pre-merge approach from the reference paper).
* `run_dynamic_multi_lora_on_modal.py` — **the headline experiment**:
  online popularity-aware routing with blue-green profile switching
  (the dynamic system).
* `lora_profile_builder.py` — offline profile builder (fused base +
  delta adapters). Padded to a single rank so vLLM's per-LoRA-rank
  assumption holds.
* `popularity_tracker.py` — sliding-window decision logic with
  `merge_threshold` / `switch_margin` / `cooldown_sec` guards.
* `dynamic_router.py` — request router + blue-green
  `ProfileSwitchManager` (vLLM subprocess lifecycle).
* `workload.py` — schedule generators for the three experiments.
* `analyze_dynamic_results.py` — CSV / Markdown / plot consolidator
  that joins **all four systems** (base-only, LRU, static, dynamic).

All four servers run on Modal A100 against the same base model
(`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) so the four systems are
directly comparable.

---

## 1. Architecture of the dynamic system

```
                                ┌──────────────────────────────┐
                                │  Workload generator (client) │
                                │  - stable_skew | hot_change  │
                                │  - thrashing                 │
                                └──────────────┬───────────────┘
                                               │ HTTP
                                               ▼
                                ┌──────────────────────────────┐
                                │   DynamicLoRARouter          │
                                │  - sliding-window popularity │
                                │  - threshold / margin /      │
                                │    cooldown switch decision  │
                                │  - routes hot → fused base   │
                                │  - routes cold → delta adptr │
                                └──┬─────────────┬─────────────┘
                          start /  │             │  HTTP forward
                          stop     │             │
                                   ▼             ▼
                ┌───────────────────────────────────────────┐
                │     vLLM (active profile, per hot)        │
                │  fused_base = base + hot LoRA  (no LoRA)  │
                │  + delta_<cold>-vs-<hot>  (rank r_h+r_c)  │
                └───────────────────────────────────────────┘
                                   ▲
                                   │
                  ┌────────────────┴────────────────┐
                  │   ProfileSwitchManager          │
                  │  - blue-green spawns vLLM with  │
                  │    new profile, then drains old │
                  └─────────────────────────────────┘
```

Key design decisions:

* **No in-place GPU weight mutation.** Profile switches happen at the
  vLLM-process level (blue-green): a new vLLM is spawned with the new
  fused base + deltas, traffic is moved to it once it is healthy, then
  the old vLLM is stopped.
* **All profiles are pre-built before serving begins.** With $n$
  candidate hots we build $n$ profiles offline. The dynamic decision
  reduces to "start the right pre-built profile".
* **Anti-thrashing is a property of the policy, not the runtime.**
  See `popularity_tracker.py`: a candidate must own >= `merge_threshold`
  of the window, beat the incumbent by >= `switch_margin`, and the
  cooldown must have elapsed since the last switch.

---

## 2. Models in use

### Base model

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

### LoRAs

| Local name      | Default HF source                                                  | Notes                          |
| --------------- | ------------------------------------------------------------------ | ------------------------------ |
| `hot_general`   | `chradden/TinyLlama-1.1B-Chat-v1.0-bf16-lora-adapter`              | r=8, alpha=16, q/v             |
| `cold_dummy`    | `thierryteisseire/TinyLlama-1.1B-Chat-v1.0-fine-tuned-adapters`    | r=8, alpha=32, q/v/k/o         |
| `cold_instruct` | `rogersam/tinyllama-instruct-lite-v1`                              | r=8, alpha=32, q/v/k           |

These three intentionally have **different `target_modules`** so
that the delta-adapter math exercises all three per-layer cases
(both / only-hot / only-cold).

Sources are looked up in this order (per adapter):

1. Env var `MULTI_LORA_<NAME>_SOURCE` (e.g. `MULTI_LORA_HOT_GENERAL_SOURCE`).
2. `benchmarks/multi_lora/lora_sources.json` (committed; the
   defaults above; shared by the LRU baseline and the dynamic system).
3. The hard-coded `DEFAULT_LORA_SOURCES` in
   `run_multi_lora_baseline_on_modal.py`.

Source format: `hf://owner/repo` or a direct `.zip`/`.tar`/`.tar.gz`
URL. Adapter directories that already exist locally are reused without
re-downloading.

Inside the Modal container, the LoRAs are mounted at
`/root/loras/<name>`, the dynamic source code at
`/root/dynamic_lora_code/`, and the baseline LRU code at
`/root/baseline_code/` (the dynamic scripts import
`wait_for_vllm_server` etc. from there).

A one-shot bootstrap is also provided:

```bash
bash scripts/download_loras.sh
```

---

## 3. How to run (in order of growing complexity)

> Run all commands from the repo root.

### 3.0  (Optional) Local sanity tests

These are pure Python (no GPU) and verify the popularity tracker
and the delta-adapter math:

```bash
python -m pytest benchmarks/dynamic_multi_lora/tests/ -q
```

Expect 23 passed in ~1 s.

### 3.1  Base-model-only baseline

```bash
modal run benchmarks/multi_lora/run_base_model_on_modal.py \
  2>&1 | tee benchmarks/multi_lora/base_model_run.log
```

### 3.2  vLLM Multi-LoRA LRU baseline

```bash
modal run benchmarks/multi_lora/run_multi_lora_baseline_on_modal.py \
  2>&1 | tee benchmarks/multi_lora/multi_lora_baseline_run.log
```

### 3.3  Static pre-merged "oracle" benchmark

```bash
modal run benchmarks/dynamic_multi_lora/run_static_premerged_on_modal.py \
  --hot-adapter-name hot_general \
  2>&1 | tee benchmarks/dynamic_multi_lora/static_premerged_run.log
```

The script will build one profile for the chosen hot adapter
inside the Modal container (a few minutes of one-time cost), then
run the same hot-ratio sweep as the LRU baseline.

### 3.4  Dynamic Multi-LoRA (the headline)

```bash
modal run benchmarks/dynamic_multi_lora/run_dynamic_multi_lora_on_modal.py \
  2>&1 | tee benchmarks/dynamic_multi_lora/dynamic_multi_lora_run.log
```

This runs three back-to-back experiments with the dynamic router:

1. **Stable skew** (`E1`): same hot-ratio sweep, drives the dynamic
   router with the existing baseline workload. Demonstrates that the
   dynamic system **does not regress** on the stable case.
2. **Hot change** (`E2`): the workload begins with `hot_general`
   dominant and switches to a different adapter halfway through.
   Demonstrates online detection + blue-green profile switch.
   **Defaults**: 600 reqs per segment at 2 req/s (≈ 5 minutes per
   segment, 10 minutes total). The blue-green switch costs ~2 minutes
   on A100, so 5-minute segments leave ≥ 2 minutes of post-switch
   fast-path observation — long enough to actually see the latency
   drop after the switch.
3. **Anti-thrashing** (`E3`): rapidly oscillating dominance, run
   twice — once with the default `(margin, cooldown)` guards
   (`guarded`) and once with both guards stripped (`naive`).

You can override sizes with CLI flags:

```bash
modal run benchmarks/dynamic_multi_lora/run_dynamic_multi_lora_on_modal.py -- \
  --initial-hot hot_general \
  --stable-skew-num-requests 100 \
  --hot-change-segment-size 600 \
  --hot-change-request-rate-per-s 2.0 \
  --thrashing-block-size 30 \
  --thrashing-num-blocks 8
```

### 3.5  Download results and analyze

```bash
mkdir -p results
modal volume get vllm-benchmark-results / ./results

python benchmarks/dynamic_multi_lora/analyze_dynamic_results.py \
  --results-dir ./results --csv --markdown --plot
```

Outputs land in `./results/analysis/`:

* `summary.csv` — one row per (system, experiment, hot_ratio).
* `summary.md` — same data as a Markdown table per experiment.
* `stable_skew.png` — latency + throughput vs hot_ratio across **all
  four** systems (base-only as a horizontal reference, LRU, static
  oracle, dynamic).
* `hot_change.png` — per-request latency on a **wall-clock** time axis
  with workload-change and system-switch markers; fast-path vs
  slow-path requests are colored separately.
* `thrashing.png` — guarded vs naive policy on the oscillating workload.

---

## 4. Source layout

```
benchmarks/dynamic_multi_lora/
├── README.md                              # this file
├── lora_profile_builder.py                # offline (fused base + delta adapters)
├── popularity_tracker.py                  # sliding-window decision logic
├── dynamic_router.py                      # router + profile switch manager
├── workload.py                            # schedule generators
├── run_static_premerged_on_modal.py       # oracle pre-merge benchmark
├── run_dynamic_multi_lora_on_modal.py     # dynamic experiments
├── analyze_dynamic_results.py             # CSV / Markdown / plots
└── tests/                                 # local pytest suite (no GPU)
    ├── test_popularity_tracker.py
    ├── test_lora_profile_builder.py
    └── test_dynamic_router.py

benchmarks/multi_lora/                     # sibling baseline (shared)
├── lora_sources.json                      # default HF mappings for LoRAs
├── run_base_model_on_modal.py             # baseline (no LoRA)
└── run_multi_lora_baseline_on_modal.py    # baseline (vLLM LRU)
```

---

## 5. Expected output files (per Modal Volume root)

Per benchmark family the volume picks up a unique prefix so we can
keep all four systems' results side-by-side:

```text
# Base only
baseline_base_model_rate*.json
baseline_base_model_summary.json

# LRU multi-LoRA baseline
baseline_multi_lora_hot_ratio*_corrected.json
baseline_multi_lora_summary_corrected.json

# Static pre-merge oracle
static_premerged_hot_<hot>_ratio*.json
static_premerged_hot_<hot>_summary.json

# Dynamic multi-LoRA
dynamic_multi_lora_E1_stable_skew_hr*.json
dynamic_multi_lora_E2_hot_change_<a>_to_<b>.json
dynamic_multi_lora_E3_thrashing_guarded.json
dynamic_multi_lora_E3_thrashing_naive.json
dynamic_multi_lora_summary.json

# vLLM startup logs and per-profile vLLM logs
vllm_logs_dynamic/vllm_<profile>_<port>.log
```

---

## 6. Tuning the dynamic policy

Knobs live in `popularity_tracker.PopularityTrackerConfig`:

| Knob               | Meaning                                                    | Default |
| ------------------ | ---------------------------------------------------------- | ------- |
| `window_size`      | Sliding window size, in requests.                          | 200     |
| `min_window_size`  | Minimum samples before any switch decision.                | 100     |
| `merge_threshold`  | Candidate must own ≥ this fraction of the window.          | 0.70    |
| `switch_margin`    | Candidate must beat current by ≥ this absolute share.      | 0.15    |
| `cooldown_sec`     | Minimum wall-clock seconds between two switches.           | 120     |

The dynamic Modal driver lowers `window_size` / `cooldown_sec` so
that the experiment fits in a single short run. For a longer
production-like run, raise both.

---

## 7. Caveats and known issues

* **Different `target_modules`** between adapters force the delta
  builder to pad with zeros along the rank axis so that vLLM's
  per-LoRA single-rank assumption holds. See
  `lora_profile_builder.build_delta_state_dict`. Tested in
  `tests/test_lora_profile_builder.py`.
* **Blue-green needs ~2× transient GPU memory** during a switch.
  TinyLlama-1.1B fp16 is small enough that this is harmless on an
  A100; for a 70B model the same architecture would need a smarter
  swap mechanism (or a single-process implementation).
* **Cold-path requests pay slightly higher per-token latency** than
  the LRU baseline because the delta adapter has rank `r_h + r_c`.
  This is the same trade-off the reference paper documents.
* **Tokenizer**: the fused profile uses the base model's tokenizer
  (we re-save it into the fused dir during profile build).
* **Switch latency on A100 is ≈ 2 minutes** (cold-start vLLM + warmup).
  E2's segments default to 5 minutes each so the post-switch window
  is ≥ 2 minutes — long enough to observe the fast-path latency
  drop. For shorter, demo-style runs, lower
  `--hot-change-segment-size` and tune `cooldown_sec` accordingly.
