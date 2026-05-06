# Final-deliverable updates (15-642 final submission)

This commit contains the artifacts and analysis added between the
midterm submission and the final report/poster.

## New experiments

- **2-adapter control** (`run_multi_lora_2adapter_control_on_modal.py`):
  vanilla `multi_lora` with only the two cold adapters loaded, run
  at `hot_ratio=0`. Used to isolate adapter-count effects from the
  popularity mechanism in the stable-skew comparison. Result:
  562 ms / 1.78 req/s — accounts for ~14% of the 305 ms gap between
  `dynamic` and `multi_lora` at `hot_ratio=0`.

- **Sensitivity sweep** (`sensitivity_sweep.py`): pure-Python replay
  of a 1500-request thrashing trace through the production
  `PopularityTracker`, varying (τ_merge, τ_margin, T_cooldown).
  Result: production defaults (0.70, 0.15, 120s) commit 0 switches;
  naive baseline commits 19. The merge threshold is the binding
  guard on this workload.

## New paper figures

- `plot_stable_skew_paper.py` → `results/analysis/stable_skew_paper.pdf`:
  paper-style Figure 2 with multi_lora / dynamic / static_premerged /
  2-LoRA-control series. Replaces the older "discovery phase"
  annotation with explicit "no promotion (below threshold)" /
  "promotion regime" shading.

- `plot_hot_change_paper.py` → `results/analysis/hot_change_paper.pdf`:
  paper-style Figure 3, log-scale per-request latency scatter
  across the three E2 phases (pre-shift fused, post-shift cold,
  post-commit fused), with vertical decided/committed markers and
  a horizontal multi_lora reference at 604 ms. Generated from the
  current E2 JSON; supersedes the figure on the original poster.

## Numerical corrections in the report

- Hot-change timestamps updated to match current JSON:
  decided ≈ 331 s, committed ≈ 486 s, switch overhead ≈ 155 s.
- Per-segment latencies reported as both raw and empty-filtered
  means (TinyLlama emits zero-token completions on ~37–41% of E2
  requests vs ~20.5% of multi_lora baseline; raw means are not
  apples-to-apples). Filtered post-commit speedup: 2.19×.
- "~7% of the low-skew gap" arithmetic error in the v3 report
  fixed to "~14%" (42 ms of 305 ms gap).
