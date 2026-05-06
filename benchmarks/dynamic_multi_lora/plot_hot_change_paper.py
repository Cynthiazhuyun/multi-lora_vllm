#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Produce a publication-quality hot-change timeseries figure (Figure 3).

Reads from the standalone E2 JSON:
  results/dynamic_multi_lora_E2_hot_change_hot_general_to_cold_dummy.json

Usage (from repo root)::

    python benchmarks/dynamic_multi_lora/plot_hot_change_paper.py

Optional flags::

    --results-dir PATH   (default: ./results)
    --output-dir  PATH   (default: ./results/analysis)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "lines.linewidth":   1.4,
    "lines.markersize":  5,
})

E2_FILE = "dynamic_multi_lora_E2_hot_change_hot_general_to_cold_dummy.json"
BASELINE_FILE = "baseline_multi_lora_summary_corrected.json"


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """Simple causal rolling mean (each point uses the preceding `window` values)."""
    out = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def plot(results_dir: Path, output_dir: Path) -> None:
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    with open(results_dir / E2_FILE) as f:
        e2 = json.load(f)

    with open(results_dir / BASELINE_FILE) as f:
        bl = json.load(f)

    started_at_wall: float = e2["started_at_wall"]
    per_req: list[dict] = e2["per_request"]
    events: list[dict] = e2["summary"]["events"]

    baseline_mean: float = bl["results"]["hot_ratio_0.0"]["mean_latency_ms"]

    # ------------------------------------------------------------------
    # Derive event offsets
    # ------------------------------------------------------------------
    def offset(ts: float) -> float:
        return ts - started_at_wall

    decided_t   = offset(next(e["timestamp"] for e in events if e["kind"] == "switch_decided"))
    committed_t = offset(next(e["timestamp"] for e in events if e["kind"] == "switch_committed"))

    # ------------------------------------------------------------------
    # Split per-request into three series
    # ------------------------------------------------------------------
    seg0    = [r for r in per_req if r["segment"] == "segment_0_hot_hot_general"]
    seg1_pre  = [r for r in per_req
                 if r["segment"] == "segment_1_hot_cold_dummy"
                 and r["send_offset_s"] < committed_t]
    seg1_post = [r for r in per_req
                 if r["segment"] == "segment_1_hot_cold_dummy"
                 and r["send_offset_s"] >= committed_t]

    def xs_ys(grp: list[dict]) -> tuple[list[float], list[float]]:
        return ([r["send_offset_s"] for r in grp],
                [r["latency_ms"] for r in grp])

    xs0,   ys0   = xs_ys(seg0)
    xs1p,  ys1p  = xs_ys(seg1_pre)
    xs1c,  ys1c  = xs_ys(seg1_post)

    ROLL = 30  # rolling-mean window in requests

    # ------------------------------------------------------------------
    # Summary statistics (printed at end)
    # ------------------------------------------------------------------
    seg0_mean  = sum(ys0)  / len(ys0)
    seg1p_mean = sum(ys1p) / len(ys1p) if ys1p else float("nan")
    seg1c_mean = sum(ys1c) / len(ys1c) if ys1c else float("nan")

    # ------------------------------------------------------------------
    # Determine y-axis range and decide log vs linear
    # ------------------------------------------------------------------
    all_lats = ys0 + ys1p + ys1c
    p99_lat = sorted(all_lats)[int(len(all_lats) * 0.99)]
    # Use log scale if p99 / baseline is > 8 (very wide range)
    use_log = (p99_lat / max(1.0, min(all_lats))) > 20

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    scatter_kw = dict(s=8, alpha=0.45, linewidths=0)

    # Series 0: pre-shift, fast path
    ax.scatter(xs0, ys0, color="tab:green", marker="o",
               label="fused-base path (pre-switch, hot=hot_general)",
               **scatter_kw)
    if xs0:
        ax.plot(xs0, _rolling_mean(ys0, ROLL),
                color="tab:green", linewidth=1.4, alpha=1.0)

    # Series 1: post-shift, pre-commit (delta adapter, new hot not yet live)
    ax.scatter(xs1p, ys1p, color="tab:red", marker="s",
               label="delta path (post-shift, pre-commit)",
               **scatter_kw)
    if xs1p:
        ax.plot(xs1p, _rolling_mean(ys1p, ROLL),
                color="tab:red", linewidth=1.4, alpha=1.0)

    # Series 2: post-commit (fused-base path, new hot fully live)
    ax.scatter(xs1c, ys1c, color="tab:blue", marker="^",
               label="fused-base path (post-commit, hot=cold_dummy)",
               **scatter_kw)
    if xs1c:
        ax.plot(xs1c, _rolling_mean(ys1c, ROLL),
                color="tab:blue", linewidth=1.4, alpha=1.0)

    # Horizontal reference: multi_lora baseline at hr=0
    ax.axhline(y=baseline_mean, linestyle=":", color="gray",
               linewidth=1.0, alpha=0.7,
               label=f"multi_lora baseline (hr=0): {baseline_mean:.0f} ms")

    # Vertical event lines
    ax.axvline(x=decided_t, linestyle="--", color="darkorange",
               linewidth=1.2, alpha=0.8)
    ax.axvline(x=committed_t, linestyle="--", color="black",
               linewidth=1.4, alpha=0.9)

    # Annotations: use axes x-transform so placement is in data coords (x)
    # but fractional coords (y), keeping labels below the top edge.
    # Place "decided" label just to the LEFT of the decided line (less
    # crowded since the bulk of the red cloud is to the right).
    ax.text(decided_t - 2, 0.06, f"decided\n(t={decided_t:.0f}s)",
            transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=7, style="italic",
            color="darkorange")
    # Place "committed" label to the LEFT of the committed line as well.
    ax.text(committed_t - 2, 0.06, f"committed\n(t={committed_t:.0f}s)",
            transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=7, style="italic",
            color="black")

    # Axes labels and style
    ax.set_xlabel("Time since experiment start (s)")
    y_label = "Latency (ms, log scale)" if use_log else "Latency (ms)"
    ax.set_ylabel(y_label)
    if use_log:
        ax.set_yscale("log")
    ax.grid(True, linestyle=":", alpha=0.3, zorder=0)

    # Legend — place upper-left where the pre-shift cloud is denser
    # but the top-left corner has more headroom than upper-right.
    ax.legend(loc="upper left", frameon=False, ncol=1, fontsize=8)

    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "hot_change_paper.pdf"
    png_path = output_dir / "hot_change_paper.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(pdf_path.resolve())
    print(png_path.resolve())

    # ------------------------------------------------------------------
    # Summary stats for caption cross-check
    # ------------------------------------------------------------------
    print()
    print(f"Decision marker:          {decided_t:.1f} s")
    print(f"Commit marker:            {committed_t:.1f} s")
    print(f"Switch overhead:          {committed_t - decided_t:.1f} s")
    print(f"Pre-shift mean latency:   {seg0_mean:.1f} ms (n={len(ys0)})")
    print(f"Pre-commit mean latency:  {seg1p_mean:.1f} ms (n={len(ys1p)})")
    print(f"Post-commit mean latency: {seg1c_mean:.1f} ms (n={len(ys1c)})")
    print(f"Speedup (pre/post mean):  {seg1p_mean / max(1.0, seg1c_mean):.2f}x")
    print(f"Speedup (baseline/post):  {baseline_mean / max(1.0, seg1c_mean):.2f}x")
    print(f"Baseline (multi_lora hr=0 mean): {baseline_mean:.1f} ms")
    print(f"Y-axis scale: {'log' if use_log else 'linear'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir",  type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir  = args.output_dir or (results_dir / "analysis")
    plot(results_dir, output_dir)


if __name__ == "__main__":
    main()
