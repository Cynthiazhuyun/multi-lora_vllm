#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Produce a publication-quality stable-skew figure for the MLSys report.

Usage (from repo root)::

    python benchmarks/dynamic_multi_lora/plot_stable_skew_paper.py

Optional flags::

    --results-dir PATH   (default: ./results)
    --output-dir  PATH   (default: ./results/analysis)
    --boundary    FLOAT  (default: 0.4)  discovery-phase right boundary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# matplotlib paper-quality style
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Lightweight loaders (mirrored from analyze_dynamic_results.py)
# ---------------------------------------------------------------------------

def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def load_multi_lora(results_dir: Path):
    """Return list of (hot_ratio, mean_latency_ms, throughput)."""
    d = _load(results_dir / "baseline_multi_lora_summary_corrected.json")
    rows = []
    for v in d["results"].values():
        rows.append((float(v["hot_ratio"]),
                     float(v["mean_latency_ms"]),
                     float(v["request_throughput"])))
    return sorted(rows)


def load_static_premerged(results_dir: Path):
    d = _load(results_dir / "static_premerged_hot_hot_general_summary.json")
    rows = []
    for v in d["results"].values():
        rows.append((float(v["hot_ratio"]),
                     float(v["mean_latency_ms"]),
                     float(v["request_throughput"])))
    return sorted(rows)


def load_dynamic(results_dir: Path):
    d = _load(results_dir / "dynamic_multi_lora_summary.json")
    rows = []
    for key, payload in d["experiments"]["stable_skew"].items():
        hr = float(key.replace("hot_ratio_", ""))
        s = payload["summary"]
        rows.append((hr,
                     float(s["mean_latency_ms"]),
                     float(s["request_throughput"])))
    return sorted(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(results_dir: Path, output_dir: Path, boundary: float) -> None:
    ml  = load_multi_lora(results_dir)
    sp  = load_static_premerged(results_dir)
    dyn = load_dynamic(results_dir)

    def unzip(rows):
        xs = [r[0] for r in rows]
        lat = [r[1] for r in rows]
        tput = [r[2] for r in rows]
        return xs, lat, tput

    ml_xs,  ml_lat,  ml_tput  = unzip(ml)
    sp_xs,  sp_lat,  sp_tput  = unzip(sp)
    dyn_xs, dyn_lat, dyn_tput = unzip(dyn)

    fig, (ax_l, ax_t) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    line_kwargs = dict(
        multi_lora=dict(color="tab:red",   marker="o", linestyle="-",  label="multi_lora"),
        dynamic   =dict(color="tab:blue",  marker="s", linestyle="-",  label="dynamic_multi_lora"),
        static    =dict(color="tab:green", marker="^", linestyle="--", label="static_premerged"),
    )

    # 2-adapter control point (hot_ratio=0.0 only)
    ctrl_lat  = 562.2
    ctrl_tput = 1.778
    ctrl_scatter_kw = dict(
        marker="X", s=70, color="black", edgecolors="black",
        linewidths=1.0, zorder=5,
    )

    for ax, yl, ml_y, sp_y, dyn_y, ctrl_y in [
        (ax_l, "Mean latency (ms)",  ml_lat,  sp_lat,  dyn_lat,  ctrl_lat),
        (ax_t, "Throughput (req/s)", ml_tput, sp_tput, dyn_tput, ctrl_tput),
    ]:
        ax.plot(ml_xs,  ml_y,  **line_kwargs["multi_lora"])
        ax.plot(dyn_xs, dyn_y, **line_kwargs["dynamic"])
        ax.plot(sp_xs,  sp_y,  **line_kwargs["static"])

        # 2-adapter control marker (first axis only for legend; second without)
        if ax is ax_l:
            ax.scatter([0.0], [ctrl_y], label="multi_lora (2 LoRAs, control)",
                       **ctrl_scatter_kw)
        else:
            ax.scatter([0.0], [ctrl_y], **ctrl_scatter_kw)

        # discovery phase region
        ax.axvspan(0, boundary, color="gray", alpha=0.08, zorder=0)
        ax.axvline(x=boundary, linestyle="--", color="gray",
                   linewidth=1.0, alpha=0.6)

        ax.set_xlabel("hot_ratio")
        ax.set_ylabel(yl)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, linestyle=":", alpha=0.3, zorder=0)

    # text annotations
    ax_l.text(0.07, 0.88, "no promotion\n(below threshold)",
              transform=ax_l.transAxes,
              ha="left", va="top", fontsize=8, style="italic",
              color="dimgray")

    ax_t.text(0.65, 0.10, "promotion regime",
              transform=ax_t.transAxes,
              ha="left", va="bottom", fontsize=8, style="italic",
              color="dimgray")

    # shared legend at top (4 entries → ncol=4)
    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "stable_skew_paper.pdf"
    png_path = output_dir / "stable_skew_paper.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(pdf_path.resolve())
    print(png_path.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir",  type=Path, default=None)
    parser.add_argument("--boundary",    type=float, default=0.4)
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir  = args.output_dir or (results_dir / "analysis")
    plot(results_dir, output_dir, args.boundary)


if __name__ == "__main__":
    main()
