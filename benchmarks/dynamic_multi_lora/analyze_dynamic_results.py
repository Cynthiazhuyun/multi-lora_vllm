# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Analyzer + report builder for the multi-LoRA experiments.

Consumes JSON files produced by:

* ``run_multi_lora_baseline_on_modal.py``  (LRU baseline)
* ``run_static_premerged_on_modal.py``     (static pre-merge oracle)
* ``run_dynamic_multi_lora_on_modal.py``   (dynamic system)

and emits

* a consolidated CSV with one row per (system, experiment, hot_ratio)
* a consolidated Markdown summary table
* optional Matplotlib plots:
    - stable-skew: latency/throughput vs hot_ratio across all 4 systems
    - hot_change: per-request latency over time with switch markers
    - thrashing:  per-request latency for guarded vs naive

Run after downloading the Modal volume to a local ``results/`` dir::

    modal volume get vllm-benchmark-results / ./results
    python benchmarks/dynamic_multi_lora/analyze_dynamic_results.py \
        --results-dir ./results --plot --csv --markdown
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _maybe_load(path: Path) -> Optional[Any]:
    if path.exists():
        return _load_json(path)
    return None


@dataclass
class SystemRow:
    system: str
    experiment: str
    hot_ratio: Optional[float]
    num_requests: int
    mean_latency_ms: float
    p99_latency_ms: float
    request_throughput: float
    extra: dict[str, Any]


def load_baseline_lru(results_dir: Path) -> list[SystemRow]:
    rows: list[SystemRow] = []
    summary_path = results_dir / "baseline_multi_lora_summary_corrected.json"
    if summary_path.exists():
        data = _load_json(summary_path)
        for key, result in (data.get("results") or {}).items():
            hot_ratio = result.get("hot_ratio")
            rows.append(
                SystemRow(
                    system="vllm_lru_baseline",
                    experiment="stable_skew",
                    hot_ratio=hot_ratio,
                    num_requests=int(result.get("completed", 0)),
                    mean_latency_ms=float(result.get("mean_latency_ms", 0.0)),
                    p99_latency_ms=float(result.get("p99_latency_ms", 0.0)),
                    request_throughput=float(
                        result.get("request_throughput", 0.0)),
                    extra={
                        "errors": result.get("errors"),
                        "empty_text_responses":
                        result.get("empty_text_responses"),
                    },
                ))
        return rows
    # Fallback: per-hot-ratio files.
    for path in sorted(
            results_dir.glob("baseline_multi_lora_hot_ratio*_corrected.json")):
        result = _load_json(path)
        rows.append(
            SystemRow(
                system="vllm_lru_baseline",
                experiment="stable_skew",
                hot_ratio=float(result.get("hot_ratio")),
                num_requests=int(result.get("completed", 0)),
                mean_latency_ms=float(result.get("mean_latency_ms", 0.0)),
                p99_latency_ms=float(result.get("p99_latency_ms", 0.0)),
                request_throughput=float(
                    result.get("request_throughput", 0.0)),
                extra={"errors": result.get("errors")},
            ))
    return rows


def load_base_only(results_dir: Path) -> list[SystemRow]:
    """Load the base-model baseline (no LoRA at all).

    This is a single-point reference (independent of hot_ratio) but
    we replicate it onto every hot_ratio sample we have so it shows
    up as a flat line in the stable_skew plot.
    """
    rows: list[SystemRow] = []
    summary_path = results_dir / "baseline_base_model_summary.json"
    if not summary_path.exists():
        return rows
    data = _load_json(summary_path)
    results = data.get("results") or {}
    if not results:
        return rows
    # Pick the lowest-rate / most-comparable run. The base-model script
    # sweeps request rates; we use the slowest (most steady-state) one
    # to avoid contaminating the comparison with rate-induced queueing.
    pick = None
    for _key, val in results.items():
        if pick is None:
            pick = val
            continue
        if float(val.get("request_rate", 0.0)) < float(
                pick.get("request_rate", 0.0)):
            pick = val
    if pick is None:
        return rows
    rows.append(
        SystemRow(
            system="base_only",
            experiment="stable_skew",
            hot_ratio=None,
            num_requests=int(pick.get("completed", 0)),
            mean_latency_ms=float(pick.get("mean_latency_ms", 0.0)),
            p99_latency_ms=float(pick.get("p99_latency_ms", 0.0)),
            request_throughput=float(pick.get("request_throughput", 0.0)),
            extra={
                "request_rate": pick.get("request_rate"),
                "note": "base model only, no LoRA",
            },
        ))
    return rows


def load_lru_hot_change(results_dir: Path) -> dict[str, Any]:
    """Load the vLLM LRU multi-LoRA running on the *same* hot_change workload.

    Produced by ``run_lru_hot_change_on_modal.py``. Returns the file
    payload (so we can overlay the per-request latency curve on the
    hot_change plot) plus a SystemRow for the table.
    """
    matches = sorted(results_dir.glob("lru_E2_hot_change_*.json"))
    if not matches:
        return {"payload": None, "rows": []}
    # We only expect one (initial,second) combo per analyzer run; if
    # there are multiple just take the most recent by mtime.
    path = max(matches, key=lambda p: p.stat().st_mtime)
    payload = _load_json(path)
    s = payload.get("summary") or {}
    rows: list[SystemRow] = [
        SystemRow(
            system="vllm_lru_baseline",
            experiment="hot_change",
            hot_ratio=None,
            num_requests=int(s.get("num_successful", 0)),
            mean_latency_ms=float(s.get("mean_latency_ms", 0.0)),
            p99_latency_ms=float(s.get("p99_latency_ms", 0.0)),
            request_throughput=float(s.get("request_throughput", 0.0)),
            extra={
                "initial_hot": s.get("initial_hot"),
                "second_hot": s.get("second_hot"),
                "num_failed": s.get("num_failed"),
                "source_file": path.name,
            },
        )
    ]
    return {"payload": payload, "rows": rows}


def load_static_premerged(results_dir: Path) -> list[SystemRow]:
    rows: list[SystemRow] = []
    for summary_path in sorted(
            results_dir.glob("static_premerged_hot_*_summary.json")):
        data = _load_json(summary_path)
        hot_name = data.get("hot_adapter_name", "<unknown>")
        for key, result in (data.get("results") or {}).items():
            rows.append(
                SystemRow(
                    system=f"static_premerged[hot={hot_name}]",
                    experiment="stable_skew",
                    hot_ratio=float(result.get("hot_ratio")),
                    num_requests=int(result.get("completed", 0)),
                    mean_latency_ms=float(result.get("mean_latency_ms", 0.0)),
                    p99_latency_ms=float(result.get("p99_latency_ms", 0.0)),
                    request_throughput=float(
                        result.get("request_throughput", 0.0)),
                    extra={"errors": result.get("errors")},
                ))
    return rows


def load_dynamic(results_dir: Path) -> dict[str, Any]:
    """Return the full dynamic summary plus per-experiment SystemRows."""
    summary_path = results_dir / "dynamic_multi_lora_summary.json"
    if not summary_path.exists():
        return {"summary": None, "rows": []}
    data = _load_json(summary_path)
    rows: list[SystemRow] = []
    experiments = data.get("experiments", {})

    # Stable_skew: one row per hot_ratio.
    for key, payload in (experiments.get("stable_skew") or {}).items():
        hot_ratio_str = key.replace("hot_ratio_", "")
        try:
            hot_ratio = float(hot_ratio_str)
        except ValueError:
            hot_ratio = None
        s = payload["summary"]
        rows.append(
            SystemRow(
                system="dynamic_multi_lora",
                experiment="stable_skew",
                hot_ratio=hot_ratio,
                num_requests=int(s.get("num_successful", 0)),
                mean_latency_ms=float(s.get("mean_latency_ms", 0.0)),
                p99_latency_ms=float(s.get("p99_latency_ms", 0.0)),
                request_throughput=float(s.get("request_throughput", 0.0)),
                extra={
                    "fast_path_share": s.get("fast_path_share"),
                    "num_switch_commits": s.get("num_switch_commits"),
                },
            ))

    # hot_change: single row.
    hc = experiments.get("hot_change")
    if hc:
        s = hc["summary"]
        rows.append(
            SystemRow(
                system="dynamic_multi_lora",
                experiment="hot_change",
                hot_ratio=None,
                num_requests=int(s.get("num_successful", 0)),
                mean_latency_ms=float(s.get("mean_latency_ms", 0.0)),
                p99_latency_ms=float(s.get("p99_latency_ms", 0.0)),
                request_throughput=float(s.get("request_throughput", 0.0)),
                extra={
                    "num_switch_commits": s.get("num_switch_commits"),
                    "fast_path_share": s.get("fast_path_share"),
                },
            ))

    # thrashing: one row per guard variant.
    for variant in ("guarded", "naive"):
        t = experiments.get(f"thrashing_{variant}")
        if t:
            s = t["summary"]
            rows.append(
                SystemRow(
                    system=f"dynamic_multi_lora[thrash_{variant}]",
                    experiment="thrashing",
                    hot_ratio=None,
                    num_requests=int(s.get("num_successful", 0)),
                    mean_latency_ms=float(s.get("mean_latency_ms", 0.0)),
                    p99_latency_ms=float(s.get("p99_latency_ms", 0.0)),
                    request_throughput=float(s.get("request_throughput", 0.0)),
                    extra={
                        "num_switch_commits": s.get("num_switch_commits"),
                    },
                ))
    return {"summary": data, "rows": rows}


# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------


def write_csv(rows: list[SystemRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "system",
            "experiment",
            "hot_ratio",
            "num_requests",
            "mean_latency_ms",
            "p99_latency_ms",
            "request_throughput",
            "extra",
        ])
        for r in rows:
            writer.writerow([
                r.system,
                r.experiment,
                "" if r.hot_ratio is None else f"{r.hot_ratio:.2f}",
                r.num_requests,
                f"{r.mean_latency_ms:.2f}",
                f"{r.p99_latency_ms:.2f}",
                f"{r.request_throughput:.4f}",
                json.dumps(r.extra),
            ])


def write_markdown(rows: list[SystemRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_exp: dict[str, list[SystemRow]] = defaultdict(list)
    for r in rows:
        by_exp[r.experiment].append(r)

    lines: list[str] = ["# Multi-LoRA Benchmark Summary", ""]
    for exp_name in sorted(by_exp):
        lines.append(f"## Experiment: {exp_name}")
        lines.append("")
        lines.append(
            "| system | hot_ratio | num_requests | mean_latency_ms | "
            "p99_latency_ms | throughput (req/s) | notes |")
        lines.append(
            "| --- | --- | --- | --- | --- | --- | --- |")
        sorted_rows = sorted(
            by_exp[exp_name],
            key=lambda r: (r.system, r.hot_ratio if r.hot_ratio is not None
                           else -1),
        )
        for r in sorted_rows:
            extras_str = ", ".join(
                f"{k}={v}" for k, v in r.extra.items() if v is not None)
            hr = "-" if r.hot_ratio is None else f"{r.hot_ratio:.2f}"
            lines.append(
                f"| {r.system} | {hr} | {r.num_requests} | "
                f"{r.mean_latency_ms:.1f} | {r.p99_latency_ms:.1f} | "
                f"{r.request_throughput:.2f} | {extras_str} |")
        lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def print_table(rows: list[SystemRow]) -> None:
    by_exp: dict[str, list[SystemRow]] = defaultdict(list)
    for r in rows:
        by_exp[r.experiment].append(r)
    for exp_name in sorted(by_exp):
        print(f"\n=== {exp_name} ===")
        sorted_rows = sorted(
            by_exp[exp_name],
            key=lambda r: (r.system, r.hot_ratio if r.hot_ratio is not None
                           else -1),
        )
        print(
            f"{'system':40s}  {'hr':>5s}  {'reqs':>5s}  "
            f"{'mean_ms':>9s}  {'p99_ms':>9s}  {'tput':>7s}")
        for r in sorted_rows:
            hr = "-" if r.hot_ratio is None else f"{r.hot_ratio:.2f}"
            print(f"{r.system:40s}  {hr:>5s}  {r.num_requests:>5d}  "
                  f"{r.mean_latency_ms:>9.1f}  {r.p99_latency_ms:>9.1f}  "
                  f"{r.request_throughput:>7.2f}")


# ----------------------------------------------------------------------------
# Plots (optional)
# ----------------------------------------------------------------------------


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def plot_stable_skew(rows: list[SystemRow], output_path: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    by_system: dict[str, list[SystemRow]] = defaultdict(list)
    base_only_row: Optional[SystemRow] = None
    for r in rows:
        if r.experiment != "stable_skew":
            continue
        if r.system == "base_only":
            base_only_row = r
            continue
        if r.hot_ratio is None:
            continue
        by_system[r.system].append(r)

    if not by_system and base_only_row is None:
        return

    # Stable, distinguishable colors for the four systems.
    color_map = {
        "vllm_lru_baseline": "tab:red",
        "dynamic_multi_lora": "tab:green",
    }
    # Static-premerged label looks like "static_premerged[hot=hot_general]".
    for sys_name in by_system:
        if sys_name.startswith("static_premerged"):
            color_map[sys_name] = "tab:blue"

    fig, (ax_l, ax_t) = plt.subplots(1, 2, figsize=(12, 4.5))
    for system, items in sorted(by_system.items()):
        items.sort(key=lambda r: r.hot_ratio)
        xs = [r.hot_ratio for r in items]
        ys_l = [r.mean_latency_ms for r in items]
        ys_t = [r.request_throughput for r in items]
        color = color_map.get(system)
        ax_l.plot(xs, ys_l, marker="o", label=system, color=color)
        ax_t.plot(xs, ys_t, marker="o", label=system, color=color)
    if base_only_row is not None:
        ax_l.axhline(base_only_row.mean_latency_ms, color="gray",
                     linestyle="--", linewidth=1.2,
                     label=f"base_only ({base_only_row.mean_latency_ms:.0f}ms)")
        ax_t.axhline(base_only_row.request_throughput, color="gray",
                     linestyle="--", linewidth=1.2,
                     label=(f"base_only "
                            f"({base_only_row.request_throughput:.2f} req/s)"))
    ax_l.set_xlabel("hot_ratio")
    ax_l.set_ylabel("mean latency (ms)")
    ax_l.set_title("Stable skew: mean latency vs hot_ratio")
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(fontsize=8)
    ax_t.set_xlabel("hot_ratio")
    ax_t.set_ylabel("request throughput (req/s)")
    ax_t.set_title("Stable skew: throughput vs hot_ratio")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """Centered rolling mean. Returns a list the same length as ``values``."""
    if not values:
        return []
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(statistics.mean(values[lo:hi]))
    return out


def plot_hot_change(dynamic_summary: dict, output_path: Path,
                    lru_payload: Optional[dict] = None) -> None:
    """Plot E2 over wall-clock time so switch markers actually line up.

    The previous version used ``request_index`` on the x-axis, which
    decouples from real time and made the system-commit marker land
    at an arbitrary spot. We use ``send_offset_s`` (set by
    ``_run_workload``) and convert event timestamps to the same
    timeline via the workload's start-of-wall clock recorded in
    each request.

    When ``lru_payload`` is provided (output of
    ``run_lru_hot_change_on_modal.py``) we overlay the LRU
    rolling-mean curve on the same axes so the win is directly
    visible: same workload, different system.
    """
    import matplotlib.pyplot as plt  # type: ignore

    if dynamic_summary is None:
        return
    hc = (dynamic_summary.get("experiments") or {}).get("hot_change")
    if not hc:
        return
    per_req = hc["per_request"]
    if not per_req:
        return

    # Time axis = seconds since the first request of E2 was sent.
    xs_t = [r["send_offset_s"] for r in per_req]
    lat = [r["latency_ms"] for r in per_req]
    routed_to_hot = [bool(r["routed_to_hot"]) for r in per_req]

    window = max(1, len(xs_t) // 30)
    rolling = _rolling_mean(lat, window)

    events = hc["summary"].get("events") or []

    fig, ax = plt.subplots(figsize=(11, 4.8))

    # Color each per-request dot by whether it took the fast path
    # (routed to the fused-base hot) or the slow path (delta adapter).
    fast_xs = [t for t, h in zip(xs_t, routed_to_hot) if h]
    fast_ys = [v for v, h in zip(lat, routed_to_hot) if h]
    slow_xs = [t for t, h in zip(xs_t, routed_to_hot) if not h]
    slow_ys = [v for v, h in zip(lat, routed_to_hot) if not h]
    ax.scatter(fast_xs, fast_ys, s=10, alpha=0.35, color="tab:green",
               label="dynamic: fast path (hot, fused base)")
    ax.scatter(slow_xs, slow_ys, s=10, alpha=0.35, color="tab:red",
               label="dynamic: slow path (delta adapter)")
    ax.plot(xs_t, rolling, linewidth=2.2, color="black",
            label=f"dynamic rolling mean ({window})")

    # Overlay LRU multi-LoRA on the SAME workload, if we have it.
    # We plot only the rolling mean (and a light scatter underlay so
    # readers can see the spread) so the comparison is legible.
    lru_summary_for_anno: Optional[dict] = None
    if lru_payload is not None:
        lru_per_req = lru_payload.get("per_request") or []
        if lru_per_req:
            lru_xs = [r["send_offset_s"] for r in lru_per_req]
            lru_lat = [r["latency_ms"] for r in lru_per_req]
            lru_window = max(1, len(lru_xs) // 30)
            lru_rolling = _rolling_mean(lru_lat, lru_window)
            # Skip the per-request scatter for LRU: it sits in a flat
            # ~1100ms band of p99 spikes that visually dominates the
            # rolling-mean curve. The rolling mean alone tells the story.
            ax.plot(lru_xs, lru_rolling, linewidth=2.4, color="tab:red",
                    linestyle="--",
                    label=f"LRU baseline rolling mean ({lru_window})")
            lru_summary_for_anno = lru_payload.get("summary") or {}

    # Workload's hot adapter changes (segment boundaries on the time axis).
    seg_changes = []
    last_seg = None
    for r in per_req:
        seg = r.get("segment")
        if seg != last_seg:
            seg_changes.append((r["send_offset_s"], seg))
            last_seg = seg
    for t_offset, seg in seg_changes[1:]:
        ax.axvline(t_offset, color="orange", linestyle="--", alpha=0.8,
                   label=f"workload hot change @ t={t_offset:.0f}s ({seg})")

    # System switch markers, mapped from absolute timestamp -> E2-relative.
    # E2 starts at the wall time of its first request:
    #   wall_time(first_req) = ts(first_event_after_t0_only_if_in_window) - send_offset_s
    # The router emits an event with absolute ts; the per-request entries
    # only carry relative offsets, so we need ``started_at_wall`` from
    # the experiment payload. ``_run_workload`` records that as a top
    # level field.
    started_at_wall = hc.get("started_at_wall")
    if started_at_wall is None:
        # Best-effort recovery: take the earliest event ts and walk
        # back by the first send_offset_s.
        if events:
            started_at_wall = (min(e["timestamp"] for e in events) -
                               per_req[0]["send_offset_s"])
    if started_at_wall is not None:
        for ev in events:
            ts_rel = ev["timestamp"] - started_at_wall
            if ts_rel < xs_t[0] or ts_rel > xs_t[-1] + 5:
                continue
            kind = ev["kind"]
            if kind == "switch_decided":
                ax.axvline(ts_rel, color="purple", linestyle=":",
                           alpha=0.7,
                           label=(f"switch_decided @ t={ts_rel:.0f}s "
                                  f"-> {ev['payload'].get('target')}"))
            elif kind == "switch_committed":
                ax.axvline(ts_rel, color="green", linestyle="-",
                           alpha=0.8, linewidth=1.5,
                           label=(f"switch_committed @ t={ts_rel:.0f}s "
                                  f"-> {ev['payload'].get('target')}"))

    # Annotate before/after switch_committed mean latency to make the
    # win obvious. We use the system's own commit moment as the
    # cutoff (rather than the workload boundary) so the comparison is
    # apples-to-apples.
    commit_ts = next(
        (e["timestamp"] - started_at_wall for e in events
         if e["kind"] == "switch_committed" and started_at_wall is not None),
        None,
    )
    if commit_ts is not None:
        # Restrict to "segment 2" requests: those whose intended hot is
        # the new one. We approximate via the second segment label.
        seg_changes_for_anno = [r["send_offset_s"] for r in per_req
                                if "1" in str(r.get("segment", ""))]
        seg2_start = min(seg_changes_for_anno) if seg_changes_for_anno else 0.0
        before = [v for t, v in zip(xs_t, lat)
                  if seg2_start <= t < commit_ts]
        after = [v for t, v in zip(xs_t, lat) if t >= commit_ts]

        anno_lines: list[str] = []
        if before and after:
            mean_before = sum(before) / len(before)
            mean_after = sum(after) / len(after)
            anno_lines.append(
                f"dynamic seg2 pre-switch  = {mean_before:.0f} ms")
            anno_lines.append(
                f"dynamic seg2 post-switch = {mean_after:.0f} ms")
            anno_lines.append(
                f"  -> {mean_before / max(1.0, mean_after):.2f}x faster "
                "(within dynamic)")

        # Head-to-head with LRU on the SAME post-switch window.
        if lru_summary_for_anno is not None and after:
            lru_per_req = (lru_payload or {}).get("per_request") or []
            lru_after = [r["latency_ms"] for r in lru_per_req
                         if r["send_offset_s"] >= commit_ts]
            if lru_after:
                lru_mean_after = sum(lru_after) / len(lru_after)
                dyn_mean_after = sum(after) / len(after)
                anno_lines.append("")
                anno_lines.append(
                    "post-switch (same window):")
                anno_lines.append(
                    f"  LRU baseline    = {lru_mean_after:.0f} ms")
                anno_lines.append(
                    f"  dynamic         = {dyn_mean_after:.0f} ms")
                anno_lines.append(
                    f"  -> {lru_mean_after / max(1.0, dyn_mean_after):.2f}x "
                    "faster vs LRU")

        if anno_lines:
            ax.annotate(
                "\n".join(anno_lines),
                xy=(commit_ts, 0),
                xytext=(0.02, 0.62),
                textcoords="axes fraction",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="black", alpha=0.9),
            )

    # Clip y so the 0-2000ms band where the actual story lives stays
    # visible. We keep outliers as-is in the data but cap the view.
    if lat:
        cap = max(2000.0, sorted(lat)[int(len(lat) * 0.99)] * 1.05)
        ax.set_ylim(0, cap)

    ax.set_xlabel("time since start of E2 (s)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Hot LoRA change: latency over time with switch markers")
    ax.grid(True, alpha=0.3)
    # Dedup legend entries.
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, Any] = {}
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen[lab] = h
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="upper center",
              ncol=3, bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_thrashing(dynamic_summary: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    if dynamic_summary is None:
        return
    experiments = dynamic_summary.get("experiments") or {}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    have_anything = False
    for variant, color in [("guarded", "tab:blue"), ("naive", "tab:red")]:
        t = experiments.get(f"thrashing_{variant}")
        if not t:
            continue
        per_req = t["per_request"]
        xs = [r["request_index"] for r in per_req]
        lat = [r["latency_ms"] for r in per_req]
        ax.plot(xs, lat, color=color, alpha=0.4, label=f"{variant} per-req")
        if xs:
            window = max(1, len(xs) // 30)
            rolling = []
            for i in range(len(xs)):
                lo = max(0, i - window // 2)
                hi = min(len(xs), i + window // 2 + 1)
                rolling.append(statistics.mean(lat[lo:hi]))
            ax.plot(xs, rolling, color=color, linewidth=2.0,
                    label=f"{variant} rolling")
            n_commits = t["summary"].get("num_switch_commits", 0)
            ax.text(0.99, 0.95 if variant == "guarded" else 0.85,
                    f"{variant}: {n_commits} commits",
                    transform=ax.transAxes,
                    color=color, ha="right", va="top",
                    fontsize=10, weight="bold")
            have_anything = True
    if not have_anything:
        plt.close(fig)
        return
    ax.set_xlabel("request index")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Thrashing workload: guarded vs naive policy")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-LoRA results into CSV / Markdown / plots.")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path,
                        help="Defaults to <results-dir>/analysis")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.results_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[SystemRow] = []
    rows.extend(load_base_only(args.results_dir))
    rows.extend(load_baseline_lru(args.results_dir))
    rows.extend(load_static_premerged(args.results_dir))
    lru_hc = load_lru_hot_change(args.results_dir)
    rows.extend(lru_hc["rows"])
    dyn = load_dynamic(args.results_dir)
    rows.extend(dyn["rows"])

    print_table(rows)

    if args.csv:
        path = output_dir / "summary.csv"
        write_csv(rows, path)
        print(f"\nWrote CSV: {path}")
    if args.markdown:
        path = output_dir / "summary.md"
        write_markdown(rows, path)
        print(f"Wrote Markdown: {path}")
    if args.plot:
        if not _has_matplotlib():
            print("matplotlib not installed; install it to enable --plot")
            return
        plot_stable_skew(rows, output_dir / "stable_skew.png")
        print(f"Wrote {output_dir / 'stable_skew.png'}")
        plot_hot_change(dyn["summary"], output_dir / "hot_change.png",
                        lru_payload=lru_hc.get("payload"))
        print(f"Wrote {output_dir / 'hot_change.png'}")
        plot_thrashing(dyn["summary"], output_dir / "thrashing.png")
        print(f"Wrote {output_dir / 'thrashing.png'}")


if __name__ == "__main__":
    main()
