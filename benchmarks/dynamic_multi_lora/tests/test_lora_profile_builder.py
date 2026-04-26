# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness tests for the delta-adapter math in
``lora_profile_builder.build_delta_state_dict``.

Why these tests matter
----------------------
The whole dynamic system rests on the claim that for any hidden state
``x`` and any cold adapter C against a base fused with hot adapter H,
the relation::

    W(x) + C(x)  ==  (W + H)(x) + delta_{C-H}(x)

holds *exactly* (modulo floating-point error). If the delta math is
off, our cold-path responses become silently wrong — which would
totally invalidate the throughput / latency story.

We exercise the three per-layer cases independently using small
synthetic LoRAs on CPU (no GPU, no PEFT, no real model), so these
tests run in a few milliseconds and catch regressions at the unit
level.

Run from the repo root::

    python -m pytest benchmarks/dynamic_multi_lora/tests/test_lora_profile_builder.py -q
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import pytest

torch = pytest.importorskip("torch")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from lora_profile_builder import build_delta_state_dict  # noqa: E402


def _make_lora(layer_to_rank: dict[str, int],
               *,
               d_in: int,
               d_out: int,
               alpha: int,
               seed: int) -> dict[str, torch.Tensor]:
    """Build a fake LoRA state dict with the given layers."""
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    for layer_name, r in layer_to_rank.items():
        prefix = f"base_model.model.{layer_name}"
        sd[f"{prefix}.lora_A.weight"] = torch.randn(r,
                                                    d_in,
                                                    generator=g,
                                                    dtype=torch.float32)
        sd[f"{prefix}.lora_B.weight"] = torch.randn(d_out,
                                                    r,
                                                    generator=g,
                                                    dtype=torch.float32)
    return sd


def _config(r: int, alpha: int, target_modules: list[str]) -> dict[str, Any]:
    return {
        "r": r,
        "lora_alpha": alpha,
        "target_modules": target_modules,
        "peft_type": "LORA",
    }


def _lora_output(state_dict: dict[str, torch.Tensor], scale: float, layer: str,
                 x: torch.Tensor, *, d_out: int) -> torch.Tensor:
    """Apply one LoRA layer: ``y = scale * B @ A @ x`` (zero if absent)."""
    prefix = f"base_model.model.{layer}"
    a_key = f"{prefix}.lora_A.weight"
    b_key = f"{prefix}.lora_B.weight"
    if a_key not in state_dict or b_key not in state_dict:
        return torch.zeros(d_out, dtype=x.dtype)
    A = state_dict[a_key]
    B = state_dict[b_key]
    return scale * (B @ (A @ x))


def _delta_output_for_layer(delta_sd: dict[str, torch.Tensor], layer: str,
                            x: torch.Tensor, *,
                            d_out: int) -> torch.Tensor:
    prefix = f"base_model.model.{layer}"
    a_key = f"{prefix}.lora_A.weight"
    b_key = f"{prefix}.lora_B.weight"
    if a_key not in delta_sd or b_key not in delta_sd:
        return torch.zeros(d_out, dtype=x.dtype)
    A = delta_sd[a_key]
    B = delta_sd[b_key]
    # By construction, alpha == r in the delta config so scale = 1.
    return B @ (A @ x)


@pytest.mark.parametrize(
    "case",
    ["both", "only_hot", "only_cold", "mixed"],
)
def test_delta_matches_cold_minus_hot(case: str) -> None:
    """For every per-layer case, ``delta(x) == cold(x) - hot(x)``."""
    d_in = 16
    d_out = 8
    hot_alpha = 16
    cold_alpha = 8

    if case == "both":
        layer_to_rank_hot = {"q_proj": 4}
        layer_to_rank_cold = {"q_proj": 8}
        target_layers = ["q_proj"]
    elif case == "only_hot":
        layer_to_rank_hot = {"q_proj": 4}
        layer_to_rank_cold = {"v_proj": 4}
        target_layers = ["q_proj", "v_proj"]
    elif case == "only_cold":
        layer_to_rank_hot = {"v_proj": 4}
        layer_to_rank_cold = {"q_proj": 8}
        target_layers = ["q_proj", "v_proj"]
    else:  # mixed: q in both, k only in hot, v only in cold
        layer_to_rank_hot = {"q_proj": 4, "k_proj": 4}
        layer_to_rank_cold = {"q_proj": 8, "v_proj": 8}
        target_layers = ["q_proj", "k_proj", "v_proj"]

    hot_w = _make_lora(layer_to_rank_hot,
                       d_in=d_in,
                       d_out=d_out,
                       alpha=hot_alpha,
                       seed=1)
    cold_w = _make_lora(layer_to_rank_cold,
                        d_in=d_in,
                        d_out=d_out,
                        alpha=cold_alpha,
                        seed=2)
    # Use the largest rank from this LoRA to set its r/alpha consistently.
    hot_r = next(iter(layer_to_rank_hot.values()))
    cold_r = next(iter(layer_to_rank_cold.values()))

    hot_cfg = _config(hot_r, hot_alpha, ["q_proj", "k_proj", "v_proj"])
    cold_cfg = _config(cold_r, cold_alpha, ["q_proj", "k_proj", "v_proj"])

    delta_sd, delta_cfg = build_delta_state_dict(hot_w, hot_cfg, cold_w,
                                                 cold_cfg)

    assert delta_cfg["r"] == hot_r + cold_r
    assert delta_cfg["lora_alpha"] == delta_cfg["r"], (
        "alpha must equal rank so PEFT default scale = 1")

    g = torch.Generator().manual_seed(123)
    x = torch.randn(d_in, generator=g, dtype=torch.float32)

    hot_scale = hot_cfg["lora_alpha"] / hot_cfg["r"]
    cold_scale = cold_cfg["lora_alpha"] / cold_cfg["r"]

    for layer in target_layers:
        cold_y = _lora_output(cold_w, cold_scale, layer, x, d_out=d_out)
        hot_y = _lora_output(hot_w, hot_scale, layer, x, d_out=d_out)
        expected = cold_y - hot_y
        actual = _delta_output_for_layer(delta_sd, layer, x, d_out=d_out)
        torch.testing.assert_close(
            actual,
            expected,
            atol=1e-5,
            rtol=1e-5,
            msg=lambda m: f"Mismatch on layer {layer} ({case}): {m}",
        )


def test_delta_target_modules_is_union():
    """The delta config must include every target module from hot and cold."""
    hot_w = _make_lora({"q_proj": 4}, d_in=8, d_out=4, alpha=8, seed=1)
    cold_w = _make_lora({"v_proj": 4}, d_in=8, d_out=4, alpha=8, seed=2)
    hot_cfg = _config(4, 8, ["q_proj"])
    cold_cfg = _config(4, 8, ["v_proj"])

    _, delta_cfg = build_delta_state_dict(hot_w, hot_cfg, cold_w, cold_cfg)
    assert sorted(delta_cfg["target_modules"]) == ["q_proj", "v_proj"]


def test_delta_with_identical_hot_and_cold_is_zero():
    """If H == C, the delta should be all-zero so adding it is a no-op."""
    layers = {"q_proj": 4, "k_proj": 4}
    sd = _make_lora(layers, d_in=8, d_out=4, alpha=8, seed=1)
    cfg = _config(4, 8, ["q_proj", "k_proj"])

    delta_sd, _ = build_delta_state_dict(sd, cfg, sd, cfg)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(8, generator=g, dtype=torch.float32)
    for layer in ["q_proj", "k_proj"]:
        y = _delta_output_for_layer(delta_sd, layer, x, d_out=4)
        torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-5,
                                   rtol=1e-5)


def test_delta_layers_share_a_single_rank_after_padding():
    """vLLM expects every targeted layer in a LoRA to have the same rank
    (the global ``r`` in adapter_config.json). For asymmetric hot/cold
    coverage we pad the missing side with zeros, so the resulting tensor
    shape on the rank axis must equal ``r_h + r_c`` everywhere.
    """
    d_in, d_out = 16, 8
    hot_r, cold_r = 4, 6
    hot_w = _make_lora({"q_proj": hot_r, "k_proj": hot_r},
                       d_in=d_in,
                       d_out=d_out,
                       alpha=hot_r * 2,
                       seed=1)
    cold_w = _make_lora({"q_proj": cold_r, "v_proj": cold_r},
                        d_in=d_in,
                        d_out=d_out,
                        alpha=cold_r * 2,
                        seed=2)
    hot_cfg = _config(hot_r, hot_r * 2, ["q_proj", "k_proj"])
    cold_cfg = _config(cold_r, cold_r * 2, ["q_proj", "v_proj"])

    delta_sd, delta_cfg = build_delta_state_dict(hot_w, hot_cfg, cold_w,
                                                 cold_cfg)
    expected_rank = hot_r + cold_r
    assert delta_cfg["r"] == expected_rank
    for key, tensor in delta_sd.items():
        if "lora_A" in key:
            assert tensor.shape[0] == expected_rank, (
                f"{key} has rank {tensor.shape[0]}, expected {expected_rank}")
        elif "lora_B" in key:
            assert tensor.shape[1] == expected_rank, (
                f"{key} has rank {tensor.shape[1]}, expected {expected_rank}")
