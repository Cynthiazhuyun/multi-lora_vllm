# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline LoRA profile builder for dynamic multi-LoRA serving.

For each *candidate* hot adapter, this module builds a self-contained
"profile":

  * a fused base model (= base + hot LoRA merged into the weights)
  * a set of delta adapters for the OTHER (cold) LoRAs, expressed against
    the fused base

The fused model is consumed by vLLM as a regular base model. The delta
adapters are consumed by vLLM as regular LoRA adapters via
``--lora-modules``.

A delta adapter for cold ``C`` against fused ``(base + hot H)`` satisfies::

    (W + H) + delta_{C-H}   ==   W + C

In rank-r LoRA notation with ``H(x) = scale_H * B_H @ A_H @ x`` and
``C(x) = scale_C * B_C @ A_C @ x``, the delta is constructed so that
``delta(x) = C(x) - H(x)`` and the new adapter's ``lora_alpha`` is set
equal to its rank, which makes PEFT's default ``scale = alpha/r = 1``
(no double scaling at runtime).

Three per-layer cases are handled:

1. hot has the layer, cold does not   -> delta = -H
2. cold has the layer, hot does not   -> delta = +C
3. both have the layer                -> rank-stacked: ``delta = (scale_C*B_C | -scale_H*B_H) @ (A_C; A_H)``

Layers that neither touches are skipped.

Why a separate file from ``tools/pre_merge_hot_lora/generate_weights.py``?
That CLI builds ONE profile (one hot, several colds). For the dynamic
system we need a small Python API that can build *N* profiles (one per
candidate hot) from inside a Modal container, with progress reporting and
safetensors output that vLLM can load directly.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Keep heavy deps (torch / peft / safetensors) out of import-time so that
# this module can be inspected from a thin client process.
def _import_torch():
    import torch  # type: ignore

    return torch


def _import_safetensors():
    from safetensors.torch import load_file, save_file  # type: ignore

    return load_file, save_file


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------


@dataclass
class LoraProfile:
    """A single pre-built serving profile keyed by which adapter is HOT."""

    hot_name: str
    fused_model_dir: Path
    delta_adapter_dirs: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hot_name": self.hot_name,
            "fused_model_dir": str(self.fused_model_dir),
            "delta_adapter_dirs": {
                name: str(path)
                for name, path in self.delta_adapter_dirs.items()
            },
            "metadata": self.metadata,
        }


# ----------------------------------------------------------------------------
# IO helpers (work with either local dirs or HF repo IDs)
# ----------------------------------------------------------------------------


def _resolve_adapter_file(adapter_path: str | Path,
                          filename: str) -> Optional[Path]:
    """Find ``filename`` under ``adapter_path`` (local dir or HF repo)."""
    p = Path(str(adapter_path))
    local_candidate = p / filename
    if local_candidate.exists():
        return local_candidate

    # Treat as HF repo if it doesn't look like a path that should exist.
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        return None

    try:
        return Path(
            hf_hub_download(repo_id=str(adapter_path), filename=filename))
    except Exception:
        return None


def _load_adapter_weights(adapter_path: str | Path) -> dict[str, Any]:
    """Load ``adapter_model.{safetensors,bin}`` from a LoRA dir or HF repo.

    Returns a state dict on CPU.
    """
    torch = _import_torch()

    safetensors_path = _resolve_adapter_file(adapter_path,
                                             "adapter_model.safetensors")
    if safetensors_path is not None:
        load_file, _ = _import_safetensors()
        return load_file(str(safetensors_path), device="cpu")

    bin_path = _resolve_adapter_file(adapter_path, "adapter_model.bin")
    if bin_path is not None:
        return torch.load(str(bin_path), map_location="cpu")

    raise FileNotFoundError(
        f"Cannot find adapter_model.safetensors or adapter_model.bin under "
        f"{adapter_path}")


def _load_adapter_config(adapter_path: str | Path) -> dict[str, Any]:
    config_path = _resolve_adapter_file(adapter_path, "adapter_config.json")
    if config_path is None:
        raise FileNotFoundError(
            f"Cannot find adapter_config.json under {adapter_path}")
    with open(config_path) as f:
        return json.load(f)


def _save_adapter(state_dict: dict[str, Any], config: dict[str, Any],
                  output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, save_file = _import_safetensors()

    # Make tensors contiguous before saving (safetensors requirement).
    contiguous_sd = {k: v.contiguous() for k, v in state_dict.items()}
    save_file(contiguous_sd, str(output_dir / "adapter_model.safetensors"))

    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)


# ----------------------------------------------------------------------------
# Core math
# ----------------------------------------------------------------------------


def _module_prefixes(state_dict: dict[str, Any]) -> set[str]:
    """Extract the module prefixes shared between ``lora_A`` and ``lora_B``."""
    prefixes: set[str] = set()
    for key in state_dict:
        if ".lora_A.weight" in key:
            prefixes.add(key.replace(".lora_A.weight", ""))
        elif ".lora_A." in key:  # default_0 etc.
            prefixes.add(key.split(".lora_A.")[0])
    return prefixes


def _layer_keys(prefix: str,
                state_dict: dict[str, Any]) -> Optional[tuple[str, str]]:
    """Find the ``lora_A`` and ``lora_B`` keys that live under ``prefix``."""
    a_key = None
    b_key = None
    for key in state_dict:
        if not key.startswith(prefix + "."):
            continue
        if ".lora_A." in key and key.endswith(".weight"):
            a_key = key
        elif ".lora_B." in key and key.endswith(".weight"):
            b_key = key
    if a_key is None or b_key is None:
        return None
    return a_key, b_key


def build_delta_state_dict(
    hot_weights: dict[str, Any],
    hot_config: dict[str, Any],
    cold_weights: dict[str, Any],
    cold_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct the delta state dict and adapter config for ``cold - hot``.

    Returns ``(delta_state_dict, delta_config)``.

    The delta is mathematically equivalent to: ``cold(x) - hot(x)``. With
    the returned config (``r' = r_h + r_c``, ``alpha' = r'``), PEFT/vLLM
    will apply scale = 1 to the delta, so the output sums correctly.

    Important: vLLM's LoRA loader treats ``r`` from the config as the
    *single* rank that holds for every targeted module. To stay compatible
    we therefore pad every per-layer tensor to rank ``r_h + r_c`` even
    when the layer is only present in one side -- the padded slot is
    filled with zeros so the LoRA contribution is unchanged.
    """
    torch = _import_torch()

    hot_r = int(hot_config["r"])
    hot_alpha = float(hot_config["lora_alpha"])
    cold_r = int(cold_config["r"])
    cold_alpha = float(cold_config["lora_alpha"])
    hot_scale = hot_alpha / hot_r
    cold_scale = cold_alpha / cold_r
    new_rank = hot_r + cold_r

    delta_sd: dict[str, Any] = {}

    all_prefixes = _module_prefixes(hot_weights) | _module_prefixes(
        cold_weights)

    for prefix in sorted(all_prefixes):
        hot_keys = _layer_keys(prefix, hot_weights)
        cold_keys = _layer_keys(prefix, cold_weights)

        if hot_keys is None and cold_keys is None:
            continue

        if cold_keys is None and hot_keys is not None:
            # Hot has this layer, cold doesn't.
            #   delta(x) = -H(x) = (0, -scale_h * B_h) @ (A_c_zero; A_h) @ x
            # Pad to (r_c + r_h) on the rank axis with zeros for the
            # would-be cold half so the rank matches the global config.
            a_key, b_key = hot_keys
            A_h = hot_weights[a_key]  # (r_h, d_in)
            B_h = hot_weights[b_key]  # (d_out, r_h)
            d_in = A_h.shape[1]
            d_out = B_h.shape[0]
            A_pad = torch.zeros(cold_r, d_in, dtype=A_h.dtype,
                                device=A_h.device)
            B_pad = torch.zeros(d_out, cold_r, dtype=B_h.dtype,
                                device=B_h.device)
            new_A = torch.cat([A_pad, A_h], dim=0)
            new_B = torch.cat([B_pad, (-hot_scale) * B_h], dim=1)
            delta_sd[a_key] = new_A
            delta_sd[b_key] = new_B
            continue

        if hot_keys is None and cold_keys is not None:
            # Cold has this layer, hot doesn't.
            #   delta(x) = C(x) = (scale_c * B_c, 0) @ (A_c; A_h_zero) @ x
            a_key, b_key = cold_keys
            A_c = cold_weights[a_key]  # (r_c, d_in)
            B_c = cold_weights[b_key]  # (d_out, r_c)
            d_in = A_c.shape[1]
            d_out = B_c.shape[0]
            A_pad = torch.zeros(hot_r, d_in, dtype=A_c.dtype,
                                device=A_c.device)
            B_pad = torch.zeros(d_out, hot_r, dtype=B_c.dtype,
                                device=B_c.device)
            new_A = torch.cat([A_c, A_pad], dim=0)
            new_B = torch.cat([cold_scale * B_c, B_pad], dim=1)
            delta_sd[a_key] = new_A
            delta_sd[b_key] = new_B
            continue

        # Both have it: rank-stack into a single rank-(r_c + r_h) adapter.
        a_key_h, b_key_h = hot_keys
        a_key_c, b_key_c = cold_keys
        A_h = hot_weights[a_key_h]
        B_h = hot_weights[b_key_h]
        A_c = cold_weights[a_key_c]
        B_c = cold_weights[b_key_c]

        # Use the cold key naming so target_modules from the cold config
        # keep matching downstream.
        a_out_key = a_key_c
        b_out_key = b_key_c

        new_A = torch.cat([A_c, A_h], dim=0)  # (r_c + r_h, d_in)
        new_B = torch.cat([cold_scale * B_c, (-hot_scale) * B_h],
                          dim=1)  # (d_out, r_c + r_h)

        delta_sd[a_out_key] = new_A
        delta_sd[b_out_key] = new_B

    delta_config = dict(cold_config)
    delta_config["r"] = new_rank
    delta_config["lora_alpha"] = new_rank

    # Union of target modules across hot and cold so the LoRA can be
    # applied wherever either touched.
    hot_targets = set(hot_config.get("target_modules") or [])
    cold_targets = set(cold_config.get("target_modules") or [])
    union_targets = sorted(hot_targets | cold_targets)
    if union_targets:
        delta_config["target_modules"] = union_targets

    return delta_sd, delta_config


# ----------------------------------------------------------------------------
# Profile building
# ----------------------------------------------------------------------------


def merge_hot_into_base(base_model: str | Path, hot_adapter_path: str | Path,
                        output_dir: Path) -> Path:
    """Merge the hot LoRA into the base model with PEFT and save it.

    Returns the directory containing the fused model.
    """
    torch = _import_torch()
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)

    if (output_dir / "config.json").exists() and any(
            output_dir.glob("*.safetensors")):
        logger.info("Fused model already present at %s; skipping merge.",
                    output_dir)
        return output_dir

    logger.info("Loading base model %s ...", base_model)
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    logger.info("Loading hot adapter from %s ...", hot_adapter_path)
    model = PeftModel.from_pretrained(base, str(hot_adapter_path))
    logger.info("Merging hot adapter into base ...")
    model = model.merge_and_unload()

    logger.info("Saving fused model to %s ...", output_dir)
    model.save_pretrained(str(output_dir), safe_serialization=True)

    # Tokenizer is needed by vLLM to serve from this directory.
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(base_model),
                                                  trust_remote_code=True)
        tokenizer.save_pretrained(str(output_dir))
    except Exception as exc:
        logger.warning("Failed to copy tokenizer to fused model: %s", exc)

    # Free GPU memory before the next merge / build step.
    del model
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_dir


def build_profile(
    base_model: str | Path,
    hot_name: str,
    hot_adapter_path: str | Path,
    cold_adapters: dict[str, str | Path],
    output_root: str | Path,
    *,
    skip_existing: bool = True,
) -> LoraProfile:
    """Build one profile for ``hot_name`` and a set of cold adapters.

    Layout::

        output_root/profile_<hot_name>/
            fused_model/
            deltas/
                <cold_name>-vs-<hot_name>/
                    adapter_model.safetensors
                    adapter_config.json
            profile.json
    """
    output_root = Path(output_root)
    profile_dir = output_root / f"profile_{hot_name}"
    fused_dir = profile_dir / "fused_model"
    deltas_dir = profile_dir / "deltas"
    deltas_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fused base = base + hot.
    if skip_existing and (fused_dir / "config.json").exists() and any(
            fused_dir.glob("*.safetensors")):
        logger.info("Fused model already exists for hot=%s; reusing.",
                    hot_name)
    else:
        merge_hot_into_base(base_model, hot_adapter_path, fused_dir)

    # 2. Build delta adapters for all colds (skip the hot itself).
    hot_weights = _load_adapter_weights(hot_adapter_path)
    hot_config = _load_adapter_config(hot_adapter_path)

    delta_dirs: dict[str, Path] = {}
    for cold_name, cold_path in cold_adapters.items():
        if cold_name == hot_name:
            continue

        delta_out = deltas_dir / f"{cold_name}-vs-{hot_name}"
        if skip_existing and (delta_out /
                              "adapter_model.safetensors").exists():
            logger.info("Delta %s already exists; reusing.", delta_out)
            delta_dirs[cold_name] = delta_out
            continue

        cold_weights = _load_adapter_weights(cold_path)
        cold_config = _load_adapter_config(cold_path)
        delta_sd, delta_config = build_delta_state_dict(
            hot_weights, hot_config, cold_weights, cold_config)
        _save_adapter(delta_sd, delta_config, delta_out)
        logger.info("Wrote delta adapter to %s (%d tensors, r=%d).",
                    delta_out, len(delta_sd), delta_config["r"])
        delta_dirs[cold_name] = delta_out

    profile = LoraProfile(
        hot_name=hot_name,
        fused_model_dir=fused_dir,
        delta_adapter_dirs=delta_dirs,
        metadata={
            "base_model": str(base_model),
            "hot_adapter_path": str(hot_adapter_path),
            "cold_adapters": {n: str(p)
                              for n, p in cold_adapters.items()},
        },
    )
    with open(profile_dir / "profile.json", "w") as f:
        json.dump(profile.to_dict(), f, indent=2)
    return profile


def build_all_profiles(
    base_model: str | Path,
    adapters: dict[str, str | Path],
    output_root: str | Path,
    *,
    candidates: Optional[list[str]] = None,
    skip_existing: bool = True,
) -> dict[str, LoraProfile]:
    """Build a profile for every adapter in ``candidates`` (default: all).

    ``adapters`` is the full map of ``name -> path``. For each candidate
    name we treat that adapter as hot and build deltas for the rest.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = list(candidates) if candidates is not None else list(
        adapters)

    profiles: dict[str, LoraProfile] = {}
    for hot_name in candidates:
        if hot_name not in adapters:
            raise KeyError(f"Candidate hot {hot_name!r} not in adapters dict")
        logger.info("Building profile for hot=%s ...", hot_name)
        profiles[hot_name] = build_profile(
            base_model=base_model,
            hot_name=hot_name,
            hot_adapter_path=adapters[hot_name],
            cold_adapters=adapters,
            output_root=output_root,
            skip_existing=skip_existing,
        )
    return profiles


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _cli() -> None:
    import argparse

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build dynamic-multi-LoRA profiles offline.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Name=path mapping for an adapter; pass once per adapter.",
    )
    parser.add_argument(
        "--candidate-hot",
        action="append",
        default=None,
        help=("Adapter name to treat as hot in one profile. "
              "May be passed multiple times. Defaults to ALL adapters."),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    adapters: dict[str, str] = {}
    for spec in args.adapter:
        if "=" not in spec:
            raise SystemExit(f"--adapter must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        adapters[name] = path

    profiles = build_all_profiles(
        base_model=args.base_model,
        adapters=adapters,
        output_root=args.output_root,
        candidates=args.candidate_hot,
        skip_existing=not args.rebuild,
    )
    print(json.dumps({n: p.to_dict() for n, p in profiles.items()}, indent=2))


if __name__ == "__main__":
    _cli()
