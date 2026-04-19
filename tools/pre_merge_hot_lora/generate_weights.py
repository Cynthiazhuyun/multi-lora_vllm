# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generate a new base model with the hot LoRA adapter merged into it.
Generate delta lora weights equivalent to the cold LoRA - hot LoRA adapter.

Example:
    python tools/pre_merge_hot_lora/generate_weights.py \
        --base-model baffo32/decapoda-research-llama-7B-hf \
        --hot-adapter apaca-lora \
        --cold-adapters guanaco-lora wizard-lora \
        --output-dir models
"""
import os
import json
import argparse
from tqdm import tqdm
from typing import Dict, Any
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from peft import PeftModel


def merge_hot_lora(base_model_path: str, hot_adapter_path: str, output_dir: str):
    """Merge hot LoRA adapter into base model and save the fused model."""
    base_model_name = Path(base_model_path).name
    hot_adapter_name = Path(hot_adapter_path).name
    merged_model_path = Path(output_dir) / f"{base_model_name}-{hot_adapter_name}-fused"
    if os.path.exists(merged_model_path):
        print(f"Fused model already exists at {merged_model_path}, skipping merge.")
        return
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, hot_adapter_path)
    model = model.merge_and_unload()
    
    # save the merged model
    print(f"Saving fused model to {merged_model_path}...")
    model.save_pretrained(merged_model_path)
    print(f"Saved fused model to {merged_model_path}")

def load_adapter_weights(lora_id_or_path: str) -> Dict[str, torch.Tensor]:
    file_path = None
    
    filenames = ["adapter_model.bin", "adapter_model.safetensors"]
    
    for fname in filenames:
        # Try local path
        local_p = os.path.join(lora_id_or_path, fname)
        if os.path.exists(local_p):
            file_path = local_p
            break
        
        # Try HuggingFace Repo ID
        try:
            file_path = hf_hub_download(repo_id=lora_id_or_path, filename=fname)
            print(f"Loaded {fname} from HuggingFace: {file_path}")
            break
        except Exception:
            # If download failed, continue to try the next file
            continue

    if file_path is None:
        raise FileNotFoundError(f"Cannot find {lora_id_or_path} in local or HuggingFace")

    return torch.load(file_path, map_location="cpu")

def load_adapter_config(lora_id_or_path: str) -> dict:
    file_path = None
    filename = "adapter_config.json"
    
    # Try local path
    local_p = os.path.join(lora_id_or_path, filename)
    if os.path.exists(local_p):
        file_path = local_p
    else:
        # Try HuggingFace
        try:
            file_path = hf_hub_download(repo_id=lora_id_or_path, filename=filename)
        except Exception:
            # If download failed, continue to try the next file
            pass

    if file_path is None:
        raise FileNotFoundError(f"Cannot find {lora_id_or_path} in local or HuggingFace")

    with open(file_path, 'r') as f:
        return json.load(f)
        
def build_delta_lora(
    cold_name: str,
    hot_weights: Dict[str, torch.Tensor],
    cold_weights: Dict[str, torch.Tensor],
    hot_config: Dict[str, Any],
    cold_config: Dict[str, Any],
    output_dir: str,
):

    hot_scale = hot_config['lora_alpha'] / hot_config['r']
    cold_scale = cold_config['lora_alpha'] / cold_config['r']
    delta_state_dict = {}

    # We need to process the union of all keys involved
    all_keys = set(hot_weights.keys()) | set(cold_weights.keys())

    # Extract module prefixes
    module_prefixes = set()
    for k in all_keys:
        if "lora_A" in k:
            module_prefixes.add(k.replace(".lora_A.weight", ""))

    for prefix in tqdm(module_prefixes, desc="Building the delta LoRA weights"):
        key_A = f"{prefix}.lora_A.weight"
        key_B = f"{prefix}.lora_B.weight"

        has_hot = key_A in hot_weights
        has_cold = key_A in cold_weights

        if has_hot and not has_cold:
            A_h = hot_weights[key_A]
            B_h = hot_weights[key_B]

            delta_state_dict[key_A] = A_h
            delta_state_dict[key_B] = -1 * hot_scale * B_h

        elif has_cold and not has_hot:
            A_c = cold_weights[key_A]
            B_c = cold_weights[key_B]

            delta_state_dict[key_A] = A_c
            delta_state_dict[key_B] = cold_scale * B_c

        elif has_hot and has_cold:
            A_h = hot_weights[key_A]
            B_h = hot_weights[key_B]
            A_c = cold_weights[key_A]
            B_c = cold_weights[key_B]

            # Shape: [r_c + r_h, dim_in]
            new_A = torch.cat([A_c, A_h], dim=0)
            # Shape: [dim_out, r_c + r_h]
            new_B = torch.cat([cold_scale * B_c, -1 * hot_scale * B_h], dim=1)

            delta_state_dict[key_A] = new_A
            delta_state_dict[key_B] = new_B

    # Config Update
    new_rank = hot_config['r'] + cold_config['r']
    delta_config = cold_config.copy()
    delta_config['r'] = new_rank
    delta_config['lora_alpha'] = new_rank 

    # Save Config and Weights
    delta_save_dir = os.path.join(output_dir, f"{cold_name}-delta")
    os.makedirs(delta_save_dir, exist_ok=True)
    with open(os.path.join(delta_save_dir, "adapter_config.json"), 'w') as f:
        json.dump(delta_config, f, indent=2)

    # Save Weights
    print(f"Saving Delta Adapter to {delta_save_dir}...")
    torch.save(delta_state_dict, os.path.join(delta_save_dir, "adapter_model.bin"))
    print(f"Saved Delta Adapter to {delta_save_dir}")

def main(args: argparse.Namespace):
    # # 1 - Merge the hot LoRA adapter into the base model
    print("==============================================")
    print("  Merging the hot LoRA adapter into the base model...")
    print("==============================================")
    merge_hot_lora(args.base_model, args.hot_adapter, args.output_dir)
    
    # 2 - Build the delta LoRA weights
    print("==============================================")
    print("  Building the delta LoRA weights...")
    print("==============================================")
    hot_weights = load_adapter_weights(args.hot_adapter)
    hot_config = load_adapter_config(args.hot_adapter)

    for cold_path in args.cold_adapters:
        print(f"  Building the delta LoRA weights for {cold_path}...")
        cold_weights = load_adapter_weights(cold_path)
        cold_config = load_adapter_config(cold_path)
        cold_name = Path(cold_path).name
        if os.path.exists(os.path.join(args.output_dir, f"{cold_name}-delta")):
            print(f"Delta adapter already exists at {os.path.join(args.output_dir, f'{cold_name}-delta')}, skipping build.")
            continue
        build_delta_lora(cold_name, hot_weights, cold_weights, hot_config, cold_config, args.output_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True, help="Name of the base model.")
    parser.add_argument("--hot-adapter", type=str, required=True, help="Name of the hot LoRA adapter.")
    parser.add_argument("--cold-adapters", type=str, nargs="+", required=True, help="Names of the cold LoRA adapters.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory.")
    args = parser.parse_args()
    main(args)