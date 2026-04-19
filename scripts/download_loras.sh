#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -c 'from benchmarks.multi_lora.run_multi_lora_baseline_on_modal import ensure_local_loras; ensure_local_loras()'
