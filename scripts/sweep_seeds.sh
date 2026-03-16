#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/default.json}"
ROOT_DIR="${2:-outputs_sweep_default}"
EPISODES="${3:-40}"
if [ "$#" -ge 4 ]; then
  SEEDS=("${@:4}")
else
  SEEDS=(7 11 19)
fi
PYTHONPATH=src python3 -m exo_rl.experiments.sweep_seeds --config "$CONFIG" --root-dir "$ROOT_DIR" --episodes "$EPISODES" --seeds "${SEEDS[@]}"
