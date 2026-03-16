#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python3 -m exo_rl.eval.evaluate --config "${1:-configs/default.json}" --model "${2:-outputs/models/best.pt}"
