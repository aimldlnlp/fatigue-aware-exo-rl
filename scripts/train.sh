#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python3 -m exo_rl.agents.train_ppo --config "${1:-configs/default.json}"
