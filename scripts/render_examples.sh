#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${1:-outputs/models/best.pt}"
PYTHONPATH=src python3 -m exo_rl.viz.plot_training --history outputs/logs/training_history.json --outdir outputs/figures
PYTHONPATH=src python3 -m exo_rl.viz.animate_episode --model "$MODEL_PATH" --mp4 outputs/videos/policy_episode.mp4 --gif outputs/videos/policy_episode.gif
PYTHONPATH=src python3 -m exo_rl.viz.render_policy_comparison --model "$MODEL_PATH" --mp4 outputs/videos/policy_comparison.mp4 --gif outputs/videos/policy_comparison.gif
PYTHONPATH=src python3 -m exo_rl.viz.plot_assist_breakdown --model "$MODEL_PATH" --outdir outputs/figures
PYTHONPATH=src python3 -m exo_rl.viz.plot_trajectory_comparison --model "$MODEL_PATH" --outdir outputs/figures
PYTHONPATH=src python3 -m exo_rl.viz.plot_workspace_heatmap --model "$MODEL_PATH" --outdir outputs/figures --grid-points 7
