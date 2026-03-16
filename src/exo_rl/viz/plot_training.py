from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exo_rl.utils.io import ensure_dir, load_json
from exo_rl.viz.style import METRIC_COLORS, POLICY_COLORS, apply_plot_style, style_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, default="outputs/logs/training_history.json")
    parser.add_argument("--outdir", type=str, default="outputs/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = load_json(args.history)
    outdir = ensure_dir(args.outdir)
    apply_plot_style()
    updates = history["updates"]
    metrics = [
        ("reward", "Reward"),
        ("success_rate", "Success Rate"),
        ("distance", "Avg Distance"),
        ("human_effort", "Human Effort"),
        ("assist_ratio", "Assist Ratio"),
        ("final_fatigue", "Final Fatigue"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), constrained_layout=True)
    fig.suptitle("Training Dashboard", fontsize=17, fontweight="normal", y=1.02)
    color_map = {
        "reward": POLICY_COLORS["ppo_rl"],
        "success_rate": POLICY_COLORS["ppo_rl"],
        "distance": METRIC_COLORS["distance"],
        "human_effort": METRIC_COLORS["human"],
        "assist_ratio": METRIC_COLORS["assist"],
        "final_fatigue": METRIC_COLORS["fatigue"],
    }
    for ax, (key, title) in zip(axes.flat, metrics, strict=True):
        color = color_map[key]
        ax.plot(updates, history[key], linewidth=2.2, color=color)
        ax.fill_between(updates, history[key], alpha=0.08, color=color)
        ax.set_title(title)
        ax.set_xlabel("Update")
        style_axis(ax)
    fig.savefig(Path(outdir) / "training_dashboard.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
