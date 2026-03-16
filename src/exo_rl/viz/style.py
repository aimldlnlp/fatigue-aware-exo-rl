from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICY_ORDER = ["ppo_rl", "zero_assist", "fixed_assist", "rule_based_fatigue"]
POLICY_LABELS = {
    "ppo_rl": "PPO RL",
    "zero_assist": "Zero Assist",
    "fixed_assist": "Fixed Assist",
    "rule_based_fatigue": "Fatigue Heuristic",
}
POLICY_COLORS = {
    "ppo_rl": "#2f6f68",
    "zero_assist": "#58759a",
    "fixed_assist": "#86679f",
    "rule_based_fatigue": "#a56f44",
}
METRIC_COLORS = {
    "fatigue": "#a87a52",
    "assist": "#7a6c93",
    "distance": "#58707c",
    "human": "#6a8fbe",
    "exo": "#3f7d76",
}


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "axes.edgecolor": "#d9d9d9",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "normal",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.titlepad": 10,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "text.color": "#222222",
            "grid.color": "#e8e8e8",
            "grid.alpha": 0.55,
            "font.family": "DejaVu Serif",
            "font.weight": "normal",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d9d9d9")
    ax.spines["bottom"].set_color("#d9d9d9")
    ax.grid(alpha=0.3)


def style_minimal_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(alpha=0.18)
