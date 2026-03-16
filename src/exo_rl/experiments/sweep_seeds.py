from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from exo_rl.utils.io import ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--root-dir", type=str, default="outputs_sweep")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    return parser.parse_args()


def run_command(command: list[str], workdir: Path) -> None:
    subprocess.run(command, cwd=workdir, check=True)


def summarize_policy_across_seeds(seed_summaries: list[dict], policy_name: str) -> dict[str, float]:
    metrics = seed_summaries[0]["metrics"][policy_name].keys()
    return {
        key: float(np.mean([summary["metrics"][policy_name][key] for summary in seed_summaries]))
        for key in metrics
    } | {
        f"{key}_std": float(np.std([summary["metrics"][policy_name][key] for summary in seed_summaries]))
        for key in metrics
    }


def rl_score(metrics: dict[str, float]) -> float:
    return (
        2.5 * metrics["success_rate"]
        - 0.35 * metrics["avg_distance"]
        - 0.15 * metrics["avg_assist_ratio"]
        - 0.10 * metrics["final_fatigue"]
    )


def make_markdown_report(seed_summaries: list[dict], aggregate: dict[str, dict[str, float]], best_seed: int) -> str:
    lines = [
        "# Multi-Seed Sweep",
        "",
        f"Seeds: {', '.join(str(summary['seed']) for summary in seed_summaries)}",
        f"Best PPO seed: {best_seed}",
        "",
        "## Aggregate Metrics",
        "",
        "| Policy | Success mean±std | Distance mean±std | Effort mean±std | Assist mean±std | Fatigue mean±std |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for policy, metrics in aggregate.items():
        lines.append(
            "| "
            + policy
            + " | "
            + f"{metrics['success_rate']:.3f}±{metrics['success_rate_std']:.3f}"
            + " | "
            + f"{metrics['avg_distance']:.3f}±{metrics['avg_distance_std']:.3f}"
            + " | "
            + f"{metrics['avg_human_effort']:.3f}±{metrics['avg_human_effort_std']:.3f}"
            + " | "
            + f"{metrics['avg_assist_ratio']:.3f}±{metrics['avg_assist_ratio_std']:.3f}"
            + " | "
            + f"{metrics['final_fatigue']:.3f}±{metrics['final_fatigue_std']:.3f}"
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-Seed PPO",
            "",
            "| Seed | Success | Distance | Effort | Assist | Fatigue | Score |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for summary in seed_summaries:
        metrics = summary["metrics"]["ppo_rl"]
        lines.append(
            f"| {summary['seed']} | {metrics['success_rate']:.3f} | {metrics['avg_distance']:.3f} | "
            f"{metrics['avg_human_effort']:.3f} | {metrics['avg_assist_ratio']:.3f} | "
            f"{metrics['final_fatigue']:.3f} | {summary['ppo_score']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    base_config = load_json(args.config)
    root_dir = ensure_dir(args.root_dir)
    workdir = Path.cwd()
    python = sys.executable
    seed_summaries: list[dict] = []

    for seed in args.seeds:
        run_dir = ensure_dir(root_dir / f"seed_{seed}")
        print(f"[seed {seed}] training -> {run_dir}")
        run_command(
            [
                python,
                "-m",
                "exo_rl.agents.train_ppo",
                "--config",
                args.config,
                "--seed",
                str(seed),
                "--outputs-dir",
                str(run_dir),
            ],
            workdir,
        )
        print(f"[seed {seed}] evaluating")
        run_command(
            [
                python,
                "-m",
                "exo_rl.eval.evaluate",
                "--config",
                args.config,
                "--seed",
                str(seed),
                "--outputs-dir",
                str(run_dir),
                "--model",
                str(run_dir / "models" / "best.pt"),
                "--episodes",
                str(args.episodes),
            ],
            workdir,
        )
        metrics = load_json(run_dir / "eval" / "metrics_summary.json")
        seed_summaries.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "metrics": metrics,
                "ppo_score": rl_score(metrics["ppo_rl"]),
            }
        )

    aggregate = {
        policy: summarize_policy_across_seeds(seed_summaries, policy)
        for policy in seed_summaries[0]["metrics"].keys()
    }
    best_summary = max(seed_summaries, key=lambda item: item["ppo_score"])
    summary_payload = {
        "base_config": str(Path(args.config)),
        "episodes": args.episodes,
        "seeds": list(args.seeds),
        "best_seed": best_summary["seed"],
        "best_run_dir": best_summary["run_dir"],
        "seed_runs": seed_summaries,
        "aggregate": aggregate,
    }
    save_json(root_dir / "summary.json", summary_payload)
    report = make_markdown_report(seed_summaries, aggregate, int(best_summary["seed"]))
    (root_dir / "summary.md").write_text(report, encoding="utf-8")

    print("\nAggregate PPO:")
    ppo = aggregate["ppo_rl"]
    print(
        f"success={ppo['success_rate']:.3f}±{ppo['success_rate_std']:.3f} "
        f"distance={ppo['avg_distance']:.3f}±{ppo['avg_distance_std']:.3f} "
        f"assist={ppo['avg_assist_ratio']:.3f}±{ppo['avg_assist_ratio_std']:.3f}"
    )
    print(f"best seed={best_summary['seed']} run_dir={best_summary['run_dir']}")


if __name__ == "__main__":
    main()
