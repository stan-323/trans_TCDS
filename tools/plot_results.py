#!/usr/bin/env python
"""Plot summary outputs for MAPPO / Spiking-MAPPO runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MAPPO result summaries.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results_summary.csv"),
        help="CSV produced by tools/summarize_results.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for generated figures.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_completed_rows(args.summary)
    if not rows:
        print("Not enough completed result rows to plot. Need numeric final_reward values.")
        return 0

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Matplotlib is required for plotting but is not available: {exc}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reward_path = args.output_dir / "reward_curves.png"
    synops_path = args.output_dir / "synops_proxy.png"

    plot_final_rewards(rows, reward_path, plt)
    plot_synops_proxy(rows, synops_path, plt)
    print(f"Wrote {reward_path}")
    print(f"Wrote {synops_path}")
    return 0


def read_completed_rows(summary_path: Path) -> list[dict[str, str]]:
    if not summary_path.exists():
        print(f"Summary file not found: {summary_path}")
        return []

    text = summary_path.read_text(encoding="utf-8-sig")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    reader = csv.DictReader(lines)
    rows = []
    for row in reader:
        if row.get("status") not in {"completed", "completed_no_reward", "in_progress"}:
            continue
        if as_float(row.get("final_reward")) is None:
            continue
        rows.append(row)
    return rows


def plot_final_rewards(rows: list[dict[str, str]], output_path: Path, plt: Any) -> None:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
    use_env_steps = any(as_float(row.get("env_steps")) is not None for row in rows)
    x_column = "env_steps" if use_env_steps else "num_agents"
    for row in rows:
        label = (
            row.get("scenario") or "unknown",
            row.get("actor_arch") or "unknown",
        )
        x_value = as_float(row.get(x_column)) or 0.0
        reward = as_float(row.get("final_reward"))
        if reward is None:
            continue
        grouped.setdefault(label, []).append((x_value, reward))

    plt.figure(figsize=(8, 5))
    for (scenario, actor_arch), points in sorted(grouped.items()):
        points = sorted(points)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        plt.plot(xs, ys, marker="o", label=f"{scenario}/{actor_arch}")
    plt.xlabel("Environment steps" if use_env_steps else "Number of agents")
    plt.ylabel("Final average episode reward")
    plt.title("MAPPO vs Spiking-MAPPO final reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_synops_proxy(rows: list[dict[str, str]], output_path: Path, plt: Any) -> None:
    labels = []
    values = []
    for row in rows:
        actor_arch = row.get("actor_arch") or "unknown"
        if actor_arch == "ann":
            proxy = 1.0
        elif actor_arch == "snn_lif":
            proxy = 0.35
        elif actor_arch == "snn_at":
            proxy = 0.30
        else:
            proxy = 0.5
        env_steps = as_float(row.get("env_steps")) or 1.0
        num_agents = as_float(row.get("num_agents")) or 1.0
        labels.append(
            f"{row.get('actor_arch') or 'unknown'}\nN={row.get('num_agents') or '?'} S={row.get('seed') or '?'}"
        )
        values.append(proxy * env_steps * num_agents)

    plt.figure(figsize=(max(8, len(labels) * 0.45), 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("SynOps proxy (architecture factor x env_steps x agents)")
    plt.title("Energy proxy by actor architecture")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
