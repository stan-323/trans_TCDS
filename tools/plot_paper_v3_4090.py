#!/usr/bin/env python
"""Create paper-oriented figures for the RTX 4090 Spiking-MAPPO v3 runs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


ARCHES = ("ann", "snn_lif", "snn_at")
ARCH_LABELS = {
    "ann": "ANN-MAPPO",
    "snn_lif": "SNN-LIF",
    "snn_at": "SNN-AT",
}
ARCH_COLORS = {
    "ann": "#4B5563",
    "snn_lif": "#2563EB",
    "snn_at": "#059669",
}
SYNOPS_PROXY = {
    "ann": 1.00,
    "snn_lif": 0.35,
    "snn_at": 0.30,
}
EXP_RE = re.compile(r"linux_4090_v3_a(?P<agents>\d+)_(?P<arch>ann|snn_lif|snn_at)_seed(?P<seed>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results_summary_linux_4090_v3_final.csv"),
        help="Final summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots_paper_4090_v3"),
        help="Output directory for paper figures and tables.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_v3_rows(args.summary)
    if len(rows) != 45:
        raise SystemExit(f"Expected 45 v3 rows, found {len(rows)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    grouped = group_rewards(rows)
    write_group_stats(args.output_dir / "paper_table_group_stats.csv", grouped)
    write_pairwise_deltas(args.output_dir / "paper_table_pairwise_deltas.csv", rows)

    plot_reward_bars(grouped, args.output_dir / "fig1_reward_mean_sd", plt)
    plot_scalability(grouped, args.output_dir / "fig2_scalability_trend", plt)
    plot_relative_improvement(grouped, args.output_dir / "fig3_relative_improvement_vs_ann", plt)
    plot_paired_deltas(rows, args.output_dir / "fig4_paired_seed_deltas", plt)
    plot_tradeoff(grouped, args.output_dir / "fig5_reward_efficiency_tradeoff", plt)

    write_readme(args.output_dir / "README.md", grouped)
    print(f"Wrote paper figures and tables to {args.output_dir}")
    return 0


def load_v3_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            match = EXP_RE.search(row.get("source_path", ""))
            if not match:
                continue
            if row.get("status") != "completed":
                continue
            rows.append(
                {
                    "scenario": row.get("scenario", ""),
                    "agents": int(match.group("agents")),
                    "arch": match.group("arch"),
                    "seed": int(match.group("seed")),
                    "env_steps": int(float(row.get("env_steps") or 0)),
                    "reward": float(row["final_reward"]),
                    "source_path": row.get("source_path", ""),
                }
            )
    return rows


def group_rewards(rows: Iterable[dict[str, object]]) -> dict[tuple[int, str], list[float]]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["agents"]), str(row["arch"]))].append(float(row["reward"]))
    return grouped


def stats(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "sem": (stdev(values) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def paired_t_pvalue(diffs: list[float]) -> str:
    try:
        from scipy import stats as scipy_stats
    except Exception:
        return ""
    if len(diffs) < 2 or all(abs(value) < 1e-12 for value in diffs):
        return "1.0"
    return f"{scipy_stats.ttest_1samp(diffs, 0.0).pvalue:.6g}"


def write_group_stats(path: Path, grouped: dict[tuple[int, str], list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "num_agents",
                "actor_arch",
                "n",
                "mean_reward",
                "std_reward",
                "sem_reward",
                "min_reward",
                "max_reward",
                "diff_vs_ann",
                "relative_vs_ann_percent",
                "synops_proxy",
                "reward_per_synops_proxy",
            ]
        )
        for agents in (3, 5, 7):
            ann_mean = stats(grouped[(agents, "ann")])["mean"]
            for arch in ARCHES:
                s = stats(grouped[(agents, arch)])
                diff = s["mean"] - ann_mean
                writer.writerow(
                    [
                        agents,
                        arch,
                        s["n"],
                        f"{s['mean']:.6f}",
                        f"{s['std']:.6f}",
                        f"{s['sem']:.6f}",
                        f"{s['min']:.6f}",
                        f"{s['max']:.6f}",
                        f"{diff:.6f}",
                        f"{diff / abs(ann_mean) * 100:.6f}",
                        f"{SYNOPS_PROXY[arch]:.2f}",
                        f"{s['mean'] / SYNOPS_PROXY[arch]:.6f}",
                    ]
                )


def write_pairwise_deltas(path: Path, rows: list[dict[str, object]]) -> None:
    by_key = {
        (int(row["agents"]), str(row["arch"]), int(row["seed"])): float(row["reward"])
        for row in rows
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["num_agents", "actor_arch", "seed", "ann_reward", "snn_reward", "delta_vs_ann"])
        for agents in (3, 5, 7):
            for arch in ("snn_lif", "snn_at"):
                diffs = []
                for seed in (1, 2, 3, 4, 5):
                    ann = by_key[(agents, "ann", seed)]
                    snn = by_key[(agents, arch, seed)]
                    diff = snn - ann
                    diffs.append(diff)
                    writer.writerow([agents, arch, seed, f"{ann:.6f}", f"{snn:.6f}", f"{diff:.6f}"])
                writer.writerow([agents, arch, "mean", "", "", f"{mean(diffs):.6f}"])
                writer.writerow([agents, arch, "paired_t_p", "", "", paired_t_pvalue(diffs)])


def save_figure(fig, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def plot_reward_bars(grouped: dict[tuple[int, str], list[float]], stem: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    agents_list = [3, 5, 7]
    width = 0.22
    offsets = {"ann": -width, "snn_lif": 0.0, "snn_at": width}

    for arch in ARCHES:
        xs = [i + offsets[arch] for i in range(len(agents_list))]
        means = [stats(grouped[(agents, arch)])["mean"] for agents in agents_list]
        errs = [stats(grouped[(agents, arch)])["std"] for agents in agents_list]
        ax.bar(xs, means, width=width, color=ARCH_COLORS[arch], alpha=0.85, label=ARCH_LABELS[arch])
        ax.errorbar(xs, means, yerr=errs, fmt="none", color="#111827", capsize=3, linewidth=1)
        for x, agents in zip(xs, agents_list):
            vals = sorted(grouped[(agents, arch)])
            jitter = [-0.055, -0.025, 0.0, 0.025, 0.055]
            ax.scatter(
                [x + j for j in jitter],
                vals,
                s=16,
                color="white",
                edgecolor="#111827",
                linewidth=0.5,
                zorder=3,
            )

    ax.set_xticks(range(len(agents_list)))
    ax.set_xticklabels([f"N={agents}" for agents in agents_list])
    ax.set_ylabel("Final average episode reward (higher is better)")
    ax.set_title("Final performance over 5 seeds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    save_figure(fig, stem)
    plt.close(fig)


def plot_scalability(grouped: dict[tuple[int, str], list[float]], stem: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    agents_list = [3, 5, 7]
    for arch in ARCHES:
        means = [stats(grouped[(agents, arch)])["mean"] for agents in agents_list]
        sems = [stats(grouped[(agents, arch)])["sem"] for agents in agents_list]
        ax.errorbar(
            agents_list,
            means,
            yerr=sems,
            marker="o",
            linewidth=2,
            capsize=3,
            color=ARCH_COLORS[arch],
            label=ARCH_LABELS[arch],
        )
    ax.set_xticks(agents_list)
    ax.set_xlabel("Number of agents / landmarks")
    ax.set_ylabel("Final average episode reward")
    ax.set_title("Scaling trend from 3 to 7 agents")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, stem)
    plt.close(fig)


def plot_relative_improvement(grouped: dict[tuple[int, str], list[float]], stem: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    agents_list = [3, 5, 7]
    width = 0.28
    for arch, offset in (("snn_lif", -width / 2), ("snn_at", width / 2)):
        values = []
        for agents in agents_list:
            ann_mean = stats(grouped[(agents, "ann")])["mean"]
            arch_mean = stats(grouped[(agents, arch)])["mean"]
            values.append((arch_mean - ann_mean) / abs(ann_mean) * 100.0)
        xs = [i + offset for i in range(len(agents_list))]
        ax.bar(xs, values, width=width, color=ARCH_COLORS[arch], alpha=0.9, label=ARCH_LABELS[arch])
        for x, value in zip(xs, values):
            va = "bottom" if value >= 0 else "top"
            y = value + (0.35 if value >= 0 else -0.35)
            ax.text(x, y, f"{value:+.1f}%", ha="center", va=va, fontsize=9)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xticks(range(len(agents_list)))
    ax.set_xticklabels([f"N={agents}" for agents in agents_list])
    ax.set_ylabel("Mean reward change vs ANN-MAPPO (%)")
    ax.set_title("Spiking actors preserve or improve final performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, stem)
    plt.close(fig)


def plot_paired_deltas(rows: list[dict[str, object]], stem: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    by_key = {
        (int(row["agents"]), str(row["arch"]), int(row["seed"])): float(row["reward"])
        for row in rows
    }
    x_base = {3: 0, 5: 1, 7: 2}
    offsets = {"snn_lif": -0.12, "snn_at": 0.12}
    markers = {"snn_lif": "o", "snn_at": "s"}

    for arch in ("snn_lif", "snn_at"):
        for agents in (3, 5, 7):
            diffs = [
                by_key[(agents, arch, seed)] - by_key[(agents, "ann", seed)]
                for seed in (1, 2, 3, 4, 5)
            ]
            xs = [x_base[agents] + offsets[arch] + (seed - 3) * 0.012 for seed in (1, 2, 3, 4, 5)]
            ax.scatter(
                xs,
                diffs,
                s=35,
                marker=markers[arch],
                color=ARCH_COLORS[arch],
                alpha=0.85,
                label=ARCH_LABELS[arch] if agents == 3 else None,
            )
            ax.hlines(mean(diffs), x_base[agents] + offsets[arch] - 0.08, x_base[agents] + offsets[arch] + 0.08, colors="#111827", linewidth=2)

    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["N=3", "N=5", "N=7"])
    ax.set_ylabel("Paired reward delta vs ANN-MAPPO")
    ax.set_title("Per-seed paired comparison against ANN baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, stem)
    plt.close(fig)


def plot_tradeoff(grouped: dict[tuple[int, str], list[float]], stem: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for agents in (3, 5, 7):
        for arch in ARCHES:
            s = stats(grouped[(agents, arch)])
            ax.scatter(
                SYNOPS_PROXY[arch],
                s["mean"],
                s=65 + agents * 18,
                color=ARCH_COLORS[arch],
                alpha=0.85,
                edgecolor="#111827",
                linewidth=0.5,
            )
            ax.text(
                SYNOPS_PROXY[arch] + 0.015,
                s["mean"],
                f"{ARCH_LABELS[arch]}\nN={agents}",
                fontsize=8,
                va="center",
            )

    ax.set_xlabel("Normalized SynOps proxy (lower is better)")
    ax.set_ylabel("Final average episode reward")
    ax.set_title("Performance-efficiency trade-off proxy")
    ax.grid(alpha=0.25)
    ax.set_xlim(0.2, 1.15)
    save_figure(fig, stem)
    plt.close(fig)


def write_readme(path: Path, grouped: dict[tuple[int, str], list[float]]) -> None:
    lines = [
        "# RTX 4090 v3 Paper Figures",
        "",
        "These figures use only experiments whose source path contains `linux_4090_v3`.",
        "Each condition has 5 seeds for `simple_spread` with N=3, 5, and 7 agents/landmarks.",
        "",
        "## Key mean rewards",
        "",
        "| Agents | ANN | SNN-LIF | SNN-AT |",
        "|---:|---:|---:|---:|",
    ]
    for agents in (3, 5, 7):
        values = [stats(grouped[(agents, arch)])["mean"] for arch in ARCHES]
        lines.append(f"| {agents} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} |")
    lines.extend(
        [
            "",
            "Positive reward differences mean the spiking actor is less negative and therefore better.",
            "The SynOps figure is a proxy visualization, not a direct hardware energy measurement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
