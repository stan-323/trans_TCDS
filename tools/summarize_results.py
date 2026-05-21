#!/usr/bin/env python
"""Summarize MAPPO / Spiking-MAPPO training outputs into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


SUMMARY_FIELDS = [
    "scenario",
    "actor_arch",
    "num_agents",
    "seed",
    "env_steps",
    "final_reward",
    "source_path",
    "status",
]

REWARD_KEYS = (
    "eval_average_episode_rewards",
    "average_episode_rewards",
    "average_step_rewards",
    "episode_rewards",
    "reward",
)

LOG_SUFFIXES = {".log", ".txt", ".out", ".err"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan on-policy result folders and scripts_windows logs."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_summary.csv"),
        help="Output CSV path. Relative paths are resolved under --repo-root.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        action="append",
        default=None,
        help="Additional or replacement results root. Can be passed more than once.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        action="append",
        default=None,
        help="Additional or replacement scripts_windows log root. Can be passed more than once.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    rows = collect_rows(repo_root, args.results_root, args.logs_root)
    rows = sorted(
        rows,
        key=lambda row: (
            row.get("scenario") or "",
            row.get("actor_arch") or "",
            row.get("seed") or "",
            row.get("source_path") or "",
        ),
    )
    write_summary(output_path, rows)

    if not rows:
        print(
            "No completed MAPPO results found. "
            f"Wrote empty summary with headers to {output_path}."
        )
    else:
        print(f"Wrote {len(rows)} MAPPO result rows to {output_path}.")
    return 0


def collect_rows(
    repo_root: Path,
    results_roots: list[Path] | None = None,
    logs_roots: list[Path] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    smac_agents = load_smac_agent_counts(repo_root)

    for root in resolve_results_roots(repo_root, results_roots):
        rows.extend(scan_result_root(root, repo_root, smac_agents))

    for root in resolve_log_roots(repo_root, logs_roots):
        rows.extend(scan_log_root(root, repo_root, smac_agents))

    return rows


def resolve_results_roots(repo_root: Path, roots: list[Path] | None) -> list[Path]:
    if roots:
        return [resolve_under_repo(repo_root, root) for root in roots]
    candidates = [
        repo_root / "onpolicy" / "scripts" / "results",
        repo_root / "scripts" / "results",
        repo_root / "results",
    ]
    return [path for path in candidates if path.exists()]


def resolve_log_roots(repo_root: Path, roots: list[Path] | None) -> list[Path]:
    if roots:
        return [resolve_under_repo(repo_root, root) for root in roots]
    candidates = [
        repo_root / "scripts_windows",
        repo_root.parent / "scripts_windows",
    ]
    return [path for path in candidates if path.exists()]


def resolve_under_repo(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def scan_result_root(
    root: Path, repo_root: Path, smac_agents: dict[str, str]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not root.exists():
        return rows

    summary_files = set(root.rglob("summary.json"))
    summary_files.update(root.rglob("wandb-summary.json"))

    for summary_file in sorted(summary_files):
        data = load_json(summary_file)
        if not isinstance(data, dict):
            continue

        metadata = metadata_from_result_path(root, summary_file, smac_agents)
        metadata.update(extract_neighbor_metadata(summary_file))

        final_reward, reward_step = final_metric(data, REWARD_KEYS)
        env_steps = reward_step or max_scalar_step(data) or metadata.get("env_steps")
        scenario = first_present(metadata, "scenario", "map_name", "scenario_name")
        num_agents = metadata.get("num_agents") or infer_num_agents(
            scenario, metadata.get("env_name"), metadata.get("units"), smac_agents
        )

        rows.append(
            make_row(
                scenario=scenario,
                actor_arch=infer_actor_arch(metadata, summary_file),
                num_agents=num_agents,
                seed=metadata.get("seed"),
                env_steps=env_steps,
                final_reward=final_reward,
                source_path=summary_file,
                status="completed" if final_reward is not None else "completed_no_reward",
            )
        )

    return rows


def scan_log_root(
    root: Path, repo_root: Path, smac_agents: dict[str, str]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not root.exists():
        return rows

    for log_file in sorted(path for path in root.rglob("*") if path.suffix.lower() in LOG_SUFFIXES):
        try:
            text = log_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        metadata = extract_metadata_from_text(text)
        if not metadata:
            continue

        scenario = first_present(metadata, "scenario", "map_name", "scenario_name")
        final_reward = extract_last_number(
            text,
            (
                r"eval average episode rewards(?: of agent)?:?\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                r"average episode rewards(?: is|:)?\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                r"average_step_rewards['\"]?\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            ),
        )
        env_steps = metadata.get("env_steps") or extract_last_number(
            text, (r"total num timesteps\s+(\d+)\s*/\s*\d+",)
        )
        num_agents = metadata.get("num_agents") or infer_num_agents(
            scenario, metadata.get("env_name"), metadata.get("units"), smac_agents
        )

        rows.append(
            make_row(
                scenario=scenario,
                actor_arch=infer_actor_arch(metadata, log_file, text),
                num_agents=num_agents,
                seed=metadata.get("seed"),
                env_steps=env_steps,
                final_reward=final_reward,
                source_path=log_file,
                status=infer_log_status(text, final_reward),
            )
        )
    return rows


def make_row(
    *,
    scenario: Any,
    actor_arch: Any,
    num_agents: Any,
    seed: Any,
    env_steps: Any,
    final_reward: Any,
    source_path: Path,
    status: str,
) -> dict[str, str]:
    return {
        "scenario": stringify(scenario),
        "actor_arch": stringify(actor_arch),
        "num_agents": stringify(num_agents),
        "seed": stringify(seed),
        "env_steps": stringify(env_steps),
        "final_reward": stringify(final_reward),
        "source_path": str(source_path.resolve()),
        "status": status,
    }


def write_summary(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def metadata_from_result_path(
    root: Path, summary_file: Path, smac_agents: dict[str, str]
) -> dict[str, str]:
    try:
        parts = summary_file.relative_to(root).parts
    except ValueError:
        parts = summary_file.parts

    metadata: dict[str, str] = {}
    if len(parts) >= 4:
        metadata["env_name"] = parts[0]
        metadata["scenario"] = parts[1]
        metadata["algorithm_name"] = parts[2]
        metadata["experiment_name"] = parts[3]
    for part in parts:
        seed = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:$|[_-])", part, re.IGNORECASE)
        if seed:
            metadata["seed"] = seed.group(1)
        short_seed = re.search(r"(?:^|[_-])s[_-]?(\d+)(?:$|[_-])", part, re.IGNORECASE)
        if short_seed and "seed" not in metadata:
            metadata["seed"] = short_seed.group(1)
        agents = re.search(r"(?:^|[_-])(?:n|num_agents)[_-]?(\d+)(?:$|[_-])", part, re.IGNORECASE)
        if agents and "num_agents" not in metadata:
            metadata["num_agents"] = agents.group(1)
        short_agents = re.search(r"(?:^|[_-])a(\d+)(?:$|[_-])", part, re.IGNORECASE)
        if short_agents and "num_agents" not in metadata:
            metadata["num_agents"] = short_agents.group(1)
        if part.lower().startswith("run") and part[3:].isdigit():
            metadata["run"] = part[3:]

    scenario = metadata.get("scenario")
    if scenario and scenario in smac_agents:
        metadata["num_agents"] = smac_agents[scenario]
    return metadata


def extract_neighbor_metadata(summary_file: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    seen: set[Path] = set()
    for parent in [summary_file.parent, *summary_file.parents[1:5]]:
        if parent in seen:
            continue
        seen.add(parent)
        for candidate in parent.iterdir() if parent.exists() else []:
            if not candidate.is_file():
                continue
            name = candidate.name.lower()
            if name in {"config.json", "args.json", "metadata.json", "params.json"}:
                data = load_json(candidate)
                if isinstance(data, dict):
                    metadata.update(flatten_metadata(data))
            elif name in {"config.yaml", "config.yml"}:
                try:
                    metadata.update(extract_metadata_from_text(candidate.read_text(encoding="utf-8", errors="ignore")))
                except OSError:
                    pass
    return metadata


def flatten_metadata(data: dict[str, Any]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key, value in data.items():
        normalized = key.lstrip("-").replace("-", "_")
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        if isinstance(value, (str, int, float, bool)):
            metadata[normalized] = stringify(value)
    return metadata


def extract_metadata_from_text(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    option_pattern = re.compile(r"--([A-Za-z0-9_]+)(?:\s+([^\s\\]+))?")
    for key, value in option_pattern.findall(text):
        if not value or value.startswith("--"):
            continue
        metadata[key] = clean_token(value)

    pairs = {
        "scenario_name": r"\bscenario(?:\s+is)?\s*[=:]\s*([A-Za-z0-9_.-]+)",
        "map_name": r"\bmap(?:\s+is)?\s*[=:]\s*([A-Za-z0-9_.-]+)",
        "algorithm_name": r"\balgo(?:rithm)?(?:\s+is)?\s*[=:]\s*([A-Za-z0-9_.-]+)",
        "experiment_name": r"\bexp(?:eriment)?(?:\s+is)?\s*[=:]\s*([A-Za-z0-9_.-]+)",
        "seed": r"\bseed(?:\s+is)?\s*[=:]\s*(\d+)",
    }
    for key, pattern in pairs.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match and key not in metadata:
            metadata[key] = clean_token(match.group(1))

    for pattern in (
        r"\bMap\s+([^\s]+)\s+Algo\s+([^\s]+)\s+Exp\s+([^\s]+)",
        r"\bScenario\s+([^\s]+)\s+Algo\s+([^\s]+)\s+Exp\s+([^\s]+)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            scenario_key = "map_name" if pattern.startswith(r"\bMap") else "scenario_name"
            metadata.setdefault(scenario_key, clean_token(match.group(1)))
            metadata.setdefault("algorithm_name", clean_token(match.group(2)))
            metadata.setdefault("experiment_name", clean_token(match.group(3)))

    target_steps = extract_last_number(text, (r"--num_env_steps\s+(\d+)",))
    if target_steps is not None:
        metadata["env_steps"] = stringify(target_steps)
    return metadata


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def final_metric(data: dict[str, Any], preferred_keys: Iterable[str]) -> tuple[Any | None, Any | None]:
    for preferred_key in preferred_keys:
        for key, raw_points in data.items():
            if preferred_key.lower() not in key.lower():
                continue
            points = parse_points(raw_points)
            if points:
                step, value = points[-1]
                return value, step
            number = as_number(raw_points)
            if number is not None:
                return number, None
    return None, None


def max_scalar_step(data: dict[str, Any]) -> Any | None:
    max_step: float | None = None
    for raw_points in data.values():
        for step, _value in parse_points(raw_points):
            step_number = as_number(step)
            if step_number is None:
                continue
            if max_step is None or step_number > max_step:
                max_step = step_number
    return max_step


def parse_points(raw_points: Any) -> list[tuple[Any, Any]]:
    points: list[tuple[Any, Any]] = []
    if not isinstance(raw_points, list):
        return points
    for point in raw_points:
        if isinstance(point, dict):
            step = first_present(point, "step", "global_step", "_step", "x")
            value = first_present(point, "value", "y")
        elif isinstance(point, (list, tuple)) and len(point) >= 3:
            step = point[1]
            value = point[2]
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            step = point[0]
            value = point[1]
        else:
            continue
        if as_number(value) is not None:
            points.append((step, value))
    return points


def first_present(mapping: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def stringify(value: Any) -> str:
    if value is None:
        return ""
    number = as_number(value)
    if number is not None:
        if number.is_integer():
            return str(int(number))
        return f"{number:.10g}"
    return str(value)


def clean_token(value: str) -> str:
    return value.strip().strip("\"'").strip(",")


def infer_actor_arch(metadata: dict[str, str], source_path: Path, text: str = "") -> str:
    haystack = " ".join(
        [
            str(source_path).lower(),
            text.lower(),
            " ".join(f"{key}={value}" for key, value in metadata.items()).lower(),
        ]
    )
    explicit = first_present(metadata, "actor_arch", "actor_type", "policy_arch", "network_type")
    if explicit:
        return stringify(explicit)
    if re.search(r"(?:^|[^A-Za-z0-9])snn[_-]?lif(?:$|[^A-Za-z0-9])", haystack):
        return "snn_lif"
    if re.search(r"(?:^|[^A-Za-z0-9])snn[_-]?at(?:$|[^A-Za-z0-9])", haystack):
        return "snn_at"
    if re.search(r"(?:^|[^A-Za-z0-9])ann(?:$|[^A-Za-z0-9])", haystack):
        return "ann"
    if re.search(r"\b(spiking|spike|snn|lif|neuron)\b", haystack):
        return "spiking"
    if re.search(r"\b(rmappo|recurrent|rnn|gru|lstm)\b", haystack):
        return "recurrent_mappo"
    if re.search(r"\b(mappo|mlp|feedforward)\b", haystack):
        return "ann"
    return "unknown"


def infer_num_agents(
    scenario: Any,
    env_name: Any,
    units: Any,
    smac_agents: dict[str, str],
) -> str:
    scenario_text = stringify(scenario)
    if scenario_text in smac_agents:
        return smac_agents[scenario_text]
    units_text = stringify(units)
    units_match = re.match(r"(\d+)v\d+", units_text)
    if units_match:
        return units_match.group(1)
    if scenario_text == "simple_spread":
        return "3"
    if scenario_text == "simple_reference":
        return "2"
    if scenario_text == "simple_speaker_listener":
        return "2"
    if stringify(env_name).lower() == "mpe":
        return ""
    return ""


def load_smac_agent_counts(repo_root: Path) -> dict[str, str]:
    smac_maps = repo_root / "onpolicy" / "envs" / "starcraft2" / "smac_maps.py"
    if not smac_maps.exists():
        return {}
    try:
        text = smac_maps.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}

    counts: dict[str, str] = {}
    block_pattern = re.compile(
        r'"([^"]+)":\s*{(?:(?!\n\s*").)*?"n_agents":\s*(\d+)',
        re.DOTALL,
    )
    for map_name, n_agents in block_pattern.findall(text):
        counts[map_name] = n_agents
    return counts


def extract_last_number(text: str, patterns: Iterable[str]) -> float | None:
    last_value: float | None = None
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            last_value = as_number(match.group(1))
    return last_value


def infer_log_status(text: str, final_reward: Any | None) -> str:
    lower_text = text.lower()
    if "traceback" in lower_text or re.search(r"\b(error|exception)\b", lower_text):
        return "failed"
    if final_reward is not None:
        return "completed"
    if "total num timesteps" in lower_text:
        return "in_progress"
    return "log_only"


if __name__ == "__main__":
    raise SystemExit(main())
