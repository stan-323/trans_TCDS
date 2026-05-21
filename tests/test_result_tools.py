import csv
import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_summarize_writes_empty_csv_when_no_results(tmp_path, capsys):
    summarize_results = importlib.import_module("tools.summarize_results")
    output_csv = tmp_path / "results_summary.csv"

    exit_code = summarize_results.main(
        ["--repo-root", str(tmp_path), "--output", str(output_csv)]
    )

    assert exit_code == 0
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows == [summarize_results.SUMMARY_FIELDS]
    assert "No completed MAPPO results found" in capsys.readouterr().out


def test_plot_exits_cleanly_when_summary_has_no_rows(tmp_path, capsys):
    plot_results = importlib.import_module("tools.plot_results")
    summary_csv = tmp_path / "results_summary.csv"
    summary_csv.write_text(
        "scenario,actor_arch,num_agents,seed,env_steps,final_reward,source_path,status\n",
        encoding="utf-8",
    )

    exit_code = plot_results.main(
        ["--summary", str(summary_csv), "--output-dir", str(tmp_path)]
    )

    assert exit_code == 0
    assert not (tmp_path / "reward_curves.png").exists()
    assert not (tmp_path / "synops_proxy.png").exists()
    assert "Not enough completed result rows to plot" in capsys.readouterr().out
