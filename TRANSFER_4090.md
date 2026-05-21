# Spiking-MAPPO Transfer Notes

This repository contains the code changes, run scripts, tests, and final small
artifacts needed to reproduce or continue the Spiking-MAPPO v3 experiments on a
new Linux training machine.

## What is stored in git

- Spiking actor implementation and MAPPO integration.
- Linux and Windows launch scripts.
- Result summarization and plotting tools.
- Unit tests for the spiking actor, spike-stat logging, and result tools.
- Final v3 summary CSV and plots from the RTX 4090 run.
- A small final artifact archive under `final_artifacts/`.

Full raw training directories under `onpolicy/scripts/results/`, local log
folders, and W&B output are intentionally not tracked by git.

## New Linux machine setup

```bash
git clone https://github.com/stan-323/trans_TCDS.git
cd trans_TCDS
git checkout spiking-mappo-v3-4090

conda create -n spiking_mappo python=3.10 -y
conda activate spiking_mappo

# Install a CUDA-enabled PyTorch build matching the machine's driver/CUDA.
# For the previous RTX 4090 box, CUDA 12.8 worked.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install pytest matplotlib seaborn pandas absl-py imageio
```

Quick validation:

```bash
pytest tests -q
nvidia-smi
```

## Re-run the 45-run v3 suite

For a single RTX 4090:

```bash
bash scripts_linux/run_paper_queue_v3_4090.sh 2
```

Use `2` for conservative parallelism. Increase only after checking GPU
utilization, process stability, and log freshness.

## Final result files

The completed 45-run v3 summary and figures are:

- `results_summary_linux_4090_v3_final.csv`
- `plots_linux_4090_v3_final/reward_curves.png`
- `plots_linux_4090_v3_final/synops_proxy.png`
- `final_artifacts/spiking_mappo_4090_v3_final_20260521_005800.tar.gz`

The archive can be unpacked on another machine with:

```bash
tar -xzf final_artifacts/spiking_mappo_4090_v3_final_20260521_005800.tar.gz
```
