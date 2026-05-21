#!/usr/bin/env bash
set -euo pipefail

MAX_PARALLEL="${1:-4}"
ROOT="${ROOT:-/home/ubuntu/spiking_mappo_official}"
CONDA_ROOT="${CONDA_ROOT:-/home/ubuntu/anaconda3}"
ENV_NAME="${ENV_NAME:-spiking_mappo}"
TRAIN_DIR="$ROOT/onpolicy/scripts/train"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="$ROOT/linux_logs/paper_queue_v3_4090_$STAMP"
PID_FILE="$ROOT/linux_logs/paper_queue_v3_4090.pid"

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer, got: $MAX_PARALLEL" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"
echo "$$" > "$PID_FILE"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

cd "$TRAIN_DIR"

STATUS_FILE="$LOG_ROOT/queue_status.txt"
MANIFEST_FILE="$LOG_ROOT/run_manifest.csv"
RUNNING_FILE="$LOG_ROOT/running_pids.txt"

{
  echo "started_at=$(date '+%F %T')"
  echo "root=$ROOT"
  echo "log_root=$LOG_ROOT"
  echo "max_parallel=$MAX_PARALLEL"
  echo "num_env_steps=2000000"
  echo "rollout_threads=4"
  echo "total_runs=45"
} | tee "$STATUS_FILE"

echo "index,agents,actor,seed,experiment,log_file,exit_file,pid" > "$MANIFEST_FILE"
: > "$RUNNING_FILE"

running_pids=()

prune_running() {
  local next=()
  : > "$RUNNING_FILE"
  local pid
  for pid in "${running_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      next+=("$pid")
      echo "$pid" >> "$RUNNING_FILE"
    fi
  done
  running_pids=("${next[@]}")
}

run_one() {
  local index="$1"
  local agents="$2"
  local actor="$3"
  local seed="$4"
  local experiment="linux_4090_v3_a${agents}_${actor}_seed${seed}"
  local run_log="$LOG_ROOT/${index}_a${agents}_${actor}_seed${seed}.log"
  local exit_file="$LOG_ROOT/${index}_a${agents}_${actor}_seed${seed}.exit_code"

  echo "[$(date '+%F %T')] START ${index}/45 ${experiment}" | tee -a "$STATUS_FILE"
  set +e
  python -u train_mpe.py \
    --env_name MPE \
    --algorithm_name mappo \
    --experiment_name "$experiment" \
    --scenario_name simple_spread \
    --num_agents "$agents" \
    --num_landmarks "$agents" \
    --seed "$seed" \
    --n_training_threads 1 \
    --n_rollout_threads 4 \
    --num_mini_batch 1 \
    --episode_length 25 \
    --num_env_steps 2000000 \
    --ppo_epoch 10 \
    --use_ReLU \
    --gain 0.01 \
    --lr 7e-4 \
    --critic_lr 7e-4 \
    --actor_arch "$actor" \
    --use_wandb \
    --log_spike_stats \
    --save_interval 1000 \
    --log_interval 200 > "$run_log" 2>&1
  local code=$?
  set -e
  echo "$code" > "$exit_file"
  if [ "$code" -eq 0 ]; then
    echo "[$(date '+%F %T')] DONE ${index}/45 ${experiment}" | tee -a "$STATUS_FILE"
  else
    echo "[$(date '+%F %T')] FAILED ${index}/45 ${experiment} exit=${code} log=${run_log}" | tee -a "$STATUS_FILE"
  fi
  return "$code"
}

launch_one() {
  local index="$1"
  local agents="$2"
  local actor="$3"
  local seed="$4"
  local experiment="linux_4090_v3_a${agents}_${actor}_seed${seed}"
  local run_log="$LOG_ROOT/${index}_a${agents}_${actor}_seed${seed}.log"
  local exit_file="$LOG_ROOT/${index}_a${agents}_${actor}_seed${seed}.exit_code"

  run_one "$index" "$agents" "$actor" "$seed" &
  local pid=$!
  running_pids+=("$pid")
  echo "$index,$agents,$actor,$seed,$experiment,$run_log,$exit_file,$pid" >> "$MANIFEST_FILE"
  prune_running
}

index=0
for agents in 3 5 7; do
  for actor in ann snn_lif snn_at; do
    for seed in 1 2 3 4 5; do
      index=$((index + 1))
      while [ "${#running_pids[@]}" -ge "$MAX_PARALLEL" ]; do
        set +e
        wait -n
        set -e
        prune_running
      done
      launch_one "$index" "$agents" "$actor" "$seed"
    done
  done
done

while [ "${#running_pids[@]}" -gt 0 ]; do
  set +e
  wait -n
  set -e
  prune_running
done

cd "$ROOT"
PYTHONNOUSERSITE=1 python tools/summarize_results.py --repo-root . --output results_summary_linux_4090_v3.csv > "$LOG_ROOT/postprocess.log" 2>&1
PYTHONNOUSERSITE=1 MPLBACKEND=Agg python tools/plot_results.py --summary results_summary_linux_4090_v3.csv --output-dir plots_linux_4090_v3 >> "$LOG_ROOT/postprocess.log" 2>&1

failures=0
for exit_file in "$LOG_ROOT"/*.exit_code; do
  [ -e "$exit_file" ] || continue
  if [ "$(cat "$exit_file")" != "0" ]; then
    failures=$((failures + 1))
  fi
done

{
  echo "finished_at=$(date '+%F %T')"
  echo "failures=$failures"
  echo "summary=$ROOT/results_summary_linux_4090_v3.csv"
  echo "plots=$ROOT/plots_linux_4090_v3"
} | tee -a "$STATUS_FILE"

exit "$failures"
