#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/home/ubuntu/spiking_mappo_official}"
CONDA_ROOT="${CONDA_ROOT:-/home/ubuntu/anaconda3}"
ENV_NAME="${ENV_NAME:-spiking_mappo}"
CHECK_INTERVAL="${CHECK_INTERVAL:-600}"
STALE_SECONDS="${STALE_SECONDS:-1800}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="$ROOT/linux_logs/watchdog_final_v3_4090_$STAMP"
PID_FILE="$ROOT/linux_logs/watchdog_final_v3_4090.pid"

mkdir -p "$LOG_ROOT" "$ROOT/linux_logs"
echo "$$" > "$PID_FILE"
exec > >(tee -a "$LOG_ROOT/watchdog.log") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

load_env() {
  # shellcheck source=/dev/null
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  export WANDB_DISABLED=true
  export PYTHONNOUSERSITE=1
  export CUDA_VISIBLE_DEVICES=0
  export OMP_NUM_THREADS=1
  export MPLBACKEND=Agg
}

kill_tree() {
  local root="$1"
  local child
  for child in $(pgrep -P "$root" 2>/dev/null || true); do
    kill_tree "$child"
  done
  kill -TERM "$root" 2>/dev/null || true
}

force_kill_tree() {
  local root="$1"
  local child
  for child in $(pgrep -P "$root" 2>/dev/null || true); do
    force_kill_tree "$child"
  done
  kill -KILL "$root" 2>/dev/null || true
}

summarize_results() {
  cd "$ROOT" || return 1
  PYTHONNOUSERSITE=1 python tools/summarize_results.py \
    --repo-root . \
    --output results_summary_linux_4090_v3_watchdog.csv \
    > "$LOG_ROOT/summarize.log" 2>&1
}

final_postprocess() {
  cd "$ROOT" || return 1
  log "Running final summary and plots"
  PYTHONNOUSERSITE=1 python tools/summarize_results.py \
    --repo-root . \
    --output results_summary_linux_4090_v3_final.csv \
    > "$LOG_ROOT/postprocess.log" 2>&1
  local summary_code=$?
  PYTHONNOUSERSITE=1 MPLBACKEND=Agg python tools/plot_results.py \
    --summary results_summary_linux_4090_v3_final.csv \
    --output-dir plots_linux_4090_v3_final \
    >> "$LOG_ROOT/postprocess.log" 2>&1
  local plot_code=$?
  log "Final postprocess summary_code=$summary_code plot_code=$plot_code"
  return "$summary_code"
}

v3_count() {
  grep -c 'linux_4090_v3' "$ROOT/results_summary_linux_4090_v3_watchdog.csv" 2>/dev/null || echo 0
}

target_exp() {
  local agents="$1" actor="$2" seed="$3"
  echo "linux_4090_v3_a${agents}_${actor}_seed${seed}"
}

target_complete() {
  local agents="$1" actor="$2" seed="$3"
  local exp
  exp="$(target_exp "$agents" "$actor" "$seed")"
  grep -q "$exp" "$ROOT/results_summary_linux_4090_v3_watchdog.csv" 2>/dev/null
}

target_pids() {
  local agents="$1" actor="$2" seed="$3"
  local exp
  exp="$(target_exp "$agents" "$actor" "$seed")"
  pgrep -f "mappo-MPE-${exp}" 2>/dev/null || true
}

any_target_running() {
  target_pids 7 snn_lif 3
  target_pids 7 snn_lif 4
}

latest_target_log_age() {
  local now latest
  now="$(date +%s)"
  latest="$(find "$ROOT/linux_logs" -type f \( \
      -name '*a7_snn_lif_seed3.log' -o \
      -name '*a7_snn_lif_seed4.log' \) \
      -printf '%T@\n' 2>/dev/null | sort -nr | head -1)"
  if [ -z "$latest" ]; then
    echo 999999
    return
  fi
  awk -v now="$now" -v latest="$latest" 'BEGIN { printf "%d\n", now - latest }'
}

kill_stale_targets() {
  local pid
  for pid in $(any_target_running); do
    log "TERM stale target tree pid=$pid"
    kill_tree "$pid"
  done
  sleep 10
  for pid in $(any_target_running); do
    log "KILL stale target tree pid=$pid"
    force_kill_tree "$pid"
  done
  for pid in $(pgrep -f 'final_recovery_v3_4090_.*\.sh|linux_recovery_v3_4090\.sh|linux_paper_queue_v3_4090\.sh' 2>/dev/null || true); do
    if [ "$pid" != "$$" ]; then
      log "TERM stale wrapper pid=$pid"
      kill_tree "$pid"
    fi
  done
}

attempt_count() {
  local index="$1"
  find "$LOG_ROOT" -maxdepth 1 -name "watchdog_retry_${index}_*.exit_code" 2>/dev/null | wc -l
}

run_one() {
  local index="$1" agents="$2" actor="$3" seed="$4"
  local exp run_log exit_file code
  exp="$(target_exp "$agents" "$actor" "$seed")"
  run_log="$LOG_ROOT/watchdog_retry_${index}_a${agents}_${actor}_seed${seed}_$(date +%H%M%S).log"
  exit_file="${run_log%.log}.exit_code"

  cd "$ROOT/onpolicy/scripts/train" || return 1
  log "START watchdog retry index=$index exp=$exp"
  set +e
  python -u train_mpe.py \
    --env_name MPE \
    --algorithm_name mappo \
    --experiment_name "$exp" \
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
  code=$?
  set +e
  echo "$code" > "$exit_file"
  if [ "$code" -eq 0 ]; then
    log "DONE watchdog retry index=$index exp=$exp"
  else
    log "FAILED watchdog retry index=$index exp=$exp exit=$code log=$run_log"
  fi
  return "$code"
}

main() {
  log "watchdog started root=$ROOT check_interval=$CHECK_INTERVAL stale_seconds=$STALE_SECONDS max_attempts=$MAX_ATTEMPTS"
  load_env

  while true; do
    summarize_results || log "summarize failed; will retry"

    local count running age attempts missing
    count="$(v3_count)"
    running="$(any_target_running | tr '\n' ' ')"
    missing=()
    target_complete 7 snn_lif 3 || missing+=("38 7 snn_lif 3")
    target_complete 7 snn_lif 4 || missing+=("39 7 snn_lif 4")

    log "status v3_count=$count missing=${#missing[@]} running_pids=${running:-none}"

    if [ "${#missing[@]}" -eq 0 ] && [ "$count" -ge 45 ]; then
      final_postprocess
      log "DONE all v3 runs complete"
      touch "$LOG_ROOT/watchdog_done"
      exit 0
    fi

    if [ -n "$running" ]; then
      age="$(latest_target_log_age)"
      log "target process is running; latest target log age=${age}s"
      if [ "$age" -gt "$STALE_SECONDS" ]; then
        log "stale threshold exceeded; cleaning target processes"
        kill_stale_targets
      else
        sleep "$CHECK_INTERVAL"
      fi
      continue
    fi

    if [ "${#missing[@]}" -eq 0 ]; then
      log "no target process and no target missing, waiting for summary count to settle"
      sleep "$CHECK_INTERVAL"
      continue
    fi

    # Run only one missing target per loop; this preserves the v3 config while keeping
    # recovery serial and easy to audit.
    read -r index agents actor seed <<< "${missing[0]}"
    attempts="$(attempt_count "$index")"
    if [ "$attempts" -ge "$MAX_ATTEMPTS" ]; then
      log "ABORT index=$index exceeded max attempts in this watchdog session"
      touch "$LOG_ROOT/watchdog_failed"
      exit 2
    fi

    run_one "$index" "$agents" "$actor" "$seed" || true
  done
}

main "$@"
