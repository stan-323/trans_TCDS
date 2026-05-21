#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/home/ubuntu/spiking_mappo_official}"
CONDA_ROOT="${CONDA_ROOT:-/home/ubuntu/anaconda3}"
ENV_NAME="${ENV_NAME:-spiking_mappo}"
CHECK_INTERVAL="${CHECK_INTERVAL:-600}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="$ROOT/linux_logs/package_final_v3_4090_$STAMP"
PID_FILE="$ROOT/linux_logs/package_final_v3_4090.pid"
ARTIFACT_DIR="$ROOT/final_artifacts"

mkdir -p "$LOG_ROOT" "$ARTIFACT_DIR" "$ROOT/linux_logs"
echo "$$" > "$PID_FILE"
exec > >(tee -a "$LOG_ROOT/package_final.log") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

load_env() {
  # shellcheck source=/dev/null
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  export WANDB_DISABLED=true
  export PYTHONNOUSERSITE=1
  export MPLBACKEND=Agg
}

summary_count() {
  grep -c 'linux_4090_v3' "$ROOT/results_summary_linux_4090_v3_final.csv" 2>/dev/null || echo 0
}

main() {
  log "final packager started root=$ROOT check_interval=$CHECK_INTERVAL"
  load_env

  while true; do
    cd "$ROOT" || exit 1
    PYTHONNOUSERSITE=1 python tools/summarize_results.py \
      --repo-root . \
      --output results_summary_linux_4090_v3_final.csv \
      > "$LOG_ROOT/summarize.log" 2>&1
    local count
    count="$(summary_count)"
    log "v3_count=$count"

    if [ "$count" -ge 45 ]; then
      PYTHONNOUSERSITE=1 MPLBACKEND=Agg python tools/plot_results.py \
        --summary results_summary_linux_4090_v3_final.csv \
        --output-dir plots_linux_4090_v3_final \
        > "$LOG_ROOT/plot.log" 2>&1

      local ready_file archive
      ready_file="$ARTIFACT_DIR/FINAL_V3_READY.txt"
      archive="$ARTIFACT_DIR/spiking_mappo_4090_v3_final_$(date +%Y%m%d_%H%M%S).tar.gz"
      {
        echo "ready_at=$(date '+%F %T')"
        echo "root=$ROOT"
        echo "summary=$ROOT/results_summary_linux_4090_v3_final.csv"
        echo "plots=$ROOT/plots_linux_4090_v3_final"
        echo "v3_count=$count"
        echo "archive=$archive"
      } > "$ready_file"

      tar -czf "$archive" \
        results_summary_linux_4090_v3_final.csv \
        plots_linux_4090_v3_final \
        final_artifacts/FINAL_V3_READY.txt \
        linux_logs/final_recovery_v3_4090_* \
        linux_logs/watchdog_final_v3_4090_* \
        "$LOG_ROOT" \
        2>> "$LOG_ROOT/tar.log"

      log "DONE archive=$archive"
      exit 0
    fi

    sleep "$CHECK_INTERVAL"
  done
}

main "$@"
