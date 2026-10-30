#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
out_dir="${PHASEI_ML_OUTPUT_DIR:-build/phasei_ml_training_v2}"
pid_file="$out_dir/experiment.pid"
log_file="$out_dir/experiment.log"
unit="phasei-ml-training-v2.service"

status() {
    if systemctl --user is-active --quiet "$unit"; then
        systemctl --user show "$unit" -p ActiveState -p SubState -p MainPID -p ExecMainStartTimestamp --no-pager
        printf 'Run log: %s\n' "$log_file"
        latest="$(find "$out_dir" -mindepth 2 -maxdepth 2 -name train.log -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
        if [[ -n "$latest" ]]; then
            printf 'Latest training log: %s\n' "$latest"
            tail -n 2 "$latest"
        fi
    elif systemctl --user list-units --all --plain --no-legend "$unit" 2>/dev/null | rg -q "$unit"; then
        systemctl --user status "$unit" --no-pager || true
        printf 'Service journal: journalctl --user -u %s\n' "$unit"
    elif [[ -f "$out_dir/experiment.json" ]] && rg -q 'completed_unix' "$out_dir/experiment.json"; then
        printf 'Experiment complete. Report: %s/report.md\n' "$out_dir"
    else
        printf 'No active experiment process. Check %s for logs and checkpoints.\n' "$out_dir"
        return 1
    fi
}

if [[ "${1:-}" == "--status" ]]; then
    status
    exit $?
fi

if systemctl --user is-active --quiet "$unit"; then
    printf 'Experiment already active in systemd unit %s\n' "$unit" >&2
    exit 1
fi

mkdir -p "$out_dir"
systemctl --user reset-failed "$unit" >/dev/null 2>&1 || true
systemd-run --user --unit="$unit" --description="Phase I full ML training and evaluation" \
    --working-directory="$repo_root" \
    --property=Restart=on-failure --property=RestartSec=30s \
    --property=StartLimitIntervalSec=300s --property=StartLimitBurst=3 \
    --property="StandardOutput=append:$repo_root/$log_file" \
    --property="StandardError=append:$repo_root/$log_file" \
    env PYTHONPATH="$repo_root/ml/phasei/python${PYTHONPATH:+:$PYTHONPATH}" \
    "$repo_root/.venv/bin/python" -u -m phasei_ml.experiment \
    --data "$repo_root/build/phasei_ml_dataset_v1" \
    --output-dir "$out_dir" --device cuda:0
pid="$(systemctl --user show "$unit" -p MainPID --value)"
printf '%s\n' "$pid" > "$pid_file"
sleep 2
if ! systemctl --user is-active --quiet "$unit"; then
    printf 'Experiment service did not stay active; inspect %s\n' "$log_file" >&2
    systemctl --user status "$unit" --no-pager || true
    exit 1
fi
printf 'Started systemd service %s (PID %s)\nLog: %s\nStatus: %s --status\n' "$unit" "$pid" "$log_file" "$0"
