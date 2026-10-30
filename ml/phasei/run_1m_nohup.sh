#!/usr/bin/env bash
# Launch the 1M gamma + 1M positron Phase I campaign in the background.
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_dir="${PHASEI_OUTPUT_DIR:-$repo_root/build/phasei_training_1m_20261029}"
python_bin="${PHASEI_PYTHON:-python3}"
mace_bin="${PHASEI_MACE_BIN:-$repo_root/build/MACE}"
mpi_launcher="${PHASEI_MPIEXEC:-mpiexec}"
events="${PHASEI_EVENTS:-1000000}"
workers="${PHASEI_SOURCE_WORKERS:-4}"
ranks="${PHASEI_MPI_RANKS:-4}"

set_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" > "$run_dir/status.txt"
}

if [[ "${1:-}" == "--status" ]]; then
    if [[ -f "$run_dir/status.txt" ]]; then cat "$run_dir/status.txt"; else echo "not started"; fi
    if [[ -f "$run_dir/run.pid" ]] && kill -0 "$(cat "$run_dir/run.pid")" 2>/dev/null; then
        echo "worker running (PID $(cat "$run_dir/run.pid"))"
    fi
    exit 0
fi

if [[ "${1:-}" != "--worker" ]]; then
    mkdir -p "$run_dir"
    if [[ -f "$run_dir/run.pid" ]] && kill -0 "$(cat "$run_dir/run.pid")" 2>/dev/null; then
        echo "Campaign already running (PID $(cat "$run_dir/run.pid"))" >&2
        exit 1
    fi
    if [[ -e "$run_dir/gamma.root" || -e "$run_dir/positron.root" ||
          -e "$run_dir/gamma_sim" || -e "$run_dir/positron_sim" ]]; then
        echo "Campaign outputs already exist in $run_dir; refusing to overwrite" >&2
        exit 1
    fi
    # setsid keeps the worker outside an ephemeral launching session, while
    # nohup ignores a terminal hangup in ordinary interactive use.
    setsid nohup "$0" --worker > "$run_dir/run.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$run_dir/run.pid"
    sleep 3
    if ! kill -0 "$(cat "$run_dir/run.pid")" 2>/dev/null; then
        echo "Worker exited early; inspect $run_dir/run.log" >&2
        exit 1
    fi
    echo "Campaign running (PID $(cat "$run_dir/run.pid"))"
    echo "Status: $run_dir/status.txt"
    echo "Log: $run_dir/run.log"
    exit 0
fi

on_exit() {
    rc=$?
    if (( rc != 0 )); then set_status "failed exit=$rc"; fi
}
trap on_exit EXIT

cd "$repo_root"
if [[ -n "${ROOTSYS:-}" ]]; then
    export PYTHONPATH="$repo_root/ml/phasei/python:$ROOTSYS/lib${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$repo_root/ml/phasei/python${PYTHONPATH:+:$PYTHONPATH}"
fi
"$python_bin" -c 'import ROOT, numpy'
test -x "$mace_bin"
command -v "${mpi_launcher%% *}" > /dev/null

set_status "generating sources"
"$python_bin" -m phasei_ml.source \
    --particle gamma --events "$events" --seed 1 --workers "$workers" \
    --output "$run_dir/gamma.root" > "$run_dir/gamma.source.log" 2>&1 &
gamma_source_pid=$!
"$python_bin" -m phasei_ml.source \
    --particle e+ --events "$events" --seed 2 --workers "$workers" \
    --output "$run_dir/positron.root" > "$run_dir/positron.source.log" 2>&1 &
positron_source_pid=$!

source_failed=0
wait "$gamma_source_pid" || source_failed=1
wait "$positron_source_pid" || source_failed=1
if (( source_failed )); then
    echo "Source generation failed; inspect the per-particle source logs" >&2
    exit 1
fi

"$python_bin" - "$run_dir" "$events" <<'PY'
import json
from pathlib import Path
import sys
import ROOT

directory, expected = Path(sys.argv[1]), int(sys.argv[2])
for name in ("gamma", "positron"):
    manifest = json.loads((directory / f"{name}.root.json").read_text())
    assert manifest["events"] == expected, (name, manifest["events"], expected)
    handle = ROOT.TFile.Open(str(directory / f"{name}.root"), "READ")
    assert handle and not handle.IsZombie(), name
    try:
        tree = handle.Get(manifest["tree"])
        assert tree and tree.GetEntries() == expected, name
    finally:
        handle.Close()
PY

set_status "simulating gamma and positron"
"$python_bin" -m phasei_ml.simulate \
    "$run_dir/gamma.root.json" "$run_dir/gamma_sim" \
    --ranks "$ranks" --mace-bin "$mace_bin" --mpiexec "$mpi_launcher" \
    > "$run_dir/gamma.sim.log" 2>&1 &
gamma_sim_pid=$!
"$python_bin" -m phasei_ml.simulate \
    "$run_dir/positron.root.json" "$run_dir/positron_sim" \
    --ranks "$ranks" --mace-bin "$mace_bin" --mpiexec "$mpi_launcher" \
    > "$run_dir/positron.sim.log" 2>&1 &
positron_sim_pid=$!

sim_failed=0
wait "$gamma_sim_pid" || sim_failed=1
wait "$positron_sim_pid" || sim_failed=1
if (( sim_failed )); then
    echo "Simulation failed; inspect the per-particle simulation logs" >&2
    exit 1
fi

set_status "complete"
echo "Both 1M-event sources and MPI simulations completed."
