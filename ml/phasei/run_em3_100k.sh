#!/usr/bin/env bash
# Persistent, sequential Phase I EM3 production run: 100k e+ then 100k gamma.
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_dir="${PHASEI_OUTPUT_DIR:-$repo_root/build/phasei_em3_100k_20261029}"
python_bin="${PHASEI_PYTHON:-python3}"
mace_bin="${PHASEI_MACE_BIN:-$repo_root/build/phasei_em3_release/MACE}"
mpi_launcher="${PHASEI_MPIEXEC:-mpiexec}"
events=100000
ranks=56

set_status() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" > "$run_dir/status.txt"; }

if [[ "${1:-}" == "--status" ]]; then
    [[ ! -f "$run_dir/status.txt" ]] || cat "$run_dir/status.txt"
    if [[ -f "$run_dir/run.pid" ]] && kill -0 "$(cat "$run_dir/run.pid")" 2>/dev/null; then
        echo "worker running (PID $(cat "$run_dir/run.pid"))"
    fi
    for phase in positron gamma; do
        log="$run_dir/$phase.sim.log"
        if [[ -f "$log" ]]; then
            echo "$phase progress: $(rg -o 'Event [0-9]+|events processed: [0-9]+' "$log" | tail -1 || true)"
        fi
    done
    exit 0
fi

if [[ "${1:-}" != "--worker" ]]; then
    mkdir -p "$run_dir"
    if [[ -f "$run_dir/run.pid" ]] && kill -0 "$(cat "$run_dir/run.pid")" 2>/dev/null; then
        echo "Campaign already running (PID $(cat "$run_dir/run.pid"))" >&2; exit 1
    fi
    if [[ -e "$run_dir/positron_sim" || -e "$run_dir/gamma_sim" ]]; then
        echo "Simulation output directories already exist in $run_dir; refusing to overwrite" >&2; exit 1
    fi
    if [[ -e "$run_dir/positron.root" || -e "$run_dir/gamma.root" || -e "$run_dir/positron.root.json" || -e "$run_dir/gamma.root.json" ]]; then
        if [[ ! -f "$run_dir/positron.root" || ! -f "$run_dir/gamma.root" || ! -f "$run_dir/positron.root.json" || ! -f "$run_dir/gamma.root.json" ]]; then
            echo "Source outputs are incomplete in $run_dir; refusing to overwrite" >&2; exit 1
        fi
    fi
    setsid nohup "$0" --worker > "$run_dir/run.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$run_dir/run.pid"
    sleep 3
    if ! kill -0 "$(cat "$run_dir/run.pid")" 2>/dev/null; then echo "Worker exited early; inspect $run_dir/run.log" >&2; exit 1; fi
    echo "Campaign running (PID $(cat "$run_dir/run.pid")); status: $run_dir/status.txt; logs: $run_dir"
    exit 0
fi

on_exit() { rc=$?; if (( rc != 0 )); then set_status "failed exit=$rc"; fi; }
trap on_exit EXIT
cd "$repo_root"
if [[ -n "${ROOTSYS:-}" ]]; then export PYTHONPATH="$repo_root/ml/phasei/python:$ROOTSYS/lib${PYTHONPATH:+:$PYTHONPATH}"; else export PYTHONPATH="$repo_root/ml/phasei/python${PYTHONPATH:+:$PYTHONPATH}"; fi
"$python_bin" -c 'import ROOT, numpy'
test -x "$mace_bin"
command -v "${mpi_launcher%% *}" >/dev/null

if [[ -f "$run_dir/positron.root.json" && -f "$run_dir/gamma.root.json" ]]; then
    set_status "reusing verified source manifests"
elif [[ ! -e "$run_dir/positron.root" && ! -e "$run_dir/gamma.root" &&
        ! -e "$run_dir/positron.root.json" && ! -e "$run_dir/gamma.root.json" ]]; then
    set_status "generating positron source"
    "$python_bin" -m phasei_ml.source --particle 'e+' --events "$events" --seed 20261029 --workers 4 --output "$run_dir/positron.root" > "$run_dir/positron.source.log" 2>&1
    set_status "generating gamma source"
    "$python_bin" -m phasei_ml.source --particle gamma --events "$events" --seed 20261030 --workers 4 --output "$run_dir/gamma.root" > "$run_dir/gamma.source.log" 2>&1
else
    echo "Source outputs are incomplete; refusing to overwrite them" >&2
    exit 1
fi

"$python_bin" - "$run_dir" "$events" <<'PY'
import json, sys
from pathlib import Path
import ROOT
d, n = Path(sys.argv[1]), int(sys.argv[2])
for name in ("positron", "gamma"):
    m = json.loads((d / f"{name}.root.json").read_text())
    assert m["events"] == n, (name, m.get("events"))
    f = ROOT.TFile.Open(str(d / f"{name}.root"), "READ")
    assert f and not f.IsZombie(), name
    try:
        t = f.Get(m["tree"])
        assert t and t.GetEntries() == n, (name, t.GetEntries() if t else None)
    finally: f.Close()
PY

set_status "simulating positron (56 MPI ranks)"
"$python_bin" -m phasei_ml.simulate "$run_dir/positron.root.json" "$run_dir/positron_sim" --ranks "$ranks" --mace-bin "$mace_bin" --mpiexec "$mpi_launcher" > "$run_dir/positron.sim.log" 2>&1
set_status "positron complete; verifying before gamma"
"$python_bin" - "$run_dir/positron_sim/simulation_manifest.json" <<'PY'
import json, sys
m=json.load(open(sys.argv[1])); assert m["completed"] and m["events"] == 100000 and m["mpi_ranks"] == 56, m
assert m["particle"] == "e+", m.get("particle")
PY

set_status "simulating gamma (56 MPI ranks)"
"$python_bin" -m phasei_ml.simulate "$run_dir/gamma.root.json" "$run_dir/gamma_sim" --ranks "$ranks" --mace-bin "$mace_bin" --mpiexec "$mpi_launcher" > "$run_dir/gamma.sim.log" 2>&1
"$python_bin" - "$run_dir/gamma_sim/simulation_manifest.json" <<'PY'
import json, sys
m=json.load(open(sys.argv[1])); assert m["completed"] and m["events"] == 100000 and m["mpi_ranks"] == 56, m
assert m["particle"] == "gamma", m.get("particle")
PY
set_status "complete"
echo "Both sequential 100k EM3 simulations completed."
