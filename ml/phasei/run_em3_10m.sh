#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_dir="${PHASEI_10M_RUN_DIR:-$repo_root/build/phasei_em3_10m_20261030}"
dataset_dir="${PHASEI_10M_DATASET_DIR:-$repo_root/build/phasei_ml_dataset_10m_20261030}"
unit="phasei-em3-10m-20261030.service"
events=5000000
ranks=56
workers=4
python_bin="$repo_root/.venv/bin/python"
root_sys="${ROOTSYS:-/opt/root/6.38.00}"
mace_bin="$repo_root/build/phasei_em3_release/MACE"
geometry="$repo_root/src/app/simulation/MACE/PhaseI/SimMACEPhaseI/scripts/SimMACEPhaseI_geom.yaml"
status_file="$run_dir/status.txt"

set_status() {
    mkdir -p "$run_dir"
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" > "$status_file"
}

status() {
    if systemctl --user is-active --quiet "$unit"; then
        systemctl --user show "$unit" -p ActiveState -p SubState -p MainPID -p ExecMainStartTimestamp --no-pager
        [[ ! -f "$status_file" ]] || cat "$status_file"
        for log in "$run_dir/source_eplus.log" "$run_dir/source_gamma.log" \
                   "$run_dir/sim_eplus.log" "$run_dir/sim_gamma.log" \
                   "$run_dir/convert.log"; do
            if [[ -f "$log" ]]; then
                printf '\n--- %s ---\n' "${log##*/}"
                tail -n 3 "$log"
            fi
        done
    elif [[ -f "$dataset_dir/manifest.json" ]]; then
        printf 'Campaign complete. Dataset: %s\n' "$dataset_dir"
    else
        [[ ! -f "$status_file" ]] || cat "$status_file"
        printf 'No active service. Inspect %s\n' "$run_dir"
        return 1
    fi
}

archive_partial() {
    local path="$1"
    if [[ -e "$path" ]]; then
        mv "$path" "${path}.incomplete.$(date +%Y%m%d_%H%M%S)"
    fi
}

ensure_source() {
    local particle="$1" label="$2" pdg="$3" seed="$4"
    local source="$run_dir/${label}.root" manifest="$run_dir/${label}.root.json"
    if [[ -f "$manifest" ]]; then
        if "$python_bin" - "$manifest" "$source" "$particle" "$events" "$geometry" <<'PY'
import json, sys
import hashlib
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
root=Path(sys.argv[2]); geom=Path(sys.argv[5])
if m.get('particle')!=sys.argv[3] or m.get('events')!=int(sys.argv[4]): raise SystemExit(1)
if not root.is_file() or m.get('source_file')!=str(root.resolve()): raise SystemExit(1)
if hashlib.sha256(root.read_bytes()).hexdigest()!=m.get('sha256'): raise SystemExit(1)
if hashlib.sha256(geom.read_bytes()).hexdigest()!=m.get('geometry_sha256'): raise SystemExit(1)
PY
        then return; fi
        archive_partial "$source"
        archive_partial "$manifest"
    fi
    if [[ ! -f "$manifest" ]]; then archive_partial "$source"; fi
    set_status "generating $particle source ($events events, $workers workers)"
    "$python_bin" -m phasei_ml.source --particle "$particle" --events "$events" \
        --seed "$seed" --workers "$workers" --chunk-size 10000 --geometry "$geometry" \
        --output "$source" > "$run_dir/source_${label}.log" 2>&1
}

ensure_simulation() {
    local label="$1" particle="$2"
    local source_manifest="$run_dir/${label}.root.json"
    local sim_dir="$run_dir/${label}_sim"
    local sim_manifest="$sim_dir/simulation_manifest.json"
    if [[ -f "$sim_manifest" ]]; then
        if "$python_bin" - "$sim_manifest" "$particle" "$events" "$ranks" \
            "$source_manifest" "$sim_dir" <<'PY'
import json, sys
import hashlib
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
source_path=Path(sys.argv[5]); sim_dir=Path(sys.argv[6])
s=json.loads(source_path.read_text())
if m.get('completed') is not True or m.get('particle')!=sys.argv[2]: raise SystemExit(1)
if m.get('events')!=int(sys.argv[3]) or m.get('mpi_ranks')!=int(sys.argv[4]): raise SystemExit(1)
if m.get('source_sha256')!=s.get('sha256'): raise SystemExit(1)
if m.get('source_manifest_sha256')!=hashlib.sha256(source_path.read_bytes()).hexdigest(): raise SystemExit(1)
outputs=m.get('output_files',[])
if len(outputs)!=int(sys.argv[4]): raise SystemExit(1)
if any(not (sim_dir/p).is_file() or (sim_dir/p).stat().st_size==0 for p in outputs): raise SystemExit(1)
PY
        then return; fi
        archive_partial "$sim_dir"
    else
        archive_partial "$sim_dir"
    fi
    set_status "simulating $particle ($events events, $ranks MPI ranks)"
    "$python_bin" -m phasei_ml.simulate "$source_manifest" "$sim_dir" \
        --ranks "$ranks" --mace-bin "$mace_bin" --mpiexec /usr/bin/mpiexec \
        > "$run_dir/sim_${label}.log" 2>&1
}

worker() {
    trap 'rc=$?; if (( rc != 0 )); then set_status "failed exit=$rc"; fi' EXIT
    cd "$repo_root"
    source "$root_sys/bin/thisroot.sh"
    source /opt/geant4/11.3.2/bin/geant4.sh
    export GEANT4_DIR=/opt/geant4/11.3.2/lib/cmake/Geant4
    export ROOTSYS="$root_sys"
    export PYTHONPATH="$repo_root/ml/phasei/python:$root_sys/lib${PYTHONPATH:+:$PYTHONPATH}"
    "$python_bin" -c 'import ROOT, numpy; print("ROOT", ROOT.gROOT.GetVersion())'
    [[ -x "$mace_bin" && -r "$geometry" ]]
    command -v /usr/bin/mpiexec >/dev/null

    set_status "starting source generation for gamma/e+; EM3 Release; 56 MPI ranks per simulation"
    ensure_source 'e+' eplus -11 2026103001
    ensure_source gamma gamma 22 2026103002
    set_status "source generation complete; validating both ROOT trees"
    "$python_bin" - "$run_dir" "$events" <<'PY'
import json, sys
from pathlib import Path
import ROOT
d=Path(sys.argv[1]); n=int(sys.argv[2])
for name in ('eplus','gamma'):
    m=json.loads((d/f'{name}.root.json').read_text())
    assert m['events']==n
    f=ROOT.TFile.Open(str(d/f'{name}.root'),'READ')
    assert f and not f.IsZombie()
    try:
        tree=f.Get(m['tree'])
        assert tree and tree.GetEntries()==n
    finally: f.Close()
PY

    ensure_simulation eplus 'e+'
    ensure_simulation gamma gamma
    set_status "both simulations complete; converting unmerged rank files"
    if [[ -f "$dataset_dir/manifest.json" ]]; then
        "$python_bin" - "$dataset_dir/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert sum(p['events'] for p in m['particles'])==10_000_000
PY
    else
        archive_partial "$dataset_dir"
        set_status "converting 10M events (direct unmerged ROOT shards; no hadd)"
        "$python_bin" -m phasei_ml.convert \
            --pair "$run_dir/eplus.root.json" "$run_dir/eplus_sim/simulation_manifest.json" \
            --pair "$run_dir/gamma.root.json" "$run_dir/gamma_sim/simulation_manifest.json" \
            --output "$dataset_dir" > "$run_dir/convert.log" 2>&1
    fi
    set_status "complete dataset=$dataset_dir"
    trap - EXIT
}

if [[ "${1:-}" == "--status" ]]; then
    status
    exit $?
fi
if [[ "${1:-}" == "--worker" ]]; then
    worker
    exit $?
fi

mkdir -p "$run_dir"
if systemctl --user is-active --quiet "$unit"; then
    echo "Campaign already active: $unit" >&2
    exit 1
fi
systemctl --user reset-failed "$unit" >/dev/null 2>&1 || true
systemd-run --user --unit="$unit" --description="Phase I EM3 10M training data campaign" \
    --working-directory="$repo_root" \
    --property=Restart=on-failure --property=RestartSec=30s \
    --property=StartLimitIntervalSec=300s --property=StartLimitBurst=3 \
    --property="StandardOutput=append:$run_dir/service.log" \
    --property="StandardError=append:$run_dir/service.log" \
    "$repo_root/ml/phasei/run_em3_10m.sh" --worker
sleep 3
if ! systemctl --user is-active --quiet "$unit"; then
    systemctl --user status "$unit" --no-pager || true
    exit 1
fi
printf 'Started %s\nStatus: %s --status\nOutput: %s\nDataset: %s\n' \
    "$unit" "$0" "$run_dir" "$dataset_dir"
