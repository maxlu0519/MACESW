# Dependencies and build boundary

> 中文摘要：本目录不改变默认编译。Phase I Python 环境已在仓库根目录 `.venv/` 安装并验证；PyROOT 使用系统 ROOT，PyTorch CUDA 12.4 张量运算及 22 项接口测试均通过。

## Existing MACESW environment

| Component | Role in this project | Current status |
| --- | --- | --- |
| MACESW / MACE Phase I | Geometry, simulation, detector and event data definitions | Existing repository code |
| Mustard | Tuple models and ROOT output conventions used by MACESW | Existing C++ dependency |
| ROOT | Simulation file format and C++ I/O when a native adapter is needed | Existing MACESW environment |
| CMake and C++ toolchain | Build MACESW and a possible optional native adapter | Existing build system |

The simulation currently writes `G4Run*/TTCSimHit`, `SciFiSimHit`, and `ECALSimHit`; `SimPrimaryVertex` is written when primary-vertex saving is enabled. The relevant data declarations are in `src/lib/data/MACE/Data/SimHit.h++`, `src/lib/data/MACE/PhaseI/Data/SimHit.h++`, and `src/lib/data/MACE/Data/SimVertex.h++`. These SimHit headers currently carry deprecation notices, so an implementation must verify the actual ROOT schema and current replacement APIs before fixing an adapter contract.

## Current Python interface (validated)

| Component | Intended role | Decision status |
| --- | --- | --- |
| Python | Source generation, simulation orchestration, dataset reading | Python 3.13.5 in repository-root `.venv/` |
| PyROOT | Write reader-compatible source TTrees and read Sim output shards | Imports from `/opt/root/6.38.00/lib`; `ROOT.gROOT.GetVersion()` reports 6.36.06; do not pip-install ROOT |
| NumPy | Parallel sampling and numerical preprocessing | 2.2.4 |
| MPI launcher (`mpiexec`) | Run the existing `MACE PhaseI SimMACEPhaseI` program with rank-separated outputs | Required; exercised with two ranks in small pilots |
| PyYAML | Parse arbitrary supported geometry YAML; a restricted parser handles the checked-in Phase I geometry | 6.0.2; optional for the checked-in geometry |
| PyTorch | Neural-network training | 2.6.0+cu124; CUDA 12.4 runtime detected on GTX 1660 SUPER, driver 550.163.01 |
| pytest | Run the written contract tests | 8.3.5; development only |

`ml/phasei/requirements.txt` records the Python package versions. System ROOT/PyROOT is intentionally not installed through pip and is not part of that file.

## Repository-local Python environment

The Phase I environment lives at the repository root in `.venv/`. From the repository root, create it once with `python3 -m venv .venv`, then install the pinned packages with `.venv/bin/python -m pip install -r ml/phasei/requirements.txt`.

Activate it with `source .venv/bin/activate`. PyROOT is supplied by the existing ROOT installation, so preserve these environment values when using the venv:

```sh
export ROOTSYS=/opt/root/6.38.00
export PYTHONPATH="$ROOTSYS/lib${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOTSYS/lib:/opt/geant4/11.3.2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The package is currently used from source; make it importable with `export PYTHONPATH="$PWD/ml/phasei/python:$PYTHONPATH"`. Run the Python contract tests with `python -m pytest ml/phasei/tests` after activating the environment and exporting these paths.

The CUDA wheel was selected against the detected driver and GPU. A 256-event forward/backward batch on the GTX 1660 SUPER used 40.5 MB peak allocated and 52.4 MB peak reserved memory for the V1 model. The full existing 200,000-event dataset converted in 8.68 seconds with 735,372 KiB peak RSS. The two-epoch CPU smoke run completed and its checkpoint reloaded. The throughput-focused batch-4096 run used 2.28 GiB peak CUDA reserved memory. All 40 Phase I Python tests pass. No new Geant4 simulation is required to reproduce these training results.

These Python packages are **not** default MACESW build requirements. The pins validate the Python contract tests, ROOT conversion, and initial model interface. Any future native adapter must remain an explicit opt-in and must not add training dependencies to the default build.

## Integration rule for later code

The initial reader uses PyROOT to preserve MACE/Mustard TTree field semantics and reads MPI output shards without merging. If that proves insufficient, place a small native converter under `ml/phasei/`, reuse MACE/Mustard definitions, and add an explicit CMake opt-in whose default is off. Training framework dependencies must not enter the default MACESW configuration or targets. Inference integration into C++ is a later decision, after validated models and an interface contract exist.

See the [README](README.md) for the intended data flow and [TASKS.md](TASKS.md) for the first compatibility checks.
