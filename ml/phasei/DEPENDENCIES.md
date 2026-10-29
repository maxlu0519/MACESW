# Dependencies and build boundary

> 中文摘要：本轮不安装依赖，也不改变默认编译。静态代码使用 Python、NumPy 和 PyROOT 生成源并读取模拟结果，以 MPI 运行现有 MACE 程序；这些接口尚未运行验证。PyTorch 和 uproot/awkward 仍是后续候选工具。

## Existing MACESW environment

| Component | Role in this project | Current status |
| --- | --- | --- |
| MACESW / MACE Phase I | Geometry, simulation, detector and event data definitions | Existing repository code |
| Mustard | Tuple models and ROOT output conventions used by MACESW | Existing C++ dependency |
| ROOT | Simulation file format and C++ I/O when a native adapter is needed | Existing MACESW environment |
| CMake and C++ toolchain | Build MACESW and a possible optional native adapter | Existing build system |

The simulation currently writes `G4Run*/TTCSimHit`, `SciFiSimHit`, and `ECALSimHit`; `SimPrimaryVertex` is written when primary-vertex saving is enabled. The relevant data declarations are in `src/lib/data/MACE/Data/SimHit.h++`, `src/lib/data/MACE/PhaseI/Data/SimHit.h++`, and `src/lib/data/MACE/Data/SimVertex.h++`. These SimHit headers currently carry deprecation notices, so an implementation must verify the actual ROOT schema and current replacement APIs before fixing an adapter contract.

## Current Python interface (not yet executed)

| Component | Intended role | Decision status |
| --- | --- | --- |
| Python | Source generation, simulation orchestration, dataset reading | Static code present; compatibility not yet validated |
| PyROOT | Write reader-compatible source TTrees and read Sim output shards | Required by the static interface; not yet validated |
| NumPy | Parallel sampling and numerical preprocessing | Required for source generation; not yet validated |
| MPI launcher (`mpiexec`) | Run the existing `MACE PhaseI SimMACEPhaseI` program with rank-separated outputs | Required to launch simulation; not invoked in this change |
| PyYAML | Parse arbitrary supported geometry YAML; a restricted parser handles the checked-in Phase I geometry | Optional |
| pytest | Run the written contract tests | Development only; tests not run in this change |

PyTorch for training and uproot/awkward for an alternative, potentially faster ROOT reader remain proposed for later work. Neither is required by the current static interfaces.

These Python packages are **not** default MACESW build requirements. No compatible version range or accelerator requirement is claimed by this static change. The next validation step must test representative ROOT files, choose compatible versions, and record a reproducible Python environment before claiming a working pipeline.

## Integration rule for later code

The initial reader uses PyROOT to preserve MACE/Mustard TTree field semantics and reads MPI output shards without merging. If that proves insufficient, place a small native converter under `ml/phasei/`, reuse MACE/Mustard definitions, and add an explicit CMake opt-in whose default is off. Training framework dependencies must not enter the default MACESW configuration or targets. Inference integration into C++ is a later decision, after validated models and an interface contract exist.

See the [README](README.md) for the intended data flow and [TASKS.md](TASKS.md) for the first compatibility checks.
