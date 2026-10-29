# Dependencies and build boundary

> 中文摘要：本轮不安装依赖，也不改变默认编译。现有模拟与数据定义属于 MACESW、Mustard、ROOT/CMake；第一阶段拟用 Python、PyTorch、uproot/awkward、NumPy 做离线训练和评估。实现时再验证兼容性并锁定版本。

## Existing MACESW environment

| Component | Role in this project | Current status |
| --- | --- | --- |
| MACESW / MACE Phase I | Geometry, simulation, detector and event data definitions | Existing repository code |
| Mustard | Tuple models and ROOT output conventions used by MACESW | Existing C++ dependency |
| ROOT | Simulation file format and C++ I/O when a native adapter is needed | Existing MACESW environment |
| CMake and C++ toolchain | Build MACESW and a possible optional native adapter | Existing build system |

The simulation currently writes `G4Run*/TTCSimHit`, `SciFiSimHit`, and `ECALSimHit`; `SimPrimaryVertex` is written when primary-vertex saving is enabled. The relevant data declarations are in `src/lib/data/MACE/Data/SimHit.h++`, `src/lib/data/MACE/PhaseI/Data/SimHit.h++`, and `src/lib/data/MACE/Data/SimVertex.h++`. These SimHit headers currently carry deprecation notices, so an implementation must verify the actual ROOT schema and current replacement APIs before fixing an adapter contract.

## Proposed Stage 1 training environment

| Component | Intended role | Decision status |
| --- | --- | --- |
| Python | Dataset preparation, training, and evaluation | Proposed |
| uproot + awkward | Read variable-length ROOT event/hit data without linking Python training to MACESW | Proposed; verify against real output |
| NumPy | Numerical preprocessing and baseline calculations | Proposed |
| PyTorch | Simple neural baseline and training loop | Proposed |

These packages are **not** MACESW build requirements today. No package manifest, compatible version range, accelerator requirement, or installation command is claimed by this documentation-only change. The first implementation must test representative ROOT files, choose compatible versions, record a reproducible Python environment, and document CPU execution before optional accelerator support.

## Integration rule for later code

Prefer a Python reader when it preserves the event and field semantics of MACESW output. If Python I/O is insufficient, place a small native converter under `ml/phasei/`, reuse MACE/Mustard definitions, and add an explicit CMake opt-in whose default is off. Training framework dependencies must not enter the default MACESW configuration or targets. Inference integration into C++ is a later decision, after validated models and an interface contract exist.

See the [README](README.md) for the intended data flow and [TASKS.md](TASKS.md) for the first compatibility checks.
