# Phase I event reconstruction with machine learning

> 中文摘要：本目录规划 Phase I 的端到端重建研究，现已加入单粒子 ROOT 源、MPI 模拟启动器和未合并 ROOT 分片的数据集读取接口。这些代码尚未运行验证，也没有模型训练或推理程序。

## Purpose and current status

This project will study reconstruction of Phase I events using the TTC, SciFi tracker, and ECAL together. It belongs to MACESW and should reuse its detector definitions, simulation outputs, and Mustard data conventions. The first study uses simulated energy deposits and hit times to measure an **idealized, simulation-conditioned performance bound**. It does not predict the final performance of optical sensors, electronics, or a calibrated detector.

**Current status: static implementation, not yet run or validated.** The source generator, MPI launcher, and dataset reader have been written but have not been executed against real ROOT files. There is no training program, model weight, or inference target. The commands below describe the intended interface, not a verified workflow.

## Intended data flow

```text
one source ROOT TTree per particle (gamma or e+)
       |  MPI SimMACEPhaseI
       v
unmerged per-rank SimMACEPhaseI ROOT files
  G4Run*/TTCSimHit, SciFiSimHit, ECALSimHit
  G4Run*/SimPrimaryVertex (when enabled; labels only)
       |
       v
event grouping + detector-observable feature selection + truth labels
       |
       v
reproducible train/validation/test datasets
       |
       v
conventional baselines + simple neural baseline -> predictions and metrics
```

The event key is for grouping and splitting, not a model feature. A SimHit record contains both detector-like quantities and MC truth. Only explicitly approved detector-observable fields may reach a model; the primary-vertex record is reserved for labels and evaluation. In particular, a generator vertex is not automatically the point where a particle enters a detector. Target positions and energy conventions must be stated before training.

## Intended command interface (unverified)

Run from the repository root with a Python environment containing PyROOT and NumPy. The generator samples the configured Phase I cylinder uniformly in volume, an isotropic direction, and kinetic energy from 1 keV to 60 MeV. It uses up to four sampling processes by default, followed by one ROOT writer. Generate separate files for `gamma` and `e+`:

```sh
PYTHONPATH=ml/phasei/python python -m phasei_ml.source \
  --particle gamma --events 10000 --seed 1 --output gamma.root
PYTHONPATH=ml/phasei/python python -m phasei_ml.source \
  --particle e+ --events 10000 --seed 2 --output positron.root
```

Each source has an adjacent `.root.json` manifest. The simulation launcher reads that manifest and, by default, uses up to four MPI ranks. Each particle needs its own output directory:

The source TTree has one entry and one primary particle per generated event. Its required branches follow the current Mustard reader: scalar `w` (`float`), `t` (`double`, ns), `x/y/z` (`float`, mm), and `std::vector<int> pdgID` plus `std::vector<float> E/px/py/pz` (MeV). `E` is **total** energy in the source; `SimPrimaryVertex.Ek0` is kinetic energy in the simulation output. The generator sets `w=1` and `t=0`.

```sh
PYTHONPATH=ml/phasei/python python -m phasei_ml.simulate \
  gamma.root.json gamma_sim
PYTHONPATH=ml/phasei/python python -m phasei_ml.simulate \
  positron.root.json positron_sim
```

The launcher writes a simulation manifest listing the actual rank files. Import `iter_events` from `phasei_ml.dataset` and pass the source and simulation manifest paths to read those files directly; no `hadd` step is required. The reader groups events across ranks by run and `EvtID`, retains zero-hit events, and checks the source against saved `SimPrimaryVertex` truth. It yields in rank-file/primary-tree order, not numeric `EvtID` order. In the current Mustard reader, event `i` consumes source entry `(i + 1) % N` for a single run of `N` events; the reader checks this mapping against every saved primary vertex.

These Python tools do not change the default MACESW build. If a native adapter is needed later, it should live under this directory, reuse MACE/Mustard data types, and be built only by an explicit opt-in. Training and evaluation can remain separate from the C++ build. No graph neural network or waveform model is required for the initial single-particle baseline.

## Documents

- [Roadmap](ROADMAP.md): research stages, deliverables, and progression criteria.
- [Dependencies](DEPENDENCIES.md): existing software and proposed training tools.
- [Current-stage tasks](TASKS.md): the Stage 1 implementation and acceptance checklist.

The repository-wide build and contribution guidance remains in the [root README](../../README.md) and [contribution guide](../../CONTRIBUTING.md).
