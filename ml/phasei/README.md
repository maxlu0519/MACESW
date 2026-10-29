# Phase I event reconstruction with machine learning

> 中文摘要：本目录规划 Phase I 的端到端重建研究。第一阶段只用理想化的 SimHit 估计单粒子重建能力边界；后续逐步加入多粒子事例、样机响应和实验标定。目前只有文档，没有可运行的训练或推理程序。

## Purpose and current status

This project will study reconstruction of Phase I events using the TTC, SciFi tracker, and ECAL together. It belongs to MACESW and should reuse its detector definitions, simulation outputs, and Mustard data conventions. The first study uses simulated energy deposits and hit times to measure an **idealized, simulation-conditioned performance bound**. It does not predict the final performance of optical sensors, electronics, or a calibrated detector.

**Current status: documentation only.** This directory contains no dataset converter, training program, model weights, inference target, or supported command to run. The documents describe work to be implemented on this branch in later changes.

## Intended data flow

```text
SimMACEPhaseI ROOT output
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

The first implementation should read existing ROOT output without changing the default MACESW build. If ROOT reading needs a native adapter, it should live under this directory, reuse MACE/Mustard data types, and be built only by an explicit opt-in. Python training and evaluation can remain separate from the C++ build. No graph neural network or waveform model is required for the initial single-particle baseline.

## Documents

- [Roadmap](ROADMAP.md): research stages, deliverables, and progression criteria.
- [Dependencies](DEPENDENCIES.md): existing software and proposed training tools.
- [Current-stage tasks](TASKS.md): the Stage 1 implementation and acceptance checklist.

The repository-wide build and contribution guidance remains in the [root README](../../README.md) and [contribution guide](../../CONTRIBUTING.md).
