# Phase I reconstruction roadmap

> 中文摘要：路线分为理想单粒子 SimHit、多粒子与 pileup、样机响应与弱监督、最终实验标定四阶段。每阶段都要保留独立测试集并明确数据选择、真值定义和性能指标；理想模拟结果不能直接当作真实探测器性能。

## Shared rules

- Record the simulation, geometry, generator, and selection configuration with each dataset and result. Keep MC truth out of model inputs.
- Split by complete generated event before deriving examples or augmentations. Report generated-event denominators, including events with no selected hit, so an ECAL coincidence requirement cannot silently inflate efficiency.
- Quote performance versus particle type, energy, position, direction, and detector acceptance. A rotated or otherwise asymmetric apparatus must be evaluated in its actual azimuthal geometry; do not assume rotational symmetry.
- Distinguish conditional resolution for reconstructed events from total efficiency and acceptance. Keep statistical and detector-response limitations separate.

## Stage 1 — Single-particle ideal SimHit baseline

**Goal:** Establish a reproducible, simulation-conditioned performance bound for one incident γ, e⁺, or e⁻. Compare conventional estimators with a simple neural baseline before considering a graph model.

**Data:** `SimMACEPhaseI` TTC, SciFi, and ECAL SimHit ROOT trees, with saved primary-particle truth for labels. Generate multiple monoenergetic points and vary incident position, direction, and azimuth. Preserve empty/no-hit events. Use only an approved detector-observable feature whitelist; do not treat simulated deposits as calibrated optical or electronic readout.

**Outputs:** A versioned dataset specification, generation and split manifests, baseline predictions, and held-out reports of energy and position bias/resolution, angular error, PID confusion, and detection/reconstruction efficiency. State whether each target is generator-level or detector-entry truth; use kinetic-energy labels consistently.

**Exit criterion:** Another developer can reproduce the splits, baselines, and reports from recorded inputs; truth-leakage checks pass; performance is shown across the sampled phase space and on held-out energy/angle regions. There is no fixed resolution target before those studies establish a reference.

## Stage 2 — Multi-particle simulation

**Goal:** Extend the validated single-particle pipeline to event multiplicity, particle association, overlapping showers/tracks, relevant decay channels, and pileup. Estimate multiplicity or topology probabilities before per-particle reconstruction where useful.

**Data:** Generated multi-particle and pileup samples with distinct truth records, mixed detector hits, and recorded channel, timing, and overlap conditions. Include single-particle and background controls.

**Outputs:** Event-level topology/PID probabilities, matched particle candidates, reconstructed kinematics, and efficiency and fake-rate studies versus overlap and pileup. Compare against Stage 1 baselines and conventional reconstruction.

**Exit criterion:** Stable event-level matching and uncertainty evaluation on held-out channel and overlap conditions, with documented failure modes and no MC association identifiers in model inputs.

## Stage 3 — Prototype-informed detector response

**Goal:** Replace idealized deposits with a measured or validated response model, including optical transport, sensors, electronics, thresholds, timing, and calibration variation. Use prototype data and weak supervision where labels are incomplete.

**Data:** Prototype measurements and metadata, calibration/control samples, and simulation passed through a response or digitization model. Record differences between prototype and final geometry/readout.

**Outputs:** Validated response model, simulation-to-data comparisons, updated reconstruction and uncertainty estimates, and a documented calibration strategy. Weakly supervised constraints must be anchored by external calibration or control data for absolute scales.

**Exit criterion:** Relevant observables and control measurements agree within stated uncertainties across operating conditions; the remaining simulation-to-data differences and their effect on reconstruction are quantified.

## Stage 4 — Experimental calibration and physics use

**Goal:** Calibrate the installed detector, validate reconstruction on experimental control data, and provide analysis-ready event quantities and uncertainties.

**Data:** Channel-level calibration constants, detector conditions, independent control samples, and the final readout format.

**Outputs:** Versioned reconstruction and calibration artifacts, systematic studies, validated selection efficiencies, and inputs suitable for multidimensional physics likelihood analyses.

**Exit criterion:** Independent calibration and control samples validate the claimed performance and uncertainty model over the analysis acceptance; changes are traceable to conditions and software versions.

The actionable work for the current stage is in [TASKS.md](TASKS.md); tool boundaries are in [DEPENDENCIES.md](DEPENDENCIES.md).
