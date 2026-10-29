# Current-stage development tasks: Stage 1

> 中文摘要：当前任务是建立可复现的单粒子理想 SimHit 基线。先核对 ROOT 模式与入射真值定义，再制定特征白名单、生成覆盖不同粒子和相空间的数据、按事例划分数据集，最后比较传统方法与简单神经网络，并同时报告分辨率和效率。

Stage 1 is defined in the [roadmap](ROADMAP.md). This checklist describes future implementation work; none of the items below is delivered by this documentation-only change.

## 1. Verify data and label semantics

- [ ] Inspect representative `SimMACEPhaseI` ROOT files and record the actual branches, units, types, event identifiers, and empty-event behavior of `G4Run*/TTCSimHit`, `SciFiSimHit`, `ECALSimHit`, and `SimPrimaryVertex`.
- [ ] Check generator and analysis settings that save `SimPrimaryVertex` and apply `/MACE/Analysis/CoincidenceWithECAL`; produce unfiltered samples for generated-event efficiency. Verify that no-hit events remain countable even if hit trees have no rows for them.
- [ ] Define the targets precisely: particle PDG ID, initial kinetic energy `Ek0`, direction from `p0`, and a named position target. `x0` is a primary vertex, not automatically a detector-entry or impact point; add a suitable truth source before claiming entry/impact-position resolution.
- [ ] Check whether the deprecated SimHit declarations are still the authoritative schema for generated ROOT files, and record the applicable MACESW commit and geometry version.

**Acceptance:** A field/units table and a label definition can be checked against a real ROOT sample; each target has a documented source and physical meaning.

## 2. Specify model inputs and construct datasets

- [ ] Publish an explicit per-detector feature whitelist before training. Candidate idealized inputs are detector/channel IDs and `t`/`Edep` from the hit's raw-observable part, plus geometry derived from channel IDs. Validate these fields in real files; do not use them merely because they appear in a SimHit record.
- [ ] Keep `EvtID` for grouping and splitting only. Exclude `TrkID`, `MotherID`, `PDGID`, `x`, `Ek`, `p`, `t0`, `x0`, `Ek0`, `p0`, `CreatProc`, and any other MC truth or generator-derived field from model inputs. Treat optical-photon counts and simulated `ADC` as outside the initial `t`/`Edep` baseline until their response semantics are validated.
- [ ] Generate γ, e⁺, and e⁻ separately at several monoenergetic points and over sampled positions, polar angles, and azimuths; record generator distributions, geometry, random seeds, and event-selection settings. Include events with zero hits and detector-specific misses.
- [ ] Split complete generated events into train/validation/test before transformations. Keep disjoint held-out energy and angle regions as a generalization check; ensure the same event, reused seed, or derived example cannot cross splits.

**Acceptance:** A reproducible manifest identifies input files, generator settings, feature whitelist, target definitions, splits, and event counts before and after selection. A truth-leakage audit verifies every model input column.

## 3. Establish baselines and evaluation

- [ ] Implement conventional detector-wise estimators where meaningful, including an ECAL energy sum and energy-weighted position baseline; document their assumptions and unavailable outputs.
- [ ] Train a simple set/vector or fixed-size neural baseline on the same approved observables and splits. Start without a graph model; compare architectures only if the simpler baseline exposes a concrete limitation.
- [ ] Evaluate energy bias and resolution, position residuals, angular error, PID confusion/calibration, and uncertainty coverage where predicted. Report each metric versus particle species, energy, position, direction, and azimuth.
- [ ] Report hit/detection and successful-reconstruction efficiencies relative to all generated events, alongside conditional resolution for reconstructed events. Show no-hit and out-of-acceptance populations separately; inspect failures rather than hiding them with event cuts.

**Acceptance:** Reproducible predictions and held-out plots/tables compare conventional and neural baselines with the same event selection. Reports clearly label the results as idealized SimHit performance, not optical/electronic or final experimental performance.

Implementation dependencies and build constraints are in [DEPENDENCIES.md](DEPENDENCIES.md); the [README](README.md) links the project documents.
