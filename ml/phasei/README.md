# Phase I event reconstruction with machine learning

> 中文摘要：Phase I 的 SimHit 转换、首版概率神经网络、训练与评估接口已实现。转换器可直接处理未合并的 MPI ROOT 分片，并保留零命中事例。本版训练结果仅用于软件验收，不代表探测器性能。

## Purpose and current status

This project will study reconstruction of Phase I events using the TTC, SciFi tracker, and ECAL together. It belongs to MACESW and should reuse its detector definitions, simulation outputs, and Mustard data conventions. The first study uses simulated energy deposits and hit times to measure an **idealized, simulation-conditioned performance bound**. It does not predict the final performance of optical sensors, electronics, or a calibrated detector.

**Current status: V1 software acceptance and a throughput-focused full-data training run completed.** The converter processed 100,000 positrons and 100,000 photons directly from 56 rank shards per particle, without `hadd`. It writes event-indexed NumPy arrays, train-only normalization, and deterministic 80/10/10 splits. A vectorized loader and batch-4096 validation run are available below. Neither the initial smoke test nor validation loss alone establishes physics performance.

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
`phasei_ml.convert`: validation + event grouping + whitelisted features + labels
       |
       v
reproducible train/validation/test datasets
       |
       v
conventional baselines + simple neural baseline -> predictions and metrics
```

The event key is for grouping and splitting, not a model feature. A SimHit record contains both detector-like quantities and MC truth. Only explicitly approved detector-observable fields may reach a model; the primary-vertex record is reserved for labels and evaluation. In particular, a generator vertex is not automatically the point where a particle enters a detector. Target positions and energy conventions must be stated before training.

The first implemented model embeds channel IDs and uses hit time and deposited energy from TTC, SciFi, and ECAL. A detector-wise set encoder feeds PID, kinetic-energy, vertex-position, and direction probability heads. It includes a learned branch for events with no hits. No geometry coordinates, MC-truth input fields, waveforms, optical response, or electronics response are model inputs.

## Source and simulation commands

Run from the repository root with a Python environment containing PyROOT and NumPy. The generator samples the configured Phase I cylinder uniformly in volume, an isotropic direction, and kinetic energy from 1 keV to 60 MeV. It uses up to four sampling processes by default, followed by one ROOT writer. Generate separate files for `gamma` and `e+`:

```sh
PYTHONPATH=ml/phasei/python python -m phasei_ml.source \
  --particle gamma --events 10000 --seed 1 --output gamma.root
PYTHONPATH=ml/phasei/python python -m phasei_ml.source \
  --particle e+ --events 10000 --seed 2 --output positron.root
```

Each source has an adjacent `.root.json` manifest. The simulation launcher reads that manifest and, by default, uses up to four MPI ranks. Each particle needs its own output directory:

The source TTree has one entry and one primary particle per generated event. Its required branches follow the current Mustard reader: scalar `w` (`float`), `t` (`double`, ns), `x/y/z` (`float`, mm), and `std::vector<int> pdgID` plus `std::vector<float> E/px/py/pz` (MeV). `E` is **total** energy in the source; `SimPrimaryVertex.Ek0` is kinetic energy in the simulation output. The generator sets `w=1` and `t=0`.

For photons, the source places the stored `float` energy one representable step below the magnitude of the stored momentum. This prevents Geant4 from inferring a spurious nonzero mass from independent float rounding; the adjustment is below 10 eV at 60 MeV.

```sh
PYTHONPATH=ml/phasei/python python -m phasei_ml.simulate \
  gamma.root.json gamma_sim
PYTHONPATH=ml/phasei/python python -m phasei_ml.simulate \
  positron.root.json positron_sim
```

The launcher writes a simulation manifest listing the actual rank files. Import `iter_events` from `phasei_ml.dataset` and pass the source and simulation manifest paths to read those files directly; no `hadd` step is required. The reader groups events across ranks by run and `EvtID`, retains zero-hit events, and checks the source against saved `SimPrimaryVertex` truth. It yields in rank-file/primary-tree order, not numeric `EvtID` order. In the current Mustard reader, event `i` consumes source entry `(i + 1) % N` for a single run of `N` events; the reader checks this mapping against every saved primary vertex.

Convert source/simulation pairs into a dataset directory:

```sh
export ROOTSYS=/opt/root/6.38.00
export PYTHONPATH="$PWD/ml/phasei/python:$ROOTSYS/lib${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOTSYS/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv/bin/python -m phasei_ml.convert \
  --pair build/phasei_em3_100k_20261029/positron.root.json build/phasei_em3_100k_20261029/positron_sim/simulation_manifest.json \
  --pair build/phasei_em3_100k_20261029/gamma.root.json build/phasei_em3_100k_20261029/gamma_sim/simulation_manifest.json \
  --output build/phasei_ml_dataset_v1
```

The converter validates provenance, ROOT branches, event coverage, duplicate/missing event IDs, and the source-row offset. It sorts by `EvtID`; each detector stores channel IDs, times, deposited energy, and `N+1` offsets. Equal adjacent offsets encode events with no hits. The input whitelist is channel ID, `t`, and `Edep`, plus counts and sums derived from those fields. `EvtID`, `PDGID`, `t0`, `x0`, `Ek0`, and `p0` are used only for joins, splits, targets, and evaluation.

Train and evaluate the initial routed-expert model:

```sh
.venv/bin/python -m phasei_ml.train --data build/phasei_ml_dataset_v1 \
  --output build/phasei_ml_dataset_v1/model.pt --epochs 20 --batch-size 256
.venv/bin/python -m phasei_ml.evaluate --checkpoint build/phasei_ml_dataset_v1/model.pt \
  --data build/phasei_ml_dataset_v1 --batch-size 256
```

Training defaults are recorded in [default.yaml](config/default.yaml). Use `--no-experts` for the same-input ablation. A quick smoke run can use `--train-limit-per-class 2048 --eval-limit-per-class 256 --epochs 2`. Checkpoints include model and normalization configuration, seed, training settings, and the dataset-manifest hash.

For throughput-focused training on the local GTX 1660 SUPER, use [fast_4096.yaml](config/fast_4096.yaml) with the vectorized batch loader. It gathers all selected hit rows through NumPy offsets, keeps zero-hit events and dynamic padding, and accepts the same feature whitelist as the original loader. The original per-event loader remains available with `--input-pipeline legacy` for comparisons. Training reads only the train and validation splits; call `phasei_ml.evaluate` separately after choosing a checkpoint.

```sh
.venv/bin/python -m phasei_ml.train --data build/phasei_ml_dataset_v1 \
  --config ml/phasei/config/fast_4096.yaml \
  --output build/phasei_ml_fast_new/best.pt \
  --last-output build/phasei_ml_fast_new/last.pt
```

The measured batch-4096 run on the existing 200,000-event dataset stopped after 118 epochs, with best validation objective 2.2390 at epoch 103, 31,206 training events/s including validation and checkpoint overhead, and 2.28 GiB peak CUDA reserved memory. This is a validation result; no held-out test result was used to select that checkpoint. The run artifact is [training_result.json](../../build/phasei_ml_training_batch4096_v1/training_result.json).

To start one million gamma and one million positron events with `nohup`, run from the repository root:

```sh
./ml/phasei/run_1m_nohup.sh
./ml/phasei/run_1m_nohup.sh --status
```

The script writes under `build/phasei_training_1m_20261029` by default. It generates both sources concurrently using four workers each, then launches both simulations concurrently with four MPI ranks each. `run.log`, per-particle logs, `status.txt`, and `run.pid` remain in that directory. Set `PHASEI_OUTPUT_DIR`, `PHASEI_SOURCE_WORKERS`, or `PHASEI_MPI_RANKS` to override defaults before launching. Run in an environment with PyROOT, NumPy, and MPI; the script preserves the existing ROOT `PYTHONPATH`.

These Python tools do not change the default MACESW build. The optional C++ geometry exporter reuses MACE/Mustard detector descriptions and is built only by an explicit opt-in. Training and evaluation remain separate from the C++ build. The V1 single-particle baseline remains available; the graph study below is a separate model path.

## Opt-in geometry-aware graph model

The graph model reads the same 100k-per-species converted dataset and its existing 80/10/10 event split. It adds no SimHit features: one node represents each observed detector channel in an event, channel energy deposits are summed, and the earliest channel hit time is measured relative to the event's earliest hit. Events with no hits remain in the dataset. Edges are built within each event only; no hit truncation is applied. Truth arrays supply training targets only.

Build the optional C++ geometry exporter with the explicit CMake switch. It imports the simulation geometry YAML alongside Phase I detector defaults, then exports all SciFi fibers, TTC tiles, ECAL modules, and the ECAL neighbor table. The Python wrapper records both the YAML and canonical export hashes; GNN data loading checks the YAML hash against the converted dataset provenance.

```sh
cmake -S . -B build/phasei_gnn_export -DMACESW_BUILD_PHASEI_GNN_GEOMETRY_EXPORT=ON
cmake --build build/phasei_gnn_export --target phasei_gnn_geometry_export -j2
PYTHONPATH=ml/phasei/python .venv/bin/python -m phasei_ml.export_geometry \
  --exporter build/phasei_gnn_export/phasei_gnn_geometry_export \
  --geometry-yaml src/app/simulation/MACE/PhaseI/SimMACEPhaseI/scripts/SimMACEPhaseI_geom.yaml \
  --output build/phasei_gnn_geometry/geometry.json
```

The GNN has detector-specific gated edge networks (SciFi 16-d channel embedding/160 hidden/4 blocks, TTC 8/96/2, ECAL 8/160/3), two attention pools per detector, and a seven-token two-layer Transformer fusion block. It predicts two-class PID, a five-component Gaussian energy mixture, a four-component full-covariance position mixture, and a four-component von Mises–Fisher direction mixture. A learned event prior handles the all-zero-hit case.

SciFi cross-layer edges are candidate neighbors inferred from fiber geometry,
not exact helical intersections at a reconstructed longitudinal position. TTC
tile adjacency is approximated by the 12 nearest tile centers because the
Phase I description does not provide a neighbor table; ECAL uses its native
crystal-neighbor list.

SciFi cross-layer edges are geometry-based candidate adjacencies derived from fiber local IDs, rotation, pitch, and radius over at most four layers. They approximate neighborhood structure and do not represent exact helix intersections at a reconstructed z; this approximation may affect direction metrics.

Training fits all feature and target normalizations on the training split only and does not load test hit values. For the current 100k-per-species dataset, use seed 42, FP32 AdamW at `3e-4` with weight decay `0.01`, effective batch 2048, at most 80 epochs, and patience 10:

```sh
PYTHONPATH=ml/phasei/python .venv/bin/python -m phasei_ml.gnn_train \
  --data build/phasei_ml_dataset_v1 \
  --geometry build/phasei_gnn_geometry/geometry.json \
  --output build/phasei_gnn_pilot_seed42_mb512_workers2_20261030/best.pt \
  --last-output build/phasei_gnn_pilot_seed42_mb512_workers2_20261030/last.pt \
  --seed 42 --epochs 80 --patience 10 --batch-size 512 --grad-accum 4 \
  --num-workers 2 --device cuda:0
```

A full forward/backward benchmark on the GTX 1660 SUPER compared 8,192 training-split events at effective batch 2,048. With `512×4`, two persistent graph-loader workers improved throughput from 4,249 to 5,520 events/s (29.8%) versus zero workers, including worker startup; both settings peaked at 482 MiB reserved. Full-epoch time improved from 32–33 seconds to 19 seconds. The full-data pilot uses a fresh run directory for `num_workers=2`; earlier `256×8` and zero-worker `512×4` runs are preserved as preliminary configurations.

| Microbatch × accumulation | Optimizer-step time | CPU collation time | Peak reserved GPU memory |
|---|---:|---:|---:|
| 256 × 8 | 0.356 s | 0.182 s | 258 MiB |
| 512 × 4 | 0.188 s | 0.177 s | 452 MiB |
| 1024 × 2 | 0.213 s | 0.176 s | 816 MiB |
| 2048 × 1 | 0.247 s | 0.179 s | 1,560 MiB |

The optimizer-step benchmark preserves effective batch size across rows; the 8,192-event worker comparison measures the complete forward/backward/optimizer path and includes worker startup amortized over four updates.

The held-out evaluator runs only after selecting a checkpoint. It writes per-particle, energy-bin, and detector-hit-combination metrics, confusion matrices, compressed test predictions, and a Markdown report. `--baseline-checkpoint` runs the existing Phase I model on the same held-out split for a fair comparison.

```sh
PYTHONPATH=ml/phasei/python .venv/bin/python -m phasei_ml.gnn_evaluate \
  --checkpoint build/phasei_gnn_pilot_seed42_mb512_workers2_20261030/best.pt \
  --data build/phasei_ml_dataset_v1 \
  --geometry build/phasei_gnn_geometry/geometry.json \
  --baseline-checkpoint build/phasei_ml_training_batch4096_v1/best.pt \
  --output-dir build/phasei_gnn_pilot_seed42_mb512_workers2_20261030/evaluation --device cuda:0
```

The geometry export is opt-in and the normal MACESW build is unchanged when `MACESW_BUILD_PHASEI_GNN_GEOMETRY_EXPORT` is left OFF.

The completed 100k-per-species pilot, selected checkpoint, geometry export,
same-data baseline checkpoint, and held-out report are archived in
[artifacts/gnn_pilot_seed42/EXPERIMENT.md](artifacts/gnn_pilot_seed42/EXPERIMENT.md).

## Documents

- [Roadmap](ROADMAP.md): research stages, deliverables, and progression criteria.
- [Dependencies](DEPENDENCIES.md): existing software and proposed training tools.
- [Current-stage tasks](TASKS.md): the Stage 1 implementation and acceptance checklist.

The repository-wide build and contribution guidance remains in the [root README](../../README.md) and [contribution guide](../../CONTRIBUTING.md).
