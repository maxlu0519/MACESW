# Phase I geometry-aware GNN pilot (seed 42)

This artifact records a simulation-only study with 100,000 generated gamma and
100,000 generated positron events. The existing split contains 159,996 training,
19,994 validation, and 20,010 held-out test events. Inputs are channel IDs,
summed SimHit deposited energy, earliest channel time relative to the earliest
event hit, and exported detector geometry. SimHit truth fields are excluded from
the input. The target position is the primary vertex inside the target.

## Training

- Model: 3,111,629 parameters; detector-specific gated graph blocks and a
  two-layer, eight-head fusion Transformer. ECAL nodes include crystal center,
  unit normal, and native neighbor count.
- Seed 42, FP32, AdamW at `3e-4`, weight decay `0.01`, batch 512 with four
  gradient-accumulation steps (effective batch 2,048), two data-loader workers.
- Early stopping patience 10, maximum 80 epochs. Training ended at epoch 54;
  the selected checkpoint is epoch 44 with validation total loss 2.7595.
- Elapsed time 1,028.4 seconds; approximately 9,451 training plus validation
  events per second averaged over the run. Peak CUDA reserved memory was
  512 MiB on the 6 GiB GTX 1660 SUPER.
- The held-out test split was first read after checkpoint selection.

The initial 256×8 run and a subsequent 512×4 single-worker run were stopped
after completed checkpoints and kept in ignored `build/` directories. A
2,048-event benchmark measured 0.356 s/update for 256×8, 0.188 s/update for
512×4, 0.213 s/update for 1024×2, and 0.247 s/update for 2048×1. An
8,192-event full-step benchmark found two data-loader workers 29.8% faster
than the single-worker 512×4 setting. The selected run uses 512×4 with two
workers throughout; no checkpoints from the earlier settings were mixed in.

## Held-out comparison

The same 20,010 events were evaluated with the GNN and the existing model
trained on this same 100k-per-species dataset. The old model used its own
training schedule and hyperparameters, so this is a same-data comparison,
not an architecture-only controlled ablation. The old checkpoint is archived
as `old_100k_best.pt`. The 10M-event epoch-57 model is not used as a comparator.

| Metric | GNN | Old 100k model |
|---|---:|---:|
| PID accuracy | 95.87% | 98.06% |
| Absolute energy error, 68% quantile | 5.32 MeV | 4.53 MeV |
| Primary-vertex 3D error, 68% quantile | 28.11 mm | 26.76 mm |
| 3D direction angle, 68% quantile | 12.37° | 13.65° |

The GNN improves the aggregate direction angle but trails the old model on
PID, energy, and primary-vertex position. The full confusion matrices,
particle/energy/hit-combination counts and metrics are in [report.md](report.md)
and [metrics.json](metrics.json). Per-event predictions with particle and
particle-local event index are in `test_predictions.npz`.

SciFi cross-layer edges represent geometry-based candidate neighborhoods,
not exact helix intersections at a reconstructed longitudinal position. TTC
tile adjacency is approximated by the 12 nearest tile centers because the
Phase I description has no explicit tile-neighbor list. The preliminary
GNN run without ECAL crystal normals remains in ignored `build/`; it reached
12.01° direction q68 but was not used as the final artifact because its
geometry export omitted a required field. Adding normals did not improve
every metric in this single-seed pilot.
All performance numbers describe idealized SimHit inputs; they do not include
optical or electronic readout effects.

`manifest.json` records file hashes, dataset and geometry hashes, and the
training code hash. The committed Python graph/model code matches the
training-time hash.
