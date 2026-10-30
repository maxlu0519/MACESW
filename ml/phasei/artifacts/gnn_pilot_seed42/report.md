# Phase-I GNN held-out evaluation

Only the selected checkpoint is evaluated here; these values describe the SimHit simulation study.

SciFi cross-layer edges are geometry-based candidate adjacencies derived from fiber local IDs, rotation, pitch, and radius over at most four layers. They approximate neighborhood structure and do not represent exact helix intersections at a reconstructed z; this may affect direction metrics.

Test events: 20010

## Aggregate and grouped metrics

| Group | N | PID accuracy | Energy bias (MeV) | Energy abs q68 (MeV) | Primary vertex x0 error q68 (mm) | Direction q68 (deg) | Total NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 20010 | 0.9587 | 0.03923 | 5.32 | 28.11 | 12.37 | 2.7931 |
| particle:e+ | 10006 | 0.9487 | -0.007482 | 4.594 | 24.2 | 13.44 | 2.2787 |
| particle:gamma | 10004 | 0.9687 | 0.08596 | 6.187 | 30.3 | 9.389 | 3.3076 |
| energy_mev:0-5 | 1696 | 0.8921 | 5.612 | 3.446 | 30.05 | 45.12 | 4.1617 |
| energy_mev:5-10 | 1674 | 0.9558 | 5.085 | 3.376 | 29.32 | 24.37 | 3.4345 |
| energy_mev:10-20 | 3310 | 0.9583 | 4.26 | 4.805 | 28.37 | 16.01 | 2.9639 |
| energy_mev:20-40 | 6733 | 0.9687 | 1.323 | 5.111 | 27.59 | 10.67 | 2.6015 |
| energy_mev:40-60 | 6597 | 0.9667 | -6.102 | 7.803 | 27.33 | 8.225 | 2.3883 |
| hit_combination:none | 2716 | 0.8778 | -0.08611 | 20.36 | 30.49 | 110.4 | 7.7681 |
| hit_combination:TTC | 41 | 0.7805 | 0.9565 | 18.6 | 32.44 | 8.585 | 4.6955 |
| hit_combination:SciFi | 221 | 0.8778 | 0.5431 | 13.32 | 28.75 | 43.18 | 4.9179 |
| hit_combination:TTC+SciFi | 587 | 0.9796 | 0.543 | 12.98 | 24.57 | 9.273 | 2.7822 |
| hit_combination:ECAL | 6641 | 0.9804 | -0.003791 | 4.274 | 30.33 | 6.091 | 1.8117 |
| hit_combination:TTC+ECAL | 369 | 0.9214 | 0.4614 | 4.357 | 29.8 | 9.327 | 2.8455 |
| hit_combination:SciFi+ECAL | 1777 | 0.9426 | -0.1433 | 3.078 | 28.37 | 24.77 | 3.6911 |
| hit_combination:TTC+SciFi+ECAL | 7658 | 0.9758 | 0.08493 | 4.31 | 22.74 | 10.84 | 1.5982 |

## Same-data old 100k baseline

Baseline checkpoint: [`old_100k_best.pt`](old_100k_best.pt) (best epoch 103).
Both checkpoints use dataset manifest `2696e754c26482a7e309b3a66a0afffe52196802f67dbe843aa3429b7db2b573` and the same held-out event split.
Cross-model NLL is omitted because the probabilistic head parameterizations and likelihood scales differ. `test_predictions.npz` stores each event's particle and particle-local event index; `data_manifest_sha256` binds those indices to the source dataset.

| Metric | GNN | Old baseline |
|---|---:|---:|
| PID accuracy | 0.95872 | 0.98061 |
| Energy bias (MeV) | 0.039232 | -0.2745 |
| Energy abs q68 (MeV) | 5.3198 | 4.525 |
| Primary-vertex x0 3D error q68 (mm) | 28.106 | 26.759 |
| Direction q68 (deg) | 12.366 | 13.645 |

### PID confusion matrices

Rows are true gamma and e+, in that order; columns are predicted gamma and e+.

| Model | gamma→gamma | gamma→e+ | e+→gamma | e+→e+ |
|---|---:|---:|---:|---:|
| GNN | 9691 | 313 | 513 | 9493 |
| Old 100k baseline | 9972 | 32 | 356 | 9650 |
