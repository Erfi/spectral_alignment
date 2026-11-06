# Discrete appendix

| Output | Data file | Script |
|---|---|---|
| Realization, action-repeat, and calibrated spectral tables | `assets/bound_tightness/soft_policy_tau0p3_alpha0p5_bounds/soft_policy_tau0p3_results.csv` | `soft_policy_tau03_sweep.py` |
| Discrete layouts figure | — | `plot_layouts_and_matrices.py` |

The sweep imports Q and SR routines from `soft_policy_subspace_experiment.py` and grid and fitting routines from `matrix_fb_stagewise_experiment.py`. The latter's hard-policy results are not reported.

Run from the repository root: `uv run --with numpy --with scipy scripts/error_bound_experiments/soft_policy_tau03_sweep.py`.
