# Examples

Run any numbered example from the repo root:

```bash
python examples/01_line.py
```

Notes:
- `matplotlib` is used for plots (CI uses `MPLBACKEND=Agg`).
- Some examples require `ultranest` (they will fail/exit if it is not installed).
- CI/test runner only executes numbered scripts (`[0-9][0-9]_*.py`). Unnumbered scripts in this folder are helpers.

## Index

- `examples/01_line.py` — basic line fit + plotting + band.
- `examples/02_batch_common_x.py` — batch fits on a shared x-grid (subplot grid).
- `examples/03_batch_ragged_x.py` — ragged batch fits (each dataset has its own x).
- `examples/04_backend_swap.py` — swapping fit backends.
- `examples/05_derived_params.py` — derived parameters (`.derive`) on a Gaussian fit.
- `examples/06_fit_on_off.py` — seed-only mode (`optimise=False`) vs fitting.
- `examples/07_results_indexing.py` — `Results`/`ParamsView` indexing patterns (batched).
- `examples/08_hierarchical_indexing.py` — “fits of fits” (fit per-system, then fit trends).
- `examples/09_seed_demo.py` — guessers, `Model.seed`, `seed_override`, and seed vs fit curves.
- `examples/10_uncertainty_correlations.py` — correlated vs independent uncertainty propagation.
- `examples/11_binomial.py` — binomial likelihood fit (`scipy.minimize`) with Jeffreys intervals.
- `examples/12_gaussian_2d.py` — 2D Gaussian fit (flattened vs grid inputs).
- `examples/13_ultranest.py` — `ultranest` backend on a sinusoid (posterior samples + band).
- `examples/14_plot_helpers.py` — high-level `Run.plot` defaults using `FitData` metadata.
- `examples/15_rabi_oscillation.py` — Rabi model fit (minimize vs ultranest).
- `examples/16_rabi_band_compare.py` — uncertainty band comparison (minimize vs ultranest).
- `examples/17_ramsey_multimodal.py` — multimodal Ramsey posterior + posterior sample curves + next-point suggestion.
- `examples/18_sequential_design.py` — sequential design loop for a Ramsey-style sinusoid.
- `examples/19_sequential_gaussian.py` — sequential design loop for a Gaussian spectral feature.

## Helper scripts

- `examples/ultranest_sanity.py` — standalone UltraNest sanity check (writes logs under `ultranest_sanity_run/`).

