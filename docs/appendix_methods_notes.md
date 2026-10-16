# Experiment Details appendix: provenance and author checks

The manuscript fragment is [appendix_methods.tex](appendix_methods.tex).
Include it with `\input{docs/appendix_methods}` after the manuscript’s `\appendix`,
and load `amsmath` and `amssymb`. The fragment declares `\section{Experiment Details}`
and five subsections corresponding to F.1–F.5 in the earlier figure specification.
It uses automatic numbering; it does not force the section letter to F.
Training details and the complete solver/complexity analysis are retained.
It does not describe the optional fine-tuned CelebA backbone as the main experiment.

The text was checked against both executable code and the retained run
configurations. It distinguishes configured choices, implemented selection rules,
and facts observed in saved artifacts. No training or plotting code was changed
for this documentation task.


## Included assets and manuscript integration

The appendix now directly includes the **17 files** in
`paper_plotting/paper_figures/appendix/`: 15 vector PDFs and two LaTeX tables.
That folder contains no previews, intermediate data, README, or manifests.
Every file is referenced in `appendix_methods.tex`, and every referenced asset
exists. Load `amsmath`, `amssymb`, `graphicx`, and `booktabs`. Override
`\appendixfigdir` before including the fragment if the manuscript is compiled
outside this repository root.

The retained slots are F.2-a, the covariance part of F.2-b, F.4-a/b/d,
F.5-a/b/c/d/e/f, and the rank heatmaps, plus the requested per-seed crossover
audit table. The per-task and rank groups each use separate Gaussian/code PDFs.
Code histograms show seed 42 for the selected specialist/generalist at every
budget; they are regenerated rather than copied from the older all-bank plot.

The role-swap manifest has 42/42 completed jobs with factors `posY,posX`.
The fine-tuned figure was recovered from retained per-image losses for the
selected pair and task-aware baseline, with 2,000 paired bootstrap resamples
and six fitted seeds. Its backbone was trained on all 40 CelebA attributes,
including evaluation labels. It must be described as task-informed pretraining.

No complete null-control results, shrinkage-sensitivity sweep, or constructed
worst-case/SNR-cap experiment is available. Their figure placeholders have been
replaced with explicit availability statements, not fabricated graphics.
The covariance spectrum is a real training-data diagnostic and is not a
substitute for a shrinkage sweep.

Run `python paper_plotting/prepare_appendix.py` from the repository root to
rebuild the assets and `docs/appendix_values.json`. The latter records extracted
configurations, numerical statistics, crossover brackets, checkpoint identity,
and SHA-256 hashes of every bundled asset. It is deliberately outside the
asset folder. Caption text is filled in the manuscript; if run inputs change,
review captions against the regenerated JSON before publication.

## What can be said about hyperparameter selection

| Choice | What the repository supports |
|---|---|
| CelebA logistic regularization | Per-attribute selection over `C = 0, 0.1, 1, 10` using an internal 80/20 training split; refit using all training representations. |
| CelebA code checkpoint | Minimum hard-code pretext/task validation loss over all 30 epochs. |
| CelebA MLP probe checkpoint | Validation NLL, at most 100 epochs, patience 10. |
| CelebA pretext selection | Separate 1,000-image selection subset; seed-averaged scores, with either per-budget or globally fixed selection. |
| dSprites representation checkpoint | Minimum validation pretext distortion plus beta times KL, over 100 epochs. |
| dSprites probe checkpoint | Minimum independent probe-validation NLL over 80 epochs. |
| dSprites learning rate | Saved grid contains only 0.001. The code supports validation selection if more rates are supplied; this is not evidence that such a search was run. |
| Widths, learning rates, ridge, variance floors, epochs | Recorded fixed settings. Their historical selection rationale is not established by the saved configuration alone. |
| dSprites beta grid | Fourteen values spanning a broad range of achieved rates. The README explains the desire to cover both sides of a possible crossing; no separate search history establishes how the final grid was chosen. |

If you performed pilot sweeps outside these artifacts, add their candidate sets,
selection data, and criteria before describing settings as “tuned,” “optimal,”
or “chosen a priori.” The fragment deliberately makes none of those claims.

## Source map

Paths below are relative to the repository root.

| Appendix topic | Primary implementation |
|---|---|
| CelebA image partitions, cached views, frozen backbone | `celeba_experiment/celeba_experiment/data.py`: `extract`, `load_backbone`, `view_transform`, `RegionCrop` |
| CelebA validation split and downstream pretext selection | `celeba_experiment/celeba_experiment/rate_selection.py`: `split_validation`, `evaluate_run`, `select_policies` |
| Whitening, five-fold ridge fits, relevant-span solve | `celeba_experiment/celeba_experiment/rate_models.py`: `fit_pca`, `fit_gaussian_tasks`, `task_channel` |
| Feature proxies and pretext component weights | `celeba_experiment/celeba_experiment/fitting.py`: `fit_cross_view`; `pretexts.py`: `build_candidate`, `proxy_channel` |
| Hard-bit encoder and optimizer | `celeba_experiment/celeba_experiment/finite_codes.py`: `BinaryEncoder`, `CodePredictor`, `fit_codes` |
| Logistic probe tuning | `celeba_experiment/celeba_experiment/probes.py`: `fit_probes`, `one_fit`; `analysis.py`: `fit_channel_probes` |
| MLP diagnostic and bootstrap | `celeba_experiment/celeba_experiment/robustness.py`: `fit_mlp_losses`, `fit_bootstrap`, `report`; `robust_statistics.py`: `paired_bootstrap` |
| dSprites grouping, fold sizes, normalization | `dsprites_experiment/data.py`: `grouped_split`, `Sprites`, `make_data` |
| dSprites architecture and loss | `dsprites_experiment/model.py`: `VIB`, `Predictor`, `objective`, `excess_nll` |
| dSprites optimization and probe evaluation | `dsprites_experiment/train.py`: `optimize_model`, `fit_probe`, `evaluate` |
| dSprites grid and learning-rate selection | `dsprites_experiment/launch.py`; `summarize.py`: `summarize` |
| Gaussian solver, derivatives, projection, certificate, water filling | `celeba_experiment/gaussian_rate/nd_solvers.py`: `GaussianFamily`, `s_value_grad`, `project`, `certificate`, `solve`, `single_task` |

The saved-run evidence is:

- `celeba_experiment/runs/robust_rate_selection/config.json`,
  `split_indices.npz`, `feature_manifest.json`, `certificates.json`, and
  `pretext_certificates.json`.
- `celeba_experiment/runs/robust_rate_selection/robustness/moment_bootstrap/status.json`.
- `dsprites_experiment/runs/dsprites_1/manifest.json` and each job's
  `data_provenance.json`.
- The current specialist choice and measured-versus-proxy band selection in
  `paper_plotting/paper_experiment_plots.ipynb` and `plot_helpers.py`.

## Distinctions to preserve when editing the manuscript

1. CelebA has one fixed data split and five representation seeds. dSprites has
   three split/training seeds, with paired partitions within each seed.
2. CelebA downstream probes reuse representation-training images. Only dSprites
   has a completely independent probe-training image fold.
3. The historical CelebA configuration names `expression_attributes` as the
   robustness specialist. The current figures explicitly select
   `joint_attributes` (five attributes) and recompute pair-dependent diagnostics.
4. The Gaussian solver optimizes a fitted surrogate. Its certificate is not a
   guarantee about neural code optimality, true image mutual information, or
   statistical generalization. A theorem-level justification of the geodesic
   residual bound should be cited to the corresponding result in your paper;
   the code is an implementation of a certificate, not a proof by itself.
5. Main measured Gaussian bands use test-image resampling. The separate moment
   bootstrap has 199/200 successful fits and cannot be reported as a completed
   200-resample interval. The bootstrap refits PCA across resamples, while the
   ordinary five-fold ridge cross-fit conditions on a shared training PCA.
6. Code budget, nominal Gaussian rate, and variational KL rate are different
   quantities. The dSprites MSE index is not a literal allocation of bits.
7. Full role-swap/null results are not established by pipeline smoke tests.
   Verify the current run manifests and provenance before filling their result
   placeholders; the existence of a figure PDF alone does not establish completion.
8. The solver's API has defaults of `tol=1e-6, maxiter=5000`, but the main
   experiment explicitly uses `1e-5` and `3000`. Larger bootstrap recovery limits
   are a separate retry procedure, not the base-run setting.
9. The solver supports warm starts, but the main wrappers do not use them.
10. The code's `temperature=0.2` pertains to an optional InfoNCE loss, absent
    from the eight active pretexts. It is not a hard-bit sigmoid temperature.

## Complexity summary

Let `d` be the original whitened dimension, `q` the retained relevant dimension,
`T` the number of task matrices, `N` the number of outer iterations, and `L <= 60`
the maximum number of backtracking trials used per iteration.

| Operation | Arithmetic cost |
|---|---|
| Rank-one-family solve, excluding setup | `O(N (1+L) (q^3 + T q^2))` |
| Full-rank-family solve, excluding setup | `O(N (1+L) T q^3)` |
| Weighted matrix and span eigendecomposition | `O(T d^2 + d^3)` |
| Current unoptimized three-operand basis contraction | `O(T d^2 q^2)` |
| Reduced task factorization | `O(T q^3)` |
| Single-matrix water filling | `O(d^3 + J d)` for `J` scalar root evaluations |

Resident storage inside the reduced solver is `O(T q^2)`, since dense task
matrices are retained even for rank-one tasks. Including wrapper inputs and
outputs gives `O(T d^2 + T q^2 + d q)` for one solve. This excludes cached image
features, neural models, losses, and stored grids of covariance solutions.

There is no justified `O(log(1/epsilon))`, `O(1/epsilon)`, or similar total
convergence bound in the implementation. Keep `N` explicit. The saved
positive-rate task-aware solves each used 11 iterations; this does not imply
that all objectives, dimensions, or bootstrap samples converge in 11 steps.

## Documentation validation

The rate grids, seeds, and CelebA partition counts were checked against saved
artifacts. The dSprites fold sizes were checked for consistency across all 168
jobs (56 per seed). Saved matrices have relevant dimensions 4 for the task-aware
objective and 5 for the plotted five-attribute specialist. The LaTeX fragment's
brace and environment structure was checked. No TeX compiler is installed in
this environment, so a PDF compilation was not performed.


The complete CelebA bank now appears as measured-NLL curves for Gaussian and
binary-code representations, including all eight pretexts and both reference
objectives. The decoder-capacity figure shows both mechanisms side by side.
Only proxy-versus-measured NLL receives a regression-fit score: OLS with an
intercept gives in-sample R² = 0.981512, slope 0.701817, intercept -0.004594.
Accounting and code-entropy plots retain identity lines without R² annotations.
