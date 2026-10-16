# Rate-dependent pretext selection on dSprites

The primary experiment now trains on **the official dSprites dataset**, not the two-blob renderer. dSprites is an established procedurally generated visual benchmark, not natural photographs. Images contain a sprite whose shape, scale, orientation, and position vary. The Gaussian renderer remains available to correctness tests.

## Setup and launch

Use Python 3.10+ with a CUDA-compatible PyTorch installation, then install `../requirements.txt`. Run commands from this directory.

```bash
python -m pip install -r ../requirements.txt
# Downloads the official ~18.4 MiB archive and builds a shared ~360 MiB image cache.
python prepare_dsprites.py --out data/dsprites
# Or prepare an archive you already downloaded:
python prepare_dsprites.py --npz /path/to/dsprites.npz --out data/dsprites

# Correctness checks, including checks on actual dSprites images:
DSPRITES_TEST_CACHE=data/dsprites python -m unittest discover -s tests -p 'test_*.py' -v

# End-to-end smoke run on actual dSprites, 8 jobs and 2 epochs per stage.
python launch.py --config configs/smoke.json --data-dir data/dsprites --cpu --out runs/smoke

# Main experiment: M=4, 4 pretexts × 14 betas × 3 seeds = 168 independent runs.
python launch.py --data-dir data/dsprites --gpus 0,1,2,3 --out runs/dsprites

# Secondary sweep (504 jobs).
python launch.py --data-dir data/dsprites --gpus 0,1,2,3 --Ms 4 6 8 --out runs/dsprites_M

# Swap the roles of horizontal and vertical position to check visual asymmetry.
python launch.py --data-dir data/dsprites --gpus 0,1,2,3 --factors posY posX --out runs/dsprites_swap

# Optional Gaussian renderer control; no dataset download required.

```

`--download-data` on the launcher prepares the cache **once before workers start**. Without it, a missing cache produces an error. `--dry-run` enumerates jobs without downloading data or requiring a prepared cache. GPU IDs can be UUIDs. Omitting `--gpus` honors `CUDA_VISIBLE_DEVICES` or detects all GPUs; explicit IDs refer to CUDA device identifiers rather than indices into an existing visibility list. Each worker sees one GPU as `cuda:0`. The launcher runs one job per GPU, replacing it when finished.

Rerun the same command to skip completed runs. Failed or interrupted jobs restart from scratch; per-epoch optimizer resumption is not implemented. `result.json` is written atomically after all stages finish. Per-run logs, checkpoints, history, configurations, and test arrays are retained. Configuration and source hashes prevent mixing revisions in one run directory. A launcher lock blocks overlapping schedulers; verify the recorded process has exited before removing a stale lock after a crash. Failures produce a nonzero exit status and a `failures.json` list.

The default uses fourteen betas because achieved rate is unknown before training, and estimating a crossover requires coverage on both sides. All pretexts receive identical architecture, schedule, beta grid, learning-rate grid, and seeds. Add learning rates through `lrs` in JSON; candidates are selected using their own pretext validation objective, never downstream test scores. Use a new output directory after changing configuration or source. Three seeds give coarse uncertainty; increase to five or more for final results.

## Dataset and task construction

The official dataset contains 737,280 binary 64×64 images over a full Cartesian product of shape (3), scale (6), orientation (40), horizontal position (32), and vertical position (32); color is fixed. The dataset is downloaded directly from Google's dSprites repository, verified against a pinned SHA256, and read without unpickling metadata. Images are streamed from the archive into a packed memory map. Worker processes share the OS file cache and unpack only requested images; no worker loads a separate 3 GiB image array.

By default, task 1 uses `posX` and task 2 uses `posY`. Their class indices 0,...,31 are standardized using the known uniform-factor mean 15.5 and standard deviation sqrt((32²−1)/12). These are two independent unit-variance **discrete** factors in the complete dataset. Shape, scale, and orientation vary without being task targets. They therefore introduce visual nuisance variation and, for reconstruction, additional information worth preserving. No images are resized, augmented, clipped, or normalized per image.

For standardized factors S1,S2, tasks are

```
Y1 = S1 + epsilon1,  Var(epsilon1) = M^-2
Y2 = S2 + epsilon2,  Var(epsilon2) = M^-4
evaluation weights = (1 - 1/M, 1/M).
```

All task noises are independent. Both tasks are evaluated on each training example and weighted exactly, which has the same expected objective as sampling task prevalence and lower gradient variance. The factor order can be swapped. `scale` is also supported as a scalar factor, but has only six levels and changes the scientific comparison; do not mix such runs with the default positional experiment. Shape is categorical and orientation is circular/partially ambiguous, so neither is silently treated as a scalar regression label.

### Splits and duplicate protection

There are 715,706 unique pixel images in the verified archive: some annotations render identical pixels. SHA256 hashes of exact pixels group these copies. Every group stays entirely within one fold, preventing the same visual observation from appearing in both training and test folds.

For each seed, randomize groups and assign five disjoint folds: representation training (at least 32,768 rows), representation validation (4,096), probe training (16,384), probe validation (4,096), and test (8,192). Because groups remain intact, actual sizes may slightly exceed requested sizes. No rows are shared across folds within a seed. Seeds use independent partitions and can reuse images across seeds. Samples and splits are paired across pretexts, betas, M, and factor-order controls. Finite folds need not have perfectly uniform factors.

`split_indices.npz` records exact source indices, and `data_provenance.json` records dataset checksum, factor order, split hash, and actual fold sizes. Standardization uses the public generation grid, not empirical test labels. This is a held-out-observation experiment; the task-aware mixture uses the same tasks during training and evaluation, not unseen downstream tasks.

## Model, losses, and measured rate

A fixed CNN maps an image to a diagonal Gaussian posterior q(c|U) with width 8. Sampling uses the reparameterization trick. All four pretexts instantiate the same CNN, channel, two-output Gaussian task decoder, and image decoder. Train the entire CNN separately for each pretext/beta/seed; no backbone is pretrained or frozen during representation learning. Bottleneck width stays fixed to avoid confounding channel budget with parameter count.

The objective is `D_pretext + beta * E KL_2(q(c|U) || N(0,I))`, in bits:

| Pretext | Distortion |
|---|---|
| specialist1 | Task-1 Gaussian excess NLL |
| specialist2 | Task-2 Gaussian excess NLL |
| mixture | Weighted sum with (1−1/M,1/M) |
| reconstruction | Mean binary cross-entropy per pixel, in bits |

For binary dSprites reconstruction, the image decoder outputs logits and the loss is mean `binary_cross_entropy_with_logits / ln(2)`. This is the image Bernoulli NLL divided by the fixed pixel count. Its scale is fixed across runs. The Gaussian-renderer control retains the original fixed-variance MSE reconstruction objective. Beta is not directly comparable across pretexts; compare achieved KL rates.

Task decoders predict a mean mu and variance `v = noise_variance + softplus(raw) + 1e-7`. Known independent task noise is integrated analytically:

```
excess NLL_i = [log(v_i / sigma_i²)
                + ((S_i - mu_i)² + sigma_i²) / v_i - 1] / (2 ln 2).
```

This is exactly expected noisy-label Gaussian NLL minus irreducible noise entropy. It remains valid with discrete factors and avoids additional noisy-label sampling variance. The Gaussian conditional decoder is a variational approximation, not a claim that the dSprites posterior is Gaussian.

Rate is measured as the test average analytic KL in bits. `E KL = I(U;C) + KL(q(C)||N(0,I))`, so it is an upper bound, not measured mutual information or a hard code length. Representation evaluation always uses stochastic channel samples; posterior means are not used as deterministic codes.

## Independent probe evaluation

Select the representation checkpoint on its validation `D+beta*KL`. Freeze it, then train a fresh nonlinear Gaussian probe for both tasks on independent probe data. Equal task weights during probe fitting prevent evaluator neglect of the rare task. Select its checkpoint on probe validation NLL. Report the original prevalence-weighted score on test data.

Use eight independent channel draws per test image and average **losses**, never predictions before scoring. Report achieved KL, both task excess NLLs, weighted distortion, per-factor MSE and R², and `-0.5 log2(MSE)` recoverability indices. Negative indices from poor probes are retained. `test_metrics.npz` includes per-row losses, KL, and identical-image group IDs. Rate and distortion standard errors cluster duplicate pixels; seed-bootstrap bands resample whole training seeds.

For dSprites, the recoverability index is only a monotone transform of MSE, **not allocated bits or a Gaussian rate-distortion lower bound**. Each positional factor has at most 5 bits of entropy, yet its recoverability index can diverge as MSE approaches zero. Do not sum these indices as a rate estimate. Nonlinear-probe approximation error and optimization failure can hide or mimic a reversal.

## Hypothesis and Gaussian reference

The primary dSprites hypothesis is qualitative: the mixture-trained representation initially predicts the common positional factor more accurately, then increasingly favors the rare, higher-precision factor as the bottleneck relaxes. Test this via `delta = -0.5 log2(MSE2) + 0.5 log2(MSE1)`. At population unit variance, an ordering reversal in this statistic agrees with a latent-factor R² ordering reversal. The two test empirical variances need not match exactly.

The exact Gaussian control has

```
R1 = log2 M
R2 = 4 log2 M
R_cross = log2(M^4 * (M-2) / (M²-M+1)).
```

For M=4,6,8 the Gaussian crossover is 5.300, 7.386, 8.752 bits. **These are not exact predictions or lower bounds for dSprites.** Its factors are discrete and bounded; information about each factor saturates at 5 bits. At M=8 the high-precision task noise is already smaller than one factor-grid step. Precision saturation and nuisance information may materially change the behavior. The Gaussian recoverability indices cannot be interpreted as dSprites allocations. Report absence of a reversal rather than tuning it into existence.

For dSprites, the default per-M plots omit Gaussian oracle curves and witness-rate markers. An explicit `--show-gaussian-reference` option displays them as a surrogate, with no bound claim. `crossover_vs_M.png` labels the Gaussian trend as heuristic. The optional Gaussian-renderer control is the place to test exact analytic predictions and the M=256 relative-gap calculation. The neural pretexts are not the theorem's full class of fixed Gaussian pretexts; this experiment does not verify that universal lower bound.

## Outputs and uncertainty

Aggregation runs after the launcher completes successfully:

```bash
python summarize.py --run runs/dsprites
python summarize.py --run runs/dsprites --show-gaussian-reference
# Explicit opt-in for an unfinished sweep:
python summarize.py --run runs/dsprites --allow-partial
```

`summary/` contains `all_runs.csv`, validation-selected `selected_runs.csv`, `curve_bands.csv`, and PNG/PDF figures for rate–distortion, mixture factor recoverability, and relative priority. Factor names are used in legends. `crossovers.json` includes crossing estimates, interpolation brackets, conditional intervals, support ranges, and detection fractions. `diagnostics.json` records missing jobs, per-seed crossings, and R² decreases larger than 0.01. `theory_predictions.json` clearly marks Gaussian applicability.

Seed curves are interpolated only on their common achieved-rate support. Whole-seed bootstrap resampling preserves dependence across rate settings. Bands are pointwise percentile intervals, conditional on the candidate grid and validation selection; they are not simultaneous bands. A one-seed smoke run produces degenerate bands and is not scientific evidence.

A reversal requires negative and positive differences beyond ±0.05 recoverability-index units. All supported negative-to-positive crossings remain visible. Crossover intervals are conditional on exactly one crossing; missing/multiple-crossing bootstrap fractions are reported. There is no extrapolation, isotonic fitting, monotonicity correction, or forced reversal. Check raw seed curves, convergence histories, probe quality, and common-rate coverage before interpreting a crossover or its absence.

## Dataset attribution

Official source: https://github.com/google-deepmind/dsprites-dataset

The archive is downloaded by the user; it is not redistributed in this code package. Refer to the upstream repository's license and citation when using the data.

```bibtex
@misc{dsprites17,
  author = {Loic Matthey and Irina Higgins and Demis Hassabis and Alexander Lerchner},
  title = {dSprites: Disentanglement testing Sprites dataset},
  howpublished = {https://github.com/deepmind/dsprites-dataset/},
  year = {2017}
}
```

## Paper appendix controls

`configs/role_swap.json` and `configs/null_control.json` each contain the main
14-beta, three-seed grid for the mixture objective (42 jobs). The swap exchanges
`posX,posY` for `posY,posX`. The null sets both prevalence weights to 0.5 and both
noise variances to 1/16. Explicit `task_weights` and `noise_variances` apply to
training, held-out probe noise, and weighted evaluation. Results record them;
custom task settings disable the original M-family Gaussian reference.

```bash
python launch.py --config configs/role_swap.json --data-dir data/dsprites --gpus 0,1,2,3 --out runs/dsprites_role_swap
python launch.py --config configs/null_control.json --data-dir data/dsprites --gpus 0,1,2,3 --out runs/dsprites_null_control
```

Then run `../paper_plotting/paper_experiment_plots.ipynb`. See the
[plotting README](../paper_plotting/README.md#role-swap-and-null-control) for the
physical-axis gap convention, input checks, and uncertainty interpretation.
These configurations require actual training; existing main-run results cannot
be relabeled to provide control evidence.
