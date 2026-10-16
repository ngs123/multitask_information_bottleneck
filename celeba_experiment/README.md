# CelebA paper experiment

This folder contains the CelebA training pipeline. Shared plotting code for both
paper experiments is in `../paper_plotting/`. The dSprites training pipeline is in `../dsprites_experiment`.
The primary CelebA specialist is `joint_attributes` (five targets), compared with
`reconstruction` on the four held-out evaluation attributes.

Run the following commands from this folder.

```bash
python -m pip install -r ../requirements.txt
python run_rate_selection.py --config configs/rate_selection_robust.json \
  --cache cache/celeba_paper --output runs/celeba_paper --device cuda:0
python -m unittest discover -s tests -p 'test_*.py' -v
```

For a quick integration check without images or downloads:

```bash
python run_rate_selection.py --config configs/rate_selection_robust.json \
  --smoke --device cpu --output /tmp/celeba-smoke
```

Use a new output directory for each new configuration. This cleanup did not
launch a full scientific retraining run.

## Recreate figures and tables

Open `../paper_plotting/paper_experiment_plots.ipynb` and run all cells. It works
from the repository root, this folder, or `../paper_plotting/` and uses the shared `plot_helpers.py`.
The notebook writes PDF/PNG figures,
CSV/LaTeX tables, captions, and provenance under `../paper_plotting/paper_figures`.

Inputs retained locally:

- `runs/robust_rate_selection`: main frozen-backbone run, saved losses,
  Gaussian fits, trained codes, bootstrap draws, and report recovery state.
- `cache/rate_selection_semantic`: exact cache needed to recover that run.
- `../paper_plotting/inputs/dsprites_all_runs.csv`: canonical dSprites results, 168 runs, matching
  `../dsprites_experiment/runs/dsprites_1/summary/all_runs.csv`.
- `runs/celeba_multitask`, `cache/rate_selection_multitask`, and
  `runs/CelebA/rate_selection/20260925T151945964363Z`: fine-tuned checkpoint,
  cache, and latest sweep retained for the requested appendix replication.
- `data/celeba`: raw dataset required for retraining.

The original run configurations remain intact. The plotting default explicitly
selects the five-target specialist and recalculates pair-dependent quantities.

## Available outputs and remaining gaps

Main Figure 3, Figure 4, Table 1, per-task curves, ranks, entropy diagnostics,
inside/outside allocation, proxy fidelity, linear/MLP decoder comparison,
theory-to-data table, dSprites bit accounting, and context-objective plots can
be regenerated from retained results. Per-run code-usage histograms also remain
in the main CelebA run.

The supplied files do not contain role-swap results, an equal-weight/equal-SNR
null experiment, a shrinkage sensitivity sweep, or the constructed-worst-case
and SNR-cap CSVs requested by the appendix specification. The fine-tuned run
also lacks the bootstrap reports needed for the full uncertainty layout.
Their notebook input slots are retained; missing results are not fabricated.
The dSprites launcher supports a role-swap run through `--factors posY posX`.

The original Gaussian moment bootstrap has **199/200** successful fits.
The default measured-NLL plot uses separate **2,000 test-image resamples**;
these are not moment-fitting intervals. Proxy mode rejects the incomplete
moment bootstrap. Recovery uses the original configuration and cache:

```bash
python resume_rate_selection_report.py --run runs/robust_rate_selection \
  --cache cache/rate_selection_semantic --bootstrap-max-iterations 48000
```

See `RECOVERY.md` for details. Recovery has not been run during cleanup.

## Numbers requiring manuscript updates

The selected five-target comparison retains the saved main crossings of
16.8182 bits (Gaussian measured NLL) and 21.2035 bits (codes). The old notebook
configuration incorrectly defaulted to the two-target specialist.

The old fitted-quantity table also used that two-target specialist. Recomputed
for five targets, its subspace rank is 5, Delta is **0.0181922**, and D_miss is
**0.215195 bits**, rather than the paper's 0.033 and 0.262. The sufficient bound
is 16.2447 bits. These quantities are now calculated for the requested pair.

The canonical dSprites CSV reproduces the paper's six-row table and gives a
mean-curve crossing of **5.26382 bits**, with all three seeds reversing. The
Gaussian reference is 5.29956 bits. The paper's 5.32 is not the crossing of this
current mean curve.

## Cleanup

Unused exploratory searches, the old InfoNCE pipeline, unrelated synthetic
experiments, superseded run directories/caches, stale documentation, and the
older dSprites CSV have been permanently removed from this folder. There is
no cleanup archive here. The dSprites sibling keeps its current `dsprites_1`
run; its superseded `dsprites` sweep has been removed.

The retained Python packages are `celeba_experiment/` (features, fits, codes,
probes, resampling) and `gaussian_rate/` (solver and diagnostics). Entry points
are `run_rate_selection.py`, `prepare_celeba.py`,
`resume_rate_selection_report.py`, `train_celeba_multitask.py`, and
`plot_gaussian_support.py`. Tests are in `tests/`.
