# Multitask information bottleneck

CelebA and dSprites experiments with shared paper plotting code. Training scripts,
configurations, tests, datasets, caches, and saved runs live in their respective
experiment directories. Shared dependencies and `paper_plotting/` live here.

## Installation

Use Python 3.10+ and run from this repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# On GPU machines, install a CUDA-compatible torch/torchvision pair first.
python -m pip install -r requirements.txt
```

The shared requirements combine both experiments' minimum versions; torchvision
also constrains its compatible torch version. They are not a frozen environment.

## Training

Run each experiment from its own directory so existing relative data and saved
run paths retain their meaning. Use a new output directory for new configurations.

CelebA (requires the local CelebA dataset; see its README for preparation):

```bash
cd celeba_experiment
python run_rate_selection.py --config configs/rate_selection_robust.json \
  --cache cache/celeba_paper --output runs/celeba_paper --device cuda:0
```

For a CPU integration check without downloading images, replace the cache option
with `--smoke`, use `--device cpu`, and choose a temporary output directory.

dSprites, starting from the repository root:

```bash
cd dsprites_experiment
python prepare_dsprites.py --out data/dsprites
python launch.py --data-dir data/dsprites --gpus 0,1,2,3 --out runs/dsprites
```

See [CelebA](celeba_experiment/README.md), [report recovery](celeba_experiment/RECOVERY.md),
and [dSprites](dsprites_experiment/README.md) for scientific details and other runs.
Datasets, caches, and runs stay local and are excluded by `.gitignore`.

## Paper figures

From the repository root:

```bash
jupyter lab paper_plotting/paper_experiment_plots.ipynb
```

Run all cells, or execute without the UI:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 paper_plotting/paper_experiment_plots.ipynb
```

The notebook finds the repository root from its working directory or ancestors.
For an external working directory, set `MIB_REPO_ROOT` to its absolute path.
Outputs go to `paper_plotting/paper_figures/`. CelebA figures require retained
`celeba_experiment/runs/robust_rate_selection` results. The small tracked
[dSprites input snapshot](paper_plotting/inputs/dsprites_all_runs.csv) allows
reproducing dSprites figures without retraining. See [plotting notes](paper_plotting/README.md)
for provenance and optional inputs.

## Validation

Run these separately from the repository root:

```bash
(cd celeba_experiment && python -m unittest discover -s tests -p 'test_*.py' -v)
(cd dsprites_experiment && DSPRITES_TEST_CACHE=data/dsprites python -m unittest discover -s tests -p 'test_*.py' -v)
```

The real-image dSprites check needs the prepared cache. Run the notebook command
above to validate figure regeneration against local saved results. These checks
do not perform full scientific retraining.

Appendix training details and Gaussian-solver complexity are documented in
[the LaTeX methods fragment](docs/appendix_methods.tex), with
[source references and author checks](docs/appendix_methods_notes.md).
