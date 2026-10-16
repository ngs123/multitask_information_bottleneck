# Recover a report after a bootstrap solve fails

The previous implementation raised an exception if any bootstrap sample failed,
which stopped the later tables and diagnostics. Successful resamples already
exist at `robustness/moment_bootstrap/resample_NNNN/curves.npz` and can be reused.

For the reported gap `1.25561556e-5` at a requested tolerance of `1e-5`, increase
the solver iteration allowance on the **same fitted bootstrap problem**. There
is no need to discard that sample or rerun the other 199 samples.

Extract this update over the project code, then run from the project directory:

```bash
python resume_rate_selection_report.py \
  --run runs/robust_rate_selection \
  --cache cache/rate_selection_semantic \
  --bootstrap-max-iterations 12000
```

Use the exact cache path supplied to your original run. The command verifies its
manifest and original split IDs, uses saved PCA and task/pretext models, reuses
completed bootstrap curves, and retries missing curves. It changes neither
training seeds nor moment-bootstrap samples. It does not train bottlenecks or
extract backbone features. Optional MLP probes are fitted only when missing or
when their recorded settings differ.

The initial iteration limit is 12,000. If necessary, the same solve is retried
with 48,000 and then 192,000 iterations. The gap tolerance stays unchanged.
Each retried sample records `solver_attempts.json`. The updated `status.json`
records how many curves were reused and which resamples were recomputed.

If some samples still fail, the report completes the other diagnostics but
withholds **all moment-bootstrap intervals** and marks the bootstrap incomplete.
It retains successful draws for another resume attempt. Test-bootstrap bands
remain available because they are a separate uncertainty calculation.

Tables are now written and printed before the moment bootstrap, then updated
with the crossover summary. Open these after resumption:

- `runs/robust_rate_selection/robustness/tables.html` — readable tables;
- `runs/robust_rate_selection/robustness/tables.md` — Markdown tables;
- `runs/robust_rate_selection/robustness/tables.tex` — LaTeX tabular blocks;
- `fitted_quantities.csv` and `rank_instability.csv` — machine-readable tables.

The table headers and scope note distinguish sufficient rate bounds from
crossing estimates and conditional crossing intervals from unconditional ones.

Validation: 16 tests passed, including retry/cache reuse, unchanged tolerance,
persistent failure handling, and table export. A synthetic CLI recovery fixture
reused two completed resamples and solved only the missing third one. Your
specific resample 173 was not available here, so its convergence at the larger
iteration limit has not been verified.
