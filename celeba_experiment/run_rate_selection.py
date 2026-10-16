#!/usr/bin/env python3
"""Compare task/pretext-aware Gaussian channels and learned deterministic codes."""
import argparse
import json
import platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from celeba_experiment.analysis import dump
from celeba_experiment.rate_selection import evaluate_run, validate_config
from celeba_experiment.merge_runs import validate_merge, import_previous, combined_config
from celeba_experiment.data import set_backbone_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', help='Defaults to the merge source config, otherwise configs/rate_selection_robust.json')
    ap.add_argument('--merge-run', type=Path, help='Existing run to extend; import completed cells and regenerate combined plots in the new output')
    ap.add_argument('--cache', default='cache/celeba_paper')
    ap.add_argument('--backbone-checkpoint', type=Path, help='Multitask backbone.pt; use a new feature cache')
    ap.add_argument('--output', help='New directory; default timestamped run')
    ap.add_argument('--representation', choices=['gaussian', 'quantized', 'both'])
    ap.add_argument('--fit', choices=['task', 'pretext', 'both'])
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--all-tasks', action='store_true')
    group.add_argument('--tasks', nargs='+')
    ap.add_argument('--bits', nargs='+', type=int)
    ap.add_argument('--rates', nargs='+', type=float)
    ap.add_argument('--seeds', nargs='+', type=int)
    ap.add_argument('--device', help='cpu, cuda, cuda:0, etc.')
    ap.add_argument('--robustness', action='store_true', help='Enable moment/test bootstraps and V1–V7 reports')
    ap.add_argument('--specialist', help='Prespecified specialist pretext name for paired comparisons')
    ap.add_argument('--generalist', help='Prespecified generalist pretext name (default reconstruction)')
    ap.add_argument('--moment-bootstrap', type=int, help='Full moment pipeline resamples (default 200)')
    ap.add_argument('--test-bootstrap', type=int, help='Paired image resamples (default 2000)')
    ap.add_argument('--mlp-probes', action='store_true', help='Add one-hidden-layer probe comparisons (V8)')
    ap.add_argument('--diagnostics-only', action='store_true', help='Require --merge-run; error instead of training missing bottlenecks')
    ap.add_argument('--smoke', action='store_true', help='Synthetic cached features; no download. Exercises every configured branch, not scientific evidence.')
    args = ap.parse_args()
    config_path = Path(args.config) if args.config else (args.merge_run / 'config.json' if args.merge_run else Path('configs/rate_selection_robust.json'))
    config = json.loads(config_path.read_text())
    set_backbone_checkpoint(config, args.backbone_checkpoint)
    if args.representation:
        config['representations'] = ['gaussian', 'quantized'] if args.representation == 'both' else [args.representation]
    if args.fit: config['fit_modes'] = ['task', 'pretext'] if args.fit == 'both' else [args.fit]
    if args.all_tasks: config['tasks'] = {'reduced': False}
    if args.tasks: config['tasks'] = {'reduced': True, 'names': args.tasks}
    if args.bits is not None: config['quantized']['bits'] = args.bits
    if args.rates is not None: config['gaussian']['rates'] = args.rates
    if args.seeds is not None: config['seeds'] = args.seeds
    if args.device: config['device'] = args.device
    robust = dict(config.get('robustness', {}))
    if args.robustness or args.mlp_probes: robust['enabled'] = True
    for key, value in [('specialist',args.specialist),('generalist',args.generalist),
                       ('moment_resamples',args.moment_bootstrap),('test_resamples',args.test_bootstrap)]:
        if value is not None: robust[key] = value
    if args.mlp_probes: robust['mlp'] = {**robust.get('mlp', {}), 'enabled': True}
    if robust: config['robustness'] = robust
    if args.diagnostics_only:
        if not args.merge_run: ap.error('--diagnostics-only requires --merge-run')
        if not robust.get('enabled'): ap.error('--diagnostics-only requires robustness.enabled or --robustness')
        config['_diagnostics_only'] = True
    if args.smoke:
        config['input_dim'] = config['target_dim'] = 4
        config['gaussian'].update(rates=[0, 1, 4], noise_draws=2)
        config['quantized'].update(bits=[0, 2, 4], epochs=3, hidden_dim=16, batch_size=64)
        config['probes'].update(C_grid=[0, 1], max_iter=100)
        config['seeds'] = [config['seed']]
        config['smoke_test'] = True
        if config.get('robustness', {}).get('enabled'):
            config['robustness'].update(moment_resamples=3, test_resamples=30)
            config['robustness']['mlp'] = {**config['robustness'].get('mlp', {}), 'epochs':3, 'patience':2}
    validate_config(config)
    old_rows = validate_merge(args.merge_run, config) if args.merge_run else None
    out = Path(args.output or 'runs/CelebA/rate_selection/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'config.json', config)
    dump(out / 'environment.json', {'python': platform.python_version(), 'torch': str(torch.__version__),
                                    'numpy': np.__version__, 'cuda_available': torch.cuda.is_available()})
    torch.set_num_threads(config.get('threads', 4))
    with threadpool_limits(limits=config.get('threads', 4)):
        if config.get('smoke_test', False):
            from celeba_experiment.smoke_data import synthetic_features
            splits, manifest = synthetic_features(config)
        else:
            from celeba_experiment.data import extract
            manifest = extract(config, args.cache)
            splits = {}
            for s in ['train', 'valid', 'test']:
                with np.load(Path(args.cache) / f'{s}.npz') as data:
                    splits[s] = {k: data[k] for k in data.files}
        dump(out / 'feature_manifest.json', manifest)
        imported = import_previous(args.merge_run, out, manifest, old_rows) if args.merge_run else None
        if args.merge_run:
            dump(out / 'extension_config.json', config)
            dump(out / 'merge_source.json', {'run': str(args.merge_run.resolve()), 'imported_cells': len(old_rows)})
        if old_rows:
            # Reuse completed cells, but evaluate new references at old rates too.
            for mode, section, key in [('gaussian', 'gaussian', 'rates'), ('quantized', 'quantized', 'bits')]:
                if mode in config['representations']:
                    config[section][key] = sorted(set(config[section][key]) |
                        {r['budget'] for r in old_rows if r['mode'] == mode})
        rows = evaluate_run(splits, manifest, config, out, imported=imported)
        dump(out / 'config.json', combined_config(config, rows))
    print(f'Run saved to {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
