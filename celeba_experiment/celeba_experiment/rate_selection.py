"""Rate-aware pretext comparison with common inputs and validation-only selection."""
import csv
from pathlib import Path
import numpy as np
from .analysis import dump, fit_channel_probes
from .merge_runs import cell_key
from .fitting import fit_cross_view
from .probes import fit_probes, evaluate
from .tasks import select_tasks
from .pretexts import components, check_source_labels, build_candidate, proxy_channel
from .rate_models import fit_pca, transform, fit_gaussian_tasks, task_channel, compact_channel
from gaussian_rate.nd_solvers import GaussianFamily, single_task


def validate_config(config):
    from .robustness import validate
    validate(config)
    candidates = config['pretexts']
    names = [c['name'] for c in candidates]
    if len(names) != len(set(names)) or not names or bool({'task_aware', 'average_task'} & set(names)):
        raise ValueError('Pretext names must be nonempty and unique')
    for name in names:
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Pretext names must be filesystem-safe identifiers')
    if config['gaussian'].get('noise_draws', 1) < 1:
        raise ValueError('Need positive noise_draws')
    if not 0 < config['gaussian']['rho_max'] < 1:
        raise ValueError('rho_max must lie strictly between zero and one')
    policies = [p['name'] for p in config['policies']]
    if 'anchor' in policies or any(not n or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in n) for n in policies):
        raise ValueError('Policy names must be safe identifiers; anchor is reserved')
    if len(policies) != len(set(policies)):
        raise ValueError('Duplicate view policy names')
    for c in candidates:
        parts = components(c)
        if not parts or any(not np.isfinite(p.get('weight', 1.)) or p.get('weight', 1.) <= 0 for p in parts):
            raise ValueError('Pretext components must have positive finite weights')
        for part in parts:
            objective = part.get('objective', 'gaussian_nll')
            if objective not in ('gaussian_nll', 'infonce', 'attribute_bce'):
                raise ValueError(f'Unsupported pretext objective: {objective}')
            if objective == 'attribute_bce':
                if not part.get('attributes'):
                    raise ValueError('attribute_bce needs source attributes')
            elif part.get('target') not in ['anchor'] + policies:
                raise ValueError(f'Unknown target: {part.get("target")}')
    modes = config.get('representations', ['gaussian', 'quantized'])
    if not modes or set(modes) - {'gaussian', 'quantized'}:
        raise ValueError('Unknown representation mode')
    if not config.get('fit_modes', ['task', 'pretext']) or set(config.get('fit_modes', ['task', 'pretext'])) - {'task', 'pretext'}:
        raise ValueError('Unknown fitting mode')
    rates = config['gaussian']['rates']
    bits = config['quantized']['bits']
    if not rates or any(r < 0 or not np.isfinite(r) for r in rates) or len(set(rates)) != len(rates):
        raise ValueError('Gaussian rates must be unique finite nonnegative numbers')
    if not bits or any(type(b) is not int or b < 0 for b in bits) or len(set(bits)) != len(bits):
        raise ValueError('Quantized budgets must be unique nonnegative integers')
    seeds = config.get('seeds', [config['seed']])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and unique')


def split_validation(data, seed):
    """Separate encoder checkpoint validation from candidate-selection validation."""
    n = len(data['anchor'])
    if n < 8:
        raise ValueError('Need at least eight validation examples')
    order = np.random.default_rng(seed + 9182).permutation(n)
    cut = n // 2
    return ({k: v[order[:cut]] for k, v in data.items()},
            {k: v[order[cut:]] for k, v in data.items()})


def probe_and_score(encoded, labels, config, seed, directory, stochastic=False):
    if stochastic:
        model = fit_channel_probes(encoded['train'], labels['train'],
                                   config['_n_train'], {**config, 'seed': seed})
    else:
        model = fit_probes(encoded['train'], labels['train'], config['probes'], seed)
    np.savez_compressed(directory / 'probes.npz', **model)
    metrics = {s: evaluate(model, encoded[s], labels[s]) for s in ['select', 'test']}
    # Retain per-image losses for paired bootstrap, averaging losses over noise
    # draws (NOT averaging predictions, which would evaluate a multi-view decoder).
    per_image = {}
    for s in ['select', 'test']:
        logits = encoded[s] @ model['weights'].T + model['bias']
        loss = (np.logaddexp(0., logits) - labels[s] * logits) / np.log(2.)
        if stochastic:
            loss = loss.reshape(config['gaussian']['noise_draws'], -1, loss.shape[1]).mean(0)
        per_image[s + '_ce_bits'] = loss
    np.savez_compressed(directory / 'per_image_losses.npz', **per_image)
    return metrics


def evaluate_run(raw_splits, manifest, config, out, imported=None):
    validate_config(config)
    out = Path(out)
    names, ids = select_tasks(manifest['attribute_names'], config)
    check_source_labels(config['pretexts'], manifest['attribute_names'], names)
    dump(out / 'selected_tasks.json', {'names': names, 'indices': ids, 'weights': [1/len(ids)]*len(ids)})
    raw = {s: {k: np.asarray(v) for k, v in data.items()} for s, data in raw_splits.items()}
    raw['valid'], raw['select'] = split_validation(raw['valid'], config['seed'])
    np.savez_compressed(out / 'split_indices.npz', **{s: data['indices'] for s, data in raw.items()})
    labels = {s: data['labels'][:, ids] for s, data in raw.items()}
    anchor_fit = fit_pca(raw['train']['anchor'], config.get('input_dim', 32))
    np.savez_compressed(out / 'anchor_transform.npz', **anchor_fit)
    features = {s: transform(data['anchor'], anchor_fit) for s, data in raw.items()}
    cfg = {**config, '_n_train': len(features['train'])}
    uncompressed = out / 'uncompressed'
    if not uncompressed.exists():
        uncompressed.mkdir()
        dump(uncompressed / 'metrics.json', probe_and_score(features, labels, cfg, config['seed'], uncompressed))
    tasks = fit_gaussian_tasks(features['train'], labels['train'], seed=config['seed'], **config.get('task_proxy', {}))
    np.savez_compressed(out / 'gaussian_tasks.npz', **tasks)
    family = GaussianFamily(tasks['H'])
    from .specht_reference import average_task_candidate
    average_candidate, reference_info = average_task_candidate(features, tasks['H'], config['seed'])
    dump(out / 'specht_reference.json', reference_info)
    np.savez_compressed(out / 'average_task_proxy.npz', H=average_candidate['H'])
    targets, proxies = {}, {}
    feature_parts = [p for c in config['pretexts'] for p in components(c) if p.get('objective') != 'attribute_bce']
    for candidate in feature_parts:
        target = candidate['target']
        if target in targets:
            continue
        if target == 'anchor':
            keys = ['anchor']
        else:
            j = [p['name'] for p in config['policies']].index(target)
            keys = sorted(k for k in raw['train'] if k.startswith(f'target_{j}_'))
        if not keys:
            raise ValueError(f'No cached features for {target}')
        fit = fit_pca(np.concatenate([raw['train'][k] for k in keys]), config.get('target_dim', 32))
        np.savez_compressed(out / f'target_transform_{target}.npz', **fit)
        targets[target] = {s: np.stack([transform(data[k], fit) for k in keys]) for s, data in raw.items()}
        proxy = fit_cross_view(features['train'], list(targets[target]['train']), config['gaussian']['rho_max'])
        proxies[target] = proxy
        np.savez_compressed(out / f'cross_view_{target}.npz', **proxy)
    candidates = {c['name']: build_candidate(c, raw, features, manifest['attribute_names'], targets, proxies, config, out)
                  for c in config['pretexts']}
    candidates['average_task'] = average_candidate
    pretext_certificates = []
    imported = imported or {}
    rows = list(imported.get('rows', []))
    certificates = list(imported.get('certificates', []))
    completed = {cell_key(row) for row in rows}
    seeds = config.get('seeds', [config['seed']])
    fit_modes = config.get('fit_modes', ['task', 'pretext'])
    channels = {}
    # All proxy curves are inexpensive and used for the Gaussian selection rule,
    # including in quantized-only runs. No downstream validation/test fitting.
    all_rates = sorted(set(config['gaussian']['rates']) | set(config['quantized']['bits']))
    predicted = dict(imported.get('predicted', {}))
    for rate in all_rates:
        predicted.setdefault(rate, {})
        for c in config['pretexts'] + [{'name': 'average_task'}]:
            K, cert = proxy_channel(candidates[c['name']], rate, config['gaussian'])
            if cert is not None:
                pretext_certificates.append({'candidate': c['name'], 'budget': rate, **cert})
            channels[c['name'], rate] = K
            predicted[rate][c['name']] = family.value(K)
    dump(out / 'pretext_certificates.json', pretext_certificates)
    dump(out / 'gaussian_predicted_distortions.json', predicted)
    for mode in config.get('representations', ['gaussian', 'quantized']):
        budgets = config['gaussian']['rates'] if mode == 'gaussian' else config['quantized']['bits']
        for budget in budgets:
            specs = [{'name': 'task_aware', 'objective': 'task_bce'},
                     {'name': 'average_task', 'objective': 'gaussian_nll'}]
            if 'pretext' in fit_modes:
                if mode == 'gaussian':
                    # The joint Gaussian proxy depends only on the target, not
                    # on the neural training loss. Evaluate duplicate targets once.
                    seen = set()
                    for candidate in config['pretexts']:
                        part = components(candidate)
                        key = ('feature', part[0].get('target')) if len(part) == 1 and part[0].get('objective') != 'attribute_bce' else ('candidate', candidate['name'])
                        if key not in seen or config.get('robustness', {}).get('enabled', False):
                            specs.append(candidate)
                            seen.add(key)
                else:
                    specs += config['pretexts']
            for spec in specs:
                name = spec['name']
                if all((mode, name, budget, seed) in completed for seed in seeds):
                    print(f'Reusing {mode}: {name}, budget={budget:g}', flush=True)
                    continue
                if config.get('_diagnostics_only', False):
                    raise ValueError(f'Missing completed cells for {mode}/{name}/{budget}. Omit --diagnostics-only to fill them.')
                aware = name == 'task_aware'
                if mode == 'gaussian':
                    if aware:
                        K, cert = task_channel(tasks['H'], budget,
                            config['gaussian'].get('gap_tolerance', 1e-5), config['gaussian'].get('max_iterations', 3000))
                        certificates = [c for c in certificates if c['budget'] != budget]
                        certificates.append({'budget': budget, **cert})
                        if not cert['converged'] and not config['gaussian'].get('allow_unconverged', False):
                            dump(out / 'certificates.json', certificates)
                            raise RuntimeError(f'Gaussian task solve at R={budget} failed gap tolerance; see certificates.json')
                    else:
                        K = channels[name, budget]
                    A = compact_channel(K)
                for seed in seeds:
                    if (mode, name, budget, seed) in completed:
                        continue
                    directory = out / mode / name / f'budget_{budget:g}' / f'seed_{seed}'
                    directory.mkdir(parents=True)
                    print(f'{mode}: {name}, budget={budget:g}, seed={seed}', flush=True)
                    if mode == 'gaussian':
                        draws = config['gaussian']['noise_draws']
                        encoded, yy = {}, {}
                        for j, s in enumerate(['train', 'valid', 'select', 'test']):
                            # Common random numbers across candidates and budgets.
                            rng = np.random.default_rng(seed + 10000 + j)
                            encoded[s] = np.tile(features[s] @ A.T, (draws, 1))
                            if np.any(A):
                                encoded[s] += rng.normal(size=encoded[s].shape)
                            yy[s] = np.tile(labels[s], (draws, 1))
                        np.savez_compressed(directory / 'channel.npz', K=K, A=A,
                                            nominal_rate=-np.linalg.slogdet(K)[1]/(2*np.log(2)))
                        metrics = probe_and_score(encoded, yy, cfg, seed, directory, True)
                        stats = {'gaussian_distortion': family.value(K)}
                    else:
                        from .finite_codes import fit_codes, pack_codes
                        tt = {s: labels[s][None].astype(np.float32) for s in features} if aware else candidates[name]['targets']
                        train_cfg = {**config['quantized'], 'threads': config.get('threads', 4)}
                        if not aware and ('components' in spec or spec.get('objective') == 'attribute_bce'):
                            train_cfg['heads'] = candidates[name]['heads']
                        if 'device' in config: train_cfg['device'] = config['device']
                        encoded, history = fit_codes(features, tt, budget, spec.get('objective', 'gaussian_nll'),
                                                     train_cfg, seed, directory / 'encoder.pt')
                        dump(directory / 'history.json', history)
                        np.savez_compressed(directory / 'hard_codes.npz', bits=budget,
                                            **{s: pack_codes(z, budget) for s, z in encoded.items()})
                        stats = {}
                        for s, z in encoded.items():
                            _, counts = np.unique(z, axis=0, return_counts=True)
                            p = counts / counts.sum()
                            stats[s + '_unique_codes'] = len(counts)
                            stats[s + '_empirical_code_entropy_bits'] = float(-(p*np.log2(p)).sum())
                        metrics = probe_and_score(encoded, labels, cfg, seed, directory)
                    dump(directory / 'metrics.json', {'metrics': metrics, 'diagnostics': stats})
                    rows.append({'mode': mode, 'candidate': name, 'budget': budget, 'seed': seed,
                                 'fit': 'task' if aware else ('reference' if name == 'average_task' else 'pretext'),
                                 'objective': 'gaussian_proxy' if mode == 'gaussian' else spec.get('objective', 'gaussian_nll'),
                                 'select_ce_bits': metrics['select']['mean_ce_bits'],
                                 'test_ce_bits': metrics['test']['mean_ce_bits'],
                                 'directory': str(directory.relative_to(out))})
                    dump(out / 'results.json', rows)
    dump(out / 'certificates.json', certificates)
    with (out / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Backfilled reference cells use the union of old and new budgets (CLI merge).
    from .specht_reference import save_reference_curves
    save_reference_curves(rows, out, reference_info)
    selections = select_policies(rows, predicted)
    dump(out / 'selection.json', selections)
    plot_curves(rows, selections, out)
    from .robustness import report
    report(raw, manifest, ids, features, labels, tasks, candidates, rows, config, out)
    support_cfg = config['gaussian'].get('support_diagnostics', {})
    if support_cfg.get('enabled', True) and any(r['mode'] == 'gaussian' and r['candidate'] == 'task_aware' for r in rows):
        from gaussian_rate.support import diagnose_saved_run
        diagnose_saved_run(out, active_rate_tol=support_cfg.get('active_rate_tol', 1e-5))
    return rows


def select_policies(rows, predicted):
    """Fixed vs per-rate rules aggregate seeds; held-out test never selects."""
    result = {}
    for mode in sorted(set(r['mode'] for r in rows)):
        rr = [r for r in rows if r['mode'] == mode and r['fit'] == 'pretext']
        if not rr:
            continue
        names = sorted(set(r['candidate'] for r in rr))
        budgets = sorted(set(r['budget'] for r in rr))
        def score(name, budget, split):
            return float(np.mean([r[split+'_ce_bits'] for r in rr if r['candidate'] == name and r['budget'] == budget]))
        fixed = min(names, key=lambda n: np.mean([score(n, b, 'select') for b in budgets]))
        output = []
        for b in budgets:
            chosen = min(names, key=lambda n: score(n, b, 'select'))
            gp = min(names, key=lambda n: predicted[b][n])
            output.append({'budget': b, 'rate_selected': chosen, 'fixed_selected': fixed, 'gaussian_selected': gp,
                           'rate_selected_test_ce_bits': score(chosen, b, 'test'),
                           'fixed_selected_test_ce_bits': score(fixed, b, 'test'),
                           'gaussian_selected_test_ce_bits': score(gp, b, 'test'),
                           'improvement_over_fixed_bits': score(fixed, b, 'test')-score(chosen, b, 'test')})
        result[mode] = output
    return result


def plot_curves(rows, selections, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for mode in sorted(set(r['mode'] for r in rows)):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for name in sorted(set(r['candidate'] for r in rows if r['mode'] == mode and r['candidate'] != 'average_task')):
            rr = [r for r in rows if r['mode'] == mode and r['candidate'] == name]
            budgets = sorted(set(r['budget'] for r in rr))
            means = [np.mean([r['test_ce_bits'] for r in rr if r['budget'] == b]) for b in budgets]
            std = [np.std([r['test_ce_bits'] for r in rr if r['budget'] == b]) for b in budgets]
            axes[0].plot(budgets, means, marker='o', label=name)
            axes[0].fill_between(budgets, np.array(means)-std, np.array(means)+std, alpha=.12)
        for key, label in [('fixed_selected', 'Fixed pretext'), ('rate_selected', 'Rate-aware selection'), ('gaussian_selected', 'Gaussian-predicted selection')]:
            ss = selections.get(mode, [])
            if ss: axes[1].plot([s['budget'] for s in ss], [s[key+'_test_ce_bits'] for s in ss], marker='o', label=label)
        import json
        reference_path = out / 'specht_curves.json'
        if reference_path.exists():
            refs = json.loads(reference_path.read_text()).get(mode, [])
            for ax in axes:
                ax.plot([r['budget'] for r in refs], [r['average_task_ce_bits'] for r in refs],
                        color='black', marker='s', label='Average-task pretext')
                ax.plot([r['budget'] for r in refs], [r['specht_ce_reference_bits'] for r in refs],
                        color='black', linestyle='--', label='Specht reference (heuristic CE)')
        for ax in axes:
            ax.set_xlabel('Nominal Gaussian rate (bits/image)' if mode == 'gaussian' else 'Hard code budget (bits/image)')
            ax.set_ylabel('Test log loss (bits/task)'); ax.grid(alpha=.2)
            if ax.lines: ax.legend(fontsize=8)
        axes[0].set_title('Candidates (shading: seed standard deviation)')
        axes[1].set_title('Selection uses validation only')
        fig.tight_layout(); fig.savefig(out / f'{mode}_selection.png', dpi=180)
        fig.savefig(out / f'{mode}_selection.pdf'); plt.close(fig)
