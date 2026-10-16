"""Average-task single pretext and explicitly heuristic probe-loss reference."""
import json
import numpy as np
from .analysis import dump


def specht_factor(L):
    if not np.isfinite(L) or L < 0:
        raise ValueError('L must be finite and nonnegative')
    x = np.log1p(L)
    if x < 1e-4:
        return float(np.exp(x*x/8 - x**4/576))
    return float(np.exp(np.log(np.expm1(x)/x) + x/np.expm1(x) - 1))


def average_task_candidate(features, H, seed):
    avg = np.mean(H, axis=0)
    eig, u = np.linalg.eigh((avg + avg.T)/2)
    keep = eig > max(1e-12, eig.max()*1e-10)
    # Drop numerical null directions, so no bits are wasted on irrelevant modes.
    root = u[:, keep] * np.sqrt(eig[keep])
    effective = root @ root.T
    # Quantized analogue: predict one synthetic Gaussian target conditional on X.
    # Coordinate whitening is invertible and improves NLL optimization conditioning.
    scales = np.sqrt(1 + eig[keep])
    targets = {}
    for j, (split, x) in enumerate(features.items()):
        rng = np.random.default_rng(seed + 78413 + j)
        if keep.any():
            targets[split] = np.stack([(x @ root + rng.normal(size=(len(x), len(scales)))) / scales
                                       for _ in range(2)]).astype(np.float32)
        else:
            targets[split] = np.zeros((1, len(x), 1), np.float32)
    L = float(max(0., np.linalg.eigvalsh(H).max()))
    info = {'L': L, 'specht_factor': specht_factor(L), 'weights': [1/len(H)]*len(H),
            'effective_rank': int(keep.sum()),
            'quantized_target': 'Y = sqrt(Hbar) X + independent standard Gaussian noise; coordinate-whitened; two cached draws',
            'ce_reference': 'uncompressed_CE + S(1+L)*(task_aware_CE - uncompressed_CE); heuristic, no clipping',
            'scope': 'The theorem bounds Gaussian proxy excess distortion only, not empirical probe CE or binary codes.'}
    return {'H': effective[None], 'weights': np.ones(1), 'targets': targets}, info


def save_reference_curves(rows, out, info):
    baseline = json.loads((out / 'uncompressed' / 'metrics.json').read_text())['test']['mean_ce_bits']
    result = {}
    for mode in sorted({r['mode'] for r in rows}):
        curves = []
        for b in sorted({r['budget'] for r in rows if r['mode'] == mode}):
            def mean(name):
                values = [r['test_ce_bits'] for r in rows if r['mode'] == mode and r['budget'] == b and r['candidate'] == name]
                return float(np.mean(values)) if values else None
            task, avg = mean('task_aware'), mean('average_task')
            if task is None or avg is None:
                continue
            bound = baseline + info['specht_factor']*(task-baseline)
            curves.append({'budget': b, 'task_aware_ce_bits': task, 'average_task_ce_bits': avg,
                           'uncompressed_ce_bits': baseline, 'specht_ce_reference_bits': bound,
                           'average_below_ce_reference': bool(avg <= bound)})
        result[mode] = curves
    dump(out / 'specht_curves.json', result)
    # A separate plot tests the actual theorem in its native proxy units.
    native = []
    for b in sorted({r['budget'] for r in rows if r['mode'] == 'gaussian'}):
        values = {}
        for name in ['task_aware', 'average_task']:
            matches = [r for r in rows if r['mode'] == 'gaussian' and r['budget'] == b and r['candidate'] == name]
            if matches:
                values[name] = json.loads((out / matches[0]['directory'] / 'metrics.json').read_text())['diagnostics']['gaussian_distortion']
        if len(values) == 2:
            bound = info['specht_factor']*values['task_aware']
            native.append({'budget': b, **values, 'specht_upper': bound,
                           'violation': max(0., values['average_task']-bound)})
    dump(out / 'specht_proxy_curves.json', native)
    if native:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, label, style in [('task_aware', 'Task-aware Gaussian optimum', '-o'),
                                  ('average_task', 'Average-task Gaussian pretext', '-s'),
                                  ('specht_upper', 'Specht upper bound', '--')]:
            ax.plot([r['budget'] for r in native], [r[key] for r in native], style, label=label)
        ax.set(xlabel='Nominal Gaussian rate (bits/image)', ylabel='Gaussian proxy excess loss (bits/task)')
        ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(out / ('specht_proxy.' + ext))
        plt.close(fig)
