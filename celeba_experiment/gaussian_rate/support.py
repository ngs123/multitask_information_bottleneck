"""Active-support diagnostics in a common whitened input coordinate system.

Support means range(I-K), NOT range(K); finite-rate K is positive definite.
Per-mode rates are -.5*log2(eig(K)). The threshold is in bits per mode.
"""
import csv
import json
from pathlib import Path
import numpy as np


def support_statistics(rates, covariances, distortions, active_rate_tol=1e-5):
    rates, K, D = np.asarray(rates, float), np.asarray(covariances, float), np.asarray(distortions, float)
    if not np.isfinite(active_rate_tol) or active_rate_tol <= 0:
        raise ValueError('active_rate_tol must be finite and positive')
    if rates.ndim != 1 or len(rates) == 0 or K.ndim != 3 or K.shape[0] != len(rates) or K.shape[1] != K.shape[2] or D.shape != rates.shape:
        raise ValueError('Expected rates (n,), covariances (n,d,d), distortions (n,)')
    if not np.isfinite(K).all() or not np.isfinite(rates).all() or not np.isfinite(D).all() or np.any(rates < 0):
        raise ValueError('Inputs must be finite; rates must be nonnegative')
    order = np.argsort(rates); rates, K, D = rates[order], K[order], D[order]
    if np.any(np.diff(rates) <= 0):
        raise ValueError('Rates must be distinct; duplicate seeds must be combined before analysis')
    if not np.allclose(K, K.transpose(0, 2, 1), atol=1e-9, rtol=0):
        raise ValueError('Covariances must be symmetric')
    eig, vectors = np.linalg.eigh((K+K.transpose(0, 2, 1))/2)
    if np.any(eig <= 0) or np.any(eig > 1+1e-7):
        raise ValueError('Need 0 < K <= I; check solver convergence')
    allocations = np.maximum(0, -.5*np.log2(eig))
    bases = [v[:, r > active_rate_tol] for v, r in zip(vectors, allocations)]
    ranks = np.array([q.shape[1] for q in bases])
    n = len(rates)
    overlap = np.full((n, n), np.nan)
    angle = np.full((n, n), np.nan)
    retention = np.full(n, np.nan)
    for i, u in enumerate(bases):
        for j in range(i, n):
            v = bases[j]
            if min(ranks[i], ranks[j]) == 0:
                continue  # The empty support has no principal angles.
            s = np.clip(np.linalg.svd(u.T@v, compute_uv=False), 0, 1)
            s[np.abs(1-s) < 1e-12] = 1
            overlap[i, j] = overlap[j, i] = float(np.mean(s*s))
            angle[i, j] = angle[j, i] = float(np.degrees(np.arccos(s.min())))
        if i and ranks[i-1] > 0:
            retention[i] = float(np.clip(np.linalg.norm(bases[i-1].T@u, 'fro')**2/ranks[i-1], 0, 1))
    changes = [{'left_rate': float(rates[i-1]), 'right_rate': float(rates[i]),
                'old_rank': int(ranks[i-1]), 'new_rank': int(ranks[i]),
                'kind': 'activation' if ranks[i] > ranks[i-1] else 'deactivation'}
               for i in range(1, n) if ranks[i] != ranks[i-1]]
    return dict(rates=rates, K=K, D=D, ranks=ranks, mode_rates=allocations,
                overlap=overlap, max_principal_angle=angle, previous_support_retained=retention,
                changes=changes, active_rate_tol=float(active_rate_tol))


def write_support_diagnostics(rates, covariances, distortions, out, active_rate_tol=1e-5,
                              title='Task-aware Gaussian representations'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    stats = support_statistics(rates, covariances, distortions, active_rate_tol)
    R, D, rank = stats['rates'], stats['D'], stats['ranks']
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout='constrained')
    ax.plot(R, D, '-o', color='#1764ab', ms=3, lw=2, label='Task-aware Gaussian RD')
    ax.set(xlabel='Nominal Gaussian rate (bits / input)', ylabel='Gaussian proxy distortion (bits / task)',
           title=title)
    ax.grid(alpha=.18)
    rank_ax = ax.twinx()
    rank_ax.step(R, rank, where='post', color='#687481', ls=':', lw=1.3, alpha=.7)
    rank_ax.set(ylabel='Number of active modes', ylim=(-.15, max(rank.max()+.8, 1)))
    rank_ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for i, c in enumerate(stats['changes']):
        color = '#38824b' if c['kind'] == 'activation' else '#ba4343'
        ax.axvspan(c['left_rate'], c['right_rate'], color=color, alpha=.065)
        ax.axvline(c['right_rate'], color=color, ls='--', lw=1.1)
        ax.text(c['right_rate'], .96-(i % 3)*.095, f" {c['old_rank']}→{c['new_rank']}",
                transform=ax.get_xaxis_transform(), color=color, fontsize=9, va='top')
    ax.text(.02, .02, 'Dashed: first sampled rate with a changed rank; shaded: preceding interval',
            transform=ax.transAxes, fontsize=8, color='#555555')
    for ext in ['png', 'pdf']: fig.savefig(out/f'rd_mode_activations.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout='constrained')
    if len(R) > 1:
        edges = np.r_[R[0]-(R[1]-R[0])/2, (R[:-1]+R[1:])/2, R[-1]+(R[-1]-R[-2])/2]
    else:
        edges = np.array([R[0]-.5, R[0]+.5])
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#dddddd')
    image = axes[0].pcolormesh(edges, edges, np.ma.masked_invalid(stats['overlap']),
                             cmap=cmap, vmin=0, vmax=1, shading='flat')
    axes[0].set(xlabel='Rate (bits / input)', ylabel='Rate (bits / input)',
                title='Pairwise active-support overlap')
    fig.colorbar(image, ax=axes[0], label='Mean squared cosine of principal angles')
    angles = np.r_[np.nan, np.diag(stats['max_principal_angle'], 1)]
    axes[1].plot(R, angles, color='#1764ab', lw=1.5, alpha=.7)
    equal = np.r_[False, np.diff(rank) == 0] & np.isfinite(angles)
    changed = ~equal & np.isfinite(angles)
    axes[1].scatter(R[equal], angles[equal], color='#1764ab', s=24, label='Same support dimension')
    axes[1].scatter(R[changed], angles[changed], color='#cc651f', marker='D', s=32, label='Support dimension changes')
    axes[1].set(xlabel='Rate (bits / input)', ylabel='Largest principal angle (degrees)',
                title='Change from preceding sampled rate', ylim=(-1, max(5., np.nanmax(angles)*1.2) if np.isfinite(angles).any() else 5.))
    axes[1].grid(alpha=.18); axes[1].legend(fontsize=8, frameon=False)
    if not np.isfinite(stats['overlap']).any():
        axes[0].text(.5, .5, 'No nonempty support at this threshold', transform=axes[0].transAxes, ha='center')
    fig.suptitle(f'Support = modes with rate > {active_rate_tol:g} bits; gray = empty support', fontsize=11)
    fig.supxlabel('Overlap 1 means the smaller support is contained in the larger. Full-span overlap cannot detect internal rotation.', fontsize=9)
    for ext in ['png', 'pdf']: fig.savefig(out/f'subspace_rotation.{ext}', dpi=180)
    plt.close(fig)
    np.savez_compressed(out/'support_metrics.npz', **{k: v for k, v in stats.items() if isinstance(v, np.ndarray)})
    with (out/'support_metrics.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['rate', 'distortion', 'active_rank', 'max_angle_from_previous_degrees', 'previous_support_retained'])
        w.writerows(zip(R, D, rank, angles, stats['previous_support_retained']))
    summary = {k: v for k, v in stats.items() if not isinstance(v, np.ndarray)}
    summary.update(ambient_dimension=int(stats['K'].shape[1]),
                   units='rates and distortion in bits; principal angles in degrees',
                   rank_threshold_sensitivity={str(t): np.sum(stats['mode_rates'] > t, axis=1).tolist()
                                               for t in (active_rate_tol/10, active_rate_tol, active_rate_tol*10)},
                   notes=['Rank-change locations are sampled brackets, not exact critical rates.',
                          'Supports and comparisons use the same whitened input coordinates.',
                          'Positive angle at equal rank is rotation; nested rank growth has angle zero.',
                          'Support overlap cannot detect changes inside the full active span.',
                          'Adjacent angles depend on rate-grid spacing; use the pairwise heatmap for larger-scale changes.'])
    (out/'support_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return stats


def load_saved_run(run):
    """Read one task-family run; never combine different feature coordinate systems."""
    run = Path(run)
    if (run/'results.json').exists():
        rows = json.loads((run/'results.json').read_text())
        cells = [r for r in rows if r['mode'] == 'gaussian' and r['candidate'] == 'task_aware']
        if not cells: raise ValueError('No saved task-aware Gaussian channels in this run')
        with np.load(run/'gaussian_tasks.npz') as data: H = data['H']
        weights = np.ones(len(H))/len(H)
        by_rate = {}
        for row in cells:
            with np.load(run/row['directory']/'channel.npz') as data: K = data['K']
            R = float(row['budget'])
            if R in by_rate and not np.allclose(K, by_rate[R], atol=1e-8, rtol=1e-8):
                raise ValueError(f'Task-aware covariances differ across seeds at rate {R}; analyze separately')
            by_rate[R] = K
        rates = np.array(sorted(by_rate)); Ks = np.array([by_rate[r] for r in rates])
        if (run/'certificates.json').exists():
            bad = [c for c in json.loads((run/'certificates.json').read_text()) if not c.get('converged', True)]
        else: bad = []
    elif (run/'gaussian_channels.npz').exists():
        report = json.loads((run/'rate_distortion.json').read_text()); rows = report['rows']
        with np.load(run/'gaussian_channels.npz') as data: Ks = np.array([data[f'optimal_K_{i}'] for i in range(len(rows))])
        with np.load(run/'gaussian_tasks.npz') as data: H = data['H_ce' if report['task_model'] == 'ce_gain' else 'H_moment']
        rates = np.array([r['R'] for r in rows]); weights = np.ones(len(H))/len(H)
        bad = [r for r in rows if not r.get('converged', True)]
    elif (run/'covariances.npz').exists():
        with np.load(run/'covariances.npz') as data:
            rates, Ks, H, weights = [data[k] for k in ['rates', 'K_optimal', 'task_matrices', 'task_weights']]
        if (run/'source.json').exists():
            source = json.loads((run/'source.json').read_text())
            bad = source.get('source_unconverged', []) + [c for c in source.get('additional_gaussian_solves', []) if not c.get('converged', True)]
        else: bad = []
    else:
        raise ValueError('Expected a rate-selection run, one legacy head directory, or a synthetic run with saved covariances')
    from .nd_solvers import GaussianFamily
    family = GaussianFamily(H, weights)
    return dict(rates=rates, K=Ks, D=np.array([family.value(k) for k in Ks]), H=H, weights=weights,
                unconverged=bad)


def diagnose_saved_run(run, out=None, active_rate_tol=1e-5):
    data = load_saved_run(run)
    return write_support_diagnostics(data['rates'], data['K'], data['D'],
                                     out or Path(run)/'support_diagnostics', active_rate_tol,
                                     title='Task-aware Gaussian representations' if not data['unconverged'] else 'Task-aware Gaussian solver outputs (some unconverged)')
