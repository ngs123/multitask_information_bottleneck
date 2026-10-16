"""Paired uncertainty and geometry diagnostics; no training dependencies."""
import itertools
import numpy as np
from scipy.stats import rankdata, kendalltau

LN2 = np.log(2.)


def percentile(a, level=.95, axis=0):
    tail = (1-level)/2
    return np.quantile(a, [tail, 1-tail], axis=axis)


def crossover(rates, differences, tol=1e-10):
    """First strictly positive-to-negative crossing; initial R=0 tie is ignored.

    Linear interpolation is a grid-based estimator, not a root solve. Keep all
    direction-changing roots and the noncrossing status for honest reporting.
    """
    r, y = np.asarray(rates), np.asarray(differences)
    keep = np.flatnonzero(np.abs(y) > tol)
    roots = []
    for a, b in zip(keep[:-1], keep[1:]):
        if y[a] > 0 and y[b] < 0:
            roots.append(float(r[a] + (r[b]-r[a])*y[a]/(y[a]-y[b])))
    if roots:
        return {'rate': roots[0], 'status': 'interior', 'roots': roots}
    status = ('all_tied' if not len(keep) else
              'specialist_better_on_grid' if np.all(y[keep] > 0) else
              'generalist_better_on_grid' if np.all(y[keep] < 0) else
              'no_positive_to_negative_crossing')
    return {'rate': None, 'status': status, 'roots': []}


def crossover_summary(rates, point, draws, level=.95):
    result = crossover(rates, point)
    samples = [crossover(rates, y) for y in draws]
    roots = [x['rate'] for x in samples if x['rate'] is not None]
    result.update(n_resamples=len(samples), n_interior=len(roots),
                  statuses={s: sum(x['status'] == s for x in samples)
                            for s in sorted({x['status'] for x in samples})},
                  interior_fraction=len(roots)/len(samples) if samples else None,
                  conditional_percentile_ci=percentile(roots, level).tolist() if roots else None,
                  ci_condition='Conditional on an observed interior positive-to-negative crossing; inspect interior_fraction.',
                  interpolation='Linear between grid values; zero-rate tie excluded.',
                  multiple_crossing_resamples=sum(len(x['roots']) > 1 for x in samples))
    # Deliberately never imply that a conditional interval proves every draw crossed.
    return result


def paired_bootstrap(loss, n_resamples, seed, resample_seeds=False):
    """loss shape: seed, candidate, rate, ORIGINAL IMAGE, task.

    Same image resample and seed resample for every candidate, rate and task.
    Seeds and test images are crossed factors, not independent seed-image rows.
    Caller must average channel-noise losses within each original image first.
    """
    a = np.asarray(loss)
    if a.ndim != 5 or not np.isfinite(a).all():
        raise ValueError('Expected finite [seed,candidate,rate,image,task] losses')
    rng = np.random.default_rng(seed)
    out = np.empty((n_resamples, a.shape[1], a.shape[2], a.shape[4]))
    seed_mean = a.mean(0) if not resample_seeds else None
    for b in range(n_resamples):
        ss = rng.integers(a.shape[0], size=a.shape[0]) if resample_seeds else np.arange(a.shape[0])
        ii = rng.integers(a.shape[3], size=a.shape[3])
        averaged = a[ss].mean(0) if resample_seeds else seed_mean
        out[b] = averaged[:, :, ii, :].mean(2)
    return out


def rank_statistics(values, rates, tol=1e-10):
    """Rows=candidates; tie-aware ranks. Use lowest POSITIVE rate by default."""
    x = np.asarray(values)
    ranks = np.stack([rankdata(x[:, j], method='average') for j in range(x.shape[1])], axis=1)
    positive = np.flatnonzero(np.asarray(rates) > 0)
    low = int(positive[0]) if len(positive) else 0
    high = len(rates)-1
    a, b = x[:, low], x[:, high]
    reversed_pairs = comparable = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        da, db = a[i]-a[j], b[i]-b[j]
        if abs(da) > tol and abs(db) > tol:
            comparable += 1
            reversed_pairs += int(da*db < 0)
    tau = float(kendalltau(ranks[:, low], ranks[:, high]).statistic) if len(x)>1 else float('nan')
    return ranks, {'low_rate': float(rates[low]), 'high_rate': float(rates[high]),
                   'kendall_tau_b': tau if np.isfinite(tau) else None,
                   'reversed_pairs': reversed_pairs, 'comparable_untied_pairs': comparable,
                   'total_pairs': len(x)*(len(x)-1)//2,
                   'max_rank_change': float(np.max(np.ptp(ranks, axis=1))),
                   'note': 'Zero-rate ties excluded from endpoint comparison when positive rates exist.'}


def distortion_tasks(H, K):
    values = []
    for h in H:
        eig, u = np.linalg.eigh((h+h.T)/2)
        root = (u*np.sqrt(np.maximum(eig, 0)))@u.T
        values.append(np.linalg.slogdet(np.eye(len(K))+root@K@root)[1]/(2*LN2))
    return np.array(values)


def specialist_geometry(H, candidate, tolerance=1e-9):
    """Theory quantities use the support of the specified specialist proxy."""
    mean_source = np.einsum('t,tij->ij', candidate['weights'], candidate['H'])
    eig, u = np.linalg.eigh(mean_source)
    selected = eig > max(1e-12, eig.max()*tolerance)
    Q, Qout = u[:, selected], u[:, ~selected]
    P = Q@Q.T
    d, k = len(P), Q.shape[1]
    hbar = np.mean(H, axis=0)
    C = np.array([h@np.linalg.solve(np.eye(d)+h, np.eye(d)) for h in H])
    cbar = C.mean(0)
    miss = float(distortion_tasks(H, np.eye(d)-P).mean())
    delta = float(np.trace(P@cbar)/k-np.trace(cbar)/d) if k else None
    threshold = float(d/2*np.log2(np.trace(hbar)/(2*LN2*miss))) if miss>1e-14 else None
    # Uniform restricted allocation is an additional assumption, not implied by support.
    isotropic_parts = all(np.linalg.norm(h-(np.trace(h)/k)*P) <= tolerance*max(1., np.linalg.norm(h))
                          for h in candidate['H']) if k else False
    return Q, Qout, {'dimension':d, 'specialist_rank':k,
                    'task_snr':np.linalg.eigvalsh(H)[:, -1].tolist(),
                    'delta_uniform_subspace':delta, 'D_miss_bits':miss,
                    'isotropic_generalist_sufficient_rate_bound':threshold,
                    'specialist_components_isotropic_on_support':bool(isotropic_parts),
                    'uniform_reference_hypotheses': bool(delta is not None and delta>0 and miss>0 and k<d),
                    'scope':'Delta tests a uniform-on-S reference. A fitted anisotropic specialist need not obey that low-rate criterion. The threshold is a sufficient bound for isotropic reconstruction to beat D_miss, not an estimate of the crossing.'}


def allocation(K, Q, Qout):
    """Additive chain rule: I(X_S;Z) + I(X_perp;Z|X_S).

    Coordinates of whitened X in the two orthogonal spaces are independent.
    Marginal subspace informations alone do not generally add to total rate.
    """
    total = float(-np.linalg.slogdet(K)[1]/(2*LN2))
    inside = float(-np.linalg.slogdet(Q.T@K@Q)[1]/(2*LN2)) if Q.shape[1] else 0.
    outside_marginal = float(-np.linalg.slogdet(Qout.T@K@Qout)[1]/(2*LN2)) if Qout.shape[1] else 0.
    return {'total':total, 'inside_marginal':inside, 'outside_conditional':total-inside,
            'outside_marginal':outside_marginal}
