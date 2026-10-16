"""Shared coordinates, OOF Gaussian task fits, and certified Gaussian channels."""
import numpy as np
from sklearn.model_selection import KFold
from gaussian_rate.nd_solvers import GaussianFamily, solve


def fit_pca(x, dimension, floor=1e-5):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(0)
    eig, u = np.linalg.eigh(np.cov(x - mean, rowvar=False, bias=True))
    keep = np.flatnonzero(eig > max(eig.max() * floor, 1e-12))[-dimension:]
    if not len(keep):
        raise ValueError('No nonconstant input coordinates')
    return {'mean': mean, 'W': u[:, keep] / np.sqrt(eig[keep]), 'eigenvalues': eig[keep]}


def transform(x, fit):
    return ((np.asarray(x) - fit['mean']) @ fit['W']).astype(np.float32)


def fit_gaussian_tasks(x, y, folds=5, ridge=1e-3, variance_floor=1e-4, seed=0, groups=None):
    """OOF residual MSE, final train-only ridge mean model; equal task weights.

    Cross-fitting is conditional on the shared unlabeled preprocessing, fit on
    all training X. No validation/test labels enter H.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if folds < 2 or len(x) < folds:
        raise ValueError('Need at least two folds and at least folds training samples')

    def regress(a, b):
        xm, ym = a.mean(0), b.mean(0)
        ac = a - xm
        coef = np.linalg.solve(ac.T @ ac / len(a) + ridge * np.eye(a.shape[1]),
                               ac.T @ (b - ym) / len(a))
        return coef, ym - xm @ coef

    predictions = np.empty_like(y)
    if groups is None:
        splits = KFold(folds, shuffle=True, random_state=seed).split(x)
    else:
        groups = np.asarray(groups)
        unique = np.unique(groups)
        if len(groups) != len(x) or len(unique) < folds:
            raise ValueError('Need at least folds distinct original images for grouped cross-fitting')
        # Bootstrap copies of one image must remain in the same OOF fold.
        splits = ((np.flatnonzero(np.isin(groups, unique[a])), np.flatnonzero(np.isin(groups, unique[b])))
                  for a, b in KFold(folds, shuffle=True, random_state=seed).split(unique))
    for train, hold in splits:
        coef, bias = regress(x[train], y[train])
        predictions[hold] = x[hold] @ coef + bias
    residual = np.mean((y - predictions)**2, axis=0)
    variance = np.maximum(residual, variance_floor)
    coef, bias = regress(x, y)
    H = np.einsum('it,jt->tij', coef, coef) / variance[:, None, None]
    return {'H': H, 'coefficient': coef, 'bias': bias, 'noise_variance': variance,
            'oof_residual_mse': residual, 'prevalence': y.mean(0)}


def task_channel(H, rate, tolerance=1e-5, max_iterations=3000):
    """Solve only on the relevant span; irrelevant coordinates receive no bits."""
    eig, u = np.linalg.eigh(np.mean(H, axis=0))
    keep = eig > max(1e-12, eig.max() * 1e-10)
    if not keep.any():
        return np.eye(H.shape[1]), {'converged': True, 'gap_bound': 0., 'rate': 0., 'D': 0.}
    basis = u[:, keep]
    reduced = np.einsum('ia,tij,jb->tab', basis, H, basis)
    result = solve(GaussianFamily(reduced), rate, tol=tolerance, maxiter=max_iterations)
    K = np.eye(H.shape[1]) + basis @ (result['K'] - np.eye(keep.sum())) @ basis.T
    info = {k: result[k] for k in ['converged', 'gap_bound', 'rate', 'D', 'iterations', 'kkt_residual']}
    return K, info


def compact_channel(K):
    """Discard independent pure-noise output coordinates without losing information."""
    k, u = np.linalg.eigh((K + K.T) / 2)
    if k.min() <= 0 or not np.isfinite(k).all():
        raise ValueError('Residual covariance is numerically singular; lower the Gaussian rate')
    gain = np.maximum(1 / k - 1, 0)
    active = gain > 1e-9
    if not active.any():
        return np.zeros((1, len(k)))
    return (u[:, active] * np.sqrt(gain[active])).T
