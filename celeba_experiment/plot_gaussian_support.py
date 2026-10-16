#!/usr/bin/env python3
"""Plot active-support diagnostics from a saved run, without encoder/probe training."""
import argparse
import json
from pathlib import Path
import numpy as np
from gaussian_rate.support import load_saved_run, write_support_diagnostics
from gaussian_rate.nd_solvers import GaussianFamily, solve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--active-rate-tol', type=float, default=1e-5, help='Active iff per-mode rate exceeds this many bits')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--rates', nargs='+', type=float, help='Add missing rates by solving cached Gaussian tasks only')
    group.add_argument('--rate-grid', nargs=3, type=float, metavar=('START', 'STOP', 'COUNT'))
    p.add_argument('--gap-tolerance', type=float, default=1e-5)
    p.add_argument('--max-iterations', type=int, default=5000)
    p.add_argument('--allow-unconverged', action='store_true')
    args = p.parse_args()
    if not np.isfinite(args.gap_tolerance) or args.gap_tolerance <= 0 or args.max_iterations < 1:
        raise ValueError('Require positive finite gap tolerance and positive iteration count')
    data = load_saved_run(args.run)
    if data['unconverged'] and not args.allow_unconverged:
        raise ValueError('Saved run has unconverged Gaussian solves; inspect certificates or explicitly use --allow-unconverged')
    extra = args.rates or []
    if args.rate_grid:
        start, stop, count = args.rate_grid
        if not np.isfinite(args.rate_grid).all() or start < 0 or stop <= start or count != int(count) or count < 2:
            raise ValueError('Require finite 0 <= START < STOP and integer COUNT >= 2')
        extra = np.linspace(start, stop, int(count)).tolist()
    if not np.isfinite(extra).all() or any(r < 0 for r in extra):
        raise ValueError('Rates must be finite and nonnegative')
    family = GaussianFamily(data['H'], data['weights'])
    saved = {float(r): k for r, k in zip(data['rates'], data['K'])}
    missing = [float(r) for r in np.unique(extra) if not any(np.isclose(r, s, atol=1e-12, rtol=1e-12) for s in saved)]
    certificates = []
    if missing:
        Hbar = np.einsum('t,tij->ij', data['weights'], data['H'])
        eig, U = np.linalg.eigh(Hbar); U = U[:, eig > max(1e-12, eig.max()*1e-10)]
        reduced = GaussianFamily(np.einsum('ia,tij,jb->tab', U, data['H'], U), data['weights']) if U.shape[1] else None
        warm = None
        for R in sorted(missing):
            if reduced is None:
                saved[R] = np.eye(family.d); continue
            result = solve(reduced, R, warm_S=warm, tol=args.gap_tolerance, maxiter=args.max_iterations)
            if not result['converged'] and not args.allow_unconverged:
                raise RuntimeError(f'Gaussian solve did not converge at R={R:g}; gap={result["gap_bound"]:g}')
            warm = result['S']; saved[R] = np.eye(family.d)+U@(result['K']-np.eye(U.shape[1]))@U.T
            certificates.append({'requested_rate': R, **{k: result[k] for k in ['converged', 'gap_bound', 'rate', 'iterations']}})
            print(f'Gaussian solve only: R={R:g}, gap={result["gap_bound"]:.2g}', flush=True)
    rates = np.array(sorted(saved)); Ks = np.array([saved[r] for r in rates]); D = np.array([family.value(k) for k in Ks])
    out = args.output or args.run/'support_diagnostics'
    unconverged = bool(data['unconverged']) or any(not c['converged'] for c in certificates)
    write_support_diagnostics(rates, Ks, D, out, args.active_rate_tol,
                               title='Task-aware Gaussian representations' if not unconverged else 'Task-aware Gaussian solver outputs (some unconverged)')
    np.savez_compressed(out/'covariances.npz', rates=rates, K_optimal=Ks,
                        task_matrices=data['H'], task_weights=data['weights'])
    (out/'source.json').write_text(json.dumps({'run': str(args.run.resolve()), 'additional_gaussian_solves': certificates,
                                             'source_unconverged': data['unconverged']}, indent=2)+'\n')
    print(f'Saved diagnostics to {out.resolve()}')


if __name__ == '__main__': main()
