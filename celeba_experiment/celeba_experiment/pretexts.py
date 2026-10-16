"""Explicit pretext components: feature prediction and disjoint source labels."""
import numpy as np
from .rate_models import fit_gaussian_tasks
from gaussian_rate.nd_solvers import GaussianFamily, solve, single_task


def components(spec):
    if 'components' in spec:
        return spec['components']
    return [{k: v for k, v in spec.items() if k != 'name'}]


def check_source_labels(pretexts, attribute_names, evaluation_names):
    for spec in pretexts:
        for part in components(spec):
            if part.get('objective') != 'attribute_bce':
                continue
            names = part.get('attributes', [])
            if not names or len(set(names)) != len(names):
                raise ValueError('Source attribute lists must be nonempty and unique')
            unknown = set(names) - set(attribute_names)
            overlap = set(names) & set(evaluation_names)
            if unknown:
                raise ValueError(f'Unknown source attributes: {sorted(unknown)}')
            if overlap:
                raise ValueError(f'Pretext/evaluation label overlap: {sorted(overlap)}')


def build_candidate(spec, raw, features, attribute_names, targets, proxies, config, out):
    """Concatenate independently sampled target draws for separate decoder heads.

    Each component keeps its own loss and Gaussian task model. In particular,
    multiple attributes use an average of scalar task distortions, not the
    information of their concatenated joint target.
    """
    from .analysis import dump
    parts = components(spec)
    total = sum(p.get('weight', 1.) for p in parts)
    blocks, heads, matrices, weights = [], [], [], []
    for i, part in enumerate(parts):
        weight = part.get('weight', 1.) / total
        objective = part.get('objective', 'gaussian_nll')
        if objective == 'attribute_bce':
            ids = [attribute_names.index(n) for n in part['attributes']]
            block = {s: data['labels'][:, ids][None].astype(np.float32) for s, data in raw.items()}
            fit = fit_gaussian_tasks(features['train'], block['train'][0], seed=config['seed'],
                                     groups=config.get('_proxy_groups'), **config.get('task_proxy', {}))
            np.savez_compressed(out / f'source_proxy_{spec["name"]}_{i}.npz', **fit)
            matrices.extend(fit['H']); weights.extend([weight / len(ids)] * len(ids))
            objective = 'task_bce'
        else:
            block = targets[part['target']]
            matrices.append(proxies[part['target']]['H'])
            # Feature NLL is averaged over coordinates in the neural objective.
            weights.append(weight / block['train'].shape[-1])
        heads.append({'objective': objective, 'target_dim': block['train'].shape[-1], 'weight': weight})
        blocks.append(block)
    # Cartesian product represents independent view draws and validates every
    # combination; small cached view banks keep this tractable.
    import itertools
    combined = {}
    for s in raw:
        combinations = itertools.product(*[range(b[s].shape[0]) for b in blocks])
        combined[s] = np.stack([np.concatenate([b[s][j] for b, j in zip(blocks, combo)], axis=-1)
                                for combo in combinations])
    weights = np.asarray(weights, float); weights /= weights.sum()
    H = np.asarray(matrices, dtype=np.float64)
    H = (H + H.transpose(0, 2, 1)) / 2
    np.savez_compressed(out / f'pretext_proxy_{spec["name"]}.npz', H=H, weights=weights)
    dump(out / f'pretext_definition_{spec["name"]}.json', {'components': parts, 'heads': heads})
    return {'targets': combined, 'heads': heads, 'H': H, 'weights': weights}


def proxy_channel(candidate, rate, config):
    H, weights = candidate['H'], candidate['weights']
    if len(H) == 1:
        return single_task(H[0], rate), None
    # Restrict to the relevant span. Reusing task_channel with weighted matrices
    # would change each logdet; solve the weighted objective explicitly.
    avg = np.einsum('t,tij->ij', weights, H)
    eig, u = np.linalg.eigh(avg)
    basis = u[:, eig > max(1e-12, eig.max()*1e-10)]
    if not basis.shape[1]:
        return np.eye(H.shape[-1]), {'converged': True, 'gap_bound': 0.}
    reduced = np.einsum('ia,tij,jb->tab', basis, H, basis)
    reduced = (reduced + reduced.transpose(0, 2, 1)) / 2
    result = solve(GaussianFamily(reduced, weights), rate,
                   tol=config.get('gap_tolerance', 1e-5), maxiter=config.get('max_iterations', 3000))
    if not result['converged'] and not config.get('allow_unconverged', False):
        raise RuntimeError(f'Pretext Gaussian solve at R={rate} did not converge: gap={result["gap_bound"]}')
    K = np.eye(H.shape[-1]) + basis @ (result['K']-np.eye(basis.shape[1])) @ basis.T
    cert = {k: result[k] for k in ['converged','gap_bound','rate','iterations']}
    return K, cert
