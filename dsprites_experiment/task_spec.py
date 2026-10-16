"""Task weights and noise shared by training, probes, and reporting."""
import math


def task_parameters(M, weights=None, noise_variances=None):
    weights = list(weights if weights is not None else [1 - 1/M, 1/M])
    noise = list(noise_variances if noise_variances is not None else [M**-2, M**-4])
    if (len(weights) != 2 or not all(math.isfinite(x) and x > 0 for x in weights)
            or not math.isclose(sum(weights), 1., abs_tol=1e-9)):
        raise ValueError('task_weights must be two positive finite values summing to one')
    if len(noise) != 2 or not all(math.isfinite(x) and x > 0 for x in noise):
        raise ValueError('noise_variances must be two positive finite values')
    return weights, noise


def configured_tasks(cfg):
    return task_parameters(cfg['M'], cfg.get('task_weights'), cfg.get('noise_variances'))
