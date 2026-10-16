"""Synthetic fixture for integration checks; never presented as CelebA results."""
import numpy as np
from .tasks import DEFAULT_TASKS


def synthetic_features(config):
    rng = np.random.default_rng(config['seed'])
    names = DEFAULT_TASKS + ['Mouth_Slightly_Open','High_Cheekbones','Wavy_Hair','Straight_Hair','Receding_Hairline'] + [f'synthetic_task_{i}' for i in range(31)]
    weights = rng.normal(size=(8, 40))
    splits = {}
    for j, (split, n) in enumerate([('train', 256), ('valid', 128), ('test', 128)]):
        x = rng.normal(size=(n, 8))
        logits = x @ weights + rng.normal(size=(n, 40))
        labels = (logits > 0).astype(np.int8)
        data = {'anchor': x.astype(np.float32), 'labels': labels, 'indices': np.arange(n)+j*10000}
        for i, policy in enumerate(config['policies']):
            for v in range(config['data'].get('views_per_policy', 2)):
                gain = np.ones(8)
                gain[(i % 4)*2:(i % 4)*2+2] = .1
                data[f'target_{i}_{v}'] = (x*gain + rng.normal(size=x.shape)*.3).astype(np.float32)
        splits[split] = data
    return splits, {'attribute_names': names, 'synthetic': True, 'note': 'Integration fixture, not CelebA'}
