"""CPU checks for custom backbone loading, all cached views, and cache identity."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import models
from celeba_experiment.data import extract, load_backbone, set_backbone_checkpoint


class TinyCelebA:
    attr_names = ['a', 'b']
    def __init__(self, *args, **kwargs): pass
    def __len__(self): return 2
    def __getitem__(self, i):
        rgb = np.random.default_rng(i).integers(0, 255, (218, 178, 3), dtype=np.uint8)
        return Image.fromarray(rgb), torch.tensor([i, 1-i])


class BackboneTests(unittest.TestCase):
    def test_custom_views_reuse_and_replacement(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encoder = models.resnet18(weights=None); encoder.fc = nn.Identity(); encoder.eval()
            checkpoint = root/'backbone.pt'
            torch.save(dict(state_dict=encoder.state_dict(), architecture='resnet18', feature_dim=512), checkpoint)
            config = dict(backbone='resnet18_imagenet1k_v1', seed=3, device='cpu', threads=1,
                          data=dict(root='unused', download=False, max_train=2, max_valid=2, max_test=2,
                                    batch_size=2, workers=0, views_per_policy=1),
                          policies=[dict(name='mild', crop_scale=[.9, 1.], flip=0)])
            set_backbone_checkpoint(config, checkpoint)
            loaded, transform = load_backbone(config)
            self.assertFalse(loaded.training)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            x = transform(TinyCelebA()[0][0]).unsqueeze(0)
            with torch.no_grad():
                torch.testing.assert_close(loaded(x), encoder(x))
            with patch('celeba_experiment.data.datasets.CelebA', TinyCelebA):
                manifest = extract(config, root/'cache')
            self.assertIn('backbone_checkpoint', manifest)
            with np.load(root/'cache/train.npz') as z:
                self.assertEqual(z['anchor'].shape, (2, 512))
                self.assertEqual(z['target_0_0'].shape, (2, 512))
                np.testing.assert_allclose(z['anchor'][0], loaded(x).detach().numpy()[0], atol=1e-5)
            with patch('celeba_experiment.data.load_backbone', side_effect=AssertionError('cache should be reused')):
                self.assertEqual(extract(config, root/'cache'), manifest)
            with torch.no_grad(): encoder.conv1.weight.add_(.01)
            torch.save(dict(state_dict=encoder.state_dict(), architecture='resnet18', feature_dim=512), checkpoint)
            with self.assertRaisesRegex(ValueError, 'Cache configuration changed'):
                extract(config, root/'cache')

    def test_old_imagenet_cache_remains_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(backbone='resnet18_imagenet1k_v1', seed=3, data={}, policies=[])
            fields = dict(transform_version=2, **config)
            signature = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
            manifest = dict(signature=signature, attribute_names=['a'])
            (root/'manifest.json').write_text(json.dumps(manifest))
            for s in ['train','valid','test']: np.savez(root/f'{s}.npz', anchor=np.zeros((2,512)))
            with patch('celeba_experiment.data.load_backbone', side_effect=AssertionError('must reuse')):
                self.assertEqual(extract(config, root), manifest)


if __name__ == '__main__': unittest.main()
