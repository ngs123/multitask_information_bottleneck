"""Run with DSPRITES_TEST_CACHE=/path/to/cache for real-dataset checks."""
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from data import Sprites, grouped_split, make_data, to_batch, FOLDS
from model import VIB, objective
from prepare_dsprites import check_cache

class SplitTests(unittest.TestCase):
    def test_groups_are_disjoint_reproducible(self):
        groups=np.repeat(np.arange(100),[1,2,3,4]*25)
        a=grouped_split(groups,[20]*5,1729); b=grouped_split(groups,[20]*5,1729)
        seen=set()
        for name in FOLDS:
            self.assertTrue(np.array_equal(a[name],b[name]))
            current=set(groups[a[name]].tolist())
            self.assertFalse(seen & current); seen |= current
            self.assertGreaterEqual(len(a[name]),20)
            self.assertLess(len(a[name]),24)
    def test_binary_reconstruction_gradient(self):
        torch.set_num_threads(2)
        net=VIB(16,4); x=torch.randint(0,2,(4,1,16,16)).float(); z=torch.randn(4,2)
        loss,_,_=objective(net,x,z,4,'reconstruction',.01,256,'bce')
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(net.reconstruction[-1].weight.grad.abs().sum().item(),0)

@unittest.skipUnless(os.environ.get('DSPRITES_TEST_CACHE'),'Set DSPRITES_TEST_CACHE to test actual images')
class ActualDataTests(unittest.TestCase):
    def test_schema_and_batch_and_splits(self):
        root=Path(os.environ['DSPRITES_TEST_CACHE']); meta=check_cache(root)
        self.assertEqual(meta['n_images'],737280)
        self.assertLess(meta['n_unique_images'],meta['n_images'])
        cfg=dict(dataset='dsprites',data_dir=str(root),factors=['posX','posY'],image_size=64,
                 n_train=128,n_val=64,n_probe=128,n_test=64,seed=0)
        with tempfile.TemporaryDirectory() as folder:
            folds,renderer,provenance=make_data(cfg,Path(folder),torch.device('cpu'))
            seen=set()
            for ds in folds.values():
                ids=set(ds.group_ids.tolist()); self.assertFalse(ids & seen); seen |= ids
                x,z=ds[0]
                self.assertEqual(tuple(x.shape),(1,64,64)); self.assertEqual(tuple(z.shape),(2,))
                self.assertTrue(set(x.unique().tolist())<={0.,1.})
                self.assertLess(z.abs().max().item(),2)
            orig=folds['test']; swap=Sprites(root,orig.indices,['posY','posX'])
            self.assertTrue(torch.equal(orig[0][0],swap[0][0]))
            self.assertTrue(torch.equal(orig[0][1].flip(0),swap[0][1]))
            self.assertIsNone(renderer)
            self.assertTrue((Path(folder)/'split_indices.npz').exists())

if __name__=='__main__': unittest.main()
