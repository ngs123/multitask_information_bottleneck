"""Control settings reach objectives and evaluation without changing defaults."""
import unittest
import numpy as np
import torch
from model import objective, excess_nll
from task_spec import configured_tasks, task_parameters
from train import evaluate
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory


class ControlTests(unittest.TestCase):
    def test_defaults_and_invalid_settings(self):
        self.assertEqual(task_parameters(4), ([.75,.25],[1/16,1/256]))
        for weights, noise in [([.5,.6],None), ([0,1],None), (None,[0,1]), (None,[float('nan'),1])]:
            with self.assertRaises(ValueError): task_parameters(4,weights,noise)

    def test_null_objective_matches_equal_noise_and_weights(self):
        class Net:
            def encode(self,x): return x, torch.zeros(len(x))
            def task(self,c): return torch.zeros_like(c), torch.ones_like(c)
        z=torch.tensor([[1.,2.],[3.,1.]])
        observed=objective(Net(),z,z,4,'mixture',.1,1,task_weights=[.5,.5],noise_variances=[1/16,1/16])[0]
        expected=excess_nll(torch.zeros_like(z),torch.ones_like(z),z,torch.tensor([1/16,1/16])).mean()
        self.assertTrue(torch.allclose(observed,expected))
        swapped=objective(Net(),z,z.flip(1),4,'mixture',.1,1,task_weights=[.5,.5],noise_variances=[1/16,1/16])[0]
        self.assertTrue(torch.allclose(observed,swapped))

    def test_evaluation_uses_configured_noise_and_weights(self):
        class Probe:
            def __call__(self,c): return torch.zeros_like(c), torch.ones_like(c)
        z=torch.tensor([[1.,2.],[2.,4.]])
        cfg=dict(M=4,eval_draws=1,batch_size=2,task_weights=[.5,.5],noise_variances=[1/16,1/16])
        with TemporaryDirectory() as tmp, patch('train.extract',return_value=(z,z,torch.zeros(2))):
            result=evaluate(None,Probe(),None,z,cfg,torch.device('cpu'),Path(tmp))
        expected=excess_nll(torch.zeros_like(z),torch.ones_like(z),z,torch.tensor(configured_tasks(cfg)[1])).mean().item()
        self.assertAlmostEqual(result['distortion_bits'],expected,places=5)
