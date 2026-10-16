"""Numerical/theoretical and stochastic-channel correctness checks."""
import math
import unittest
from theory import optimum, predictions, marginal, distortion

class TheoryTests(unittest.TestCase):
    def test_reference(self):
        t=predictions(256)
        self.assertAlmostEqual(t['relative_gap_lower_bound'],4.6751351154,places=8)
        self.assertAlmostEqual(t['reference_points'][1]['r1'],14.0209945678,places=8)
    def test_reversal_and_stationarity(self):
        for M in (4,6,8,256):
            t=predictions(M)
            self.assertEqual(t['reference_points'][0]['r2'],0)
            self.assertGreater(t['reference_points'][1]['r2'],t['reference_points'][1]['r1'])
            o=optimum(M,t['crossover'])
            self.assertAlmostEqual(o['r1'],o['r2'],places=9)
            last=(0,0)
            for i in range(201):
                o=optimum(M,i*t['R2']/200)
                self.assertGreaterEqual(o['r1']+1e-10,last[0]); self.assertGreaterEqual(o['r2']+1e-10,last[1])
                self.assertLessEqual(o['distortion'],o['equal_distortion']+1e-12)
                if 1e-9<o['r1']<o['rate']-1e-9:
                    self.assertAlmostEqual(marginal(o['r1'],M**2,1-1/M),marginal(o['r2'],M**4,1/M),places=10)
                last=(o['r1'],o['r2'])
    def test_against_grid(self):
        for M in (4,8):
            for R in (1,3,6,10):
                opt=optimum(M,R)['distortion']
                brute=min(distortion(M,R*i/1000,R*(1-i/1000)) for i in range(1001))
                self.assertLessEqual(opt,brute+1e-12)
                self.assertLess(brute-opt,1e-5)

try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'PyTorch not installed')
class ModelTests(unittest.TestCase):
    def test_gaussian_nll_and_kl(self):
        from model import excess_nll, VIB, Renderer
        torch.manual_seed(1)
        # For zero information, E excess NLL = .5 log2(1+1/noise).
        z=torch.randn(200000,2); noise=torch.tensor([1/16,1/256])
        observed=excess_nll(torch.zeros_like(z),torch.ones_like(z),z,noise).mean(0)
        expected=.5*torch.log2(1+1/noise)
        self.assertTrue(torch.allclose(observed,expected,atol=.01))
        net=VIB(16,4)
        with torch.no_grad(): net.channel.weight.zero_(); net.channel.bias.zero_()
        c,kl=net.encode(Renderer(16)(z[:16]))
        self.assertTrue(torch.equal(kl,torch.zeros_like(kl)))
        self.assertGreater(c.std().item(),.5)
    def test_renderer_invertible(self):
        from model import Renderer
        r=Renderer(16); z=torch.randn(100,2)
        images=r(z).flatten(1)
        recovered=torch.linalg.lstsq((r.basis/2).flatten(1).T, images.T).solution.T
        self.assertTrue(torch.allclose(z,recovered,atol=1e-5))
    def test_all_objectives_backward(self):
        from model import VIB, Renderer, objective
        torch.set_num_threads(2)
        z=torch.randn(4,2); x=Renderer(16)(z)
        for task in ('specialist1','specialist2','mixture','reconstruction'):
            net=VIB(16,4)
            loss,_,_=objective(net,x,z,4,task,.01,256)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(net.channel.weight.grad.abs().sum().item(),0)

if __name__=='__main__': unittest.main()
