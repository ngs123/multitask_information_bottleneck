"""Model/optimizer checks; no images, weights, or network access required."""
import unittest
import numpy as np
from gaussian_rate.nd_solvers import GaussianFamily,solve,single_task,s_value_grad,channel_matrix
from celeba_experiment.fitting import fit_tasks
from celeba_experiment.probes import entropy

class GaussianChecks(unittest.TestCase):
    def test_single_task_matches_closed_form(self):
        rng=np.random.default_rng(4);A=rng.normal(size=(6,6));H=A@A.T
        family=GaussianFamily([H])
        for R in [.1,1,8]:
            result=solve(family,R,tol=1e-6)
            self.assertTrue(result['converged'])
            self.assertLess(abs(result['D']-family.value(single_task(H,R))),1e-7)
            self.assertAlmostEqual(result['rate'],R,places=8)
    def test_frechet_gradient(self):
        rng=np.random.default_rng(2);v=rng.normal(size=(10,5));family=GaussianFamily(np.einsum('ti,tj->tij',v,v))
        B=rng.normal(size=(5,5));S=B@B.T/5;E=rng.normal(size=(5,5));E=(E+E.T)/2
        _,g,_=s_value_grad(family,S);eps=1e-5
        numeric=(s_value_grad(family,S+eps*E)[0]-s_value_grad(family,S-eps*E)[0])/(2*eps)
        self.assertAlmostEqual(numeric,float(np.sum(g*E)),places=7)
    def test_channel_recovers_K(self):
        K=np.diag([.05,.2,.8,1]);A=channel_matrix(K)
        np.testing.assert_allclose(np.linalg.inv(np.eye(4)+A.T@A),K)
    def test_ce_calibration_endpoints(self):
        x=np.array([[-1.,0],[1.,0]]);y=np.array([[0,0],[1,1]])
        probe={'weights':np.array([[1.,0],[0,1.]])};metrics={'label_entropy_bits':[1.,1.],'ce_bits':[.2,1.2]}
        tasks=fit_tasks(x,y,probe,metrics);family=GaussianFamily(tasks['H_ce'])
        self.assertAlmostEqual(family.value(np.eye(2)),.4)
        self.assertAlmostEqual(float(tasks['ce_floor_bits'].mean())+family.value(np.eye(2)),1.)
        self.assertEqual(tasks['gamma'][1],0)
    def test_candidate_below_certificate(self):
        rng=np.random.default_rng(8);v=rng.normal(size=(10,4));f=GaussianFamily(np.einsum('ti,tj->tij',v,v))
        result=solve(f,3,maxiter=1,tol=1e-12);best=solve(f,3,tol=1e-8)
        self.assertLessEqual(result['D']-best['D'],result['gap_bound']+1e-10)
        self.assertLess(best['gap_bound'],1e-8)

if __name__=='__main__':unittest.main()
