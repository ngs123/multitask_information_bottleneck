"""Meaningful statistical checks; runs without PyTorch or CelebA."""
import unittest
import numpy as np
from celeba_experiment.robust_statistics import (crossover, crossover_summary,
    paired_bootstrap, allocation, specialist_geometry, rank_statistics)
from celeba_experiment.rate_models import fit_gaussian_tasks


class RobustnessTests(unittest.TestCase):
    def test_zero_rate_tie_and_censoring(self):
        r=np.array([0.,1.,2.,3.])
        self.assertEqual(crossover(r,[0,2,0,-2])['rate'],2.)
        self.assertIsNone(crossover(r,[0,-1,-2,-3])['rate'])
        summary=crossover_summary(r,[0,2,0,-2],[[0,2,0,-2],[0,2,1,.5]])
        self.assertEqual(summary['interior_fraction'],.5)
        self.assertEqual(summary['conditional_percentile_ci'],[2.,2.])
        self.assertEqual(summary['statuses']['specialist_better_on_grid'],1)

    def test_paired_resampling_preserves_constant_difference(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(3,1,4,30,2));a=np.concatenate([x,x+2],axis=1)
        for seeds in [False,True]:
            result=paired_bootstrap(a,25,7,seeds)
            np.testing.assert_allclose(result[:,1]-result[:,0],2,atol=1e-12)

    def test_chain_rule_rotated_covariance(self):
        K=np.array([[.6,.2],[.2,.3]])
        r=allocation(K,np.eye(2)[:,:1],np.eye(2)[:,1:])
        self.assertAlmostEqual(r['inside_marginal']+r['outside_conditional'],r['total'])
        self.assertGreater(r['outside_conditional'],r['outside_marginal'])

    def test_crossfit_groups_keep_duplicate_images_together(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(30,5));y=rng.normal(size=(30,2))
        a=fit_gaussian_tasks(x,y,folds=3,seed=4)
        b=fit_gaussian_tasks(np.repeat(x,3,axis=0),np.repeat(y,3,axis=0),
                             folds=3,seed=4,groups=np.repeat(np.arange(30),3))
        np.testing.assert_allclose(a['oof_residual_mse'],b['oof_residual_mse'],rtol=1e-12)

    def test_geometry_known_subspace(self):
        h=np.array([np.diag([4.,0,0]),np.diag([0,1.,0])])
        candidate={'H':np.array([np.diag([1.,0,0])]),'weights':np.ones(1)}
        q,qo,g=specialist_geometry(h,candidate)
        self.assertAlmostEqual(g['D_miss_bits'],.25)
        self.assertAlmostEqual(g['delta_uniform_subspace'],.4-(.4+.25)/3)
        self.assertEqual(g['specialist_rank'],1)
        self.assertTrue(g['specialist_components_isotropic_on_support'])

    def test_rank_ties_and_reversal(self):
        ranks,s=rank_statistics([[1,1,3],[1,2,2],[1,3,1]],[0,1,2])
        self.assertAlmostEqual(s['kendall_tau_b'],-1)
        self.assertEqual(s['reversed_pairs'],3)
        self.assertEqual(s['low_rate'],1)
        self.assertTrue(np.all(ranks[:,0]==2))


if __name__=='__main__':unittest.main()
