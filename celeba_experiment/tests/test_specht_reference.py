import unittest
import numpy as np
from celeba_experiment.specht_reference import specht_factor, average_task_candidate
from gaussian_rate.nd_solvers import GaussianFamily, single_task, solve

class SpechtTests(unittest.TestCase):
    def test_limits(self):
        self.assertEqual(specht_factor(0), 1)
        self.assertAlmostEqual(specht_factor(2), 1.160, places=3)
        self.assertAlmostEqual((specht_factor(.001)-1)/.001**2, 1/8, delta=.0002)

    def test_noncommuting_bound(self):
        rng = np.random.default_rng(82)
        for _ in range(4):
            a = rng.normal(size=(3, 2, 2))
            H = a @ a.transpose(0, 2, 1)
            H *= 8 / np.linalg.eigvalsh(H).max()
            family = GaussianFamily(H)
            for R in [0., .5, 2., 5.]:
                optimum = solve(family, R, tol=1e-7)
                self.assertTrue(optimum['converged'])
                avg = family.value(single_task(H.mean(0), R))
                self.assertLessEqual(avg, specht_factor(8)*optimum['D'] + 1e-7)

    def test_synthetic_target_information(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(100000, 3))
        H = np.array([np.diag([2., 0., 0.]), np.diag([0., 4., 0.])])
        candidate, info = average_task_candidate({'train': x}, H, 12)
        y = candidate['targets']['train'][0]
        np.testing.assert_allclose(candidate['H'][0], H.mean(0), atol=1e-12)
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        noise = y-x@coef
        fitted = coef @ np.linalg.inv(noise.T@noise/len(x)) @ coef.T
        np.testing.assert_allclose(fitted, H.mean(0), atol=.03)
        self.assertEqual(info['effective_rank'], 2)

if __name__ == '__main__': unittest.main()
