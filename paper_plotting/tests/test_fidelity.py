"""Regression-score semantics and accounting plots without a spurious R²."""
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_plotting.plot_helpers import fit_linear_predictor, identity_scatter


class FidelityTests(unittest.TestCase):
    def test_affine_predictor_includes_intercept(self):
        x=np.arange(6,dtype=float); y=2*x+3
        score=fit_linear_predictor(x,y)
        self.assertAlmostEqual(score['slope'],2)
        self.assertAlmostEqual(score['intercept'],3)
        self.assertAlmostEqual(score['linear_fit_R2'],1)
        identity_r2=1-np.sum((y-x)**2)/np.sum((y-y.mean())**2)
        self.assertLess(identity_r2,0)

    def test_ols_r2_matches_squared_correlation(self):
        x=np.array([0.,1,2,3,4]); y=np.array([3.,3,7,6,10])
        score=fit_linear_predictor(x,y)
        self.assertAlmostEqual(score['linear_fit_R2'],np.corrcoef(x,y)[0,1]**2)
        with self.assertRaises(ValueError):fit_linear_predictor(np.ones(5),y)

    def test_accounting_keeps_scatter_and_identity_without_score(self):
        fig,ax=plt.subplots();identity_scatter(ax,[1,2,3],[1,1.5,1.7])
        self.assertEqual(len(ax.texts),0)
        self.assertEqual(len(ax.collections),1)
        self.assertEqual(len(ax.lines),1)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(),ax.lines[0].get_ydata())
        plt.close(fig)
