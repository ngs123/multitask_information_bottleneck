"""Detect control relabeling errors and false crossing claims."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paper_plotting.plot_helpers import dsprites_control, roots, crossing_summary


class ControlPlotTests(unittest.TestCase):
    def fixture(self, folder, control):
        rows=[]
        for seed in range(3):
            for beta,rate,gap in [(1,1,1),(.1,3,-1)]:
                # task 1 is always common in the swap configuration.
                rows.append(dict(M=4,pretext='mixture',seed=seed,beta=beta,rate_kl_bits=rate+seed*.01,
                    mse1=2.**(-2*(2+gap/2)),mse2=2.**(-2*(2-gap/2)),
                    factor1='posY' if control=='role_swap' else 'posX',
                    factor2='posX' if control=='role_swap' else 'posY',
                    task_weight1=.75 if control=='role_swap' else .5,task_weight2=.25 if control=='role_swap' else .5,
                    noise_variance1=1/16,noise_variance2=1/256 if control=='role_swap' else 1/16))
        path=Path(folder)/'input.csv'; pd.DataFrame(rows).to_csv(path,index=False)
        return path

    def test_swap_keeps_horizontal_minus_vertical_and_detects_reverse_direction(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            source=self.fixture(tmp,'role_swap')
            rows,mean,summary=dsprites_control(source,Path(tmp)/'out','role_swap')
            self.assertTrue((rows[rows.beta==1].gap<0).all())
            self.assertEqual(summary['reversing_seeds'],3)
            self.assertEqual(summary['crossing_direction'],-1)
            self.assertAlmostEqual(summary['observed_mean_roots'][0],2.01)
        plt.close('all')

    def test_original_results_cannot_be_used_as_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=self.fixture(tmp,'null_control')
            with self.assertRaisesRegex(ValueError,'factor1=posY'):
                dsprites_control(source,Path(tmp)/'out','role_swap')

    def test_null_does_not_force_zero_or_accept_missing_provenance(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            source=self.fixture(tmp,'null_control')
            _,_,summary=dsprites_control(source,Path(tmp)/'out','null_control')
            self.assertFalse(any(summary['seed_range_contains_zero_by_beta'].values()))
            df=pd.read_csv(source).drop(columns='task_weight1'); df.to_csv(source,index=False)
            with self.assertRaisesRegex(ValueError,'task_weight1'):
                dsprites_control(source,Path(tmp)/'bad','null_control')
        plt.close('all')

    def test_crossing_conditions_are_explicit(self):
        self.assertEqual(roots([0,1,2,3],[0,1,0,-1]),[2.])
        result=crossing_summary([1,2],[1,-1],np.array([[1,-1],[1,1]]))
        self.assertEqual(result['crossing_fraction'],.5)
        self.assertEqual(result['conditional_ci'],[1.5,1.5])
