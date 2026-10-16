import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from celeba_experiment.robustness import moment_bootstrap, csv_write
from celeba_experiment.report_tables import write_tables


class RecoveryTests(unittest.TestCase):
    def test_retry_reuses_sample_and_preserves_tolerance_then_reuses_curve(self):
        H=np.array([np.diag([1.,0.])]);bank={'s':{'H':H,'weights':np.ones(1)},'average_task':{'H':H,'weights':np.ones(1)}}
        cfg={'seed':1,'pretexts':[{'name':'s'}],'gaussian':{'rho_max':.9,'gap_tolerance':1e-5,'max_iterations':10},
             'robustness':{'enabled':True,'specialist':'s','moment_resamples':1}}
        raw={'train':{'anchor':np.eye(2),'indices':np.arange(2)}}
        names=['s','average_task','task_aware'];point=np.zeros((3,1,1));samples=[];limits=[]
        def fit(raw,manifest,ids,config,sample,directory):
            samples.append(sample.copy());np.savez(directory/'gaussian_tasks.npz',H=H)
            np.savez(directory/'pretext_proxy_s.npz',H=H,weights=np.ones(1))
            return H,bank
        def solve(H,bank,rates,config):
            limits.append(config['gaussian']['max_iterations'])
            self.assertEqual(config['gaussian']['gap_tolerance'],1e-5)
            if len(limits)==1:raise RuntimeError('gap slightly above tolerance')
            return names,point,{}
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            with patch('celeba_experiment.robustness.fit_bootstrap',side_effect=fit),patch('celeba_experiment.robustness.model_curves',side_effect=solve):
                moment_bootstrap(raw,{'attribute_names':['y']},[0],cfg,out,[1.],names,point)
            self.assertEqual(limits,[10,40]);self.assertEqual(len(samples),1)
            with patch('celeba_experiment.robustness.fit_bootstrap',side_effect=AssertionError('refit')),patch('celeba_experiment.robustness.model_curves',side_effect=AssertionError('resolve')):
                moment_bootstrap(raw,{'attribute_names':['y']},[0],cfg,out,[1.],names,point)
            status=json.loads((out/'robustness/moment_bootstrap/status.json').read_text())
            self.assertEqual(status['reused'],1)
            self.assertTrue(status['bands_available'])

    def test_persistent_failure_saves_status_without_aborting_reporting(self):
        H=np.array([np.eye(2)]);bank={'s':{'H':H,'weights':np.ones(1)}}
        cfg={'seed':1,'pretexts':[{'name':'s'}],'gaussian':{'rho_max':.9,'max_iterations':1},
             'robustness':{'specialist':'s','moment_resamples':1}}
        raw={'train':{'anchor':np.eye(2),'indices':np.arange(2)}}
        with tempfile.TemporaryDirectory() as td,patch('celeba_experiment.robustness.fit_bootstrap',return_value=(H,bank)),patch('celeba_experiment.robustness.model_curves',side_effect=RuntimeError('not converged')):
            with self.assertWarns(UserWarning):
                result=moment_bootstrap(raw,{'attribute_names':['y']},[0],cfg,Path(td),[1.],['s'],np.zeros((1,1,1)))
            self.assertEqual(len(result),0)
            status=json.loads((Path(td)/'robustness/moment_bootstrap/status.json').read_text())
            self.assertFalse(status['bands_available']);self.assertEqual(len(status['failures']),1)

    def test_readable_tables_and_latex_escaping(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);csv_write(out/'fitted_quantities.csv',[{'quantity':'D_miss','value':.125}]);write_tables(out)
            self.assertIn('D_miss',(out/'tables.html').read_text())
            self.assertIn(r'D\_miss',(out/'tables.tex').read_text())
            self.assertIn('0.125',(out/'tables.md').read_text())


if __name__=='__main__':unittest.main()
