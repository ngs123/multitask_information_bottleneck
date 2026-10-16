import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from celeba_experiment.merge_runs import validate_merge, combined_config, import_previous
from gaussian_rate.nd_solvers import single_task


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'seed':1, 'seeds':[1], 'representations':['gaussian','quantized'],
                  'tasks':{'names':['x']}, 'gaussian':{'rates':[0,1]},
                  'quantized':{'bits':[0,2], 'epochs':3}}
        self.row={'mode':'quantized','candidate':'a','budget':2,'seed':1,'directory':'quantized/a/budget_2/seed_1'}

    def prepare(self, root):
        (root/'config.json').write_text(json.dumps(self.cfg))
        (root/'results.json').write_text(json.dumps([self.row]))
        (root/self.row['directory']).mkdir(parents=True)

    def test_settings_guard_allows_only_budget_mode_device_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.prepare(root)
            new=copy.deepcopy(self.cfg)
            new['quantized']['bits']=[64,128];new['representations']=['quantized'];new['device']='cpu'
            self.assertEqual(len(validate_merge(root,new)),1)
            new['seeds']=[1,2]
            new['robustness']={'enabled':True}
            self.assertEqual(len(validate_merge(root,new)),1)
            new['seeds']=[2]
            with self.assertRaises(ValueError):validate_merge(root,new)

    def test_import_is_self_contained_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'old';root.mkdir();self.prepare(root)
            out=Path(td)/'new';out.mkdir()
            (root/self.row['directory']/'model.bin').write_bytes(b'checkpoint')
            (root/'feature_manifest.json').write_text(json.dumps({'signature':'same','device':'cuda'}))
            (root/'gaussian_predicted_distortions.json').write_text('{"2":{"a":0.1}}')
            (root/'certificates.json').write_text('[]')
            result=import_previous(root,out,{'signature':'same','device':'cpu'},[self.row])
            self.assertEqual(result['predicted'][2]['a'],.1)
            self.assertEqual((out/self.row['directory']/'model.bin').read_bytes(),b'checkpoint')
            self.assertEqual((root/self.row['directory']/'model.bin').read_bytes(),b'checkpoint')

    def test_combined_config_supports_next_extension(self):
        newrow={**self.row,'budget':128}
        cfg=combined_config(self.cfg,[self.row,newrow])
        self.assertEqual(cfg['quantized']['bits'],[2,128])
        self.assertEqual(cfg['representations'],['quantized'])

    def test_high_rate_water_level(self):
        for rate in [1.,64.,128.]:
            K=single_task(np.diag([.5,1.,2.,4.]),rate)
            actual=-np.linalg.slogdet(K)[1]/(2*np.log(2))
            self.assertAlmostEqual(actual,rate,places=8)
