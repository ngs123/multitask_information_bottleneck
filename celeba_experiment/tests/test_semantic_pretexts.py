import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from celeba_experiment.pretexts import check_source_labels, build_candidate, proxy_channel
from celeba_experiment.finite_codes import CompositePredictor, fit_codes, load_encoder
from gaussian_rate.nd_solvers import GaussianFamily


class SemanticTests(unittest.TestCase):
    def test_label_disjointness(self):
        pre=[{'name':'s','objective':'attribute_bce','attributes':['source']}]
        check_source_labels(pre,['source','eval'],['eval'])
        with self.assertRaises(ValueError):check_source_labels(pre,['source','eval'],['source'])
        pre=[{'name':'h','components':[{'objective':'attribute_bce','attributes':['eval']}]}]
        with self.assertRaises(ValueError):check_source_labels(pre,['source','eval'],['eval'])

    def test_weighted_loss_and_shared_encoder(self):
        heads=[{'objective':'task_bce','target_dim':2,'weight':.3},
               {'objective':'gaussian_nll','target_dim':3,'weight':.7}]
        m=CompositePredictor(3,8,2,heads)
        x=torch.randn(9,3);y=torch.cat([(torch.rand(9,2)>.5).float(),torch.randn(9,3)],1)
        expected=.3*m.predictors[0].loss(x,y[:,:2],True)+.7*m.predictors[1].loss(x,y[:,2:],True)
        torch.testing.assert_close(m.loss(x,y,True),expected)
        self.assertIs(m.predictors[0].encoder,m.predictors[1].encoder)
        m.loss(x,y,True).backward()
        self.assertGreater(m.encoder.net[0].weight.grad.abs().sum().item(),0)

    def test_source_proxies_ignore_evaluation_labels(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=(100,3)).astype('float32')
        y=(rng.random((100,2))>.5).astype('float32')
        raw={s:{'labels':y.copy()} for s in ['train','valid']}
        features={s:x for s in raw};spec={'name':'source','objective':'attribute_bce','attributes':['s']}
        with tempfile.TemporaryDirectory() as td:
            a=build_candidate(spec,raw,features,['s','e'],{}, {},{'seed':1},Path(td))
            for s in raw:raw[s]['labels'][:,1]=1-raw[s]['labels'][:,1]
            b=build_candidate(spec,raw,features,['s','e'],{}, {},{'seed':1},Path(td))
            np.testing.assert_array_equal(a['H'],b['H'])

    def test_hybrid_weights_match_per_coordinate_loss(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(100,3)).astype('float32')
        raw={s:{'labels':(x[:,:2]>0).astype('float32')} for s in ['train','valid']}
        targets={'anchor':{s:x[None] for s in raw}}
        spec={'name':'hybrid','components':[{'objective':'attribute_bce','attributes':['s','e'],'weight':.5},
                                           {'objective':'gaussian_nll','target':'anchor','weight':.5}]}
        with tempfile.TemporaryDirectory() as td:
            c=build_candidate(spec,raw,{s:x for s in raw},['s','e'],targets,{'anchor':{'H':np.eye(3)}},{'seed':1},Path(td))
            np.testing.assert_allclose(c['weights'],[.375,.375,.25])
            self.assertEqual(c['targets']['train'].shape,(1,100,5))
            K,cert=proxy_channel(c,1,{'gap_tolerance':1e-5,'max_iterations':3000})
            self.assertTrue(cert['converged'])
            self.assertAlmostEqual(-np.linalg.slogdet(K)[1]/(2*np.log(2)),1,places=5)
            heads=c['heads'];path=Path(td)/'encoder.pt'
            z,_=fit_codes({s:x for s in raw},c['targets'],3,'composite',{'heads':heads,'epochs':2,'hidden_dim':8,'batch_size':50},2,path)
            np.testing.assert_array_equal(z['valid'],load_encoder(path)(torch.tensor(x)).detach().numpy())

if __name__=='__main__':unittest.main()
