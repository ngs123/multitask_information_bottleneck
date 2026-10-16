import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image
from celeba_experiment.tasks import select_tasks, DEFAULT_TASKS
from celeba_experiment.rate_models import fit_gaussian_tasks, task_channel, compact_channel
from celeba_experiment.finite_codes import BinaryEncoder, CodePredictor, fit_codes, load_encoder, pack_codes
from celeba_experiment.rate_selection import select_policies, split_validation
from celeba_experiment.data import view_transform


class RateSelectionTests(unittest.TestCase):
    def test_task_subset(self):
        names=['extra']+DEFAULT_TASKS
        self.assertEqual(select_tasks(names,{}),(DEFAULT_TASKS,[1,2,3,4]))
        self.assertEqual(len(select_tasks(names,{'tasks':{'reduced':False}})[0]),5)
        with self.assertRaises(ValueError):select_tasks(names,{'tasks':{'names':['missing']}})

    def test_gaussian_residual_fit(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(2000,3))
        y=np.column_stack([2*x[:,0]+rng.normal(size=2000)*.5,rng.normal(size=2000)])
        fit=fit_gaussian_tasks(x,y)
        self.assertAlmostEqual(fit['noise_variance'][0],.25,delta=.04)
        self.assertAlmostEqual(fit['H'][0,0,0],16.,delta=2.)
        self.assertTrue(np.all(np.linalg.eigvalsh(fit['H'])>-1e-10))

    def test_task_channel_relevant_span(self):
        H=np.zeros((1,3,3));H[0,0,0]=4
        K,info=task_channel(H,2.)
        np.testing.assert_allclose(K,np.diag([1/16,1,1]),atol=1e-6)
        self.assertTrue(info['converged'])
        self.assertAlmostEqual(-np.linalg.slogdet(K)[1]/(2*np.log(2)),2.)

    def test_compact_channel_preserves_information(self):
        K=np.diag([.25,1.,.5])
        A=compact_channel(K)
        self.assertEqual(A.shape,(2,3))
        np.testing.assert_allclose(np.linalg.inv(np.eye(3)+A.T@A),K)
        np.testing.assert_array_equal(compact_channel(np.eye(3)),np.zeros((1,3)))

    def test_binary_gradients_and_packing(self):
        torch.manual_seed(0);model=BinaryEncoder(3,8,5);x=torch.randn(16,3)
        z=model(x,True)
        self.assertTrue(torch.all((z==0)|(z==1)))
        z.sum().backward()
        self.assertGreater(model.net[0].weight.grad.abs().sum().item(),0)
        np.testing.assert_array_equal(model(x).detach(),model(x).detach())
        packed=pack_codes(z.detach().numpy(),5)
        np.testing.assert_array_equal(np.unpackbits(packed,axis=1,bitorder='little')[:,:5],z.detach().numpy())
        self.assertEqual(pack_codes(np.zeros((16,1)),0).shape,(16,0))

    def test_full_covariance_nll(self):
        torch.manual_seed(2);m=CodePredictor(3,8,2,4,'gaussian_nll')
        x,y=torch.randn(7,3),torch.randn(7,4);m.raw_cholesky.data.normal_(0,.1)
        pred=m.decoder(m.encoder(x))
        L=torch.tril(m.raw_cholesky,-1)+torch.diag(torch.nn.functional.softplus(torch.diag(m.raw_cholesky))+1e-3)
        expected=-torch.distributions.MultivariateNormal(pred,scale_tril=L).log_prob(y).mean()/4
        torch.testing.assert_close(m.loss(x,y),expected)

    def test_checkpoint_roundtrip(self):
        rng=np.random.default_rng(8)
        features={s:rng.normal(size=(20,3)).astype('float32') for s in ['train','valid','test']}
        for objective in ['task_bce','gaussian_nll','infonce']:
            targets={s:(x[:,:2]>0).astype('float32')[None] if objective=='task_bce' else x[None] for s,x in features.items()}
            with tempfile.TemporaryDirectory() as td:
                path=Path(td)/'model.pt'
                z,_=fit_codes(features,targets,3,objective,{'epochs':2,'hidden_dim':8,'batch_size':10,'device':'cpu'},3,path)
                encoder=load_encoder(path)
                np.testing.assert_array_equal(z['test'],encoder(torch.tensor(features['test'])).detach().numpy())

    def test_selection_ignores_test(self):
        rows=[]
        for b,av,bv in [(1,.1,.4),(4,.5,.1)]:
            for n,v in [('a',av),('b',bv)]:
                rows.append({'mode':'quantized','fit':'pretext','budget':b,'candidate':n,'select_ce_bits':v,'test_ce_bits':1-v})
        predicted={1:{'a':1,'b':2},4:{'a':2,'b':1}}
        result=select_policies(rows,predicted)['quantized']
        self.assertEqual([r['rate_selected'] for r in result],['a','b'])
        for r in rows:r['test_ce_bits']=100*np.random.random()
        again=select_policies(rows,predicted)['quantized']
        self.assertEqual([r['rate_selected'] for r in again],['a','b'])
        self.assertEqual(result[0]['fixed_selected'],again[0]['fixed_selected'])

    def test_validation_partition(self):
        d={'anchor':np.ones((20,3)),'indices':np.arange(20),'labels':np.zeros((20,4))}
        a,b=split_validation(d,1)
        self.assertFalse(set(a['indices'])&set(b['indices']))
        self.assertEqual(len(a['indices'])+len(b['indices']),20)

    def test_image_policies(self):
        config=json.loads(Path('configs/rate_selection_quick.json').read_text())
        image=Image.fromarray(np.random.default_rng(0).integers(0,256,(218,178,3),dtype=np.uint8))
        for policy in config['policies']:
            x=view_transform(policy)(image)
            self.assertEqual(tuple(x.shape),(3,224,224))
            self.assertTrue(torch.isfinite(x).all())


if __name__=='__main__':unittest.main()
