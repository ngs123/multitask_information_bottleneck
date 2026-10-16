"""One independently scheduled run. All test metrics use sampled channel outputs."""
import argparse
import copy
import hashlib
import json
import os
import platform
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import VIB, Renderer, Predictor, objective, excess_nll
from theory import predictions
from task_spec import configured_tasks
from data import make_data, to_batch


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    temp.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def loader(dataset, cfg, shuffle=False):
    return DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=shuffle,num_workers=0)


def standard_error(values, group_ids):
    # Cluster sandwich SE for a mean; identical-image copies form one cluster.
    values=np.asarray(values,dtype=np.float64)
    _,inv=np.unique(group_ids,return_inverse=True); G=inv.max()+1
    if G<2: return None
    totals=np.bincount(inv,weights=values-values.mean())
    return float(np.sqrt(G/(G-1)*np.sum(totals**2))/len(values))


def optimize_model(net, renderer, train_z, val_z, cfg, device, folder):
    optimizer = torch.optim.Adam(net.parameters(),lr=cfg['lr'])
    best, best_epoch, history = float('inf'), -1, []
    for epoch in range(cfg['epochs']):
        net.train()
        sums = np.zeros(3); count=0
        for batch in loader(train_z,cfg,True):
            x,z=to_batch(batch,renderer,device)
            optimizer.zero_grad(set_to_none=True)
            loss,d,kl = objective(net,x,z,cfg['M'],cfg['pretext'],cfg['beta'],cfg['recon_variance'],cfg.get('recon_loss','mse'),
                                  cfg.get('task_weights'),cfg.get('noise_variances'))
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training objective')
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),10); optimizer.step()
            sums += np.array([loss.item(),d.item(),kl.item()])*len(z); count+=len(z)
        net.eval(); vals=[]
        # Common MC validation draws across epochs; fork_rng restores training RNG state.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(cfg['seed']+20000)
            with torch.no_grad():
                for batch in loader(val_z,cfg):
                    x,z=to_batch(batch,renderer,device)
                    loss,_,_ = objective(net,x,z,cfg['M'],cfg['pretext'],cfg['beta'],cfg['recon_variance'],cfg.get('recon_loss','mse'),
                                  cfg.get('task_weights'),cfg.get('noise_variances'))
                    vals.append((loss.item(),len(z)))
        val=sum(v*n for v,n in vals)/len(val_z)
        history.append({'epoch':epoch,'train_loss':sums[0]/count,'train_distortion':sums[1]/count,
                        'train_kl_bits':sums[2]/count,'val_loss':val})
        if val < best:
            best=val; best_epoch=epoch
            torch.save({'model':net.state_dict(),'config':cfg,'epoch':epoch},folder/'best.pt')
        if epoch % 10 == 0: print(f'epoch={epoch} loss={sums[0]/count:.5g} val={val:.5g}',flush=True)
    state=torch.load(folder/'best.pt',map_location=device,weights_only=False)
    net.load_state_dict(state['model']); net.eval()
    atomic_json(folder/'history.json',history)
    return best_epoch


@torch.no_grad()
def extract(net, renderer, z, cfg, device, draws=1):
    codes=[]; targets=[]; kls=[]
    for _ in range(draws):
        for batch in loader(z,cfg):
            x,target=to_batch(batch,renderer,device)
            c,kl=net.encode(x)
            codes.append(c.cpu()); targets.append(target.cpu()); kls.append(kl.cpu())
    return torch.cat(codes),torch.cat(targets),torch.cat(kls)


def fit_probe(net, renderer, train_z, val_z, cfg, device):
    # Fresh samples each epoch; frozen stochastic encoder, no posterior-mean shortcut.
    probe=Predictor(cfg['width']).to(device)
    opt=torch.optim.Adam(probe.parameters(),lr=cfg['probe_lr'])
    cv,zv,_=extract(net,renderer,val_z,cfg,device,draws=cfg['eval_draws'])
    noise=torch.tensor(configured_tasks(cfg)[1],device=device)
    best=float('inf'); best_state=None
    for epoch in range(cfg['probe_epochs']):
        probe.train()
        for batch in loader(train_z,cfg,True):
            x,z=to_batch(batch,renderer,device)
            with torch.no_grad(): c,_=net.encode(x)
            mean,residual=probe(c)
            # Equal task weighting: evaluation probe capacity does not neglect rare task.
            loss=excess_nll(mean,residual,z,noise).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(),10); opt.step()
        probe.eval(); total=0.
        with torch.no_grad():
            for i in range(0,len(cv),cfg['batch_size']):
                c=cv[i:i+cfg['batch_size']].to(device); z=zv[i:i+cfg['batch_size']].to(device)
                mean,residual=probe(c)
                total+=excess_nll(mean,residual,z,noise).sum().item()
        if total<best:
            best=total; best_state=copy.deepcopy(probe.state_dict())
    probe.load_state_dict(best_state); probe.eval()
    return probe


@torch.no_grad()
def evaluate(net, probe, renderer, test_z, cfg, device, folder):
    draws=cfg['eval_draws']; n=len(test_z)
    c,z,kl=extract(net,renderer,test_z,cfg,device,draws)
    means=[]; nlls=[]
    noise=torch.tensor(configured_tasks(cfg)[1],device=device)
    for i in range(0,len(c),cfg['batch_size']):
        mean,residual=probe(c[i:i+cfg['batch_size']].to(device))
        nll=excess_nll(mean,residual,z[i:i+cfg['batch_size']].to(device),noise)
        means.append(mean.cpu()); nlls.append(nll.cpu())
    pred=torch.cat(means)
    # Average losses, not predictions: decoder gets one stochastic code per observation.
    sq=(pred-z).square().reshape(draws,n,2).mean(0).numpy()
    enll=torch.cat(nlls).reshape(draws,n,2).mean(0).numpy()
    rate=kl.reshape(draws,n).mean(0).numpy()
    mse=sq.mean(0); variance=z[:n].var(0,unbiased=False).numpy()
    r2=1-mse/variance
    effective=-.5*np.log2(np.maximum(mse,1e-30))
    weighted=enll @ np.array(configured_tasks(cfg)[0])
    group_ids=getattr(test_z,'group_ids',np.arange(n))
    np.savez_compressed(folder/'test_metrics.npz',squared_error=sq,excess_nll_bits=enll,kl_bits=rate,group_ids=group_ids)
    out={'rate_kl_bits':float(rate.mean()),'rate_kl_se':standard_error(rate,group_ids),
         'distortion_bits':float(weighted.mean()), 'distortion_se':standard_error(weighted,group_ids),
         'mse1':float(mse[0]),'mse2':float(mse[1]),'r2_1':float(r2[0]),'r2_2':float(r2[1]),
         'effective_r1':float(effective[0]),'effective_r2':float(effective[1]),
         'priority_difference':float(effective[1]-effective[0]),
         'excess_nll1':float(enll[:,0].mean()),'excess_nll2':float(enll[:,1].mean())}
    return out


def run(cfg,folder,device):
    torch.set_num_threads(cfg.get('cpu_threads',4))
    seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'config.json',cfg)
    folds,renderer,provenance=make_data(cfg,folder,device)
    tr,va,ptr,pva,te=[folds[k] for k in ('train','val','probe_train','probe_val','test')]
    net=VIB(cfg['image_size'],cfg['width']).to(device)
    best_epoch=optimize_model(net,renderer,tr,va,cfg,device,folder)
    seed_all(cfg['seed']+30000)
    probe=fit_probe(net,renderer,ptr,pva,cfg,device)
    torch.save(probe.state_dict(),folder/'probe.pt')
    seed_all(cfg['seed']+40000)
    metrics=evaluate(net,probe,renderer,te,cfg,device,folder)
    metrics.update({**{k:cfg[k] for k in ('M','pretext','beta','seed','lr')},'best_epoch':best_epoch,
                    'theory':predictions(cfg['M']) if not any(k in cfg for k in ('task_weights','noise_variances')) else None,'dataset':cfg.get('dataset','gaussian'),
                    'factor1':provenance['factors'][0],'factor2':provenance['factors'][1],
                    'gaussian_reference_is_exact':cfg.get('dataset','gaussian')=='gaussian' and not any(k in cfg for k in ('task_weights','noise_variances')),
                    **dict(zip(('task_weight1','task_weight2'), configured_tasks(cfg)[0])),
                    **dict(zip(('noise_variance1','noise_variance2'), configured_tasks(cfg)[1])),
                    'data_provenance':provenance,'torch_version':torch.__version__,
                    'python_version':platform.python_version(),'device':str(device)})
    atomic_json(folder/'result.json',metrics) # Completion sentinel, written last.
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda'); args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text()); run(cfg,Path(args.out),torch.device(args.device))
