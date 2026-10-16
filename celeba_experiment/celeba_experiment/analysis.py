"""Rate sweeps and held-out, actual binary-log-loss checks."""
import json
from pathlib import Path
import numpy as np
from gaussian_rate.nd_solvers import GaussianFamily,solve,single_task,channel_matrix
from .probes import fit_probes,evaluate,whiten_fit


def dump(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def rate_sweep(tasks,cross,config,out):
    cfg=config['gaussian'];out=Path(out)
    which='H_ce' if cfg['task_noise']=='ce_gain' else 'H_moment'
    family=GaussianFamily(tasks[which]);rows=[];arrays={};warm=None
    for i,R in enumerate(cfg['rates']):
        solution=solve(family,R,warm_S=warm,tol=cfg['gap_tolerance'],maxiter=cfg['max_iterations'])
        warm=solution['S'];arrays[f'optimal_K_{i}']=solution['K']
        row={k:solution[k] for k in ['D','iterations','converged','gap_bound','kkt_residual','rate','lambda']}
        row['R']=R;row['pretexts']={}
        for name,fit in cross.items():
            K=single_task(fit['H'],R);arrays[f'{name}_K_{i}']=K
            row['pretexts'][name]=family.value(K)
        row['isotropic']=family.value(np.eye(family.d)*2**(-2*R/family.d))
        rows.append(row)
        print(f'  R={R:g}: D={row["D"]:.5f}, gap<={row["gap_bound"]:.2g}, iterations={row["iterations"]}',flush=True)
    np.savez_compressed(out/'gaussian_channels.npz',**arrays)
    dump(out/'rate_distortion.json',{'units':'bits per image; distortion bits per task',
        'objective':'Gaussian conditional KL excess above full-feature Bayes prediction',
        'task_model':cfg['task_noise'],'binary_ce_floor_proxy':float(tasks['ce_floor_bits'].mean()) if which=='H_ce' else None,
        'rows':rows})
    support_cfg=cfg.get('support_diagnostics',{})
    if support_cfg.get('enabled',True):
        from gaussian_rate.support import write_support_diagnostics
        write_support_diagnostics([r['R'] for r in rows],
            [arrays[f'optimal_K_{i}'] for i in range(len(rows))], [r['D'] for r in rows],
            out/'support_diagnostics',support_cfg.get('active_rate_tol',1e-5),
            title='Task-aware Gaussian representations' if all(r['converged'] for r in rows) else 'Task-aware Gaussian solver outputs (some unconverged)')
    return rows,arrays


def evaluate_channels(features,labels,rows,arrays,own_policy,config,out):
    out=Path(out);results=[];cfg=config['gaussian']
    for i,row in enumerate(rows):
        if row['R'] not in cfg.get('random_channel_eval_rates',[]):continue
        for name in ['optimal',own_policy]:
            A=channel_matrix(arrays[f'{name}_K_{i}']);encoded={};ys={}
            for j,split in enumerate(['train','valid','test']):
                rng=np.random.default_rng(config['seed']+10000+100*i+j)
                draws=cfg.get('noise_draws',3)
                encoded[split]=np.tile(features[split]@A.T,(draws,1))+rng.normal(size=(draws*len(features[split]),A.shape[0]))
                ys[split]=np.tile(labels[split],(draws,1))
            # Choose regularization on the original sample IDs, not replicas;
            # all draws of an image go to the same internal partition.
            model=fit_channel_probes(encoded['train'],ys['train'],len(features['train']),config)
            np.savez_compressed(out/f'channel_probe_{name}_R{row["R"]:g}.npz',**model,A=A)
            metrics=evaluate(model,encoded['test'],ys['test'])
            results.append({'R':row['R'],'channel':name,**metrics})
    dump(out/'binary_channel_evaluation.json',{'note':'Actual stochastic channels on non-Gaussian features. Nominal rate is Gaussian-model mutual information, not an actual code length. Monte Carlo draws are not independent images.',
        'results':results})
    return results


def fit_channel_probes(x,y,n_images,config):
    # Tune on one independent noise draw to avoid replica leakage in the
    # standard train-only split; then refit at that C on all training draws.
    from .probes import one_fit
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);scaled=(x-mean)/scale
    model=fit_probes(scaled[:n_images],y[:n_images],config['probes'],config['seed'])
    weights=[];bias=[]
    for t,C in enumerate(model['C']):
        w,b=one_fit(scaled,y[:,t],C,config['probes']['max_iter'])
        weights.append(w/scale);bias.append(b-(w/scale)@mean)
    return {'weights':np.array(weights),'bias':np.array(bias),'C':model['C']}
