"""Resampling and paper diagnostics for run_rate_selection.

All resampling is paired across objectives and rates. Moment fitting resamples
original training images (cached views remain grouped); evaluation resamples
original test images after averaging stochastic-channel losses within image.
"""
import csv
import json
from pathlib import Path
import numpy as np
from .analysis import dump
from .rate_models import fit_pca, transform, fit_gaussian_tasks, task_channel, compact_channel
from .fitting import fit_cross_view
from .pretexts import components, build_candidate, proxy_channel
from .robust_statistics import (percentile, paired_bootstrap, crossover_summary,
    rank_statistics, distortion_tasks, specialist_geometry, allocation)


def settings(config):
    return {'enabled': False, 'moment_resamples': 200, 'test_resamples': 2000,
            'confidence': .95, 'seed': config['seed']+93001,
            'specialist': None, 'generalist': 'reconstruction',
            'mlp': {'enabled': False, 'hidden_dim':64, 'epochs':100, 'patience':10,
                    'alpha':1e-4, 'learning_rate':1e-3},
            **config.get('robustness', {})}


def validate(config):
    c = settings(config)
    if not c['enabled']: return
    if 'pretext' not in config.get('fit_modes',['task','pretext']):
        raise ValueError('Robustness reports require fit_modes to include pretext')
    bank = {p['name'] for p in config['pretexts']}
    for key in ('specialist','generalist'):
        if c[key] not in bank:
            raise ValueError(f'robustness.{key} must explicitly name a configured pretext')
    if c['specialist'] == c['generalist']:
        raise ValueError('Specialist and generalist must differ')
    if not 0<c['confidence']<1:
        raise ValueError('robustness.confidence must be between zero and one')
    for key in ('moment_resamples','test_resamples'):
        if type(c[key]) is not int or c[key]<0:
            raise ValueError(f'robustness.{key} must be a nonnegative integer')
    if not c['test_resamples']:
        raise ValueError('Need positive test_resamples for paired intervals')
    if c['mlp'].get('enabled',False):
        for key,default in [('epochs',100),('patience',10),('hidden_dim',64)]:
            if type(c['mlp'].get(key,default)) is not int or c['mlp'].get(key,default)<1:
                raise ValueError(f'robustness.mlp.{key} must be a positive integer')


def csv_write(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows(rows)


def fit_bootstrap(raw_train, manifest, ids, config, sample, directory):
    """Refit the full TRAIN-only moment pipeline, including every PCA fit."""
    raw={'train':{k:np.asarray(v)[sample] for k,v in raw_train.items()}}
    cfg={**config, '_proxy_groups':np.asarray(raw_train['indices'])[sample]}
    fit=fit_pca(raw['train']['anchor'], config.get('input_dim',32))
    features={'train':transform(raw['train']['anchor'],fit)}
    np.savez_compressed(directory/'anchor_transform.npz', **fit)
    tasks=fit_gaussian_tasks(features['train'], raw['train']['labels'][:,ids],
                            seed=config['seed'],groups=cfg['_proxy_groups'],**config.get('task_proxy',{}))
    np.savez_compressed(directory/'gaussian_tasks.npz',**tasks)
    targets, proxies={},{}
    for spec in config['pretexts']:
        for part in components(spec):
            if part.get('objective')=='attribute_bce':continue
            target=part['target']
            if target in targets:continue
            if target=='anchor': keys=['anchor']
            else:
                j=[p['name'] for p in config['policies']].index(target)
                keys=sorted(k for k in raw['train'] if k.startswith(f'target_{j}_'))
            target_fit=fit_pca(np.concatenate([raw['train'][k] for k in keys]),config.get('target_dim',32))
            targets[target]={'train':np.stack([transform(raw['train'][k],target_fit) for k in keys])}
            proxies[target]=fit_cross_view(features['train'],list(targets[target]['train']),config['gaussian']['rho_max'])
    bank={c['name']:build_candidate(c,raw,features,manifest['attribute_names'],targets,proxies,cfg,directory)
          for c in config['pretexts']}
    avg=tasks['H'].mean(0)
    bank['average_task']={'H':avg[None], 'weights':np.ones(1)}
    return tasks['H'],bank


def model_curves(H, bank, rates, config):
    names=list(bank)+['task_aware']
    curves=np.empty((len(names),len(rates),len(H))); channels={}
    for i,name in enumerate(names):
        for j,r in enumerate(rates):
            if name=='task_aware':
                K,cert=task_channel(H,r,config['gaussian'].get('gap_tolerance',1e-5),
                                    config['gaussian'].get('max_iterations',3000))
                if not cert['converged']:
                    raise RuntimeError(f'Unconverged oracle at rate {r}: {cert}')
            else: K,_=proxy_channel(bank[name],r,{**config['gaussian'],'allow_unconverged':False})
            curves[i,j]=distortion_tasks(H,K)
            channels[name,float(r)]=K
    return names,curves,channels


def moment_bootstrap(raw,manifest,ids,config,out,rates,names,point):
    import hashlib
    import warnings
    c=settings(config);folder=out/'robustness'/'moment_bootstrap';folder.mkdir(parents=True,exist_ok=True)
    # A cache identity protects resumption against changed data or fitting settings.
    identity = {k:config.get(k) for k in ('seed','input_dim','target_dim','task_proxy','pretexts','policies')}
    identity.update(bootstrap_seed=c['seed'],specialist=c['specialist'],ids=list(ids),
                    attributes=manifest['attribute_names'],rates=list(map(float,rates)),names=names,
                    rho_max=config['gaussian']['rho_max'],gap_tolerance=config['gaussian'].get('gap_tolerance',1e-5))
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode())
    for key,value in sorted(raw['train'].items()):
        arr=np.ascontiguousarray(value)
        digest.update(key.encode());digest.update(str((arr.shape,arr.dtype)).encode());digest.update(arr.tobytes())
    signature=digest.hexdigest();identity_path=folder/'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text())['sha256']!=signature:
            raise ValueError('Bootstrap cache data/fitting settings differ. Use a new report directory.')
    elif any(folder.glob('resample_*/curves.npz')) and not config.get('_allow_legacy_bootstrap',False):
        raise ValueError('Legacy bootstrap cache: use resume_rate_selection_report.py to validate and migrate it.')
    draws=[]; failures=[]; geometry=[]; successful_ids=[]; reused=0; attempted=[]
    previous=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
    previous_failures={f['resample']:f['error'] for f in previous.get('failures',[])}
    for b in range(c['moment_resamples']):
        directory=folder/f'resample_{b:04d}';directory.mkdir(exist_ok=True)
        rng=np.random.default_rng(np.random.SeedSequence([c['seed'],b]))
        ii=rng.integers(len(raw['train']['anchor']),size=len(raw['train']['anchor']))
        index_path=directory/'sample_indices.npy'
        if index_path.exists() and not np.array_equal(np.load(index_path),ii):
            raise ValueError(f'Bootstrap sample indices changed for resample {b}; refusing reuse.')
        np.save(index_path,ii)
        try:
            proxy_paths=[directory/f'pretext_proxy_{spec["name"]}.npz' for spec in config['pretexts']]
            if (directory/'gaussian_tasks.npz').exists() and all(p.exists() for p in proxy_paths):
                with np.load(directory/'gaussian_tasks.npz') as f:H=f['H']
                bank={}
                for spec,path in zip(config['pretexts'],proxy_paths):
                    with np.load(path) as f:bank[spec['name']]={'H':f['H'],'weights':f['weights']}
                bank['average_task']={'H':H.mean(0)[None],'weights':np.ones(1)}
            else:
                H,bank=fit_bootstrap(raw['train'],manifest,ids,config,ii,directory)
            curve_path=directory/'curves.npz'
            if curve_path.exists():
                with np.load(curve_path) as f:
                    if list(f['names'])!=names or not np.array_equal(f['rates'],rates):
                        raise ValueError(f'Cached axes differ for resample {b}')
                    curve=f['distortion']
                if curve.shape!=point.shape or not np.isfinite(curve).all():
                    raise ValueError(f'Invalid cached curve for resample {b}')
                reused+=1
            else:
                # Retry precisely the same fitted problem with a larger iteration
                # allowance. Neither the bootstrap sample nor gap tolerance changes.
                base=config['gaussian'].get('max_iterations',3000)
                initial=c.get('bootstrap_max_iterations',base)
                limits=[initial,initial*4,initial*16]
                attempts=[]
                for limit in limits:
                    solve_config={**config,'gaussian':{**config['gaussian'],'max_iterations':limit}}
                    try:
                        nn,curve,_=model_curves(H,bank,rates,solve_config)
                        if nn!=names:raise ValueError('Bootstrap candidate order changed')
                        attempts.append({'max_iterations':limit,'status':'converged'})
                        break
                    except RuntimeError as error:
                        attempts.append({'max_iterations':limit,'status':'failed','error':str(error)})
                        if limit==limits[-1]:raise
                    finally:
                        dump(directory/'solver_attempts.json',{'previous_error':previous_failures.get(b),'attempts':attempts})
                tmp=directory/'curves.tmp.npz'
                np.savez_compressed(tmp,rates=rates,names=names,distortion=curve)
                tmp.replace(curve_path);attempted.append(b)
            _,_,geo=specialist_geometry(H,bank[c['specialist']])
            geometry.append({'resample':b,**geo});draws.append(curve);successful_ids.append(b)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            failures.append({'resample':b,'error':str(error)})
        print(f'Moment bootstrap {b+1}/{c["moment_resamples"]}; successful={len(draws)}, reused={reused}',flush=True)
        dump(folder/'status.json',{'requested':c['moment_resamples'],'successful':len(draws),
             'reused':reused,'recomputed_resamples':attempted,'failures':failures,'bands_available':not failures and len(draws)==c['moment_resamples']})
    dump(identity_path,{'sha256':signature,'settings':identity})
    dump(folder/'geometry.json',geometry)
    result=np.asarray(draws) if draws else np.empty((0,)+point.shape)
    np.savez_compressed(folder/'all_curves.npz',rates=rates,names=names,point=point,draws=result,
                        resample_ids=successful_ids,complete=not failures)
    if failures:
        warnings.warn(f'{len(failures)} bootstrap resamples still failed. Saved all successful curves; '
                      'continuing tables and evaluation diagnostics WITHOUT moment-bootstrap intervals. '
                      f'See {folder}/status.json and rerun the report to retry these samples.')
        return np.empty((0,)+point.shape)
    return result


def paired_loss_cube(rows,out,candidates=None,decoder='linear'):
    """Require a complete paired design; never fabricate absent run cells."""
    candidates=candidates or sorted({r['candidate'] for r in rows})
    rates=sorted({r['budget'] for r in rows})
    seeds=sorted({r['seed'] for r in rows})
    lookup={(r['candidate'],r['budget'],r['seed']):r for r in rows}
    blocks=[]
    for seed in seeds:
        cc=[]
        for name in candidates:
            rr=[]
            for rate in rates:
                key=(name,rate,seed)
                if key not in lookup:
                    raise ValueError(f'Incomplete paired experiment: missing {key}')
                p=out/lookup[key]['directory']/('per_image_losses.npz' if decoder=='linear' else 'mlp_per_image_losses.npz')
                with np.load(p) as f:rr.append(f['test_ce_bits'])
            cc.append(rr)
        blocks.append(cc)
    return np.asarray(blocks),candidates,np.array(rates),seeds


def plot_comparison(folder,stem,rates,names,point,draws,c,task_names,floor=None):
    import matplotlib.pyplot as plt
    # point [candidate,rate,task], draws [replicate,candidate,rate,task]
    a,b=names.index(c['specialist']),names.index(c['generalist'])
    means=point.mean(-1);samples=draws.mean(-1)
    difference=means[b]-means[a];diff_samples=samples[:,b]-samples[:,a]
    cross=crossover_summary(rates,difference,diff_samples,c['confidence'])
    dump(folder/f'{stem}_crossover.json',cross)
    fig,axes=plt.subplots(2,1,figsize=(7,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for idx,color in [(a,'tab:blue'),(b,'tab:orange')]:
        axes[0].plot(rates,means[idx],color=color,label=names[idx])
        if len(draws):
            lo,hi=percentile(samples[:,idx],c['confidence']);axes[0].fill_between(rates,lo,hi,color=color,alpha=.2)
    if floor is not None:axes[0].axhline(floor,ls='--',color='tab:blue',label='Gaussian specialist limit D_miss')
    axes[0].set_ylabel('Excess log loss (bits/task)');axes[0].legend(fontsize=8)
    axes[1].plot(rates,difference,color='purple');axes[1].axhline(0,color='black',lw=.8)
    if len(draws):
        lo,hi=percentile(diff_samples,c['confidence']);axes[1].fill_between(rates,lo,hi,color='purple',alpha=.2)
    if cross['rate'] is not None:axes[1].axvline(cross['rate'],ls=':',color='black')
    ci=cross['conditional_percentile_ci']
    if ci is not None:axes[1].axvspan(*ci,color='gray',alpha=.15)
    ci_text='no interior crossing' if cross['rate'] is None else f'crossing = {cross["rate"]:.3g}'
    if ci is not None:ci_text+=f'; conditional CI [{ci[0]:.3g}, {ci[1]:.3g}]; {cross["interior_fraction"]:.0%} crossed'
    axes[1].set_title(ci_text,fontsize=9);axes[1].set_ylabel('Generalist − specialist')
    axes[1].set_xlabel('Hard code budget B (bits/image)' if 'quantized' in stem else 'Nominal Gaussian rate (bits/image)')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/f'{stem}_comparison.png',dpi=180);fig.savefig(folder/f'{stem}_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,len(task_names),figsize=(4*len(task_names),3.5),squeeze=False)
    for t,name in enumerate(task_names):
        ax=axes[0,t]
        for idx,color in [(a,'tab:blue'),(b,'tab:orange')]:
            ax.plot(rates,point[idx,:,t],color=color,label=names[idx])
            if len(draws):
                lo,hi=percentile(draws[:,idx,:,t],c['confidence']);ax.fill_between(rates,lo,hi,color=color,alpha=.2)
        ax.set_title(name);ax.set_xlabel('Budget (bits/image)');ax.set_ylabel('Excess log loss (bits)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(folder/f'{stem}_per_task.png',dpi=180);plt.close(fig)
    records=[]
    for i,name in enumerate(names):
        for j,r in enumerate(rates):
            interval=percentile(samples[:,i,j],c['confidence']) if len(draws) else [None,None]
            records.append({'candidate':name,'rate':float(r),'excess_bits':means[i,j],'ci_low':interval[0],'ci_high':interval[1]})
    csv_write(folder/f'{stem}_curves.csv',records)
    intervals=percentile(diff_samples,c['confidence']) if len(draws) else np.full((2,len(rates)),np.nan)
    csv_write(folder/f'{stem}_difference.csv',[
        {'rate':float(r),'generalist_minus_specialist':float(d),'ci_low':float(lo),'ci_high':float(hi)}
        for r,d,lo,hi in zip(rates,difference,*intervals)])
    return cross


def rank_report(folder,stem,rates,names,values):
    import matplotlib.pyplot as plt
    ranks,stats=rank_statistics(values,rates)
    fig,ax=plt.subplots(figsize=(max(6,len(rates)*.55),max(3,len(names)*.35)))
    image=ax.imshow(ranks,aspect='auto',vmin=1,vmax=max(2,len(names)),cmap='viridis_r')
    ax.set_yticks(range(len(names)),names);ax.set_xticks(range(len(rates)),[f'{r:g}' for r in rates])
    ax.set_xlabel('Budget (bits/image)');ax.set_title(stem.replace('_',' '))
    fig.colorbar(image,ax=ax,label='Rank (average ranks for ties)');fig.tight_layout()
    fig.savefig(folder/f'{stem}_ranks.png',dpi=180);plt.close(fig)
    csv_write(folder/f'{stem}_ranks.csv',[{'candidate':n,**{str(r):float(v) for r,v in zip(rates,row)}} for n,row in zip(names,ranks)])
    return {'panel':stem,**stats}


def fit_mlp_losses(encoded,labels,config,seed,directory,stochastic=False):
    """One-hidden-layer probes with validation-NLL early stopping, per task.

    Validation uses original held-out images, never a random split of noise
    draws. All pretexts share architecture, stopping rule and hyperparameters.
    """
    from sklearn.neural_network import MLPClassifier
    import copy
    from scipy.special import expit
    c=settings(config)['mlp'];epochs=c.get('epochs',100)
    metric_path=directory/'mlp_metrics.json'
    if metric_path.exists() and (directory/'mlp_probes.npz').exists() and (directory/'mlp_per_image_losses.npz').exists():
        saved=json.loads(metric_path.read_text())
        if saved.get('config')==c and saved.get('seed')==seed:return
    mean=encoded['train'].mean(0);scale=encoded['train'].std(0);scale=np.maximum(scale,1e-6)
    x={s:(v-mean)/scale for s,v in encoded.items()}
    result={s:np.empty_like(labels[s],dtype=float) for s in ('test','select')};params={};chosen=[]
    for t in range(labels['train'].shape[1]):
        model=MLPClassifier(hidden_layer_sizes=(c.get('hidden_dim',64),),activation='relu',
            alpha=c.get('alpha',1e-4),learning_rate_init=c.get('learning_rate',1e-3),
            batch_size=min(256,len(x['train'])),random_state=seed+t,max_iter=1)
        best=None;best_value=float('inf');stale=0
        for epoch in range(epochs):
            model.partial_fit(x['train'],labels['train'][:,t],classes=np.array([0,1]))
            prob=np.clip(model.predict_proba(x['valid'])[:,1],1e-12,1-1e-12)
            y=labels['valid'][:,t];score=float(np.mean(-y*np.log(prob)-(1-y)*np.log1p(-prob)))
            if score<best_value-1e-8:
                best=copy.deepcopy(model);best_value=score;stale=0;best_epoch=epoch+1
            else:stale+=1
            if stale>=c.get('patience',10):break
        chosen.append(best_epoch)
        for layer,(w,b) in enumerate(zip(best.coefs_,best.intercepts_)):
            params[f'task_{t}_W_{layer}']=w;params[f'task_{t}_b_{layer}']=b
        for s in result:
            p=np.clip(best.predict_proba(x[s])[:,1],1e-12,1-1e-12);y=labels[s][:,t]
            result[s][:,t]=(-y*np.log(p)-(1-y)*np.log1p(-p))/np.log(2)
    if stochastic:
        result={s:v.reshape(config['gaussian']['noise_draws'],-1,v.shape[-1]).mean(0) for s,v in result.items()}
    np.savez_compressed(directory/'mlp_per_image_losses.npz',**{s+'_ce_bits':v for s,v in result.items()})
    np.savez_compressed(directory/'mlp_probes.npz',mean=mean,scale=scale,**params)
    dump(directory/'mlp_metrics.json',{'epochs':chosen,'config':c,'seed':seed,
                                      'test_ce_bits':result['test'].mean(0).tolist()})


def backfill_mlp(rows,out,features,labels,config):
    """Reuse saved channels/hard codes; no bottleneck retraining required."""
    if not settings(config)['mlp'].get('enabled',False):return
    path=out/'uncompressed'
    fit_mlp_losses(features,labels,config,config['seed'],path)
    for row in rows:
        path=out/row['directory'];encoded={}
        if row['mode']=='gaussian':
            with np.load(path/'channel.npz') as f:A=f['A']
            draws=config['gaussian']['noise_draws'];yy={}
            for j,s in enumerate(['train','valid','select','test']):
                rng=np.random.default_rng(row['seed']+10000+j)
                encoded[s]=np.tile(features[s]@A.T,(draws,1))
                if np.any(A):encoded[s]+=rng.normal(size=encoded[s].shape)
                yy[s]=np.tile(labels[s],(draws,1))
        else:
            with np.load(path/'hard_codes.npz') as f:
                bits=int(f['bits'])
                for s in features:
                    # pack_codes uses little endian bit order.
                    encoded[s]=np.unpackbits(f[s],axis=1,count=bits,bitorder='little').astype(float) if bits else np.zeros((len(features[s]),1))
            yy=labels
        print(f'MLP probe: {row["mode"]} {row["candidate"]} {row["budget"]} seed={row["seed"]}',flush=True)
        fit_mlp_losses(encoded,yy,config,row['seed'],path,row['mode']=='gaussian')


def diagnostics(rows,out,features,H,channels,Q,Qout,config):
    import matplotlib.pyplot as plt
    folder=out/'robustness';allocation_rows=[];rate_rows=[]
    for (name,r),K in channels.items():
        allocation_rows.append({'candidate':name,'budget':r,**allocation(K,Q,Qout)})
    csv_write(folder/'allocation.csv',allocation_rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in sorted({r['candidate'] for r in allocation_rows}):
        rr=sorted([r for r in allocation_rows if r['candidate']==name],key=lambda r:r['budget'])
        axes[0].plot([r['budget'] for r in rr],[r['inside_marginal'] for r in rr],label=name)
        axes[1].plot([r['budget'] for r in rr],[r['outside_conditional'] for r in rr],label=name)
    axes[0].set_title('I(X_S; Z)');axes[1].set_title('I(X_perp; Z | X_S)')
    for ax in axes:ax.set_xlabel('Nominal rate (bits/image)');ax.set_ylabel('Allocated information (bits)');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(folder/'allocation.png',dpi=180);plt.close(fig)
    covs={s:np.cov(x,rowvar=False,bias=True) for s,x in features.items()}
    dump(folder/'whitening.json',{s:{'dimension':len(cov),'n_images':len(features[s]),
         'eigenvalue_min':float(np.linalg.eigvalsh(cov).min()),'eigenvalue_max':float(np.linalg.eigvalsh(cov).max()),
         'relative_frobenius_error_to_identity':float(np.linalg.norm(cov-np.eye(len(cov)))/np.sqrt(len(cov)))}
         for s,cov in covs.items()})
    for row in rows:
        directory=out/row['directory']
        if row['mode']=='gaussian':
            with np.load(directory/'channel.npz') as f:A=f['A'];K=f['K']
            rate_rows.append({'mode':'gaussian','candidate':row['candidate'],'budget':row['budget'],'seed':row['seed'],
                'nominal_rate':float(-np.linalg.slogdet(K)[1]/(2*np.log(2))),
                **{s+'_covariance_gaussian_mi_bound':float(np.linalg.slogdet(np.eye(len(A))+A@cov@A.T)[1]/(2*np.log(2))) for s,cov in covs.items()}})
        else:
            with np.load(directory/'hard_codes.npz') as f:
                z=f['test'];bits=int(f['bits'])
                _,counts=np.unique(z,axis=0,return_counts=True)
                p=counts/counts.sum();h=float(-np.sum(p*np.log2(p)))
                record={'mode':'quantized','candidate':row['candidate'],'budget':bits,'seed':row['seed'],
                        'test_plugin_entropy_bits':h,'sample_entropy_ceiling':float(np.log2(len(z))),
                        'n_test':len(z),'unique_codes':len(counts),'singleton_observation_fraction':float(np.sum(counts==1)/len(z))}
                rate_rows.append(record)
                np.savez_compressed(directory/'code_usage.npz',sorted_counts=np.sort(counts)[::-1],
                    bit_one_fraction=np.unpackbits(z,axis=1,count=bits,bitorder='little').mean(0) if bits else np.empty(0))
                fig,ax=plt.subplots(figsize=(5,3));ax.hist(counts,bins=min(30,max(1,len(np.unique(counts)))))
                ax.set(xlabel='Test observations per observed codeword',ylabel='Number of codewords',title=f'{row["candidate"]}: B={bits}, seed={row["seed"]}')
                fig.tight_layout();fig.savefig(directory/'code_usage.png',dpi=120);plt.close(fig)
    csv_write(folder/'rate_diagnostics.csv',rate_rows)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for mode,ax in zip(('gaussian','quantized'),axes):
        for name in sorted({r['candidate'] for r in rate_rows if r['mode']==mode}):
            rr=[r for r in rate_rows if r['mode']==mode and r['candidate']==name]
            key='test_plugin_entropy_bits' if mode=='quantized' else 'train_covariance_gaussian_mi_bound'
            ax.scatter([r['budget'] for r in rr],[r[key] for r in rr],s=12,label=name)
        ax.set_xlabel('Nominal budget (bits/image)')
        if ax.collections:ax.legend(fontsize=6)
    axes[0].set_ylabel('Fitted-covariance Gaussian MI bound (bits)')
    axes[1].set_ylabel('Test plug-in code entropy (bits; biased)')
    axes[1].set_title('Not a consistent achieved-rate estimate at large B',fontsize=9)
    fig.tight_layout();fig.savefig(folder/'rate_diagnostics.png',dpi=180);plt.close(fig)


def report(raw,manifest,ids,features,labels,tasks,bank,rows,config,out):
    c=settings(config)
    if not c['enabled']:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'robustness';folder.mkdir(exist_ok=True)
    rates=np.array(sorted(set(config['gaussian']['rates']) | {r['budget'] for r in rows if r['mode']=='gaussian'}))
    names,point,channels=model_curves(tasks['H'],bank,rates,config)
    task_names=[manifest['attribute_names'][i] for i in ids]
    Q,Qout,geo=specialist_geometry(tasks['H'],bank[c['specialist']])
    generalist=bank[c['generalist']]['H'];d=tasks['H'].shape[-1]
    geo['generalist_components_isotropic_full_space']=bool(all(
        np.linalg.norm(h-np.trace(h)/d*np.eye(d))<=1e-8*max(1.,np.linalg.norm(h)) for h in generalist))
    dump(folder/'fitted_quantities.json',geo)
    from .report_tables import write_tables
    # Write point-estimate tables BEFORE any resampling or optional probe fitting.
    initial_quantities=[{'quantity':'SNR '+t,'value':v} for t,v in zip(task_names,geo['task_snr'])]
    initial_quantities += [{'quantity':key,'value':geo[key]} for key in
        ('specialist_rank','delta_uniform_subspace','D_miss_bits',
         'isotropic_generalist_sufficient_rate_bound','specialist_components_isotropic_on_support',
         'generalist_components_isotropic_full_space')]
    csv_write(folder/'fitted_quantities.csv',initial_quantities)
    bank_names=[x['name'] for x in config['pretexts']]
    initial_ranks=[]
    _,rank=rank_statistics(point[[names.index(n) for n in bank_names]].mean(-1),rates)
    initial_ranks.append({'panel':'gaussian_proxy',**rank})
    for mode in sorted({r['mode'] for r in rows}):
        rr=[r for r in rows if r['mode']==mode]
        cube,nn,rrates,_=paired_loss_cube(rr,out)
        available=[n for n in bank_names if n in nn]
        _,rank=rank_statistics(cube.mean((0,3,4))[[nn.index(n) for n in available]],rrates)
        initial_ranks.append({'panel':mode+'_measured_nll',**rank})
    csv_write(folder/'rank_instability.csv',initial_ranks)
    write_tables(folder,announce=True)
    draws=moment_bootstrap(raw,manifest,ids,config,out,rates,names,point)
    cross=plot_comparison(folder,'gaussian_proxy_moment',rates,names,point,draws,c,task_names,geo['D_miss_bits'])
    quantities=[{'quantity':'SNR '+t,'value':v} for t,v in zip(task_names,geo['task_snr'])]
    quantities += [{'quantity':key,'value':geo[key]} for key in ('specialist_rank','delta_uniform_subspace','D_miss_bits','isotropic_generalist_sufficient_rate_bound','specialist_components_isotropic_on_support','generalist_components_isotropic_full_space')]
    quantities += [{'quantity':'observed_proxy_crossover','value':cross['rate']},
                   {'quantity':'crossover_conditional_ci','value':json.dumps(cross['conditional_percentile_ci'])},
                   {'quantity':'bootstrap_interior_fraction','value':cross['interior_fraction']}]
    csv_write(folder/'fitted_quantities.csv',quantities)
    write_tables(folder)
    if len(draws):
        geometry=json.loads((folder/'moment_bootstrap'/'geometry.json').read_text())
        intervals={key:percentile([g[key] for g in geometry],c['confidence']).tolist()
                   for key in ['delta_uniform_subspace','D_miss_bits','isotropic_generalist_sufficient_rate_bound']
                   if all(g[key] is not None for g in geometry)}
        intervals['task_snr']=percentile([g['task_snr'] for g in geometry],c['confidence']).tolist()
        dump(folder/'fitted_quantities_intervals.json',intervals)
    else:
        (folder/'fitted_quantities_intervals.json').unlink(missing_ok=True)
    # Rank census is confined to the configured bank, excluding oracle/references.
    bank_names=[x['name'] for x in config['pretexts']]
    rank_rows=[rank_report(folder,'gaussian_proxy',rates,bank_names,point[[names.index(n) for n in bank_names]].mean(-1))]
    backfill_mlp(rows,out,features,labels,config)
    captions=[];fidelity=[]
    for mode in sorted({r['mode'] for r in rows}):
        rr=[r for r in rows if r['mode']==mode]
        cube,nn,rrates,seeds=paired_loss_cube(rr,out)
        with np.load(out/'uncompressed'/'per_image_losses.npz') as f:baseline=f['test_ce_bits']
        # Paired subtraction also resamples the baseline's per-image test loss.
        excess=cube-baseline[None,None,None,:,:]
        center=excess.mean((0,3))
        eval_draws=paired_bootstrap(excess,c['test_resamples'],c['seed']+1,False)
        plot_comparison(folder,f'{mode}_nll_test',rrates,nn,center,eval_draws,c,task_names)
        np.savez_compressed(folder/f'{mode}_nll_test_bootstrap.npz',names=nn,rates=rrates,point=center,draws=eval_draws)
        captions.append(f'{mode} measured NLL: {100*c["confidence"]:g}% pointwise percentile intervals over {c["test_resamples"]} paired test-image resamples, conditional on {len(seeds)} fitted seed runs. Excess is relative to the fitted uncompressed linear probe, not a known Bayes loss.')
        if mode=='quantized':
            joint=paired_bootstrap(excess,c['test_resamples'],c['seed']+2,True)
            plot_comparison(folder,'quantized_nll_seeds_test',rrates,nn,center,joint,c,task_names)
            np.savez_compressed(folder/'quantized_nll_seeds_test_bootstrap.npz',names=nn,rates=rrates,point=center,draws=joint)
            captions.append(f'Learned-code joint bands: crossed resampling of {len(seeds)} training seeds and original test images. A seed varies both bottleneck training and probe fitting; these effects are not separately identified. With one seed there is no estimate of training-seed uncertainty.')
        available=[n for n in bank_names if n in nn]
        rank_rows.append(rank_report(folder,f'{mode}_measured_nll',rrates,available,center[[nn.index(n) for n in available]].mean(-1)))
        if mode=='gaussian':
            for i,n in enumerate(nn):
                for j,r in enumerate(rrates):
                    K=channels[n,float(r)]
                    fidelity.append({'candidate':n,'rate':float(r),'proxy_excess_bits':float(distortion_tasks(tasks['H'],K).mean()),'measured_excess_bits':float(center[i,j].mean())})
        if c['mlp'].get('enabled',False):
            other,_,_,_=paired_loss_cube(rr,out,nn,'mlp')
            with np.load(out/'uncompressed'/'mlp_per_image_losses.npz') as f:mlp_base=f['test_ce_bits']
            other=other-mlp_base[None,None,None,:,:]
            mlp_center=other.mean((0,3));mlp_draws=paired_bootstrap(other,c['test_resamples'],c['seed']+1,False)
            plot_comparison(folder,f'{mode}_mlp_test',rrates,nn,mlp_center,mlp_draws,c,task_names)
            a,b=nn.index(c['specialist']),nn.index(c['generalist'])
            fig,ax=plt.subplots(figsize=(6,4));ax.axhline(0,color='black',lw=.8)
            for title,p,s in [('Linear',center,eval_draws),('One-hidden-layer MLP',mlp_center,mlp_draws)]:
                ax.plot(rrates,(p[b]-p[a]).mean(-1),label=title)
                lo,hi=percentile((s[:,b]-s[:,a]).mean(-1),c['confidence']);ax.fill_between(rrates,lo,hi,alpha=.15)
            ax.set(xlabel='Budget (bits/image)',ylabel='Generalist − specialist NLL (bits/task)');ax.legend()
            fig.tight_layout();fig.savefig(folder/f'{mode}_decoder_overlay.png',dpi=180);plt.close(fig)
    csv_write(folder/'rank_instability.csv',rank_rows)
    write_tables(folder,announce=True)
    csv_write(folder/'proxy_fidelity.csv',fidelity)
    if fidelity:
        x=np.array([r['proxy_excess_bits'] for r in fidelity]);y=np.array([r['measured_excess_bits'] for r in fidelity])
        ss=float(np.sum((y-y.mean())**2));r2=1-float(np.sum((y-x)**2))/ss if ss>0 else None
        corr=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
        dump(folder/'proxy_fidelity.json',{'identity_prediction_R2':r2,'pearson_correlation':corr,
            'note':'Both axes are excess bits/task. Measured excess uses a fitted linear-probe baseline and may be negative. R2 uses the identity prediction and can be negative; points are not independent replicates.'})
        fig,ax=plt.subplots(figsize=(5,4));ax.scatter(x,y,s=16,alpha=.65)
        lim=[float(min(x.min(),y.min())),float(max(x.max(),y.max()))];ax.plot(lim,lim,'k--')
        ax.set(xlabel='Gaussian proxy excess loss (bits/task)',ylabel='Measured excess NLL (bits/task)',title=f'Identity R² = {r2:.3g}' if r2 is not None else 'Identity R² undefined')
        fig.tight_layout();fig.savefig(folder/'proxy_fidelity.png',dpi=180);plt.close(fig)
    diagnostics(rows,out,features,tasks['H'],channels,Q,Qout,config)
    captions.insert(0,f'Gaussian proxy: {100*c["confidence"]:g}% pointwise bootstrap intervals over {len(draws)} original-training-image resamples, refitting PCA/whitening and all task/pretext moments. Conditional on the frozen backbone and cached view bank. No test data are used in moment fitting. Noise draws and augmented views are never independent resampling units.')
    if not len(draws):captions[0]='Moment bootstrap disabled or incomplete: Gaussian proxy curves have no moment-fitting uncertainty bands; see moment_bootstrap/status.json.'
    captions += ['Crossover intervals are conditional on an interior positive-to-negative crossing; report the interior-crossing fraction and no-crossing statuses alongside the interval. Curves use linear interpolation between grid points. Pointwise bands are not simultaneous confidence bands or a formal global sign test.',
                 'The isotropic rate threshold is a sufficient bound, not a predicted crossing. Delta characterizes a uniform specialist on the fitted support; inspect the isotropy flag before applying Proposition 2 to an anisotropic fitted pretext.',
                 'Empirical code entropy is a plug-in estimate bounded by log2(number of test images); sparse code occupancy can make it severely underestimate population entropy. Gaussian covariance MI bounds use fitted covariances, not a distribution-free population certificate.',
                 'Rank instability describes the prespecified candidate bank and does not prove that no fixed pretext is optimal. Proxy fidelity and decoder overlays are diagnostics, not proofs of the proposed mechanism.']
    (folder/'CAPTIONS_AND_SCOPE.txt').write_text('\n\n'.join(captions)+'\n')
    dump(folder/'metadata.json',{'config':c,'seed_count':len(config.get('seeds',[config['seed']])),
        'task_names':task_names,'moment_conditioning':'Frozen backbone and cached views; refitted input/target PCA and whiteners',
        'band_types':['moment_fitting_only','test_images_only','quantized_seed_and_test_images'],
        'baseline':'Uncompressed fitted decoder, not Bayes NLL'})
