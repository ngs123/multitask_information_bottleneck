#!/usr/bin/env python3
"""Finish robustness reports in an existing run; reuse bottlenecks and bootstraps."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from celeba_experiment.analysis import dump
from celeba_experiment.rate_models import transform
from celeba_experiment.robustness import report, settings, validate


def load_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def restore_inputs(run,cache):
    manifest=json.loads((run/'feature_manifest.json').read_text())
    cache_manifest=json.loads((cache/'manifest.json').read_text())
    if {k:v for k,v in manifest.items() if k!='device'}!={k:v for k,v in cache_manifest.items() if k!='device'}:
        raise ValueError('Feature cache manifest does not match the original run')
    wanted=load_npz(run/'split_indices.npz');raw={}
    for source in ('train','valid','test'):
        data=load_npz(cache/f'{source}.npz');indices=data['indices']
        if len(np.unique(indices))!=len(indices):raise ValueError('Cache has duplicate image IDs')
        order=np.argsort(indices);sorted_ids=indices[order]
        for split in (('valid','select') if source=='valid' else (source,)):
            ids=wanted[split];positions=np.searchsorted(sorted_ids,ids)
            if np.any(positions>=len(indices)) or not np.array_equal(sorted_ids[positions],ids):
                raise ValueError(f'Cache missing original {split} image IDs')
            take=order[positions];raw[split]={k:v[take] for k,v in data.items()}
    fit=load_npz(run/'anchor_transform.npz')
    features={s:transform(v['anchor'],fit) for s,v in raw.items()}
    selected=json.loads((run/'selected_tasks.json').read_text());ids=selected['indices']
    if selected['names']!=[manifest['attribute_names'][i] for i in ids]:
        raise ValueError('Selected task names do not match manifest')
    labels={s:v['labels'][:,ids] for s,v in raw.items()}
    return raw,manifest,ids,features,labels


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True,type=Path)
    ap.add_argument('--cache',required=True,type=Path,help='Exact feature cache used by the original run')
    ap.add_argument('--bootstrap-max-iterations',type=int,default=12000,
                    help='Initial limit for missing/failed bootstrap curves; retries use 4x and 16x this limit')
    args=ap.parse_args()
    if args.bootstrap_max_iterations<1:ap.error('Iteration limit must be positive')
    config=json.loads((args.run/'config.json').read_text())
    if not settings(config)['enabled']:ap.error('Original run did not enable robustness')
    config['robustness']={**config['robustness'],'bootstrap_max_iterations':args.bootstrap_max_iterations}
    config['_allow_legacy_bootstrap']=True
    validate(config)
    rows=json.loads((args.run/'results.json').read_text())
    raw,manifest,ids,features,labels=restore_inputs(args.run,args.cache)
    config['_n_train']=len(features['train'])
    tasks=load_npz(args.run/'gaussian_tasks.npz');bank={}
    for spec in config['pretexts']:
        model=load_npz(args.run/f'pretext_proxy_{spec["name"]}.npz')
        bank[spec['name']]={'H':model['H'],'weights':model['weights']}
    avg=load_npz(args.run/'average_task_proxy.npz')['H']
    bank['average_task']={'H':avg[None] if avg.ndim==2 else avg,'weights':np.ones(1)}
    dump(args.run/'report_resume_config.json',{'bootstrap_max_iterations':args.bootstrap_max_iterations,
         'gap_tolerance':config['gaussian'].get('gap_tolerance',1e-5),'cache':str(args.cache.resolve()),
         'note':'Original run config and trained bottlenecks are unchanged; bootstrap samples are reused.'})
    with threadpool_limits(limits=config.get('threads',4)):
        report(raw,manifest,ids,features,labels,tasks,bank,rows,config,args.run)
    print(f'Report complete: {(args.run/"robustness"/"tables.html").resolve()}',flush=True)


if __name__=='__main__':main()
