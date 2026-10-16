"""One independent experiment process per GPU, paired seeds, resumable job queue."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from task_spec import task_parameters

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default=str(ROOT/'configs/main.json'))
    ap.add_argument('--out',default='runs/main')
    ap.add_argument('--gpus',default=None,help='Comma-separated CUDA device IDs or UUIDs; defaults to CUDA_VISIBLE_DEVICES or all detected GPUs')
    ap.add_argument('--cpu',action='store_true',help='Single CPU worker for smoke tests')
    ap.add_argument('--Ms',nargs='+',type=float,help='Override the M grid')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--data-dir',help='Prepared dSprites cache directory')
    ap.add_argument('--download-data',action='store_true',help='Prepare/download dSprites once before launching workers')
    ap.add_argument('--factors',nargs=2,choices=['posX','posY','scale'],help='Ordered task factors; use posY posX for a swap control')
    args=ap.parse_args(); cfg=json.loads(Path(args.config).read_text())
    if args.Ms: cfg['Ms']=args.Ms
    for M in cfg['Ms']:
        task_parameters(M, cfg.get('task_weights'), cfg.get('noise_variances'))
    if args.factors: cfg['factors']=args.factors
    if cfg.get('dataset','gaussian')=='dsprites':
        cfg['data_dir']=str(Path(args.data_dir or cfg.get('data_dir','data/dsprites')).resolve())
        if not args.dry_run:
            from prepare_dsprites import prepare, check_cache
            meta=prepare(cfg['data_dir']) if args.download_data else check_cache(cfg['data_dir'])
            cfg['dataset_fingerprint']=meta['files_sha256']
        if cfg['image_size']!=64: raise ValueError('dSprites requires image_size=64')
        if len(set(cfg.get('factors',['posX','posY'])))!=2: raise ValueError('Task factors must differ')
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    if args.cpu: slots=['cpu']
    elif args.gpus is not None: slots=args.gpus.split(',')
    elif os.environ.get('CUDA_VISIBLE_DEVICES'): slots=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        # Lazy import allows --dry-run on machines without PyTorch.
        if args.dry_run: slots=['auto']
        else:
            import torch
            slots=[str(i) for i in range(torch.cuda.device_count())]
    if not slots or any(not s.strip() or s=='-1' for s in slots):
        raise SystemExit('No GPUs selected. Use --gpus 0,1,... or --cpu for a smoke test.')
    if len(slots)!=len(set(slots)): raise SystemExit('Duplicate GPU slots are not allowed')
    required=['Ms','pretexts','betas','seeds','lrs']
    for k in required:
        if not cfg[k]: raise ValueError(f'Empty grid: {k}')
    if min(cfg['Ms'])<=3 or min(cfg['betas'])<=0 or min(cfg['lrs'])<=0:
        raise ValueError('This reversal experiment requires M>3, beta>0, lr>0')
    if not set(cfg['pretexts']) <= {'specialist1','specialist2','mixture','reconstruction'}:
        raise ValueError('Unknown pretext')
    for k in ['epochs','probe_epochs','n_train','n_val','n_probe','n_test','batch_size','width','eval_draws']:
        if cfg[k]<1: raise ValueError(f'{k} must be positive')
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}
    jobs=[]
    for M,pretext,beta,seed,lr in itertools.product(*(cfg[k] for k in required)):
        c={k:v for k,v in cfg.items() if k not in required}
        c.update(M=M,pretext=pretext,beta=beta,seed=seed,lr=lr)
        digest=hashlib.sha256(json.dumps({'config':c,'sources':sources},sort_keys=True).encode()).hexdigest()[:12]
        name=f'M{M:g}_{pretext}_b{beta:g}_s{seed}_lr{lr:g}_{digest}'
        jobs.append({'name':name,'config':c})
    if len({j['name'] for j in jobs})!=len(jobs): raise ValueError('Duplicate jobs in grid')
    manifest={'config':cfg,'sources':sources,'jobs':jobs}
    manifest_path=out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise SystemExit('Output contains a different grid/source revision. Choose a new --out directory.')
    if args.dry_run:
        print(json.dumps({'jobs':len(jobs),'slots':slots,'out':str(out),'config':cfg},indent=2)); return
    # Atomic exclusive lock prevents overlapping schedulers writing the same jobs.
    lock=out/'launcher.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f'Launcher lock exists: {lock}. Check the recorded PID before removing a stale lock.')
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    active={}; failures=[]
    try:
        manifest_path.write_text(json.dumps(manifest,indent=2))
        queue=[]
        for j in jobs:
            folder=out/j['name']; folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists(): continue
            (folder/'job.json').write_text(json.dumps(j['config'],indent=2)); queue.append(j)
        print(f'{len(jobs)} jobs; {len(jobs)-len(queue)} complete; {len(slots)} workers',flush=True)
        while queue or active:
            for slot in slots:
                if slot in active or not queue: continue
                j=queue.pop(0); folder=out/j['name']
                env=os.environ.copy(); env['OMP_NUM_THREADS']=str(cfg.get('cpu_threads',4))
                if slot!='cpu': env['CUDA_VISIBLE_DEVICES']=slot
                log=(folder/'run.log').open('w')
                cmd=[sys.executable,str(ROOT/'train.py'),'--config',str(folder/'job.json'),'--out',str(folder),
                     '--device','cpu' if slot=='cpu' else 'cuda:0']
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[slot]=(proc,log,j); print(f'START {slot}: {j["name"]}',flush=True)
            for slot,(proc,log,j) in list(active.items()):
                code=proc.poll()
                if code is None: continue
                log.close(); del active[slot]
                if code or not (out/j['name']/'result.json').exists(): failures.append(j['name'])
                print(f'END {slot}: {j["name"]} exit={code}',flush=True)
            time.sleep(.25)
    finally:
        for proc,log,_ in active.values():
            proc.terminate()
        for proc,log,_ in active.values():
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            log.close()
        lock.unlink(missing_ok=True)
    (out/'failures.json').write_text(json.dumps(failures,indent=2))
    if failures: raise SystemExit(f'{len(failures)} jobs failed; inspect run.log and rerun the same command to retry.')
    subprocess.run([sys.executable,str(ROOT/'summarize.py'),'--run',str(out)],check=True)

if __name__=='__main__':
    # SIGTERM should clean up child processes and the launcher lock too.
    signal.signal(signal.SIGTERM,lambda *_: sys.exit(143))
    main()
