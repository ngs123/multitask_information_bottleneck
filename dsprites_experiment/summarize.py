"""Aggregate paired-seed curves without extrapolating or forcing monotonicity."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from theory import optimum, predictions

COLORS={'specialist1':'#377eb8','specialist2':'#e41a1c','reconstruction':'#984ea3','mixture':'#228833'}
METRICS=['distortion_bits','r2_1','r2_2','effective_r1','effective_r2','priority_difference']


def crossings(x,y,tolerance=.05):
    """Negative-to-positive crossings with evidence beyond a tolerance on both sides.

    Near-zero points do not establish an ordering. All supported reversals are retained.
    """
    good=np.flatnonzero(np.abs(y)>tolerance)
    found=[]
    for a,b in zip(good[:-1],good[1:]):
        if y[a]<0<y[b]:
            root=x[a]-y[a]*(x[b]-x[a])/(y[b]-y[a])
            found.append({'estimate':float(root),'bracket_low':float(x[a]),'bracket_high':float(x[b])})
    return found


def interp_seeds(frame,metrics=METRICS):
    groups=[]
    for seed,g in frame.groupby('seed'):
        # Multiple achieved rates may coincide; average exactly coincident points.
        g=g.groupby('rate_kl_bits',as_index=False)[metrics].mean().sort_values('rate_kl_bits')
        groups.append((seed,g))
    low=max(g.rate_kl_bits.min() for _,g in groups)
    high=min(g.rate_kl_bits.max() for _,g in groups)
    if high<=low: return None
    x=np.linspace(low,high,201)
    ys={m:np.array([np.interp(x,g.rate_kl_bits,g[m]) for _,g in groups]) for m in metrics}
    return x,ys,[s for s,_ in groups]


def summarize(root,allow_partial=False,bootstrap=2000,show_gaussian_reference=False):
    manifest=json.loads((root/'manifest.json').read_text())
    dataset=manifest['config'].get('dataset','gaussian')
    factor_names=manifest['config'].get('factors',['S1','S2']) if dataset=='dsprites' else ['S1','S2']
    custom_tasks=any(k in manifest['config'] for k in ('task_weights','noise_variances'))
    overlay=(dataset=='gaussian' or show_gaussian_reference) and not custom_tasks
    rows=[]; missing=[]
    for j in manifest['jobs']:
        folder=root/j['name']; file=folder/'result.json'
        if not file.exists(): missing.append(j['name']); continue
        result=json.loads(file.read_text())
        history=json.loads((folder/'history.json').read_text())
        result['selection_val_loss']=min(h['val_loss'] for h in history)
        result['job']=j['name']; result.pop('theory',None); result.pop('data_provenance',None)
        rows.append(result)
    if missing and not allow_partial: raise SystemExit(f'{len(missing)} unfinished jobs. Use --allow-partial to explicitly summarize incomplete results.')
    if not rows: raise SystemExit('No completed runs')
    out=root/'summary'; out.mkdir(exist_ok=True)
    df=pd.DataFrame(rows); df.to_csv(out/'all_runs.csv',index=False)
    # Select learning-rate candidates using pretext validation loss, never test task scores.
    idx=df.groupby(['M','pretext','beta','seed']).selection_val_loss.idxmin()
    df=df.loc[idx].sort_values(['M','pretext','seed','rate_kl_bits'])
    df.to_csv(out/'selected_runs.csv',index=False)
    rng=np.random.default_rng(1729); cross_rows=[]; bands=[]; diagnostics=[]
    predictions_all=[]
    for M,dm in df.groupby('M'):
        theory=predictions(M)
        if custom_tasks:
            theory = dict(M=M, R1=None, R2=None, crossover=None)
        theory['applicability']='exact Gaussian reference' if dataset=='gaussian' else 'Gaussian surrogate only: discrete dSprites factors do not obey the exact crossover formula'
        if custom_tasks: theory['applicability']='No M-family Gaussian reference: custom task weights/noise'
        predictions_all.append(theory)
        xmax=max(float(dm.rate_kl_bits.max()),theory['R2']*1.1) if overlay else max(float(dm.rate_kl_bits.max())*1.05,0.1)
        r=np.linspace(0,xmax,501); oracle=[optimum(M,float(t)) for t in r]
        fig,ax=plt.subplots(figsize=(7,4.5))
        fig2,axes=plt.subplots(1,2,figsize=(11,4.3))
        fig3,ax3=plt.subplots(figsize=(7,4.5))
        for pretext,group in dm.groupby('pretext'):
            color=COLORS[pretext]
            for seed,g in group.groupby('seed'):
                g=g.sort_values('rate_kl_bits')
                ax.scatter(g.rate_kl_bits,g.distortion_bits,s=12,color=color,alpha=.35)
                if pretext=='mixture':
                    roots=crossings(g.rate_kl_bits.to_numpy(),g.priority_difference.to_numpy())
                    diagnostics.append({'M':M,'seed':int(seed),'kind':'seed_crossings','roots':roots,
                                        'min_rate':float(g.rate_kl_bits.min()),'max_rate':float(g.rate_kl_bits.max())})
                    for factor in (1,2):
                        diff=np.diff(g[f'r2_{factor}'].to_numpy())
                        diagnostics.append({'M':M,'seed':int(seed),'kind':f'factor{factor}_nonmonotonicity',
                                            'number_of_R2_drops_over_0.01':int((diff<-.01).sum())})
            data=interp_seeds(group)
            if data is None:
                diagnostics.append({'M':M,'pretext':pretext,'kind':'no_common_rate_support'}); continue
            x,ys,seeds=data; ns=len(seeds)
            sample_ids=rng.integers(ns,size=(bootstrap,ns))
            stats={}
            for metric,values in ys.items():
                boot=values[sample_ids].mean(1)
                mean=values.mean(0); lo,hi=np.quantile(boot,[.025,.975],axis=0)
                stats[metric]=(mean,lo,hi,boot)
                for i,R in enumerate(x):
                    bands.append({'M':M,'pretext':pretext,'metric':metric,'rate_kl_bits':R,
                                  'mean':mean[i],'ci_low':lo[i],'ci_high':hi[i],'n_seeds':ns})
            mean,lo,hi,_=stats['distortion_bits']
            ax.plot(x,mean,color=color,label=pretext); ax.fill_between(x,lo,hi,color=color,alpha=.15)
            if pretext=='mixture':
                for factor in (1,2):
                    for axis,metric in zip(axes,[f'r2_{factor}',f'effective_r{factor}']):
                        mean,lo,hi,_=stats[metric]
                        axis.plot(x,mean,label=factor_names[factor-1]); axis.fill_between(x,lo,hi,alpha=.15)
                mean,lo,hi,boot=stats['priority_difference']
                ax3.plot(x,mean,color=color,label='mixture'); ax3.fill_between(x,lo,hi,color=color,alpha=.2)
                roots=crossings(x,mean)
                boot_roots=[crossings(x,y) for y in boot]
                single=[v[0]['estimate'] for v in boot_roots if len(v)==1]
                interval=np.quantile(single,[.025,.975]).tolist() if len(single) else [None,None]
                cross_rows.append({'M':M,'theory_crossover':theory['crossover'],'R1':theory['R1'],'R2':theory['R2'],
                                   'observed_crossings':roots,'ci_conditional_on_single_crossing':interval,
                                   'bootstrap_single_crossing_fraction':len(single)/bootstrap,
                                   'bootstrap_no_crossing_fraction':sum(not v for v in boot_roots)/bootstrap,
                                   'bootstrap_multiple_crossing_fraction':sum(len(v)>1 for v in boot_roots)/bootstrap,
                                   'dataset':dataset,'gaussian_reference_is_exact':dataset=='gaussian' and not custom_tasks,
                                   'factor1':factor_names[0],'factor2':factor_names[1],
                                   'n_seeds':ns,'support_low':float(x[0]),'support_high':float(x[-1])})
        if overlay:
            ax.plot(r,[o['distortion'] for o in oracle],'k--',label='Gaussian optimum (MI budget)' if dataset=='gaussian' else 'Gaussian surrogate (not a dSprites bound)')
            ax.plot(r,[o['equal_distortion'] for o in oracle],':',color='gray',label='equal Gaussian allocation')
        ax.set_yscale('symlog',linthresh=1e-4); ax.set_ylabel('Weighted excess NLL (bits)')
        ax.set_title(f'{dataset}, M={M:g}: learned pretexts'); ax.legend(fontsize=8)
        if overlay:
            for factor in (1,2):
                alloc=np.array([o[f'r{factor}'] for o in oracle])
                axes[0].plot(r,1-2**(-2*alloc),'--',alpha=.55,label=f'Gaussian factor {factor}')
                axes[1].plot(r,alloc,'--',alpha=.55,label=f'Gaussian r{factor}')
        axes[0].set_ylabel('Latent-factor probe R²'); axes[1].set_ylabel('Recoverability index −½ log₂ MSE' + (' (bits)' if dataset=='gaussian' else ''))
        ax3.axhline(0,color='black',lw=.8)
        if overlay: ax3.plot(r,[o['r2']-o['r1'] for o in oracle],'k--',label='Gaussian r2 − r1')
        ax3.set_ylabel(f'Recoverability index: {factor_names[1]} − {factor_names[0]}')
        for axis in [ax,*axes,ax3]:
            axis.set_xlabel('Achieved E[KL(q(c|U) || N(0,I))] (bits)')
            if overlay:
                for marker in (theory['R1'],theory['R2']): axis.axvline(marker,color='gray',ls=':',lw=1)
                if theory['crossover'] is not None: axis.axvline(theory['crossover'],color='orange',ls='-.',lw=1)
            axis.grid(alpha=.15)
        for axis in [*axes,ax3]: axis.legend(fontsize=8)
        for f,name in [(fig,'rd'),(fig2,'recoverability'),(fig3,'priority')]:
            f.tight_layout(); f.savefig(out/f'M{M:g}_{name}.png',dpi=180); f.savefig(out/f'M{M:g}_{name}.pdf'); plt.close(f)
    pd.DataFrame(bands).to_csv(out/'curve_bands.csv',index=False)
    (out/'crossovers.json').write_text(json.dumps(cross_rows,indent=2))
    (out/'diagnostics.json').write_text(json.dumps({'missing_jobs':missing,'diagnostics':diagnostics},indent=2))
    (out/'theory_predictions.json').write_text(json.dumps(predictions_all,indent=2))
    if cross_rows and not custom_tasks:
        fig,ax=plt.subplots(figsize=(6,4))
        ms=np.linspace(min(c['M'] for c in cross_rows),max(c['M'] for c in cross_rows)+.01,100)
        ax.plot(np.log2(ms),[predictions(float(m))['crossover'] for m in ms],'k--',label='exact Gaussian crossover' if dataset=='gaussian' else 'Gaussian surrogate crossover (heuristic)')
        for c in cross_rows:
            if len(c['observed_crossings'])==1:
                y=c['observed_crossings'][0]['estimate']; low,high=c['ci_conditional_on_single_crossing']
                ax.scatter(np.log2(c['M']),y,color=COLORS['mixture'])
                if low is not None: ax.vlines(np.log2(c['M']),low,high,color=COLORS['mixture'])
        ax.set_xlabel('log₂ M'); ax.set_ylabel('Crossover rate (bits; empirical uses KL bound)')
        ax.legend(); fig.tight_layout(); fig.savefig(out/'crossover_vs_M.png',dpi=180); plt.close(fig)
    print(f'Wrote {out}; {len(df)} selected runs; {len(missing)} missing',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--show-gaussian-reference',action='store_true',help='Overlay Gaussian surrogate on dSprites plots; not an exact prediction or bound'); p.add_argument('--allow-partial',action='store_true'); p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args()
    if a.bootstrap<1: p.error('--bootstrap must be positive')
    summarize(a.run,a.allow_partial,a.bootstrap,a.show_gaussian_reference)
