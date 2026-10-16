"""Assemble only manuscript-referenced appendix assets from retained runs.

Run from the repository root: python paper_plotting/prepare_appendix.py
Numerical provenance goes in docs/appendix_values.json, outside the asset folder.
"""
from pathlib import Path
import json
import re
import shutil
import tempfile
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from plot_helpers import REPO_ROOT, COLORS, figure3, roots, load_curves
from celeba_experiment.robustness import paired_loss_cube
from celeba_experiment.robust_statistics import paired_bootstrap

ROOT = REPO_ROOT
FIG = ROOT/'paper_plotting/paper_figures'
OUT = FIG/'appendix'
RUN = ROOT/'celeba_experiment/runs/robust_rate_selection'
FINE = ROOT/'celeba_experiment/runs/CelebA/rate_selection/20260925T151945964363Z'


def read(path):
    return json.loads(path.read_text())


def main():
    OUT.mkdir(exist_ok=True)
    copies = {
        'f2a_code_entropy.pdf': 'celeba/code_entropy.pdf',
        'f4a_dsprites_accounting.pdf': 'dsprites/fig4_accounting.pdf',
        'f4b_role_swap.pdf': 'role_swap/role_swap.pdf',
        'f4d_dsprites_context.pdf': 'dsprites/dsprites_context.pdf',
        'f5a_gaussian_per_task.pdf': 'celeba/gaussian_per_task.pdf',
        'f5a_codes_per_task.pdf': 'celeba/codes_per_task.pdf',
        'f5b_subspace_allocation.pdf': 'celeba/allocation.pdf',
        'f5c_proxy_fidelity.pdf': 'celeba/proxy_fidelity.pdf',
        'f5d_decoder_capacity.pdf': 'celeba/decoder_capacity.pdf',
        'f5_full_pretext_bank.pdf': 'celeba/full_pretext_bank.pdf',
        'f5_rank_gaussian.pdf': 'celeba/gaussian_rank_bank.pdf',
        'f5_rank_codes.pdf': 'celeba/codes_rank_bank.pdf',
    }
    generated = {'f2a_code_usage.pdf', 'f2b_covariance_spectrum.pdf',
                 'f4_crossovers.tex', 'f5e_finetuned.pdf', 'f5f_theory_to_data.tex'}
    expected = set(copies) | generated
    manuscript = (ROOT/'docs/appendix_methods.tex').read_text()
    references = re.findall(r'\\(?:includegraphics(?:\[[^]]*\])?|input)\{\\appendixfigdir/([^}]+)\}', manuscript)
    if len(references) != len(set(references)) or set(references) != expected:
        raise ValueError('Manuscript references must match the appendix asset set exactly')
    extra = {p.name for p in OUT.iterdir()}-expected
    if extra:
        raise ValueError(f'Unexpected appendix files; review before removing: {sorted(extra)}')
    # Verify full swap completion and factor provenance before including the plot.
    swap_run = ROOT/'dsprites_experiment/runs/dsprites_role_swap'
    manifest = read(swap_run/'manifest.json')
    assert manifest['config']['factors'] == ['posY', 'posX']
    assert len(manifest['jobs']) == 42
    for job in manifest['jobs']:
        result = read(swap_run/job['name']/'result.json')
        assert (result['factor1'],result['factor2']) == ('posY','posX')
    for name, source in copies.items():
        shutil.copy2(FIG/source, OUT/name)
    values = dict(swap_completed_jobs=len(manifest['jobs']),
                  main_config=read(RUN/'config.json'), swap_config=manifest['config'],
                  fitted=read(FIG/'celeba/fitted_quantities.json'),
                  crossovers=read(FIG/'celeba/fig3_crossovers.json'),
                  rank=pd.read_csv(FIG/'celeba/fig3_rank_instability.csv').to_dict('records'),
                  proxy=pd.read_csv(FIG/'celeba/proxy_fidelity_score.csv').iloc[0].to_dict())
    # One explicitly identified seed, both selected objectives, every code budget.
    cfg=values['main_config']; seed=cfg['seeds'][0]
    fig, axes=plt.subplots(2,3,figsize=(6.75,4.7),constrained_layout=True)
    for ax,budget in zip(axes.flat,cfg['quantized']['bits']):
        for candidate,role in [('joint_attributes','specialist'),('reconstruction','generalist')]:
            file=RUN/f'quantized/{candidate}/budget_{budget}/seed_{seed}/code_usage.npz'
            with np.load(file) as z: counts=z['sorted_counts']
            # Occupancy levels are integer counts, log bins keep sparse and dense codes visible.
            bins=np.unique(np.r_[.5,np.geomspace(.5,2000.5,18)])
            ax.hist(counts,bins=bins,histtype='step',color=COLORS[role],
                    linestyle='-' if role=='specialist' else '--',label=role.title())
        ax.set_xscale('log'); ax.set_yscale('symlog',linthresh=1)
        ax.set(title=f'B = {budget}, seed {seed}',xlabel='Images per observed codeword',ylabel='Number of codewords')
    axes.flat[0].legend(fontsize=6)
    fig.savefig(OUT/'f2a_code_usage.pdf',bbox_inches='tight');plt.close(fig)
    entropy=pd.read_csv(RUN/'robustness/rate_diagnostics.csv').dropna(subset=['test_plugin_entropy_bits'])
    values['entropy']={'max_entropy':float(entropy.test_plugin_entropy_bits.max()),
                       'ceiling':float(np.log2(2000)), 'histogram_seed':seed,
                       'n_rows':len(entropy),'budgets':sorted(entropy.budget.unique().tolist())}
    # Recover raw spectrum and verify the cached training data match the retained PCA.
    with np.load(ROOT/'celeba_experiment/cache/rate_selection_semantic/train.npz') as z:
        raw=z['anchor'].astype(float)
    with np.load(RUN/'anchor_transform.npz') as z:
        mean,W=z['mean'],z['W']
    assert np.allclose(raw.mean(0),mean,rtol=1e-5,atol=1e-6)
    covariance=np.cov(raw,rowvar=False,bias=True)
    original=np.linalg.eigvalsh(covariance)[::-1]
    whitened=np.linalg.eigvalsh(W.T@covariance@W)[::-1]
    fig,axes=plt.subplots(1,2,figsize=(6.75,2.6),constrained_layout=True)
    axes[0].semilogy(np.arange(1,len(original)+1),np.maximum(original,1e-15),color=COLORS['specialist'])
    axes[0].set(xlabel='Eigenvalue index',ylabel='Training covariance eigenvalue',title='512-dimensional backbone features')
    axes[1].plot(np.arange(1,len(whitened)+1),whitened,'o-',ms=3,color=COLORS['specialist'])
    axes[1].axhline(1,color='.5',ls='--',zorder=0)
    axes[1].set(xlabel='Retained eigenvalue index',ylabel='Whitened covariance eigenvalue',title='32-dimensional PCA coordinates',ylim=(.99,1.01))
    fig.savefig(OUT/'f2b_covariance_spectrum.pdf',bbox_inches='tight');plt.close(fig)
    values['whitening']={'raw_min':float(original.min()),'raw_max':float(original.max()),
                         'max_identity_deviation':float(abs(whitened-1).max()),'n_train':len(raw)}
    # Report crossings and their actual adjacent observed-rate brackets, not confidence intervals.
    audits=[]
    for name,source,summary_file,sign in [
        ('Primary',ROOT/'paper_plotting/inputs/dsprites_all_runs.csv',FIG/'dsprites/fig4_summary.json',1),
        ('Role swap',swap_run/'summary/selected_runs.csv',FIG/'role_swap/role_swap_summary.json',-1)]:
        df=pd.read_csv(source);df=df[(df.pretext=='mixture')&(df.M==4)].copy()
        df['i1']=-.5*np.log2(df.mse1);df['i2']=-.5*np.log2(df.mse2)
        df['gap']=df.i1-df.i2 # role-index gap has the same direction in both runs
        summary=read(summary_file)
        values[name]={'summary':summary}
        for seed,h in df.groupby('seed'):
            h=h.sort_values('rate_kl_bits');x=h.rate_kl_bits.to_numpy();g=h.gap.to_numpy()
            crossings=roots(x,g)
            assert np.allclose(crossings,summary['per_seed_roots'][str(seed)])
            for cross in crossings:
                i=np.searchsorted(x,cross)
                audits.append(dict(run=name,seed=int(seed),crossing=cross,low=float(x[i-1]),high=float(x[i]),count=len(crossings)))
    values['seed_crossovers']=audits
    tex=[r'\begin{tabular}{llrrr}',r'\toprule',r'Run & Seed & Crossover & Lower grid rate & Upper grid rate \\',r'\midrule']
    for a in audits:tex.append(f"{a['run']} & {a['seed']} & {a['crossing']:.4f} & {a['low']:.4f} & {a['high']:.4f}"+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}']
    (OUT/'f4_crossovers.tex').write_text('\n'.join(tex)+'\n')
    # Statistics for factual captions.
    allocation=pd.read_csv(FIG/'celeba/allocation.csv')
    values['allocation_at_32']=allocation[allocation.budget==32].to_dict('records')
    values['decoder']={}
    for name in ['gaussian_nll_test','gaussian_mlp_test','quantized_nll_test','quantized_mlp_test']:
        df=pd.read_csv(FIG/f'celeba/{name}_difference.csv')
        values['decoder'][name]={'crossings':roots(df.rate,df.difference_bits), 'last':df.iloc[-1].to_dict()}
    proxy=pd.read_csv(RUN/'robustness/proxy_fidelity.csv')
    values['proxy']['mean_measured_minus_predicted']=float((proxy.measured_excess_bits-proxy.proxy_excess_bits).mean())
    ds=pd.read_csv(ROOT/'paper_plotting/inputs/dsprites_all_runs.csv')
    ds=ds[ds.M==4]
    context=[]
    for name,h in ds.groupby('pretext'):
        h=h.copy();h['i1']=-.5*np.log2(h.mse1);h['i2']=-.5*np.log2(h.mse2)
        ends=h.groupby('beta')[['rate_kl_bits','i1','i2']].mean().sort_values('rate_kl_bits').iloc[[0,-1]]
        context.append({'pretext':name,'endpoints':ends.to_dict('records')})
    values['context']=context
    weights=read(RUN/'selected_tasks.json')['weights']
    curves=load_curves(RUN/'robustness/gaussian_nll_test_bootstrap.npz','test',weights)
    i=curves['names'].index('joint_attributes');j=curves['names'].index('reconstruction')
    values['gaussian_per_task']={'names':read(RUN/'selected_tasks.json')['names'],
        'low_difference':(curves['task_point'][j,1]-curves['task_point'][i,1]).tolist(),
        'high_difference':(curves['task_point'][j,-1]-curves['task_point'][i,-1]).tolist()}
    # Recompute measured-NLL bootstrap reports from existing fine-tuned per-image losses.
    print('Recovering fine-tuned measured-NLL figure from saved losses',flush=True)
    rows=read(FINE/'results.json');fcfg=read(FINE/'config.json')
    fweights=np.array(read(FINE/'selected_tasks.json')['weights'])
    with np.load(FINE/'uncompressed/per_image_losses.npz') as z:baseline=z['test_ce_bits']
    fine=[]
    assert hashlib.sha256((ROOT/'celeba_experiment/runs/celeba_multitask/backbone.pt').read_bytes()).hexdigest() == read(FINE/'feature_manifest.json')['backbone_checkpoint']['sha256']
    for mode,kind in [('gaussian','test'),('quantized','codes')]:
        cube,names,rates,seeds=paired_loss_cube([r for r in rows if r['mode']==mode],FINE,
                                            ['joint_attributes','reconstruction','task_aware'])
        excess=cube-baseline[None,None,None]
        draws=paired_bootstrap(excess,2000,93044 if mode=='gaussian' else 93045,mode=='quantized')
        point=excess.mean((0,3))
        fine.append(dict(names=names,rates=rates,point=point@fweights,draws=draws@fweights,
                         task_point=point,task_draws=draws,kind=kind,source=str(FINE)))
    with tempfile.TemporaryDirectory() as temp:
        fig,summaries=figure3(*fine,temp,'joint_attributes',name='finetuned')
        shutil.copy2(Path(temp)/'finetuned.pdf',OUT/'f5e_finetuned.pdf');plt.close(fig)
    values['finetuned']={'crossovers':summaries,'config':fcfg,
                         'backbone_config':read(ROOT/'celeba_experiment/runs/celeba_multitask/config.json'),
                         'checkpoint':read(FINE/'feature_manifest.json')['backbone_checkpoint']}
    epochs=[read(p) for p in (ROOT/'celeba_experiment/runs/celeba_multitask').glob('epoch_*.json')]
    values['finetuned']['best_epoch']=min(epochs,key=lambda x:x['valid']['loss'])['epoch']
    # Compact numeric table with unambiguous units and rounded intervals.
    f=values['fitted'];cross=values['crossovers']
    tex=[r'\begin{tabular}{lrl}',r'\toprule',r'Quantity & Value & Units / interpretation \\',r'\midrule']
    for name,snr in zip(read(RUN/'selected_tasks.json')['names'],f['task_snr']):
        tex.append(r'$\lambda_{\max}(\widehat H_{\mathrm{'+name.replace('_',r'\_')+r'}})$ & '+f'{snr:.6f}'+r' & fitted task SNR \\')
    for label,value,unit in [(r'$d$',str(f['dimension']),'PCA dimension'),(r'$\dim S$',str(f['specialist_rank']),'specialist rank'),
                             (r'$\Delta$',f"{f['delta_uniform_subspace']:.7f}",'uniform-subspace reference'),
                             (r'$D_{\mathrm{miss}}$',f"{f['D_miss_bits']:.6f}",'bits; proxy asymptote'),
                             ('Sufficient threshold',f"{f['isotropic_generalist_sufficient_rate_bound']:.4f}",'bits; isotropic reference')]:
        tex.append(f'{label} & {value} & {unit}'+r' \\')
    for name,label,unit in [('gaussian','Gaussian crossover','nominal bits'),('codes','Code crossover','bits/image')]:
        v=cross[name];lo,hi=v['conditional_ci'];tex.append(f'{label} & {v["point"]:.4f} & {unit}; 95\\% CI [{lo:.4f}, {hi:.4f}]'+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}']
    (OUT/'f5f_theory_to_data.tex').write_text('\n'.join(tex)+'\n')
    assert {p.name for p in OUT.iterdir()} == expected
    values['assets']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())}
    values['unavailable']=['null control results','shrinkage sensitivity sweep','constructed worst-case and SNR-cap sweep']
    (ROOT/'docs/appendix_values.json').write_text(json.dumps(values,indent=2)+'\n')
    print(f'Wrote exactly {len(expected)} appendix assets to {OUT}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
