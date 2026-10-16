from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
CELEBA_ROOT = REPO_ROOT / "celeba_experiment"
if str(CELEBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CELEBA_ROOT))
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import rankdata, kendalltau

COLORS = dict(specialist='#0072B2', generalist='#E69F00', mixture='#009E73', oracle='#D55E00')
plt.rcParams.update({'font.size':8, 'axes.labelsize':8, 'legend.fontsize':6.5,
 'xtick.labelsize':7, 'ytick.labelsize':7, 'pdf.fonttype':42, 'ps.fonttype':42,
 'axes.spines.top':False, 'axes.spines.right':False, 'savefig.dpi':180})

def save(fig, out, name):
    out=Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(out/f'{name}.png', bbox_inches='tight')
    return fig

def table(df, out, name, escape=True):
    out=Path(out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/f'{name}.csv', index=False)
    def fmt(v):
        s=f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v)
        if escape:
            replacements={'\\':r'\textbackslash{}','_':r'\_','%':r'\%','&':r'\&','#':r'\#','{':r'\{','}':r'\}'}
            s=''.join(replacements.get(c,c) for c in s)
        return s
    tex='\\begin{tabular}{'+'l'*len(df.columns)+'}\n\\toprule\n'
    tex+=' & '.join(map(fmt,df.columns))+r' \\'+'\n'+r'\midrule'+'\n'
    for row in df.itertuples(index=False,name=None): tex+=' & '.join(map(fmt,row))+r' \\'+'\n'
    tex+=r'\bottomrule'+'\n'+r'\end{tabular}'+'\n' 
    (out/f'{name}.tex').write_text(tex)
    print(tex)
    return df

def read_json(p):
    return json.loads(Path(p).read_text())

def roots(x, y, tol=1e-10):
    """Positive-to-negative crossings only; ignore initial zero ties.
    For an intervening zero plateau, use its first zero (first contact).
    """
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)!=len(y) or not np.all(np.isfinite(y)) or np.any(np.diff(x)<=0):
        raise ValueError('Crossings require finite values on a strictly increasing grid')
    nonzero=np.flatnonzero(np.abs(y)>tol); result=[]
    for a,b in zip(nonzero[:-1],nonzero[1:]):
        if y[a]>0 and y[b]<0:
            result.append(float(x[a+1] if b>a+1 else x[a]-y[a]*(x[b]-x[a])/(y[b]-y[a])))
    return result

def crossing_summary(x, point, draws=None, confidence=.95):
    rr=roots(x,point); allroots=[] if draws is None else [roots(x,d) for d in draws]
    first=np.array([r[0] for r in allroots if r]); alpha=(1-confidence)/2
    return dict(point=rr[0] if rr else None, roots=rr, n_draws=len(allroots),
      crossing_fraction=len(first)/len(allroots) if allroots else None,
      conditional_median=float(np.median(first)) if len(first) else None,
      conditional_ci=np.quantile(first,[alpha,1-alpha]).tolist() if len(first) else None,
      multiple_crossings=sum(len(r)>1 for r in allroots))

def load_curves(path, kind, weights=None):
    """NPZ schema: names[C], rates[R], point[C,R,T], draws[B,C,R,T]."""
    path=Path(path)
    with np.load(path,allow_pickle=False) as z: a={k:z[k] for k in z.files}
    if kind=='moment':
        status=path.parent/'status.json'
        if 'complete' in a:
            complete=bool(np.all(a['complete']))
        elif status.exists():
            s=read_json(status); complete=s.get('successful')==s.get('requested') and not s.get('failures')
        else:
            raise ValueError('Moment bootstrap completeness unknown: require complete flag or status.json')
        if not complete:
            raise ValueError('Incomplete moment bootstrap. Recover failed fits before plotting percentile bands.')
    p=np.asarray(a['point'],float); d=np.asarray(a['draws'],float)
    if p.ndim!=3 or d.ndim!=4 or d.shape[1:]!=p.shape or not np.all(np.isfinite(d)) or not np.all(np.isfinite(p)):
        raise ValueError(f'Invalid curve shapes/values in {path}')
    w=np.ones(p.shape[-1])/p.shape[-1] if weights is None else np.asarray(weights,float)
    if len(w)!=p.shape[-1] or np.any(w<0) or not np.isclose(w.sum(),1):
        raise ValueError('Task weights must match tasks and sum to one')
    return dict(names=a['names'].astype(str).tolist(), rates=a['rates'].astype(float),
                point=p@w,draws=d@w,task_point=p,task_draws=d,kind=kind,source=str(path))

def rank_summary(c, exclude=('task_aware','average_task')):
    ids=[i for i,n in enumerate(c['names']) if n not in exclude]
    rid=np.flatnonzero(c['rates']>0)
    if not ids or len(rid)<2: raise ValueError('Ranks need a bank and two positive rates')
    vals=c['point'][ids][:,rid]; ranks=rankdata(vals,axis=0,method='average')
    low,high=vals[:,0],vals[:,-1]; rev=comp=0
    for i in range(len(ids)):
        for j in range(i):
            a=np.sign(low[i]-low[j]); b=np.sign(high[i]-high[j])
            comp+=int(a!=0 and b!=0); rev+=int(a*b<0)
    stats=dict(kendall_tau_b=float(kendalltau(low,high).statistic),reversed_pairs=rev,
      comparable_untied_pairs=comp,total_pairs=len(ids)*(len(ids)-1)//2,
      max_rank_change=float(np.max(np.abs(ranks[:,0]-ranks[:,-1]))))
    order=np.argsort(ranks[:,0],kind='stable')
    return ranks[order],np.array(c['names'])[ids][order],c['rates'][rid],stats

def heatmap(ax,c,row_order=None):
    """Readable appendix-only ranking plot; dark cells denote better ranks."""
    ranks,names,rates,stats=rank_summary(c)
    if row_order is not None:
        row_order=list(row_order)
        if len(row_order)!=len(names) or set(row_order)!=set(names):
            raise ValueError('Shared heatmap order must contain every ranked pretext exactly once')
        order=[list(names).index(name) for name in row_order]
        ranks,names=ranks[order],names[order]
    im=ax.imshow(ranks,aspect='auto',cmap='Blues_r',vmin=1,vmax=max(2,len(names)))
    for j in range(len(rates)):
        for i in np.flatnonzero(ranks[:,j]==ranks[:,j].min()):
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,lw=.8,color='black'))
    ax.set_yticks(np.arange(len(names)),[n.replace('_',' ') for n in names])
    ax.set_xticks(np.arange(len(rates)),[f'{r:g}' for r in rates])
    ax.set_ylabel('Pretext objective')
    ax.set_xlabel('Code budget B (bits/image)' if c['kind']=='codes' else 'Nominal Gaussian rate (bits)')
    metric='Gaussian-proxy distortion' if c['kind']=='moment' else 'Measured excess NLL'
    mechanism='learned binary codes' if c['kind']=='codes' else 'Gaussian channel'
    ax.set_title(f'Pretext rankings: {mechanism}\n{metric}; lower rank is better')
    cb=ax.figure.colorbar(im,ax=ax,pad=.025,ticks=np.arange(1,len(names)+1))
    cb.set_label('Rank (1 = lowest distortion)')
    ax.legend(handles=[Rectangle((0,0),1,1,fill=False,edgecolor='black',lw=.8,
              label='Best at this rate (including ties)')],loc='upper left',
              bbox_to_anchor=(0,-.24),frameon=False)
    return stats

def difference(ax,c,specialist,generalist,confidence=.95):
    i=c['names'].index(specialist); j=c['names'].index(generalist)
    y=c['point'][j]-c['point'][i]; d=c['draws'][:,j]-c['draws'][:,i]
    q=(1-confidence)/2; lo,hi=np.quantile(d,[q,1-q],axis=0); x=c['rates']
    lim=max(np.max(np.abs(np.r_[y,lo,hi]))*1.3,1e-6)
    ax.axhspan(0,lim,color=COLORS['specialist'],alpha=.035,zorder=0)
    ax.axhspan(-lim,0,color=COLORS['generalist'],alpha=.035,zorder=0)
    ax.axhline(0,color='.5',lw=.6,zorder=0)
    ax.fill_between(x,lo,hi,color='.5',alpha=.2); ax.plot(x,y,'o-',color='black',ms=2.5)
    s=crossing_summary(x,y,d,confidence)
    if s['point'] is not None: ax.axvline(s['point'],color='black',ls='--',lw=.7)
    if s['conditional_ci']: ax.axvspan(*s['conditional_ci'],color='.3',alpha=.1)
    p='none' if s['point'] is None else f"{s['point']:.2f}"
    ci='none' if s['conditional_ci'] is None else ', '.join(f'{v:.2f}' for v in s['conditional_ci'])
    ax.text(.98,.96,f"Crossing: {p}\n95% conditional CI: [{ci}]\nCrosses in {100*s['crossing_fraction']:.1f}% of resamples",transform=ax.transAxes,va='top',ha='right',fontsize=6.5,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1.5))
    ax.text(.98,.04,'generalist better',ha='right',transform=ax.transAxes,fontsize=6,color='.4')
    ax.text(.98,.66,'specialist better',ha='right',transform=ax.transAxes,fontsize=6,color='.4')
    ax.set_ylim(-lim,lim); return s

def figure3(gaussian, codes, out, specialist, generalist='reconstruction', mixture=None,
            oracle='task_aware', dmiss=None, name='fig3'):
    fig,axs=plt.subplots(2,2,figsize=(6.75,4.9),gridspec_kw={'height_ratios':[2,1.45],'hspace':.16,'wspace':.30},sharex='col')
    summaries={}; ranks=[]
    for col,c in enumerate([gaussian,codes]):
        if c is None:
            for ax in axs[:,col]: ax.text(.5,.5,'Input not supplied',ha='center',transform=ax.transAxes); ax.set_axis_off()
            continue
        ax=axs[0,col]; x=c['rates']; choices=[(specialist,'specialist','-', 'o'),(generalist,'generalist','-','s')]
        if mixture: choices.append((mixture,'mixture','--','^'))
        if oracle in c['names']:
            choices.append((oracle,'oracle',':','D'))
        else:
            warnings.warn(f'Task-aware baseline {oracle!r} absent from {c["source"]}; available: {c["names"]}. No oracle curve can be drawn from these inputs.')
        for n,role,ls,m in choices:
            if n not in c['names']: raise ValueError(f'{n} absent: {c["names"]}')
            i=c['names'].index(n); lo,hi=np.quantile(c['draws'][:,i],[.025,.975],axis=0)
            ax.fill_between(x,lo,hi,color=COLORS[role],alpha=.17,lw=0)
            ax.plot(x,c['point'][i],color=COLORS[role],ls=ls,marker=m,ms=3,label=('Oracle (task-aware)' if col==0 else 'Task-aware (trained)') if role=='oracle' else role.title())
        if col==0 and dmiss is not None and c['kind']=='moment':
            ax.axhline(dmiss,color='.5',ls='--',lw=.7)
            ax.text(.98,dmiss,rf'$D_{{miss}}={dmiss:.3g}$',transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=6)
        ax.set_title('Gaussian channel' if col==0 else 'Learned binary codes')
        ax.set_ylabel('Gaussian-proxy distortion (bits/task)' if c['kind']=='moment' else 'Task-averaged excess NLL (bits)')
        ax.legend(loc='upper right',frameon=False)
        summaries['gaussian' if col==0 else 'codes']=difference(axs[1,col],c,specialist,generalist)
        axs[1,col].set_xlabel('Nominal Gaussian rate (bits)' if col==0 else 'Code budget B (bits/image)')
        axs[1,col].set_ylabel('Generalist − specialist (bits)')
        rank,ns,rs,stat=rank_summary(c); ranks.append(dict(panel='gaussian' if col==0 else 'codes',**stat))
    for axis in axs.flat:
        axis.set_axisbelow(True)
        axis.grid(True, color='0.5', alpha=.16, linewidth=.5)
    save(fig,out,name)
    Path(out,f'{name}_crossovers.json').write_text(json.dumps(summaries,indent=2))
    table(pd.DataFrame(ranks),out,f'{name}_rank_instability')
    table(pd.DataFrame([dict(panel=k,**v) for k,v in summaries.items()]),out,f'{name}_crossovers')
    caption=('Gaussian nominal rate and binary-code budget are distinct quantities. '
      'Moment bands, when selected, describe Gaussian-proxy distortion and are not intervals for measured NLL. '
      'Measured excess NLL subtracts the fitted uncompressed probe, not the Bayes loss. '
      'Crossover intervals and medians condition on a positive-to-negative crossing; fractions and all point-curve roots are reported in the JSON. '
      'Ranks exclude zero rate, task_aware, and average_task. The dotted orange task-aware curves show the Gaussian solution on the left and the trained task-aware code baseline on the right; the trained baseline is not a certified global optimum. D_miss is a proxy asymptote, not a measured-NLL floor.\n'+json.dumps(summaries,indent=2))
    caption += '\nD_miss (fitted proxy) = '+str(dmiss)+' bits.\n'
    for c in (gaussian, codes):
        if c is None: continue
        band = {'moment':'moment-fitting fold resamples, with refitting',
                'test':'paired test-image resamples conditional on fitted Gaussian channels',
                'codes':'paired training-seed and test-image resamples'}[c['kind']]
        caption += f"{c['kind']}: 95% pointwise percentile bands over {len(c['draws'])} {band}; rates={c['rates'].tolist()}.\n"
    caption += 'Endpoint ranks (zero excluded): '+json.dumps(ranks)+'\n'
    Path(out,f'{name}_caption.txt').write_text(caption)
    return fig,summaries

def dsprites(csv_path, out, pretext='mixture', M=4, common=1, precise=2,
             weights=(.75,.25), snrs=(16,256), filters=None, table_betas=None, name='fig4', main_xlim=(0,10), labels=('Common', 'Precise'),
             gap_label='Common − precise (bits)', crossing_direction=1, null_control=False):
    df=pd.read_csv(csv_path); df=df[(df.pretext==pretext)&(df.M==M)].copy()
    for k,v in (filters or {}).items(): df=df[df[k]==v]
    if df.empty: raise ValueError('No matching dSprites rows')
    if df.duplicated(['beta','seed']).any(): raise ValueError('Multiple runs per beta/seed: specify filters (e.g. lr); do not select on test losses')
    for c in [common,precise]:
        if (df[f'mse{c}']<=0).any(): raise ValueError('MSE must be positive')
    df['I_common']=-.5*np.log2(df[f'mse{common}']); df['I_precise']=-.5*np.log2(df[f'mse{precise}'])
    if crossing_direction not in (-1, 1): raise ValueError('crossing_direction must be +1 or -1')
    if not np.isfinite(df[['rate_kl_bits','I_common','I_precise']].to_numpy()).all():
        raise ValueError('Rates and recoverability indices must be finite')
    df['gap']=df.I_common-df.I_precise; df['sum_I']=df.I_common+df.I_precise
    columns=['rate_kl_bits','I_common','I_precise','gap','sum_I']
    g=df.groupby('beta')[columns].agg(['mean','min','max'])
    g=g.sort_values(('rate_kl_bits','mean')); means=g.xs('mean',axis=1,level=1)
    counts=df.groupby('beta').seed.nunique()
    if counts.nunique()>1: warnings.warn('Seed counts differ across beta; see audit table')
    p,q=weights; a,b=snrs; t=(q*b-p*a)/(a*b*(p-q)) if p!=q else np.nan
    predicted=float(-np.log2(t)) if 0<t<1 else None
    fig,axs=plt.subplots(2,1,figsize=(3.3,3.8),sharex=True,gridspec_kw={'height_ratios':[2,1],'hspace':.12})
    x=means.rate_kl_bits.to_numpy(); seed_roots={}
    for col,role,m in [('I_common','specialist','o'),('I_precise','generalist','s')]:
        y=means[col].to_numpy()
        axs[0].scatter(df.rate_kl_bits,df[col],s=9,alpha=.4,color=COLORS[role],marker=m)
        axs[0].errorbar(x,y,xerr=np.array([x-g[('rate_kl_bits','min')],g[('rate_kl_bits','max')]-x]),
          yerr=np.array([y-g[(col,'min')],g[(col,'max')]-y]),color=COLORS[role],marker=m,ms=3,lw=1,capsize=2,label=labels[0 if col=='I_common' else 1])
    axs[1].axhline(0,color='.5',lw=.6,zorder=0)
    for seed,h in df.groupby('seed'):
        h=h.sort_values('rate_kl_bits'); rr=roots(h.rate_kl_bits,crossing_direction*h.gap); seed_roots[str(seed)]=rr
        axs[1].plot(h.rate_kl_bits,h.gap,color='.6',lw=.6,alpha=.7)
    axs[1].plot(x,means.gap,'o-',color='black',ms=3)
    observed=roots(x,crossing_direction*means.gap); nrev=sum(bool(v) for v in seed_roots.values())
    if null_control:
        axs[1].errorbar(x,means.gap,yerr=[means.gap-g[('gap','min')],g[('gap','max')]-means.gap],
                        fmt='none',ecolor='.35',capsize=2)
    if observed and not null_control: axs[1].axvline(observed[0],ls='--',color='black',lw=.8)
    for ax in axs:
        if predicted is not None: ax.axvline(predicted,color='.5',ls=':',lw=.8,zorder=0)
    if predicted is not None: axs[0].text(.98,.04,rf'$R^\star={predicted:.2f}$ (Gaussian reference)',ha='right',transform=axs[0].transAxes,fontsize=6)
    axs[1].text(.02,.96,('Equal weights and SNR' if null_control else
        f"Crossing: {observed[0]:.2f}" if observed else 'No expected-direction crossing'),transform=axs[1].transAxes,va='top',fontsize=6)
    axs[1].text(.02,.04,('Bars: across-seed range' if null_control else f'{nrev}/{len(seed_roots)} seeds reverse'),transform=axs[1].transAxes,fontsize=6)
    lim=max(abs(df.gap).max()*1.25,1e-3); axs[1].set_ylim(-lim,lim)
    axs[1].axhspan(0,lim,color=COLORS['specialist'],alpha=.04,zorder=0); axs[1].axhspan(-lim,0,color=COLORS['generalist'],alpha=.04,zorder=0)
    axs[1].text(.98,.85,labels[0].split(' (')[0].lower()+' better',ha='right',transform=axs[1].transAxes,fontsize=5.5)
    axs[1].text(.98,.08,labels[1].split(' (')[0].lower()+' better',ha='right',transform=axs[1].transAxes,fontsize=5.5)
    axs[0].set_ylabel('Recoverability index\n−½ log₂ MSE (bits)'); axs[0].legend(frameon=False)
    axs[1].set_ylabel(gap_label); axs[1].set_xlabel('Variational rate bound (bits/image)')
    for axis in axs:
        axis.set_axisbelow(True)
        axis.grid(True, color='0.5', alpha=.16, linewidth=.5)
    save(fig,out,name+'_fullrange')
    if main_xlim is not None:
        axs[1].set_xlim(*main_xlim)
        visible=df[(df.rate_kl_bits>=main_xlim[0])&(df.rate_kl_bits<=main_xlim[1])]
        if len(visible): axs[0].set_ylim(min(0,visible[['I_common','I_precise']].min().min())-.1, visible[['I_common','I_precise']].max().max()*1.08)
    save(fig,out,name)
    chosen=means if table_betas is None else means.loc[table_betas].sort_values('rate_kl_bits')
    display_table=chosen.copy(); display_table.columns=[r'$\mathcal R_{\mathrm{VIB}}$',r'$\mathcal I_{\mathrm{common}}$',r'$\mathcal I_{\mathrm{precise}}$','Gap',r'$\sum_i\mathcal I_i$']
    if labels != ('Common', 'Precise'):
        display_table.columns = [r'$\mathcal R_{\mathrm{VIB}}$', 'Horizontal index', 'Vertical index', 'Horizontal - vertical', r'$\sum_i\mathcal I_i$']
    # Bold both rows bracketing every observed expected-direction reversal.
    gap=chosen.gap.to_numpy(); bold=set()
    for j in range(1,len(gap)):
        if crossing_direction*gap[j-1]>0 and crossing_direction*gap[j]<0: bold.update([j-1,j])
    display_table=display_table.map(lambda x:f'{x:.2f}') if hasattr(display_table,'map') else display_table.applymap(lambda x:f'{x:.2f}')
    for j in bold: display_table.iloc[j,3]=r'\textbf{'+display_table.iloc[j,3]+'}'
    table(display_table,out,f'{name}_table',escape=False)
    audit_names = {} if labels == ('Common', 'Precise') else {'I_common':'I_horizontal', 'I_precise':'I_vertical'}
    table(df[['beta','seed']+columns].rename(columns=audit_names).sort_values(['seed','rate_kl_bits']),out,f'{name}_per_seed')
    means.reset_index().rename(columns=audit_names).to_csv(Path(out,f'{name}_means.csv'),index=False)
    table(means.reset_index().rename(columns=audit_names),out,f'{name}_all_settings')
    summary=dict(labels=labels, gap_definition=gap_label, crossing_direction=crossing_direction,
      seed_range_contains_zero_by_beta={str(beta):bool(row[('gap','min')]<=0<=row[('gap','max')]) for beta,row in g.iterrows()} if null_control else None,
      predicted_gaussian_reference=predicted,observed_mean_roots=observed,per_seed_roots=seed_roots,
      reversing_seeds=nrev,n_seeds=len(seed_roots),n_betas=len(g),weights=weights,snrs=snrs,
      rate_grid_means=x.tolist(),max_index_excess=float((df.sum_I-df.rate_kl_bits).max()))
    Path(out,f'{name}_summary.json').write_text(json.dumps(summary,indent=2))
    Path(out,f'{name}_caption.txt').write_text(f'{pretext}; M={M}; weights={weights}; SNRs={snrs}. '+
      f'{len(seed_roots)} seeds; {len(g)} beta settings. Points show individual runs; error bars show min–max across seeds at fixed beta, not confidence intervals. '+
      'The dotted line is a Gaussian reference, not an exact prediction for non-Gaussian images or the KL upper bound. '+
      'Indices are computed from MSE per run before averaging; equality to -log2(1-R²)/2 requires unit held-out variance. '+
      f'The main view uses rate limits {main_xlim}; the full-range figure retains every setting. The index sum is a model-based diagnostic, not measured mutual information. '+json.dumps(summary))
    Path(out,f'{name}_table_caption.txt').write_text('Seed-averaged indices computed per run before averaging; bold gaps bracket an observed reversal. Per-seed values and crossovers are reported separately. The sum of indices is not mutual information or a rate lower bound; excess over the KL bound diagnoses the limits of the Gaussian index for discrete factors.')
    fig2,ax=plt.subplots(figsize=(3.3,2.5)); identity_scatter(ax,df.rate_kl_bits,df.sum_I)
    ax.set(xlabel='Variational rate bound (bits/image)',ylabel='Sum of recoverability indices (bits)'); save(fig2,out,name+'_accounting')
    return df.rename(columns=audit_names),means.rename(columns=audit_names),summary

def identity_scatter(ax,x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); v=np.isfinite(x)&np.isfinite(y); x=x[v];y=y[v]
    ax.scatter(x,y,s=12,alpha=.6); low=min(x.min(),y.min());high=max(x.max(),y.max());ax.plot([low,high],[low,high],color='.5',ls='--',lw=.7)


def fit_linear_predictor(x, y):
    """OLS of measured y on proxy x, with an intercept; descriptive in-sample R²."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        raise ValueError('Expected paired one-dimensional observations')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x) == 0:
        raise ValueError('Regression requires finite data and a varying predictor')
    slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    predicted = slope*x+intercept
    total = np.sum((y-y.mean())**2)
    return dict(linear_fit_R2=float(1-np.sum((y-predicted)**2)/total) if total > 0 else None,
                slope=float(slope), intercept=float(intercept), n_pairs=len(x))


def proxy_fidelity_plot(frame, out):
    x, y = frame.proxy_excess_bits, frame.measured_excess_bits
    score = fit_linear_predictor(x, y)
    fig, ax = plt.subplots(figsize=(3.3,2.7))
    identity_scatter(ax, x, y)
    ax.lines[0].set_label('Identity')
    grid = np.array([x.min(),x.max()])
    ax.plot(grid, score['slope']*grid+score['intercept'], color='black', lw=1, label='OLS with intercept')
    label = 'undefined' if score['linear_fit_R2'] is None else f"{score['linear_fit_R2']:.3f}"
    ax.text(.03,.96,f'Linear-fit $R^2$ = {label}',transform=ax.transAxes,va='top',fontsize=7)
    ax.legend(loc='lower right',frameon=False,fontsize=6)
    ax.set(xlabel='Gaussian-proxy distortion (bits)',ylabel='Measured excess NLL (bits)')
    save(fig,out,'proxy_fidelity')
    table(pd.DataFrame([score]),out,'proxy_fidelity_score')
    return fig, score


def full_pretext_bank(gaussian, codes, out, name='full_pretext_bank'):
    """All eight pretexts and both task-aware/average-task references, in both mechanisms."""
    if set(gaussian['names']) != set(codes['names']):
        raise ValueError('Both mechanisms must contain the same pretext bank')
    names = gaussian['names']
    palette = ['#56B4E9','#CC79A7','#009E73','#999999','#332288','#882255','#44AA99']
    styles = {}; j=0
    for n in names:
        role={'joint_attributes':'specialist','reconstruction':'generalist','task_aware':'oracle'}.get(n)
        color=COLORS[role] if role else palette[j % len(palette)]
        if not role:j+=1
        styles[n]=(color, ':' if n=='task_aware' else '--' if n=='average_task' else '-',
                   ['o','s','^','v','D','P','X','<','>','h'][len(styles)%10])
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.2))
    for ax,c,title in zip(axes,[gaussian,codes],['Gaussian channel','Learned binary codes']):
        for n in names:
            color,ls,marker=styles[n]
            ax.plot(c['rates'],c['point'][c['names'].index(n)],color=color,ls=ls,marker=marker,
                    ms=3,lw=1,label=n.replace('_',' '))
        ax.set(title=title,xlabel='Nominal Gaussian rate (bits)' if c['kind']!='codes' else 'Code budget B (bits/image)',
               ylabel='Task-averaged excess NLL (bits)')
        ax.grid(alpha=.16,lw=.5);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=7)
    fig.tight_layout(rect=(0,.23,1,1))
    save(fig,out,name)
    return fig


def decoder_comparison(out, name='decoder_capacity'):
    """Use the selected-pair difference CSVs, with their saved image-bootstrap intervals."""
    fig,axes=plt.subplots(1,2,figsize=(6.75,2.8),constrained_layout=True)
    for ax,mode in zip(axes,['gaussian','quantized']):
        ax.axhline(0,color='.5',lw=.6,zorder=0)
        for label,suffix,color,marker,ls in [('Linear','nll_test',COLORS['specialist'],'o','-'),
                                           ('MLP','mlp_test',COLORS['generalist'],'s','--')]:
            frame=pd.read_csv(Path(out)/f'{mode}_{suffix}_difference.csv')
            ax.plot(frame.rate,frame.difference_bits,color=color,marker=marker,ls=ls,ms=3,label=label)
            ax.fill_between(frame.rate,frame.ci_low,frame.ci_high,color=color,alpha=.15,lw=0)
        ax.set(title='Gaussian channel' if mode=='gaussian' else 'Learned binary codes',
               xlabel='Nominal Gaussian rate (bits)' if mode=='gaussian' else 'Code budget B (bits/image)',
               ylabel='Generalist − specialist (bits)')
        ax.legend(frameon=False);ax.grid(alpha=.16,lw=.5)
    save(fig,out,name)
    return fig


def per_task(c,out,specialist,generalist='reconstruction',task_names=None,name='per_task'):
    n=c['task_point'].shape[-1]; fig,axes=plt.subplots(int(np.ceil(n/2)),2,figsize=(6.75,2.1*np.ceil(n/2)),squeeze=False)
    for t,ax in enumerate(axes.flat):
        if t>=n: ax.set_visible(False);continue
        for cand,role,marker in [(specialist,'specialist','o'),(generalist,'generalist','s')]:
            i=c['names'].index(cand); y=c['task_point'][i,:,t]; lo,hi=np.quantile(c['task_draws'][:,i,:,t],[.025,.975],axis=0)
            ax.plot(c['rates'],y,color=COLORS[role],marker=marker,ms=3,label=role);ax.fill_between(c['rates'],lo,hi,color=COLORS[role],alpha=.15)
        ax.set_title(task_names[t] if task_names else f'Task {t+1}');ax.set_xlabel('Nominal Gaussian rate (bits)' if c['kind']!='codes' else 'Code budget B (bits/image)')
        ax.set_ylabel('Proxy distortion (bits)' if c['kind']=='moment' else 'Excess NLL (bits)');ax.legend(frameon=False)
    fig.tight_layout();save(fig,out,name)

def theory_table(run,out,crossovers=None,fitted=None):
    run=Path(run);p=run/'robustness/fitted_quantities.json'
    if not p.exists(): print('SKIP theory table:',p);return
    f=fitted if fitted is not None else read_json(p); tasks=read_json(run/'selected_tasks.json') if (run/'selected_tasks.json').exists() else {}
    names=tasks.get('names',[]); rows=[]
    for i,v in enumerate(f.get('task_snr',[])): rows.append({'Quantity':f'SNR: {names[i] if i<len(names) else i+1}','Value':v})
    for key,label in [('dimension','d'),('specialist_rank','k'),('delta_uniform_subspace','Delta'),('D_miss_bits','D_miss (bits)'),('isotropic_generalist_sufficient_rate_bound','Sufficient rate bound (bits)')]:
        rows.append({'Quantity':label,'Value':f.get(key)})
    if crossovers:
        for k,s in crossovers.items(): rows.append({'Quantity':k+' observed crossover / conditional CI','Value':str(s['point'])+' / '+str(s['conditional_ci'])})
    table(pd.DataFrame(rows),out,'theory_to_data')
    Path(out,'theory_to_data_scope.txt').write_text('The threshold is sufficient, not a predicted equality. Delta and D_miss concern the fitted proxy and specified subspace. An anisotropic specialist need not allocate uniformly.\n'+json.dumps(f,indent=2))

def appendix_run(run,out,specialist,generalist='reconstruction'):
    """Render all available existing reports. Missing reports are listed, never invented."""
    run=Path(run); r=run/'robustness';out=Path(out);out.mkdir(parents=True,exist_ok=True)
    missing=[]
    def csv(name):
        p=r/name
        if not p.exists(): missing.append(str(p));return None
        return pd.read_csv(p)
    config=read_json(run/'config.json')
    changed_pair=(specialist != config.get('robustness',{}).get('specialist') or generalist != config.get('robustness',{}).get('generalist','reconstruction'))
    a=allocation_frame(run,specialist,generalist) if changed_pair else csv('allocation.csv')
    if a is not None: a.to_csv(out/'allocation.csv',index=False)
    if a is not None:
        fig,axs=plt.subplots(1,2,figsize=(6.75,2.5))
        for n,h in a.groupby('candidate'):
            h=h.sort_values('budget');axs[0].plot(h.budget,h.inside_marginal,label=n);axs[1].plot(h.budget,h.outside_conditional,label=n)
        for ax,label in zip(axs,['Information inside S (bits)','Conditional information outside S (bits)']): ax.set(xlabel='Nominal Gaussian rate (bits)',ylabel=label)
        axs[1].legend(fontsize=5);fig.tight_layout();save(fig,out,'allocation')
        (out/'allocation_scope.txt').write_text('Chain-rule decomposition: inside marginal + outside conditional = total. Outside marginal generally does not add to inside marginal when the channel couples subspaces.')
    f=csv('proxy_fidelity.csv')
    if f is not None:
        proxy_fidelity_plot(f,out)
    d=csv('rate_diagnostics.csv')
    if d is not None:
        if 'test_plugin_entropy_bits' in d:
            h=d.dropna(subset=['test_plugin_entropy_bits']);fig,ax=plt.subplots(figsize=(3.3,2.5))
            if len(h): identity_scatter(ax,h.budget,h.test_plugin_entropy_bits)
            ax.set(xlabel='Code budget B (bits/image)',ylabel='Empirical joint code entropy (bits)');save(fig,out,'code_entropy')
        bound=[c for c in d if c.endswith('covariance_gaussian_mi_bound')]
        if bound:
            fig,ax=plt.subplots(figsize=(3.3,2.5))
            for c in bound:
                h=d.dropna(subset=[c]);ax.scatter(h.budget,h[c],s=9,label=c.split('_')[0])
            hi=float(d.budget.max());ax.plot([0,hi],[0,hi],color='.5',ls='--');ax.legend(frameon=False)
            ax.set(xlabel='Nominal Gaussian rate (bits)',ylabel='Covariance-based MI upper bound (bits)');save(fig,out,'gaussian_rate_bound')
    code_usage_histograms(run, out, (specialist, generalist))
    for mode in ['gaussian','quantized']:
        fig,ax=plt.subplots(figsize=(3.3,2.5));anyplot=False
        for decoder,prefix in [('Linear',f'{mode}_nll_test'),('MLP',f'{mode}_mlp_test')]:
            if changed_pair:
                try: h=decoder_difference(run,mode,decoder,specialist,generalist)
                except FileNotFoundError as error:
                    missing.append(str(error));h=None
            else: h=csv(prefix+'_difference.csv')
            if h is not None: h.to_csv(out/(prefix+'_difference.csv'),index=False)
            if h is None:continue
            # Existing reports use difference_bits; accommodate earlier diff_bits schema.
            yc=next((k for k in ['difference_bits','diff_bits','difference','gen_minus_spec','generalist_minus_specialist'] if k in h),None)
            if yc is None: raise ValueError(f'Unknown difference columns: {h.columns.tolist()}')
            ax.plot(h.rate,h[yc],marker='o',ms=3,label=decoder);anyplot=True
            if {'ci_low','ci_high'}<=set(h):ax.fill_between(h.rate,h.ci_low,h.ci_high,alpha=.15)
        if anyplot:
            ax.axhline(0,color='.5',lw=.6);ax.legend();ax.set(xlabel='Nominal Gaussian rate (bits)' if mode=='gaussian' else 'Code budget B (bits/image)',ylabel='Generalist − specialist (bits)');save(fig,out,mode+'_decoder_overlay')
        else:plt.close(fig)
    if all((out/f'{mode}_{suffix}_difference.csv').is_file() for mode in ['gaussian','quantized'] for suffix in ['nll_test','mlp_test']):
        decoder_comparison(out)
    (out/'missing_optional_reports.txt').write_text('\n'.join(missing));print('Missing optional reports:',*missing,sep='\n')

def whitening_plot(covariance_path,out,shrinkage_csv=None):
    """covariance_path: .npy square covariance (already in bottleneck coordinates).
    Optional CSV columns: delta, rate, candidate, distortion.
    """
    cov=np.load(covariance_path)
    if cov.ndim!=2 or cov.shape[0]!=cov.shape[1]:raise ValueError('Supply square covariance .npy')
    fig,ax=plt.subplots(figsize=(3.3,2.5));ax.plot(np.linalg.eigvalsh(cov)[::-1],'o-',ms=3);ax.axhline(1,color='.5',ls='--')
    ax.set(xlabel='Eigenvalue index',ylabel='Whitened covariance eigenvalue');save(fig,out,'whitening_spectrum')
    if shrinkage_csv:
        f=pd.read_csv(shrinkage_csv);fig,ax=plt.subplots(figsize=(3.3,2.5))
        for (cand,delta),h in f.groupby(['candidate','delta']):
            h=h.sort_values('rate');ax.plot(h.rate,h.distortion,label=f'{cand}, δ={delta:g}')
        ax.legend(fontsize=5);ax.set(xlabel='Nominal Gaussian rate (bits)',ylabel='Proxy distortion (bits)');save(fig,out,'shrinkage_sensitivity')

def worst_case(curves_csv,out,snr_csv=None):
    """CSV: rate, optimal, fixed. Optional sweep CSV: L, max_relative_regret
    (fraction, not percent). Fixed curve must use one pretext over the whole grid.
    """
    d=pd.read_csv(curves_csv).sort_values('rate')
    if (d.optimal<=0).any():raise ValueError('Relative regret requires positive optimum')
    fig,axs=plt.subplots(1,2,figsize=(6.75,2.5));axs[0].plot(d.rate,d.optimal,label='Rate-dependent optimum');axs[0].plot(d.rate,d.fixed,'--',label='Best fixed pretext');axs[0].set_yscale('log');axs[0].legend()
    axs[1].plot(d.rate,100*(d.fixed/d.optimal-1),color=COLORS['generalist']);axs[1].set_ylabel('Relative regret (%)');axs[0].set_ylabel('Proxy distortion (bits/task)')
    for ax in axs:ax.set_xlabel('Nominal Gaussian rate (bits)')
    fig.tight_layout();save(fig,out,'worst_case')
    if snr_csv:
        s=pd.read_csv(snr_csv).sort_values('L'); t=np.log1p(s.L.to_numpy()); q=np.ones_like(t); nz=t>1e-7;q[nz]=np.expm1(t[nz])/t[nz]*np.exp(t[nz]/np.expm1(t[nz])-1)
        fig,ax=plt.subplots(figsize=(3.3,2.5));ax.plot(s.L,100*s.max_relative_regret,'o-',label='Observed');ax.plot(s.L,100*(q-1),'--',label='Specht bound');ax.legend();ax.set(xlabel='Maximum task SNR L',ylabel='Maximum relative regret (%)');save(fig,out,'snr_sweep')

def dsprites_context(csv_path,out,M=4):
    df=pd.read_csv(csv_path);df=df[df.M==M].copy();df['I1']=-.5*np.log2(df.mse1);df['I2']=-.5*np.log2(df.mse2)
    fig,axs=plt.subplots(1,2,figsize=(6.75,2.5))
    for pretext,g in df.groupby('pretext'):
        h=g.groupby('beta')[['rate_kl_bits','I1','I2']].mean().sort_values('rate_kl_bits')
        for i,ax in enumerate(axs):ax.plot(h.rate_kl_bits,h[f'I{i+1}'],'o-',ms=3,label=pretext)
    for i,ax in enumerate(axs):ax.set(xlabel='Variational rate bound (bits/image)',ylabel=f'Factor {i+1} recoverability (bits)')
    axs[1].legend(fontsize=5);fig.tight_layout();save(fig,out,'dsprites_context')

def run_main(celeba_run,out,specialist,generalist='reconstruction',mixture=None,gaussian_metric='proxy',gaussian_path=None,codes_path=None,oracle='task_aware'):
    run=Path(celeba_run);r=run/'robustness';out=Path(out);out.mkdir(parents=True,exist_ok=True)
    taskfile=run/'selected_tasks.json'; tasks=read_json(taskfile) if taskfile.exists() else {};weights=tasks.get('weights')
    gp=Path(gaussian_path) if gaussian_path else r/('moment_bootstrap/all_curves.npz' if gaussian_metric=='proxy' else 'gaussian_nll_test_bootstrap.npz')
    cp=Path(codes_path) if codes_path else r/'quantized_nll_seeds_test_bootstrap.npz'
    gauss=load_curves(gp,'moment' if gaussian_metric=='proxy' else 'test',weights) if gp.exists() else None
    codes=load_curves(cp,'codes',weights) if cp.exists() else None
    if gauss is None and codes is None:raise FileNotFoundError(f'Neither {gp} nor {cp} exists')
    fit,_,_=fitted_geometry(run,specialist,generalist)
    (out/'fitted_quantities.json').write_text(json.dumps(fit,indent=2))
    if gauss is not None and codes is not None:
        full_pretext_bank(gauss,codes,out)
    fig,s=figure3(gauss,codes,out,specialist,generalist,mixture,oracle=oracle,dmiss=fit.get('D_miss_bits'))
    row_order=rank_summary(gauss if gauss is not None else codes)[1]
    for name,c in [('gaussian',gauss),('codes',codes)]:
        if c is None:continue
        per_task(c,out,specialist,generalist,tasks.get('names'),name+'_per_task')
        fig,ax=plt.subplots(figsize=(6.75,max(3.2,.30*len(c['names'])+1.6)));heatmap(ax,c,row_order=row_order);fig.tight_layout();save(fig,out,name+'_rank_bank')
    theory_table(run,out,s,fitted=fit);appendix_run(run,out,specialist,generalist)
    metadata={'specialist':specialist,'generalist':generalist,'gaussian_source':str(gp),'codes_source':str(cp),'gaussian_metric':gaussian_metric,'task_weights':weights,
      'bootstrap_draws':{n:len(c['draws']) for n,c in [('gaussian',gauss),('codes',codes)] if c is not None}}
    config=run/'config.json'
    if config.exists():metadata['run_config']=read_json(config)
    (out/'plot_provenance.json').write_text(json.dumps(metadata,indent=2))
    return s


"""Recompute pair-dependent diagnostics from saved fits, without retraining."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from celeba_experiment.robust_statistics import allocation, paired_bootstrap, specialist_geometry


def fitted_geometry(run, specialist, generalist):
    run = Path(run)
    with np.load(run / 'gaussian_tasks.npz') as saved:
        tasks = saved['H']
    with np.load(run / f'pretext_proxy_{specialist}.npz') as saved:
        candidate = {key: saved[key] for key in ('H', 'weights')}
    inside, outside, fitted = specialist_geometry(tasks, candidate)
    with np.load(run / f'pretext_proxy_{generalist}.npz') as saved:
        matrices = saved['H']
    d = tasks.shape[-1]
    fitted['generalist_components_isotropic_full_space'] = bool(all(
        np.linalg.norm(h - np.trace(h) / d * np.eye(d)) <= 1e-8 * max(1., np.linalg.norm(h))
        for h in matrices))
    fitted.update(specialist=specialist, generalist=generalist)
    return fitted, inside, outside


def allocation_frame(run, specialist, generalist):
    run = Path(run)
    _, inside, outside = fitted_geometry(run, specialist, generalist)
    rows = json.loads((run / 'results.json').read_text())
    result, seen = [], set()
    for row in rows:
        key = row['candidate'], row['budget']
        if row['mode'] != 'gaussian' or key in seen:
            continue
        seen.add(key)
        with np.load(run / row['directory'] / 'channel.npz') as saved:
            values = allocation(saved['K'], inside, outside)
        result.append(dict(candidate=key[0], budget=key[1], **values))
    return pd.DataFrame(result)


def decoder_difference(run, mode, decoder, specialist, generalist):
    """Paired image bootstrap; baseline cancels in the within-draw difference."""
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    robust = config['robustness']
    rows = [r for r in json.loads((run / 'results.json').read_text())
            if r['mode'] == mode and r['candidate'] in (specialist, generalist)]
    rates = sorted({r['budget'] for r in rows})
    seeds = sorted({r['seed'] for r in rows})
    lookup = {(r['seed'], r['candidate'], r['budget']): r for r in rows}
    filename = 'mlp_per_image_losses.npz' if decoder == 'MLP' else 'per_image_losses.npz'
    cube = []
    for seed in seeds:
        candidates = []
        for name in (specialist, generalist):
            losses = []
            for rate in rates:
                row = lookup[seed, name, rate]  # Missing paired cells must fail.
                with np.load(run / row['directory'] / filename) as saved:
                    losses.append(saved['test_ce_bits'])
            candidates.append(losses)
        cube.append(candidates)
    cube = np.asarray(cube)
    draws = paired_bootstrap(cube, robust['test_resamples'], robust['seed'] + 1)
    low, high = np.quantile((draws[:, 1] - draws[:, 0]).mean(-1), [.025, .975], axis=0)
    point = (cube[:, 1] - cube[:, 0]).mean((0, 2, 3))
    return pd.DataFrame(dict(rate=rates, difference_bits=point, ci_low=low, ci_high=high))


def dsprites_control(csv_path, out, control, **kwargs):
    """Keep physical axes fixed: horizontal minus vertical in both controls.

    Swapping ordered targets makes task 1 vertical/common and task 2
    horizontal/precise. The physical gap therefore reverses negative-to-positive;
    the role-based common-minus-precise gap still reverses positive-to-negative.
    """
    frame = pd.read_csv(csv_path)
    frame = frame[(frame.pretext == 'mixture') & (frame.M == kwargs.get('M', 4))]
    expected = ('posY', 'posX') if control == 'role_swap' else ('posX', 'posY')
    if control not in ('role_swap', 'null_control'):
        raise ValueError('Unknown control')
    for column, value in zip(('factor1', 'factor2'), expected):
        if column not in frame or frame.empty or not frame[column].eq(value).all():
            raise ValueError(f'{control} requires {column}={value}; do not relabel main-run results')
    if control == 'null_control':
        for column, value in dict(task_weight1=.5, task_weight2=.5,
                                  noise_variance1=1/16, noise_variance2=1/16).items():
            if column not in frame or not np.allclose(frame[column], value):
                raise ValueError(f'Null control requires recorded {column}={value}')
        result = dsprites(csv_path, out, weights=(.5,.5), snrs=(16,16),
                          labels=('Horizontal', 'Vertical'), gap_label='Horizontal − vertical (bits)',
                          null_control=True, name=control, **kwargs)
    else:
        for column, value in dict(task_weight1=.75, task_weight2=.25,
                                  noise_variance1=1/16, noise_variance2=1/256).items():
            if column in frame and not np.allclose(frame[column], value):
                raise ValueError(f'Role swap requires {column}={value}')
        result = dsprites(csv_path, out, common=2, precise=1, weights=(.25,.75), snrs=(256,16),
                          labels=('Horizontal (precise)', 'Vertical (common)'),
                          gap_label='Horizontal − vertical (bits)', crossing_direction=-1,
                          name=control, **kwargs)
    caption = Path(out, f'{control}_caption.txt')
    with caption.open('a') as stream:
        stream.write('\nPhysical factor identities are held fixed: horizontal minus vertical. ')
        stream.write('The expected swap reversal is negative-to-positive; its occurrence is measured, never enforced.'
                     if control == 'role_swap' else
                     'Gap bars are seed ranges, not confidence bands; containing zero is descriptive, not a calibrated null test. No outcome is required or forced.')
    return result


def code_usage_histograms(run, out, candidates):
    """Export retained occupancy counts as vector pages; never pool seeds."""
    from matplotlib.backends.backend_pdf import PdfPages
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = [r for r in read_json(run/'results.json')
               if r['mode']=='quantized' and r['candidate'] in candidates]
    missing, pages = [], 0
    with PdfPages(out/'code_usage_histograms.pdf') as pdf:
        for row in records:
            source = run/row['directory']/'code_usage.npz'
            if not source.exists():
                missing.append(str(source)); continue
            with np.load(source) as data:
                counts = data['sorted_counts']
            fig, ax = plt.subplots(figsize=(3.3,2.5))
            ax.hist(counts, bins=min(30,max(1,len(np.unique(counts)))), color='.4')
            ax.set(xlabel='Test observations per observed codeword', ylabel='Number of codewords',
                   title=f"{row['candidate']}: B={row['budget']}, seed={row['seed']}")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig); pages += 1
    (out/'code_usage_caption.txt').write_text(
        f'{pages} pages, one per candidate/budget/seed. Test-image occupancy of observed codewords; '
        'unobserved codewords are excluded. Sparse occupancy limits plug-in entropy estimates.\n'
        + '\n'.join(missing))
    return dict(pages=pages, missing=missing)
