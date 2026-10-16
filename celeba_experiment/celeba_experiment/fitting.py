"""Gaussian working models: moment matching and probe-information calibration."""
import numpy as np
from .probes import entropy,whiten_fit

def fit_tasks(x,y,probe,validation_metrics):
    h=np.asarray(validation_metrics['label_entropy_bits']);ce=np.asarray(validation_metrics['ce_bits'])
    gain=np.clip(h-ce,0,h);w=probe['weights'];norm=np.linalg.norm(w,axis=1)
    u=np.divide(w,norm[:,None],out=np.zeros_like(w),where=norm[:,None]>1e-12)
    gain=np.where(norm>1e-12,gain,0);gamma=np.expm1(2*np.log(2)*gain)
    H=gamma[:,None,None]*u[:,:,None]*u[:,None,:]
    # Ordinary Gaussian moment approximation to centered binary responses.
    p=y.mean(0);b=(x.T@(y-p)/len(x)).T
    residual=np.maximum(p*(1-p)-(b*b).sum(1),1e-6)
    moment=b[:,:,None]*b[:,None,:]/residual[:,None,None]
    return {'H_ce':H,'H_moment':moment,'gamma':gamma,'gain_bits':gain,
            'entropy_bits':h,'ce_floor_bits':h-gain,'probe_direction':u,
            'moment_noise':residual,'train_prevalence':p}

def fit_cross_view(x,targets,rho_max=.995):
    # Input x was whitened using anchor TRAIN data. Targets whiten separately.
    target=np.concatenate(targets);source=np.tile(x,(len(targets),1))
    mean,W,eig=whiten_fit(target)
    whitened=(target-mean)@W
    cross=source.T@whitened/len(source)
    U,rho,Vt=np.linalg.svd(cross,full_matrices=False)
    clipped=np.minimum(rho,rho_max)
    gamma=clipped**2/(1-clipped**2)
    return {'H':(U*gamma)@U.T,'canonical_correlations_raw':rho,
            'canonical_correlations':clipped,'cross_correlation':cross,
            'target_mean':mean,'target_whitener':W,'target_covariance_eigenvalues':eig,
            'clipped_count':np.array((rho>rho_max).sum())}
