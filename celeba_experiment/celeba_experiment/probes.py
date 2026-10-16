"""Independent binary logistic probes, with train-only hyperparameter selection."""
import numpy as np
from scipy.special import expit, xlogy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def entropy(p):return -(xlogy(p,p)+xlogy(1-p,1-p))/np.log(2)

def losses(logits,y):return (np.logaddexp(0,logits)-y*logits).mean(axis=0)/np.log(2)

def one_fit(x,y,C,max_iter):
    if C==0 or np.unique(y).size<2:
        p=(y.sum()+.5)/(len(y)+1)
        return np.zeros(x.shape[1]),float(np.log(p/(1-p)))
    model=LogisticRegression(C=C,max_iter=max_iter,solver='lbfgs',tol=1e-7)
    model.fit(x,y)
    return model.coef_[0],float(model.intercept_[0])

def fit_probes(x,y,cfg,seed=42):
    rng=np.random.default_rng(seed);perm=rng.permutation(len(x));cut=int(cfg.get('fit_fraction',.8)*len(x))
    fit,select=perm[:cut],perm[cut:]
    weights=[];bias=[];chosen=[]
    for t in range(y.shape[1]):
        candidates=[]
        for C in cfg['C_grid']:
            w,b=one_fit(x[fit],y[fit,t],C,cfg.get('max_iter',500))
            candidates.append(float(losses(x[select]@w+b,y[select,t])))
        C=cfg['C_grid'][int(np.argmin(candidates))]
        w,b=one_fit(x,y[:,t],C,cfg.get('max_iter',500))
        weights.append(w);bias.append(b);chosen.append(C)
    return {'weights':np.stack(weights),'bias':np.array(bias),'C':np.array(chosen)}

def evaluate(model,x,y):
    logits=x@model['weights'].T+model['bias'];ce=losses(logits,y)
    auc=[float(roc_auc_score(y[:,t],logits[:,t])) if np.unique(y[:,t]).size==2 else None for t in range(y.shape[1])]
    return {'ce_bits':ce.tolist(),'mean_ce_bits':float(ce.mean()),
            'accuracy':((logits>=0)==y).mean(axis=0).tolist(),'auc':auc,
            'label_entropy_bits':entropy(y.mean(axis=0)).tolist()}


def whiten_fit(x,floor=1e-7):
    mean=x.mean(axis=0);center=x-mean
    cov=center.T@center/len(x);eig,U=np.linalg.eigh(cov)
    threshold=max(float(eig.max())*floor,1e-12)
    W=(U*(1/np.sqrt(np.maximum(eig,threshold))))@U.T
    return mean,W,eig
