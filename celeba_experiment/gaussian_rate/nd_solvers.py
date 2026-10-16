"""Gaussian task-family rate distortion in arbitrary dimension (bits).

The projected iterations are in S=-log K. No claim of Euclidean convexity
in S is needed: a geodesic Lagrangian residual supplies an a posteriori gap.
"""
import numpy as np
from scipy.optimize import brentq

LN2=np.log(2.)

def sym(A):return (A+A.T)/2

def simplex(v,total):
    if total==0:return np.zeros_like(v)
    u=np.sort(v)[::-1];c=np.cumsum(u)-total
    j=np.nonzero(u-c/np.arange(1,len(u)+1)>0)[0][-1]
    return np.maximum(v-c[j]/(j+1),0)

def project(S,total):
    s,U=np.linalg.eigh(sym(S));return (U*simplex(s,total))@U.T

class GaussianFamily:
    def __init__(self,H,weights=None):
        self.H=np.asarray(H,float)
        if self.H.ndim!=3 or self.H.shape[1]!=self.H.shape[2]:raise ValueError('H must be (tasks,d,d)')
        if not np.isfinite(self.H).all() or not np.allclose(self.H,self.H.transpose(0,2,1)):raise ValueError('H must be finite symmetric')
        self.d=self.H.shape[1]
        self.weights=np.ones(len(H))/len(H) if weights is None else np.asarray(weights,float)
        if np.any(self.weights<0) or not np.isclose(self.weights.sum(),1):raise ValueError('Weights must be probabilities')
        self.factors=[]
        for h in self.H:
            a,U=np.linalg.eigh(h)
            if a.min() < -1e-8:raise ValueError('H must be PSD')
            keep=a>1e-12
            self.factors.append(U[:,keep]*np.sqrt(a[keep]))
        self.rank_one=all(f.shape[1]<=1 for f in self.factors)
        if self.rank_one:self.V=np.stack([f[:,0] if f.shape[1] else np.zeros(self.d) for f in self.factors])
    def value_grad(self,K):
        if self.rank_one:
            q=np.einsum('ti,ij,tj->t',self.V,K,self.V)
            return float(self.weights@np.log1p(q)/(2*LN2)),sym((self.V.T*(self.weights/(1+q)))@self.V/(2*LN2))
        value=0.;grad=np.zeros_like(K)
        for w,F in zip(self.weights,self.factors):
            A=np.eye(F.shape[1])+F.T@K@F
            value+=w*np.linalg.slogdet(A)[1]/(2*LN2)
            grad+=w*(F@np.linalg.solve(A,F.T))/(2*LN2)
        return float(value),sym(grad)
    def value(self,K):return self.value_grad(K)[0]

def from_S(S):
    s,U=np.linalg.eigh(sym(S));return (U*np.exp(-s))@U.T,s,U

def s_value_grad(family,S):
    K,s,U=from_S(S);v,G=family.value_grad(K)
    delta=np.abs(s[:,None]-s[None,:]);coef=np.exp(-np.minimum(s[:,None],s[None,:]))
    ratio=np.ones_like(delta);np.divide(-np.expm1(-delta),delta,out=ratio,where=delta>1e-10)
    return v,-U@((coef*ratio)*(U.T@G@U))@U.T,K

def certificate(family,K,R):
    k,U=np.linalg.eigh(sym(K));root=(U*np.sqrt(k))@U.T
    _,G=family.value_grad(K);M=2*LN2*root@G@root;MU=U.T@M@U
    active=k<1-1e-7;inactive=~active
    lam=max(0.,float(np.diag(MU)[active].mean())) if active.any() else float(np.linalg.eigvalsh(M).max())
    dual=np.zeros_like(K)
    if inactive.any():
        Ui=U[:,inactive];a,V=np.linalg.eigh((lam*np.eye(inactive.sum())-MU[np.ix_(inactive,inactive)])/(2*LN2))
        dual=Ui@(V*np.maximum(a,0))@V.T@Ui.T
    residual=(M-lam*np.eye(family.d))/(2*LN2)+root@dual@root
    rate=-np.log(k).sum()/(2*LN2)
    gap=max(0.,lam*(R-rate)+np.trace(dual@(np.eye(family.d)-K)))+2*R*LN2*np.sqrt(family.d)*np.linalg.norm(residual)
    return {'gap_bound':float(gap),'kkt_residual':float(np.linalg.norm(residual)),'rate':float(rate),'lambda':lam}

def solve(family,R,warm_S=None,tol=1e-6,maxiter=5000):
    if R<0:raise ValueError('Rate must be nonnegative')
    d=family.d;total=2*LN2*R
    if R==0:return {'K':np.eye(d),'S':np.zeros((d,d)),'D':family.value(np.eye(d)),'iterations':0,'converged':True,**certificate(family,np.eye(d),R)}
    S=np.eye(d)*total/d if warm_S is None else project(warm_S,total)
    step=10.;lastS=None;lastG=None;info={}
    for iteration in range(1,maxiter+1):
        val,G,K=s_value_grad(family,S)
        if iteration%10==1:
            info=certificate(family,K,R)
            if info['gap_bound']<=tol:break
        if lastS is not None:
            ds=S-lastS;dg=G-lastG;den=np.sum(ds*dg)
            if den>1e-18:step=np.clip(np.sum(ds*ds)/den,1e-3,1e7)
        for _ in range(60):
            candidate=project(S-step*G,total);delta=candidate-S
            newval=family.value(from_S(candidate)[0])
            if newval<=val+np.sum(G*delta)+np.sum(delta*delta)/(2*step)+1e-15:break
            step*=.5
        lastS,lastG=S,G;S=candidate
    K=from_S(S)[0];info=certificate(family,K,R)
    return {'K':K,'S':S,'D':family.value(K),'iterations':iteration,'converged':info['gap_bound']<=tol,**info}

def single_task(H,R):
    """Exact Gaussian IB for one surrogate task; returns residual covariance."""
    h,U=np.linalg.eigh(sym(np.asarray(H,dtype=float)));h=np.maximum(h,0)
    if R==0 or h.max()<1e-14:return np.eye(len(h))
    # Solve for log(lambda/(1-lambda)): direct root finding in lambda
    # loses relative accuracy at high rates when lambda is extremely small.
    positive=h>0
    logh=np.log(h[positive])
    def bits(logwater):return .5*np.maximum((logh-logwater)/LN2,0)
    upper=float(logh.max())
    lower=float(logh.min()-2*LN2*R)
    logwater=brentq(lambda a:bits(a).sum()-R,lower,upper,xtol=1e-12)
    k=np.ones_like(h);k[positive]=2**(-2*bits(logwater))
    return (U*k)@U.T

def channel_matrix(K):
    k,U=np.linalg.eigh(sym(K));return (U*np.sqrt(np.maximum(1/k-1,0)))@U.T
