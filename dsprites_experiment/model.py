"""Fixed architecture, stochastic Gaussian channel, and lossless custom renderer."""
import math
from task_spec import task_parameters
import torch
from torch import nn
from torch.nn import functional as F

LN2 = math.log(2)

class Renderer(nn.Module):
    def __init__(self, size=64, kind='linear'):
        super().__init__()
        if size < 16 or size % 8: raise ValueError('image_size must be >=16 and divisible by 8')
        x = torch.linspace(-1,1,size)
        yy, xx = torch.meshgrid(x,x,indexing='ij')
        a = torch.exp(-((xx+.4)**2+yy**2)/.09)
        b = torch.exp(-((xx-.4)**2+yy**2)/.09)
        # Identical reflected shapes: no built-in factor preference.
        basis = torch.stack([a,b])
        basis = basis / basis.square().mean((1,2),keepdim=True).sqrt()
        self.register_buffer('basis', basis)
        if kind not in ('linear','asinh'): raise ValueError('Unknown renderer')
        self.kind = kind
    def forward(self,z):
        x = torch.einsum('bi,ihw->bhw', z, self.basis).unsqueeze(1)/2
        # Optional strictly monotone, bounded transform. No clipping/quantization.
        return x if self.kind == 'linear' else torch.asinh(x)

class Predictor(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width,128), nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,4))
    def forward(self,c):
        mean, raw = self.net(c).chunk(2,-1)
        return mean, F.softplus(raw)+1e-7

class VIB(nn.Module):
    def __init__(self, size=64, width=8):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(1,16,5,2,2),nn.SiLU(),nn.Conv2d(16,32,3,2,1),nn.SiLU(),
            nn.Conv2d(32,32,3,2,1),nn.SiLU(),nn.Flatten(),nn.Linear(32*(size//8)**2,128),nn.SiLU())
        self.channel = nn.Linear(128,2*width)
        self.task = Predictor(width)
        self.reconstruction = nn.Sequential(nn.Linear(width,128),nn.SiLU(),nn.Linear(128,size*size))
        self.size = size
    def encode(self,x):
        mu,lv = self.channel(self.backbone(x)).chunk(2,-1)
        lv = lv.clamp(-24,12)
        kl = .5*(mu.square()+lv.exp()-1-lv).sum(-1)/LN2
        c = mu + (.5*lv).exp()*torch.randn_like(mu)
        return c, kl
    def decode_image(self,c):
        return self.reconstruction(c).view(-1,1,self.size,self.size)

def excess_nll(mean, residual, z, noise):
    """E_e[-log2 N(z+e;mean,residual+noise)] - h2(e).

    Integrates independent task noise exactly (Rao-Blackwellization).
    This preserves the noisy-label population objective without sampling tiny noise.
    """
    total = residual + noise
    return .5*(torch.log(total/noise) + ((z-mean).square()+noise)/total - 1)/LN2

def objective(net, x, z, M, pretext, beta, recon_variance, recon_loss="mse", task_weights=None, noise_variances=None):
    task_weights, noise_variances = task_parameters(M, task_weights, noise_variances)
    c,kl = net.encode(x)
    if pretext == 'reconstruction':
        # Per-image likelihood; variance and sum convention fixed across all runs.
        if recon_loss=='bce':
            d = F.binary_cross_entropy_with_logits(net.decode_image(c),x,reduction='none').flatten(1).mean(1)/LN2
        elif recon_loss=='mse':
            d = (net.decode_image(c)-x).square().flatten(1).sum(1)/(2*recon_variance*LN2)
        else: raise ValueError(f'Unknown reconstruction loss: {recon_loss}')
    else:
        mean,residual = net.task(c)
        noise = z.new_tensor(noise_variances)
        loss = excess_nll(mean,residual,z,noise)
        weights = {'specialist1':[1.,0.], 'specialist2':[0.,1.], 'mixture':task_weights}[pretext]
        d = loss @ z.new_tensor(weights)
    return (d+beta*kl).mean(), d.mean(),kl.mean()
