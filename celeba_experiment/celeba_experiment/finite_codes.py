"""Learned deterministic B-bit encoders; stochasticity only during optimization.

Task-aware: binary attribute NLL. Pretext-aware: Gaussian feature NLL with a
learned full residual covariance, or an asymmetric cross-view InfoNCE critic.
These are empirical optima, not globally optimal finite-code solutions.
"""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BinaryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bits):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, bits))

    def forward(self, x, straight_through=False):
        logits = self.net(x)
        hard = (logits >= 0).to(logits.dtype)
        if straight_through:
            soft = torch.sigmoid(logits)
            return soft + (hard - soft).detach()
        return hard


class CodePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, bits, target_dim, objective):
        super().__init__()
        self.objective = objective
        self.encoder = BinaryEncoder(input_dim, hidden_dim, bits)
        self.decoder = nn.Sequential(nn.Linear(bits, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, target_dim))
        if objective == 'gaussian_nll':
            self.raw_cholesky = nn.Parameter(torch.zeros(target_dim, target_dim))
        if objective == 'infonce':
            self.target_head = nn.Linear(target_dim, target_dim, bias=False)

    def loss(self, x, targets, straight_through=False, temperature=.2):
        pred = self.decoder(self.encoder(x, straight_through))
        # Each sample receives one independently cached target draw per step.
        if self.objective == 'task_bce':
            return F.binary_cross_entropy_with_logits(pred, targets)
        if self.objective == 'gaussian_nll':
            L = torch.tril(self.raw_cholesky, diagonal=-1) + torch.diag(
                F.softplus(torch.diag(self.raw_cholesky)) + 1e-3)
            residual = torch.linalg.solve_triangular(L, (targets-pred).T, upper=False)
            return (.5 * residual.square().sum(0).mean() + torch.log(torch.diag(L)).sum()
                    + .5 * targets.shape[1] * math.log(2*math.pi)) / targets.shape[1]
        if self.objective == 'infonce':
            scores = F.normalize(pred, dim=-1) @ F.normalize(self.target_head(targets), dim=-1).T / temperature
            labels = torch.arange(len(x), device=x.device)
            return .5 * (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels))
        raise ValueError(f'Unknown objective: {self.objective}')


class CompositePredictor(nn.Module):
    """Separate prediction heads with one shared hard-bit encoder."""
    def __init__(self, input_dim, hidden_dim, bits, heads):
        super().__init__()
        self.encoder = BinaryEncoder(input_dim, hidden_dim, bits)
        self.head_specs = heads
        self.predictors = nn.ModuleList()
        for head in heads:
            predictor = CodePredictor(input_dim, hidden_dim, bits, head['target_dim'], head['objective'])
            predictor.encoder = self.encoder
            self.predictors.append(predictor)

    def loss(self, x, targets, straight_through=False, temperature=.2):
        offset, loss = 0, 0.
        for spec, predictor in zip(self.head_specs, self.predictors):
            end = offset + spec['target_dim']
            loss = loss + spec['weight'] * predictor.loss(x, targets[:, offset:end], straight_through, temperature)
            offset = end
        return loss


def fit_codes(features, targets, bits, objective, cfg, seed, checkpoint):
    """targets[split] has shape (views, images, target_dim); no test fitting."""
    if not isinstance(bits, int) or bits < 0:
        raise ValueError('Finite-code budgets must be nonnegative integer bits')
    if bits == 0:
        # A one-valued feature for probe code that expects at least one column.
        return {s: np.zeros((len(x), 1), np.float32) for s, x in features.items()}, []
    torch.manual_seed(seed)
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(cfg.get('threads', 4))
    x = {s: torch.as_tensor(features[s], dtype=torch.float32, device=device) for s in ['train', 'valid']}
    y = {s: torch.as_tensor(targets[s], dtype=torch.float32, device=device) for s in ['train', 'valid']}
    hidden = cfg.get('hidden_dim', 128)
    heads = cfg.get('heads')
    model = (CompositePredictor(x['train'].shape[1], hidden, bits, heads) if heads else
             CodePredictor(x['train'].shape[1], hidden, bits, y['train'].shape[-1], objective)).to(device)
    contrastive = objective == 'infonce' or any(h['objective'] == 'infonce' for h in heads or [])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', .001), weight_decay=cfg.get('weight_decay', 1e-4))
    batch = cfg.get('batch_size', 256)
    epochs = cfg.get('epochs', 30)
    if epochs < 1 or batch < 2:
        raise ValueError('Need positive epochs and batch_size >= 2')
    best, state, history = float('inf'), None, []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x['train']), device=device)
        total, count = 0., 0
        for start in range(0, len(order), batch):
            ids = order[start:start+batch]
            if contrastive and len(ids) < 2:
                continue
            draw = torch.randint(y['train'].shape[0], (len(ids),), device=device)
            loss = model.loss(x['train'][ids], y['train'][draw, ids], True, cfg.get('temperature', .2))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite finite-code training loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            total += loss.item() * len(ids)
            count += len(ids)
        model.eval()
        vt, vn = 0., 0
        with torch.no_grad():
            for view in y['valid']:
                for start in range(0, len(x['valid']), batch):
                    xx, yy = x['valid'][start:start+batch], view[start:start+batch]
                    if contrastive and len(xx) < 2:
                        continue
                    val = model.loss(xx, yy, False, cfg.get('temperature', .2)).item()
                    vt += val * len(xx)
                    vn += len(xx)
        if not count or not vn:
            raise ValueError('Insufficient samples for objective')
        val = vt/vn
        history.append({'epoch': epoch+1, 'train_loss_nats': total/count, 'valid_hard_loss_nats': val})
        if val < best:
            best, state = val, copy.deepcopy(model.state_dict())
    if state is None:
        raise RuntimeError('No finite validation checkpoint')
    model.load_state_dict(state)
    model.cpu().eval()
    torch.save({'state_dict': model.state_dict(), 'input_dim': features['train'].shape[1],
                'hidden_dim': hidden, 'bits': bits, 'target_dim': targets['train'].shape[-1],
                'objective': objective, 'heads': heads, 'seed': seed, 'config': cfg}, checkpoint)
    codes = {}
    with torch.no_grad():
        for split, values in features.items():
            codes[split] = np.concatenate([model.encoder(torch.as_tensor(values[j:j+1024], dtype=torch.float32)).numpy()
                                           for j in range(0, len(values), 1024)])
    return codes, history


def load_encoder(checkpoint):
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = (CompositePredictor(data['input_dim'], data['hidden_dim'], data['bits'], data['heads']) if data.get('heads') else
             CodePredictor(data['input_dim'], data['hidden_dim'], data['bits'], data['target_dim'], data['objective']))
    model.load_state_dict(data['state_dict'])
    return model.encoder.eval()


def pack_codes(codes, bits):
    """Byte-padded storage of exactly B meaningful binary coordinates per image."""
    if bits == 0:
        return np.empty((len(codes), 0), dtype=np.uint8)
    if codes.shape[1] != bits or not np.isin(codes, [0, 1]).all():
        raise ValueError('Expected exactly B binary coordinates')
    return np.packbits(codes.astype(np.uint8), axis=1, bitorder='little')
