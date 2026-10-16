#!/usr/bin/env python3
"""Fine-tune ResNet-18 on ALL official CelebA training images, then cache features.

python train_celeba_multitask.py --output runs/celeba_multitask --data-root data
Independent of bottleneck configs/caches; never subsamples the training split.
Requires ../requirements.txt. See README.md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
from scipy.fft import dctn
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms


TASK_NAMES = ['luminance_mean', 'red_minus_green', 'blue_minus_green',
              'luminance_std', 'luminance_top_minus_bottom',
              'luminance_left_minus_right', 'luminance_dct_0_1',
              'luminance_dct_1_0', 'luminance_dct_1_1', 'luminance_dct_2_0']

SPLIT_COUNTS = {'train': 162770, 'valid': 19867, 'test': 19962}


def continuous_targets(crop):
    """Same definitions as the corresponding diagnostic tasks; image before normalization."""
    rgb = np.asarray(crop, dtype=np.float64)/255.
    lum = rgb @ np.array([.2126, .7152, .0722])
    small = np.asarray(crop.resize((32, 32), Image.Resampling.BILINEAR), dtype=float)/255.
    spec = dctn(small @ np.array([.2126, .7152, .0722]), norm='ortho')/32.
    return np.array([lum.mean(), (rgb[..., 0]-rgb[..., 1]).mean(),
                     (rgb[..., 2]-rgb[..., 1]).mean(), lum.std(),
                     lum[:112].mean()-lum[112:].mean(),
                     lum[:, :112].mean()-lum[:, 112:].mean(),
                     spec[0, 1], spec[1, 0], spec[1, 1], spec[2, 0]], dtype=np.float32)


class CelebATasks(Dataset):
    def __init__(self, root, split, download):
        self.base = datasets.CelebA(str(root), split=split, target_type='attr', download=download)
        if len(self.base) != SPLIT_COUNTS[split]:
            raise ValueError(f'{split}: expected full official split ({SPLIT_COUNTS[split]} images), '
                             f'got {len(self.base)}. Subsampled datasets are not accepted.')
        self.crop = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                                        transforms.CenterCrop(224)])
        self.tensor = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
        self.target_cache = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, attr = self.base[index]
        crop = self.crop(image.convert('RGB'))
        target = continuous_targets(crop) if self.target_cache is None else self.target_cache[index]
        return self.tensor(crop), attr.float(), torch.from_numpy(target), index


class MultitaskResNet(nn.Module):
    def __init__(self, imagenet=True, n_attributes=40):
        super().__init__()
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if imagenet else None)
        self.encoder.fc = nn.Identity()
        self.attribute_head = nn.Linear(512, n_attributes)
        self.continuous_head = nn.Linear(512, len(TASK_NAMES))

    def forward(self, image):
        z = self.encoder(image)
        return self.attribute_head(z), self.continuous_head(z)


def make_loader(dataset, args, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=args.workers, pin_memory=args.device.startswith('cuda'), generator=generator)


def compute_normalization(dataset, args):
    """One full train pass, no validation/test data; cache tiny targets for subsequent epochs."""
    targets = np.empty((len(dataset), len(TASK_NAMES)), dtype=np.float32)
    count = 0
    for _, _, y, idx in make_loader(dataset, args):
        targets[idx.numpy()] = y.numpy()
        count += len(idx)
        if count % (args.batch_size*100) == 0:
            print(f'Train target statistics: {count}/{len(dataset)}', flush=True)
    if count != len(dataset):
        raise RuntimeError('Normalization did not cover every training image')
    mean, scale = targets.astype(float).mean(0), targets.astype(float).std(0)
    if np.any(scale < 1e-10):
        raise ValueError('Constant continuous training target')
    dataset.target_cache = targets
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(scale, dtype=torch.float32), targets


def epoch_pass(model, loader, mean, scale, args, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    sums = np.zeros(3)
    n = 0
    # Per-task validation measurements remain in standardized target units.
    sqerr = np.zeros(len(TASK_NAMES)); ysum = sqerr.copy(); ysq = sqerr.copy()
    for step, (image, attributes, target, _) in enumerate(loader):
        image, attributes, target = [a.to(args.device, non_blocking=True) for a in (image, attributes, target)]
        target = (target-mean)/scale
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type='cuda', enabled=args.amp and args.device.startswith('cuda')):
                logits, prediction = model(image)
                attr_loss = nn.functional.binary_cross_entropy_with_logits(logits, attributes)
                cont_loss = nn.functional.mse_loss(prediction.float(), target)
                loss = args.attribute_weight*attr_loss+args.continuous_weight*cont_loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training/validation loss')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update()
        batch = len(image); n += batch
        sums += batch*np.array([loss.item(), attr_loss.item(), cont_loss.item()])
        sqerr += ((prediction.detach().float()-target)**2).sum(0).cpu().numpy()
        ysum += target.sum(0).cpu().numpy(); ysq += (target**2).sum(0).cpu().numpy()
        if training and (step+1) % 100 == 0:
            print(f'  trained {n}/{len(loader.dataset)} images; loss={sums[0]/n:.5f}', flush=True)
    if n != len(loader.dataset):
        raise RuntimeError('Epoch did not cover every image')
    variance = np.maximum(ysq/n-(ysum/n)**2, 1e-12)
    return dict(loss=float(sums[0]/n), attribute_bce=float(sums[1]/n),
                continuous_mse=float(sums[2]/n), samples=n,
                continuous_r2=(1-sqerr/n/variance).tolist())


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export_features(model, args, train, valid, checkpoint):
    """Caps affect export only, never optimizer updates or checkpoint selection."""
    out = args.output/'feature_cache'
    out.mkdir(exist_ok=False)
    fingerprint = dict(checkpoint_sha256=file_hash(checkpoint),
                       preprocessing='Resize256 bilinear; CenterCrop224; ImageNet normalize',
                       counts=[args.export_train_count, args.export_valid_count, args.export_test_count], seed=args.seed)
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    manifest = dict(signature=signature, backbone='resnet18_celeba_multitask', seed=args.seed,
                    attribute_names=[a for a in train.base.attr_names if a], policies=[],
                    splits={}, training_split_count=len(train), source=fingerprint)
    targets = {'task_names': np.array(TASK_NAMES)}
    model.eval()
    for split, dataset, cap in [('train', train, args.export_train_count),
                                ('valid', valid, args.export_valid_count),
                                ('test', None, args.export_test_count)]:
        if dataset is None:
            dataset = CelebATasks(args.data_root, split, download=not args.no_download)
        count = min(cap, len(dataset)) if cap else len(dataset)
        rng = np.random.default_rng(args.seed+{'train':0,'valid':1,'test':2}[split])
        indices = np.sort(rng.choice(len(dataset), count, replace=False)) if cap else np.arange(len(dataset))
        subset = torch.utils.data.Subset(dataset, indices.tolist())
        features = np.empty((count, 512), np.float32)
        labels = np.empty((count, 40), np.int8)
        ys = np.empty((count, len(TASK_NAMES)), np.float32)
        offset = 0
        with torch.inference_mode():
            for image, attr, target, idx in make_loader(subset, args):
                z = model.encoder(image.to(args.device, non_blocking=True)).float().cpu().numpy()
                end = offset+len(z)
                if not np.array_equal(indices[offset:end], idx.numpy()):
                    raise RuntimeError('Export ordering mismatch')
                features[offset:end], labels[offset:end], ys[offset:end] = z, attr.numpy(), target.numpy()
                offset = end
        np.savez_compressed(out/f'{split}.npz', anchor=features, labels=labels, indices=indices)
        targets[split], targets[split+'_indices'] = ys, indices
        manifest['splits'][split] = dict(count=count, official_split_count=len(dataset))
        print(f'Exported {split}: {count} frozen 512-dimensional features', flush=True)
    np.savez_compressed(out/'continuous_targets.npz', **targets)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Feature cache: {out.resolve()}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--attribute-weight', type=float, default=1.)
    p.add_argument('--continuous-weight', type=float, default=1.)
    p.add_argument('--grad-clip', type=float, default=5.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--no-download', action='store_true')
    p.add_argument('--resume', action='store_true', help='Resume last.pt in the same output directory')
    p.add_argument('--export-only', action='store_true', help='Export best.pt from an existing output directory')
    p.add_argument('--skip-export', action='store_true')
    for split in ['train', 'valid', 'test']:
        p.add_argument(f'--export-{split}-count', type=int, default=0,
                       help='Optional feature-export cap only; 0 exports all. Never changes training/validation.')
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.workers < 0:
        p.error('Invalid epoch/batch/thread/worker counts')
    if any(not np.isfinite(v) or v <= 0 for v in [args.lr, args.attribute_weight, args.continuous_weight, args.grad_clip]):
        p.error('Learning rate, loss weights, and gradient clip must be positive and finite')
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error('Weight decay must be nonnegative and finite')
    if min(args.export_train_count, args.export_valid_count, args.export_test_count) < 0:
        p.error('Export counts must be nonnegative')
    if args.export_only and args.skip_export:
        p.error('--export-only conflicts with --skip-export')
    if args.data_root.name == 'celeba':
        args.data_root = args.data_root.parent
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    continuing = args.resume or args.export_only
    if continuing:
        old = json.loads((args.output/'config.json').read_text())
        if not args.export_only:
            for key in ['epochs', 'lr', 'weight_decay', 'attribute_weight', 'continuous_weight', 'seed', 'batch_size']:
                if config[key] != old[key]:
                    p.error(f'Resume must preserve {key}={old[key]}')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    train = CelebATasks(args.data_root, 'train', download=not args.no_download)
    valid = CelebATasks(args.data_root, 'valid', download=not args.no_download)
    print(f'Using ALL {len(train)} official training images and {len(valid)} validation images. '
          'No bottleneck sample limits are read.', flush=True)
    model = MultitaskResNet(imagenet=not continuing).to(args.device)
    best_path = args.output/'best.pt'
    if not args.export_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        use_amp = args.amp and args.device.startswith('cuda')
        scaler = (torch.amp.GradScaler('cuda', enabled=use_amp) if hasattr(torch.amp, 'GradScaler')
                  else torch.cuda.amp.GradScaler(enabled=use_amp))
        start, best = 0, float('inf')
        if args.resume:
            state = torch.load(args.output/'last.pt', map_location=args.device, weights_only=True)
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
            start, best = state['epoch']+1, state['best_validation_loss']
            mean, scale = state['target_mean'], state['target_scale']
            with np.load(args.output/'train_targets.npz') as z:
                train.target_cache = z['targets']
        else:
            mean, scale, targets = compute_normalization(train, args)
            np.savez_compressed(args.output/'train_targets.npz', targets=targets, task_names=np.array(TASK_NAMES))
        mean, scale = mean.to(args.device), scale.to(args.device)
        for epoch in range(start, args.epochs):
            torch.manual_seed(args.seed+epoch)
            started = time.perf_counter()
            tr = epoch_pass(model, make_loader(train, args, shuffle=True, seed=args.seed+epoch), mean, scale, args, optimizer, scaler)
            va = epoch_pass(model, make_loader(valid, args), mean, scale, args)
            improved = va['loss'] < best
            best = min(best, va['loss']); scheduler.step()
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         scaler=scaler.state_dict(), epoch=epoch, best_validation_loss=best,
                         target_mean=mean.cpu(), target_scale=scale.cpu(), task_names=TASK_NAMES,
                         training_count=len(train), config=config)
            if improved:
                save_checkpoint(best_path, state)
            save_checkpoint(args.output/'last.pt', state)
            record = dict(epoch=epoch+1, seconds=time.perf_counter()-started, train=tr, valid=va)
            # One file per epoch avoids duplicate log rows after an interrupted resume.
            (args.output/f'epoch_{epoch+1:03d}.json').write_text(json.dumps(record, indent=2)+'\n')
            print(f'Epoch {epoch+1}/{args.epochs}: train={tr["loss"]:.5f}, valid={va["loss"]:.5f}, '
                  f'continuous valid R2={np.mean(va["continuous_r2"]):.3f}', flush=True)
    best_state = torch.load(best_path, map_location=args.device, weights_only=True)
    model.load_state_dict(best_state['model'])
    save_checkpoint(args.output/'backbone.pt', dict(state_dict=model.encoder.state_dict(), architecture='resnet18',
                    feature_dim=512, selected_epoch=best_state['epoch']+1, task_names=TASK_NAMES,
                    training_count=len(train), preprocessing='Resize256 bilinear; CenterCrop224; ImageNet normalize'))
    if not args.skip_export:
        export_features(model, args, train, valid, best_path)
    print(f'Best checkpoint epoch: {best_state["epoch"]+1}; output: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
