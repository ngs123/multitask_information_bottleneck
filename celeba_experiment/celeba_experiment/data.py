import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms as T


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def set_backbone_checkpoint(config, path):
    """CLI override; checkpoint paths in JSON otherwise resolve from the working directory."""
    if path is not None:
        config['backbone'] = 'resnet18_celeba_multitask'
        config['backbone_checkpoint'] = str(Path(path).expanduser().resolve())


def backbone_identity(config):
    name = config['backbone']
    if name == 'resnet18_imagenet1k_v1':
        if config.get('backbone_checkpoint'):
            raise ValueError('backbone_checkpoint requires backbone=resnet18_celeba_multitask')
        return None
    if name != 'resnet18_celeba_multitask':
        raise ValueError(f'Unsupported backbone: {name}')
    if not config.get('backbone_checkpoint'):
        raise ValueError('Multitask backbone requires backbone_checkpoint or --backbone-checkpoint')
    path = Path(config['backbone_checkpoint']).expanduser()
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'architecture': 'resnet18', 'feature_dim': 512}


def load_backbone(config):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    if config['backbone'] == 'resnet18_imagenet1k_v1':
        encoder = models.resnet18(weights=weights)
        encoder.fc = nn.Identity()
    elif config['backbone'] == 'resnet18_celeba_multitask':
        checkpoint = torch.load(Path(config['backbone_checkpoint']).expanduser(),
                                map_location='cpu', weights_only=True)
        if checkpoint.get('architecture') != 'resnet18' or checkpoint.get('feature_dim') != 512:
            raise ValueError('Expected backbone.pt exported by train_celeba_multitask.py (ResNet-18, 512 features)')
        encoder = models.resnet18(weights=None)
        encoder.fc = nn.Identity()
        encoder.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        raise ValueError(f'Unsupported backbone: {config["backbone"]}')
    encoder.eval().requires_grad_(False)
    # Both backbones were trained with this deterministic anchor preprocessing.
    return encoder, weights.transforms()


class RegionCrop:
    """Label-free relative crops for aligned CelebA images; picklable for workers."""
    def __init__(self, region):
        if region not in ('upper', 'lower', 'mixed'):
            raise ValueError(f'Unknown region: {region}')
        self.region = region

    def __call__(self, image):
        region = random.choice(['upper', 'lower']) if self.region == 'mixed' else self.region
        w, h = image.size
        top, bottom = (0.05, 0.62) if region == 'upper' else (0.40, 0.97)
        shift = random.uniform(-0.025, 0.025)
        return image.crop((int(.08*w), int((top+shift)*h), int(.92*w), int((bottom+shift)*h)))


def view_transform(policy):
    ops = []
    if policy.get('region'):
        ops.append(RegionCrop(policy['region']))
    ops.extend([
        T.RandomResizedCrop(224,scale=tuple(policy.get('crop_scale', [.9, 1.])),antialias=True),
        T.RandomHorizontalFlip(policy.get('flip',.5)),
        T.ColorJitter(brightness=policy.get('jitter',0),contrast=policy.get('jitter',0),
                      saturation=policy.get('jitter',0),hue=min(.1,policy.get('jitter',0)/4))])
    if policy.get('grayscale', False):
        ops.append(T.Grayscale(num_output_channels=3))
    if policy.get('low_resolution'):
        ops.extend([T.Resize((policy['low_resolution'], policy['low_resolution']), antialias=True),
                    T.Resize((224, 224), antialias=True)])
    if policy.get('blur_sigma'):
        ops.append(T.GaussianBlur(23, sigma=tuple(policy['blur_sigma'])))
    ops.extend([T.ToTensor(), T.Normalize([.485,.456,.406],[.229,.224,.225])])
    return T.Compose(ops)



class ViewDataset(Dataset):
    def __init__(self,base,indices,transform,seed):
        self.base,self.indices,self.transform,self.seed=base,indices,transform,seed
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        # Reproducible cached views, independent of worker scheduling.
        seed_all(self.seed+int(self.indices[i]))
        image,label=self.base[int(self.indices[i])]
        return self.transform(image),label


def extract(config,cache_dir):
    cfg=config['data'];cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device(config.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(config.get('threads',4))
    identity = backbone_identity(config)
    signature_fields = {'transform_version':2,'data':cfg,'seed':config['seed'],
        'policies':config['policies'],'backbone':config['backbone']}
    if identity is not None:
        signature_fields['backbone_checkpoint'] = identity
    # Preserve existing ImageNet signatures; custom checkpoints are content-addressed.
    signature=hashlib.sha256(json.dumps(signature_fields,sort_keys=True).encode()).hexdigest()
    marker=cache_dir/'manifest.json'
    if marker.exists():
        old=json.loads(marker.read_text());old['attribute_names']=[a for a in old['attribute_names'] if a]
        if old['signature']!=signature:raise ValueError('Cache configuration changed: use another cache directory.')
        if all((cache_dir/f'{s}.npz').exists() for s in ['train','valid','test']):
            return old
    encoder, anchor_transform = load_backbone(config)
    encoder.to(device)
    manifest={'signature':signature,'backbone':config['backbone'],'seed':config['seed'],
              'policies':config['policies'],'splits':{},'device':str(device)}
    if identity is not None:
        manifest['backbone_checkpoint'] = identity
    for split in ['train','valid','test']:
        base=datasets.CelebA(cfg['root'],split=split,target_type='attr',download=cfg.get('download',True))
        rng=np.random.default_rng(config['seed']+{'train':0,'valid':1,'test':2}[split])
        limit=cfg.get('max_'+split)
        indices=np.sort(rng.choice(len(base),min(limit or len(base),len(base)),replace=False))
        manifest['attribute_names']=[a for a in base.attr_names if a]
        outputs={};labels=None
        transforms=[('anchor',anchor_transform)]
        for i,policy in enumerate(config['policies']):
            for v in range(cfg.get('views_per_policy',1)):
                transforms.append((f'target_{i}_{v}',view_transform(policy)))
        for j,(name,transform) in enumerate(transforms):
            d=ViewDataset(base,indices,transform,config['seed']+j*1000000)
            loader=DataLoader(d,batch_size=cfg.get('batch_size',64),num_workers=cfg.get('workers',2),
                              shuffle=False,pin_memory=device.type=='cuda')
            arrays=[];ys=[]
            with torch.inference_mode():
                for n,(image,y) in enumerate(loader):
                    arrays.append(encoder(image.to(device)).cpu().numpy());ys.append(y.numpy())
                    if n%20==0:print(f'features {split}/{name}: {min((n+1)*loader.batch_size,len(d))}/{len(d)}',flush=True)
            outputs[name]=np.concatenate(arrays).astype(np.float32)
            labels=np.concatenate(ys).astype(np.int8)
        outputs['labels']=labels;outputs['indices']=indices
        np.savez_compressed(cache_dir/f'{split}.npz',**outputs)
        manifest['splits'][split]={'count':len(indices),'official_split_count':len(base)}
    marker.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
