"""Disjoint folds of dSprites images, or the optional Gaussian renderer control."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from model import Renderer
from prepare_dsprites import check_cache

FACTOR_COLUMNS={'scale':2,'posX':4,'posY':5}
FACTOR_SIZES={'scale':6,'posX':32,'posY':32}
FOLDS=('train','val','probe_train','probe_val','test')


def grouped_split(groups,counts,seed):
    """Randomize unique-image groups; retain all copies within exactly one fold.

    Sizes may exceed requested counts by less than one group's multiplicity.
    """
    groups=np.asarray(groups)
    order=np.argsort(groups,kind='stable')
    unique,starts,lengths=np.unique(groups[order],return_index=True,return_counts=True)
    rng=np.random.default_rng(seed); perm=rng.permutation(len(unique)); cursor=0; result={}
    for name,target in zip(FOLDS,counts):
        pieces=[]; size=0
        while size<target:
            if cursor==len(perm): raise ValueError('Requested folds exceed available unique-image groups')
            g=perm[cursor]; cursor+=1
            pieces.append(order[starts[g]:starts[g]+lengths[g]]); size+=lengths[g]
        indices=np.concatenate(pieces); rng.shuffle(indices); result[name]=indices
    return result


class Sprites(Dataset):
    def __init__(self,root,indices,factors):
        self.images=np.load(Path(root)/'images_packed.npy',mmap_mode='r',allow_pickle=False)
        self.classes=np.load(Path(root)/'classes.npy',mmap_mode='r',allow_pickle=False)
        self.indices=np.asarray(indices,dtype=np.int64); self.factors=factors
        self.group_ids=np.load(Path(root)/'groups.npy',mmap_mode='r',allow_pickle=False)[self.indices]
        self.columns=[FACTOR_COLUMNS[f] for f in factors]
        sizes=np.array([FACTOR_SIZES[f] for f in factors],dtype=np.float32)
        self.mean=(sizes-1)/2; self.std=np.sqrt((sizes**2-1)/12)
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        row=self.indices[i]
        image=np.unpackbits(self.images[row],axis=-1).astype(np.float32)
        target=(self.classes[row,self.columns].astype(np.float32)-self.mean)/self.std
        return torch.from_numpy(image[None]),torch.from_numpy(target)


def make_data(cfg,folder,device):
    sizes=[cfg[k] for k in ('n_train','n_val','n_probe','n_val','n_test')]
    seed=100000+cfg['seed']*10
    dataset=cfg.get('dataset','gaussian')
    if dataset=='gaussian':
        folds={name:TensorDataset(torch.randn(n,2,generator=torch.Generator().manual_seed(seed+i)))
               for i,(name,n) in enumerate(zip(FOLDS,sizes))}
        renderer=Renderer(cfg['image_size'],cfg['renderer']).to(device)
        return folds,renderer,{'dataset':'gaussian','factors':['S1','S2'],'split_seed':seed}
    if dataset!='dsprites': raise ValueError(f'Unknown dataset {dataset}')
    if cfg['image_size']!=64: raise ValueError('dSprites requires image_size=64; no resizing is applied')
    factors=cfg.get('factors',['posX','posY'])
    if len(factors)!=2 or len(set(factors))!=2 or not set(factors)<=set(FACTOR_COLUMNS):
        raise ValueError('Choose two distinct factors from posX, posY, scale; orientation/shape are not scalar tasks')
    root=Path(cfg['data_dir']); meta=check_cache(root)
    groups=np.load(root/'groups.npy',mmap_mode='r',allow_pickle=False)
    indices=grouped_split(groups,sizes,seed)
    np.savez_compressed(folder/'split_indices.npz',**indices)
    folds={name:Sprites(root,ix,factors) for name,ix in indices.items()}
    provenance={'dataset':'dsprites','factors':factors,'split_seed':seed,
                'source_sha256':meta['source_sha256'],'actual_fold_sizes':{k:len(v) for k,v in indices.items()},
                'split_sha256':hashlib.sha256(b''.join(indices[k].tobytes() for k in FOLDS)).hexdigest(),
                'normalization':'known full-factor uniform class mean and standard deviation',
                'duplicate_policy':meta['split_grouping'],'factor_levels':[FACTOR_SIZES[f] for f in factors]}
    (folder/'data_provenance.json').write_text(json.dumps(provenance,indent=2))
    return folds,None,provenance


def to_batch(batch,renderer,device):
    if len(batch)==1:
        z=batch[0].to(device); return renderer(z),z
    x,z=batch
    return x.to(device),z.to(device)
