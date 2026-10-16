"""Download the official dSprites archive once and build a shared, packed cache.

No pickle loading. Images are streamed from the NPZ into a ~360 MiB bit-packed
memory-mappable array; each training process decodes only the images it uses.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile
import numpy as np

URL='https://raw.githubusercontent.com/google-deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
SHA256='857dc001287e8efa226e0025fac6b29e5745c621b380b0d3363e4e33f4e28fa2'
VERSION=1

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def check_cache(root):
    root=Path(root)
    manifest=root/'dataset.json'
    if not manifest.exists(): raise FileNotFoundError(f'Missing dSprites cache: run prepare_dsprites.py --out {root}')
    meta=json.loads(manifest.read_text())
    if meta['cache_version']!=VERSION or meta['source_sha256']!=SHA256:
        raise ValueError('Incompatible dSprites cache; prepare a new cache directory')
    for file,shape in [('images_packed.npy',(737280,64,8)),('classes.npy',(737280,6)),('groups.npy',(737280,))]:
        a=np.load(root/file,mmap_mode='r',allow_pickle=False)
        if a.shape!=shape: raise ValueError(f'Unexpected shape in {file}: {a.shape}')
    return meta

def prepare(root, source=None):
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    if (root/'dataset.json').exists(): return check_cache(root)
    lock=root/'prepare.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise RuntimeError(f'Preparation already locked: {lock}; check for a stale process before deleting it')
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    try:
        archive=Path(source).resolve() if source else root/'source.npz'
        if not archive.exists():
            if source: raise FileNotFoundError(archive)
            temp=root/'download.partial'
            print('Downloading official dSprites archive...',flush=True)
            with urllib.request.urlopen(URL,timeout=120) as response, temp.open('wb') as out:
                while True:
                    block=response.read(1024*1024)
                    if not block: break
                    out.write(block)
            temp.replace(archive)
        digest=sha256_file(archive)
        if digest!=SHA256: raise ValueError(f'dSprites archive SHA256 mismatch: {digest}')
        temp=root/'images_packed.tmp.npy'
        packed=np.lib.format.open_memmap(temp,mode='w+',dtype=np.uint8,shape=(737280,64,8))
        hashes=np.empty(737280,dtype='S32')
        with zipfile.ZipFile(archive) as zf, zf.open('imgs.npy') as f:
            version=np.lib.format.read_magic(f)
            if version==(1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
            elif version==(2,0): shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
            else: raise ValueError(f'Unsupported NPY header: {version}')
            if shape!=(737280,64,64) or fortran or dtype not in (np.dtype('uint8'),np.dtype('bool')):
                raise ValueError(f'Unexpected image schema: {shape}, {dtype}, fortran={fortran}')
            for lo in range(0,737280,2048):
                n=min(2048,737280-lo)
                raw=f.read(n*4096*dtype.itemsize)
                images=np.frombuffer(raw,dtype=dtype).reshape(n,64,64)
                if np.any(images>1): raise ValueError('Expected binary dSprites images')
                block=np.packbits(images,axis=-1)
                packed[lo:lo+n]=block
                hashes[lo:lo+n]=[hashlib.sha256(row.tobytes()).digest() for row in block]
        packed.flush(); del packed; temp.replace(root/'images_packed.npy')
        with np.load(archive,allow_pickle=False) as npz:
            classes=npz['latents_classes']
            if classes.shape!=(737280,6): raise ValueError('Unexpected latent class shape')
            sizes=np.array([1,3,6,40,32,32])
            if np.any(classes<0) or np.any(classes>=sizes): raise ValueError('Invalid latent class values')
            np.save(root/'classes.npy',classes.astype(np.int16))
        _,groups=np.unique(hashes,return_inverse=True)
        np.save(root/'groups.npy',groups.astype(np.int32))
        meta={'name':'dsprites','cache_version':VERSION,'source_url':URL,'source_sha256':digest,
              'n_images':737280,'n_unique_images':int(groups.max()+1),
              'latent_names':['color','shape','scale','orientation','posX','posY'],
              'latent_sizes':[1,3,6,40,32,32],
              'split_grouping':'SHA256 of exact binary pixels; identical images stay in one fold',
              'files_sha256':{name:sha256_file(root/name) for name in ['images_packed.npy','classes.npy','groups.npy']}}
        tmp=root/'dataset.tmp.json'; tmp.write_text(json.dumps(meta,indent=2)); tmp.replace(root/'dataset.json')
        print(f'Ready: {root}; {meta["n_unique_images"]} unique images',flush=True)
        return meta
    finally: lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',default='data/dsprites'); p.add_argument('--npz')
    a=p.parse_args(); prepare(a.out,a.npz)
