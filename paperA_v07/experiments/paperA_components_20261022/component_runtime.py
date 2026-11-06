"""Lossless phase-field snapshots and shared model initialisation."""
from pathlib import Path
import time,gc
import numpy as np
import torch

def save_base(path,fields,spacing):
    start=time.perf_counter();path=Path(path)
    a=np.lib.format.open_memmap(path.with_suffix('.npy.partial'),mode='w+',dtype=np.float32,shape=(10,)+tuple(fields[0].shape))
    with torch.no_grad():
        for i,field in enumerate(fields):
            a[i]=(field.detach().cpu().numpy()*np.asarray(spacing)[:,None,None,None]).astype(np.float32)
            a.flush()
    del a
    path.with_suffix('.npy.partial').replace(path)
    elapsed=time.perf_counter()-start
    print(f'BASE SNAPSHOT: ten phases saved in {elapsed:.2f}s',flush=True)
    return elapsed

def initialise_factory(original):
    def create(*args,**kwargs):
        if kwargs.get('out_ch')!=12:return original(*args,**kwargs)
        full_kwargs=dict(kwargs,out_ch=24)
        full=original(*args,**full_kwargs)
        rng=torch.get_rng_state()
        net=original(*args,**kwargs)
        net.enc.load_state_dict(full.enc.state_dict())
        # The frozen channel layout is harmonic-major: xyz cosine, xyz sine.
        with torch.no_grad():
            net.out.weight.copy_(full.out.weight[:12]);net.out.bias.copy_(full.out.bias[:12])
        if net.affine:net.fc.load_state_dict(full.fc.state_dict())
        torch.set_rng_state(rng)
        del full
        return net
    return create
