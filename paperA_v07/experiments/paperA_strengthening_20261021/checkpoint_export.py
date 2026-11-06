"""Atomic coefficient checkpoint and bounded-memory, same-CUDA-formula field export."""
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def replace_with_retry(source, destination):
    for attempt in range(40):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.15)


def save_checkpoint(path, coef_c, coef_r, aff, shape, spacing, thetas, K, KR, K_a, result, argv):
    path = Path(path)
    state = dict(schema=1, coef_c=coef_c.detach().cpu(), coef_r=coef_r.detach().cpu(),
                 aff=aff.detach().cpu() if aff is not None else None,
                 shape=list(shape), spacing=[float(s) for s in spacing],
                 thetas=[float(t) for t in thetas], K=K, KR=KR, K_a=K_a,
                 result=json.loads(json.dumps(result)), argv=list(argv),
                 model_sha256=hashlib.sha256(Path(sys.modules['pmr_extended'].__file__).read_bytes()).hexdigest())
    temporary = path.with_suffix('.pt.tmp')
    try:
        # A Python file handle also supports Windows paths containing Chinese characters.
        with temporary.open('wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print('CHECKPOINT committed; registration can resume at field export', flush=True)


def export_checkpoint(state, destination):
    import pmr_extended as model
    destination = Path(destination)
    temporary = destination.with_suffix('.npy.partial')
    shape = tuple(state['shape'])
    coef_c = state['coef_c'].to(model.device)
    coef_r = state['coef_r'].to(model.device)
    aff = state['aff'].to(model.device) if state['aff'] is not None else None
    coords = model.make_coords((1, 1) + shape, state['K_a'])
    spacing = np.asarray(state['spacing'])[:, None, None, None]
    array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                      shape=(len(state['thetas']), 3) + shape)
    try:
        with torch.no_grad():
            for i, theta in enumerate(state['thetas']):
                # Exactly the original sequence: coarse+affine, then residual, then physical units.
                base = model.disp_full(coef_c, theta, state['K'], shape).squeeze(0)
                if aff is not None:
                    base = base + model.affine_disp_b(aff, theta, coords, state['K_a']).squeeze(0)
                field = base + model.disp_full(coef_r, theta, state['KR'], shape).squeeze(0)
                array[i] = (field.cpu().numpy() * spacing).astype(np.float32)
                array.flush()
                del base, field
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f'EXPORT phase {i+1}/{len(state["thetas"])} committed', flush=True)
        del array
        replace_with_retry(temporary, destination)
    finally:
        # Failed partial exports remain explicitly partial, never treated as complete.
        del coef_c, coef_r, aff, coords
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

