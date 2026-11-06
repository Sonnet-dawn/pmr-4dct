"""Source-linked, same-run evaluation used by both registration protocols."""
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from reliable_io import atomic_text

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = Path(os.environ.get('PMR_DATA_ROOT', ROOT.parent / 'reference_4' / 'data')).resolve()
os.environ['PMR_DATA_ROOT'] = str(DATA)
sys.path.insert(0, str(HERE / 'frozen'))
from jacobian_stats import det_jacobian_array
from recompute_tre_standard import sample_trilinear
from lung_mask import mask_at_down, raw_case


def write_json(path, value):
    path = Path(path)
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def image_path(dataset, cn, phase):
    if dataset == 'dirlab':
        return DATA / 'DIRLAB' / 'mha' / f'case{cn}' / f'case{cn}_T{phase}0_R.mha'
    return DATA / 'CREATIS' / str(cn) / f'{phase*10:02d}_R.mha'


def reference_image(dataset, cn):
    im = raw_case(cn, 0, dataset)
    assert np.allclose(im.GetDirection(), np.eye(3).ravel()), 'This study requires reoriented image data.'
    assert np.allclose(im.GetSpacing(), [1, 1, 1]), 'Expected archived 1-mm image copies.'
    rs = sitk.ResampleImageFilter()
    rs.SetSize([s // 2 for s in im.GetSize()])
    rs.SetOutputSpacing([2, 2, 2])
    rs.SetOutputOrigin(im.GetOrigin()); rs.SetOutputDirection(im.GetDirection())
    rs.SetInterpolator(sitk.sitkLinear)
    return rs.Execute(im)


def landmarks(dataset, cn):
    if dataset == 'dirlab':
        base = DATA / 'DIRLAB' / 'points' / f'case{cn}'
        paths = [base / f'case{cn}_300_T{p}0_xyz_R.txt' for p in [0, 5]]
    else:
        base = DATA / 'CREATIS' / str(cn)
        paths = [base / f'{p:02d}.pts' for p in [0, 50]]
    a, b = [np.loadtxt(p).reshape(-1, 3) for p in paths]
    assert a.shape == b.shape
    expected_count = 300 if dataset == 'dirlab' else [100, 100, 100, 100, 107, 113][cn]
    assert a.shape[0] == expected_count, (dataset, cn, a.shape[0], expected_count)
    og = np.array(raw_case(cn, 0, dataset).GetOrigin())
    # DIR-Lab landmark coordinates are relative to its zero-origin reoriented images.
    if dataset == 'dirlab':
        assert np.allclose(og, 0)
    return a - og, b - og, paths


def common_mask(dataset, cn):
    m = (mask_at_down(cn, 2, 0, dataset=dataset) > .5) | (mask_at_down(cn, 2, 5, dataset=dataset) > .5)
    return ndi.maximum_filter(m.astype(np.uint8), size=5, mode='constant') > 0


def slab_jacobian(field_mm, spacing, block=12):
    """Use the project's canonical Jacobian implementation with one-slice halos."""
    depth = field_mm.shape[1]
    out = np.empty(field_mm.shape[1:], dtype=np.float32)
    for start in range(0, depth, block):
        stop = min(depth, start + block)
        lo, hi = max(0, start-1), min(depth, stop+1)
        det = det_jacobian_array(field_mm[:, lo:hi], spacing=spacing)
        out[start:stop] = det[start-lo:stop-lo]
    return out


def evaluate(field_path, job, result, outdir):
    t0 = time.monotonic()
    dataset, cn = job['dataset'], job['case']
    im = reference_image(dataset, cn)
    spacing = np.array(im.GetSpacing())
    field = np.load(field_path, mmap_mode='r')
    if field.ndim == 4:
        field = field[None]
        phases = [5]
    else:
        phases = list(range(10))
        assert field.shape[0] == 10
    assert field.shape[1] == 3 and tuple(field.shape[2:]) == tuple(reversed(im.GetSize()))
    a, b, lm_paths = landmarks(dataset, cn)
    idx = a / spacing
    assert np.all(idx >= 0) and np.all(idx <= np.array(im.GetSize()) - 1), 'Landmarks outside fixed grid.'
    endpoint = field[phases.index(5)]
    predicted = a + sample_trilinear(endpoint, a, spacing)
    error = np.linalg.norm(predicted - b, axis=1)
    motion = np.linalg.norm(b - a, axis=1)
    expected = result.get('tre_total_STANDARD', result.get('tre_dense_mm'))
    assert abs(float(error.mean()) - expected) < 2e-5, (float(error.mean()), expected)
    mask = common_mask(dataset, cn)
    per_phase = []
    for number, phase in enumerate(phases):
        d = field[number]
        assert np.isfinite(d).all(), 'Nonfinite displacement field'
        determinant = slab_jacobian(d, spacing)
        inside = determinant[mask]
        per_phase.append(dict(phase=phase, det_min=float(determinant.min()),
                              pct_nonpositive=float((determinant <= 0).mean()*100),
                              det_min_lung=float(inside.min()),
                              pct_nonpositive_lung=float((inside <= 0).mean()*100)))
    recs = []
    for j in range(len(a)):
        row = dict(job=job['id'], dataset=dataset, case=cn, seed=job['seed'], arm=job['arm'],
                   landmark=j, motion_mm=float(motion[j]), tre_mm=float(error[j]))
        for i, axis in enumerate('xyz'):
            row[f'ref_{axis}_mm'] = float(a[j, i])
            row[f'target_{axis}_mm'] = float(b[j, i])
            row[f'pred_{axis}_mm'] = float(predicted[j, i])
        recs.append(row)
    with (outdir / 'landmarks.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(recs[0]))
        writer.writeheader(); writer.writerows(recs)
    bins = []
    for lo in range(0, int(np.floor(motion.max()/5))*5 + 1, 5):
        selected = (motion >= lo) & (motion < lo+5)
        if selected.any():
            bins.append(dict(lo_mm=lo, hi_mm=lo+5, n=int(selected.sum()),
                             mean_mm=float(error[selected].mean()),
                             median_mm=float(np.median(error[selected])),
                             errors_gt5=int((error[selected] > 5).sum())))
    diagnostics = dict(job=job, field_file=str(field_path.relative_to(HERE)),
                       convention='reference-grid to target, x+d(x), components xyz in mm',
                       geometry=dict(size=list(im.GetSize()), spacing=list(spacing),
                                     origin=list(im.GetOrigin()), direction=list(im.GetDirection())),
                       n_landmarks=len(a), tre_mean_mm=float(error.mean()),
                       tre_median_mm=float(np.median(error)), tre_p95_mm=float(np.percentile(error, 95)),
                       errors_gt5=int((error > 5).sum()), initial_mean_mm=float(motion.mean()),
                       phase_jacobian=per_phase,
                       mean_nonpositive_lung_pct=float(np.mean([r['pct_nonpositive_lung'] for r in per_phase])),
                       mean_nonpositive_whole_pct=float(np.mean([r['pct_nonpositive'] for r in per_phase])),
                       endpoint_nonpositive_lung_pct=per_phase[phases.index(5)]['pct_nonpositive_lung'],
                       displacement_bins=bins, mask_definition='T00/T50 union; cube dilation radius 2 working voxels',
                       landmark_sha256={str(p.relative_to(DATA)): sha(p) for p in lm_paths},
                       analysis_wall_s=round(time.monotonic()-t0, 3))
    write_json(outdir / 'diagnostics.json', diagnostics)
    return diagnostics
