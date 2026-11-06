"""Evaluate phase landmarks on exactly the saved new final-configuration fields."""
import csv
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
os.environ.setdefault('PMR_DATA_ROOT', str(HERE.parents[1].parent/'reference_4'/'data'))
sys.path.insert(0, str(HERE/'frozen'))
import numpy as np
import pmr_extended as model
from recompute_tre_standard import sample_trilinear


def main():
    rows = []
    for dataset, cases in [('dirlab', range(1, 11)), ('creatis', range(3))]:
        for cn in cases:
            run = HERE/'runs'/f'{dataset}_c{cn:02d}_s0_final'
            if not (run/'complete.json').exists(): continue
            caseid = f'case{cn}' if dataset == 'dirlab' else f'cr{cn}'
            fields = np.load(run/f'run_{caseid}_2mm_dvf.npy', mmap_mode='r')
            phases = range(1, 6) if dataset == 'dirlab' else range(1, 10)
            for p in phases:
                if dataset == 'dirlab':
                    a, b = model.load_lm_4d75(cn, p)
                    assert len(a) == len(b) == 75
                else:
                    a, b = model.load_lm_dataset(cn, 'creatis', p_ref=0, p_tgt=p)
                    assert len(a) == len(b) == 100
                error = np.linalg.norm(a+sample_trilinear(fields[p], a, [2,2,2])-b, axis=1)
                rows.append(dict(dataset=dataset, case=cn, phase=p, n_landmarks=len(a),
                                 mean_tre_mm=float(error.mean()), median_tre_mm=float(np.median(error)),
                                 identity_mm=float(np.linalg.norm(b-a, axis=1).mean()), source_job=run.name))
    output = HERE/'analysis'; output.mkdir(exist_ok=True)
    if rows:
        with (output/'same_run_phasewise.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f'Computed {len(rows)} available phase/case evaluations; no missing result imputed.')


if __name__ == '__main__': main()
