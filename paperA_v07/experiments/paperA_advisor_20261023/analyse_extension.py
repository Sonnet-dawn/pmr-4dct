"""Independent endpoint resampling and case-paired three-seed aggregation.

The ten patients, rather than the 30 fits, are the resampling units. No
manuscript numbers are produced until every prespecified extension is complete.
"""
from pathlib import Path
import csv, json, hashlib
import numpy as np
from audit_existing import independent_tre, sha

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'paperA_strengthening_20261021'
COMP=HERE.parent/'paperA_components_20261022'
A=HERE/'analysis'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def write(name,rows):
    with (A/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    manifest=load(HERE/'manifest.json');jobs=manifest['jobs'] if isinstance(manifest,dict) else manifest
    assert len(jobs)==40
    incomplete=[j['id'] for j in jobs if not (HERE/'runs'/j['id']/'complete.json').exists()]
    assert not incomplete, f'{len(incomplete)} prespecified jobs remain: {incomplete[:3]}'
    rows=[];audit=[]
    for model in ['full','noaffine','noresidual']:
        for case in range(1,11):
            for seed in range(3):
                root=OLD if model=='full' else COMP if seed==0 else HERE
                arm='final' if model=='full' else model
                folder=root/'runs'/f'dirlab_c{case:02d}_s{seed}_{arm}'
                d=load(folder/'diagnostics.json');r=load(folder/'result.json');c=load(folder/'complete.json')
                assert sha(folder/'result.json')==c['result_sha256'] and sha(folder/'diagnostics.json')==c['diagnostics_sha256']
                assert r['seed']==seed and r['case']==case and r['down_mm']==2
                assert r['norm']=='robust' and r['mask']=='union' and r['metric']==r['res_metric']=='local'
                assert r['K']==4 and r['win']==15 and r['ncc_stride']==7
                assert r['iters1']==r['iters2']==800 and r['res_iters']==(0 if model=='noresidual' else 300)
                assert r['res_reg_scale']==10 and r['cudnn_benchmark'] is False
                field=np.load(root/d['field_file'],mmap_mode='r')
                assert field.shape==(10,3,*d['geometry']['size'][::-1])
                assert np.max(np.abs(field[0]))==0
                del field
                if model!='full':
                    assert r['affine'] is (model!='noaffine')
                    assert r['parameters']['network_weights']==(180328 if model=='noaffine' else 187608)
                    assert (r['parameters']['residual_coefficients']==0) is (model=='noresidual')
                if root==HERE:
                    v=independent_tre(root,folder,d)
                    audit.append(dict(job=folder.name,**v,result_sha256=sha(folder/'result.json'),diagnostics_sha256=sha(folder/'diagnostics.json')))
                params=r.get('parameters',{})
                if model=='full':
                    # Actual residual lattice is saved with each fit's field geometry.
                    size=d['geometry']['size'];residual=24*int(np.prod([int(v)//2 for v in size]))
                    weights=187608
                else:
                    residual=params['residual_coefficients'];weights=params['network_weights']
                times=r.get('optimisation_stage_seconds',{})
                rows.append(dict(model=model,case=case,seed=seed,tre_mm=d['tre_mean_mm'],p95_mm=d['tre_p95_mm'],median_mm=d['tre_median_mm'],endpoint_roi_fold_pct=d['endpoint_nonpositive_lung_pct'],all_phase_roi_fold_pct=d['all_phases_mean_nonpositive_lung_pct'] if 'all_phases_mean_nonpositive_lung_pct' in d else float(np.mean([x['pct_nonpositive_lung'] for x in d['phase_jacobian']])),network_weights=weights,residual_coefficients=residual,encoder_updates=1600,residual_updates=0 if model=='noresidual' else 300,optimisation_s=sum(times.values()) if times else None,pipeline_s=r.get('compatible_registration_to_report_s',r.get('time_s')),peak_gpu_gib=r['peak_mem_gb'],source_study=root.name,source_job=folder.name,result_sha256=sha(folder/'result.json'),diagnostics_sha256=sha(folder/'diagnostics.json')))
    assert len(rows)==90 and len(audit)==40
    write('core_three_seed_runs.csv',rows);write('independent_new_tre_audit.csv',audit)
    cases=[]
    for model in ['full','noaffine','noresidual']:
        for case in range(1,11):
            rr=[r for r in rows if r['model']==model and r['case']==case];assert [r['seed'] for r in rr]==[0,1,2]
            x=np.array([r['tre_mm'] for r in rr]);fold=np.array([r['endpoint_roi_fold_pct'] for r in rr])
            cases.append(dict(model=model,case=case,seed0_mm=x[0],seed1_mm=x[1],seed2_mm=x[2],mean_mm=float(x.mean()),sd_mm=float(x.std(ddof=1)),roi_fold_mean_pct=float(fold.mean()),roi_fold_sd_pct=float(fold.std(ddof=1)),p95_mean_mm=float(np.mean([r['p95_mm'] for r in rr]))))
    write('core_three_seed_cases.csv',cases)
    summary=[];effects=[];differences=[]
    full=np.array([r['mean_mm'] for r in cases if r['model']=='full'])
    for model in ['full','noaffine','noresidual']:
        rr=[r for r in cases if r['model']==model];x=np.array([r['mean_mm'] for r in rr])
        summary.append(dict(model=model,cases=10,seeds=3,mean_tre_mm=float(x.mean()),between_case_sd_mm=float(x.std(ddof=1)),mean_within_case_seed_sd_mm=float(np.mean([r['sd_mm'] for r in rr])),mean_p95_mm=float(np.mean([r['p95_mean_mm'] for r in rr])),mean_roi_fold_pct=float(np.mean([r['roi_fold_mean_pct'] for r in rr]))))
        if model=='full':continue
        delta=x-full;rng=np.random.default_rng(20261022)
        lo,hi=np.quantile(delta[rng.integers(0,10,size=(10000,10))].mean(axis=1),[.025,.975])
        effects.append(dict(model=model,mean_difference_mm=float(delta.mean()),exploratory_low_mm=float(lo),exploratory_high_mm=float(hi),variant_lower_cases=int((delta<0).sum()),n_cases=10,seeds_per_case=3,bootstrap_draws=10000,bootstrap_seed=20261022))
        for case,v in enumerate(delta,1):differences.append(dict(model=model,case=case,seed_averaged_difference_mm=float(v)))
    write('core_three_seed_summary.csv',summary);write('core_three_seed_effects.csv',effects);write('core_three_seed_differences.csv',differences)
    report=dict(status='complete',new_runs=40,core_saved_runs=90,case_units=10,seeds=[0,1,2],independent_new_field_checks=40,maximum_new_point_error_difference_mm=max(r['max_point_discrepancy_mm'] for r in audit),summary=summary,effects=effects)
    (A/'extension_complete.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
