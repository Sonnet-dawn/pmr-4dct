"""Case-level structural ablations; never pool landmarks as independent patients."""
from pathlib import Path
import csv,json,hashlib,argparse,math
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'paperA_strengthening_20261021'
MODELS=['full','noaffine','noresidual','twoharmonics','coarseonly']
NAMES=dict(full='PMR',noaffine='PMR-NoAffine',noresidual='PMR-NoResidual',twoharmonics='PMR-2Harmonics',coarseonly='PMR-CoarseOnly')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
    records=[]
    for model in MODELS:
        for case in range(1,11):
            folder=(OLD if model=='full' else HERE)/'runs'/f'dirlab_c{case:02d}_s0_{"final" if model=="full" else model}'
            if not (folder/'complete.json').exists():continue
            stamp=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
            for name,key in [('result.json','result_sha256'),('diagnostics.json','diagnostics_sha256')]:assert sha(folder/name)==stamp[key]
            r=json.loads((folder/'result.json').read_text(encoding='utf-8'));d=json.loads((folder/'diagnostics.json').read_text(encoding='utf-8'))
            if model=='full':
                size=d['geometry']['size'];network=187608;residual=24*math.prod(max(1,n//2) for n in size);opt_s=None;runtime=r['time_s']
            else:
                network=r['parameters']['network_weights'];residual=r['parameters']['residual_coefficients']
                opt_s=sum(r['optimisation_stage_seconds'].values());runtime=None if model=='coarseonly' else r['compatible_registration_to_report_s']
            phases=d['phase_jacobian'];endpoint=next(x for x in phases if x['phase']==5)
            records.append(dict(model=model,label=NAMES[model],case=case,seed=0,
                tre_mm=d['tre_mean_mm'],median_mm=d['tre_median_mm'],p95_mm=d['tre_p95_mm'],
                endpoint_lung_fold_pct=d['endpoint_nonpositive_lung_pct'],all_phase_lung_fold_pct=d['mean_nonpositive_lung_pct'],endpoint_min_jacobian=endpoint['det_min'],
                network_weights=network,residual_coefficients=residual,total_trainable_parameters=network+residual,
                encoder_updates=1600,residual_updates=0 if model in ['noresidual','coarseonly'] else 300,
                optimisation_s=opt_s,registration_to_report_s=runtime,
                peak_gpu_gib=None if model=='coarseonly' else r['peak_mem_gb'],
                source_job=str(folder.relative_to(HERE.parent)),field_file=d['field_file'],
                resource_scope='stage-2 checkpoint; no independent peak or pipeline time' if model=='coarseonly' else 'preprocess+fit+report; extra base-field export excluded'))
    complete=len(records)==50
    assert complete or args.allow_partial,f'Incomplete: {len(records)}/50 configurations'
    out=HERE/'analysis';prefix='' if complete else 'partial_'
    save_csv(out/(prefix+'component_case_results.csv'),records)
    if not complete:print('Partial records:',len(records));return
    summary=[]
    for model in MODELS:
        rows=[r for r in records if r['model']==model]
        def average(k):
            v=[r[k] for r in rows if r[k] is not None];return float(np.mean(v)) if v else None
        def spread(k):return float(np.std([r[k] for r in rows],ddof=1))
        summary.append(dict(model=model,label=NAMES[model],cases=10,seeds_per_case=1,mean_tre_mm=average('tre_mm'),sd_case_mean_tre_mm=spread('tre_mm'),mean_case_p95_mm=average('p95_mm'),mean_endpoint_lung_fold_pct=average('endpoint_lung_fold_pct'),mean_all_phase_lung_fold_pct=average('all_phase_lung_fold_pct'),network_weights=rows[0]['network_weights'],residual_parameters_min=min(r['residual_coefficients'] for r in rows),residual_parameters_max=max(r['residual_coefficients'] for r in rows),encoder_updates=1600,residual_updates=rows[0]['residual_updates'],mean_optimisation_s=average('optimisation_s'),mean_registration_to_report_s=average('registration_to_report_s'),mean_peak_gpu_gib=average('peak_gpu_gib')))
    save_csv(out/'component_summary.csv',summary)
    baseline={r['case']:r for r in records if r['model']=='full'};pairs=[];effects=[];rng=np.random.default_rng(20261022)
    for model in MODELS[1:]:
        rows=[r for r in records if r['model']==model];v=np.array([r['tre_mm']-baseline[r['case']]['tre_mm'] for r in rows])
        draws=v[rng.integers(0,10,(10000,10))].mean(axis=1)
        effects.append(dict(model=model,comparison='variant minus full',cases=10,mean_difference_mm=float(v.mean()),exploratory_low_mm=float(np.quantile(draws,.025)),exploratory_high_mm=float(np.quantile(draws,.975)),cases_with_lower_tre=int((v<0).sum()),unit='case',inference='single-seed, exploratory, unadjusted intervals'))
        pairs.extend(dict(model=model,case=r['case'],difference_mm=float(x)) for r,x in zip(rows,v))
    save_csv(out/'component_paired_differences.csv',pairs);save_csv(out/'component_effects.csv',effects)
    manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    assert all((HERE/'runs'/j['id']/'complete.json').exists() for j in manifest['jobs'])
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='complete',dispatches=30,derived_coarse_only=10,reused_baselines=10,case_configurations=50,all_result_hashes_verified=True),indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
