"""Reformat the verified 162-job study for the new manuscript; no optimisation."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'paperA_strengthening_20261021'
ROOT=HERE.parents[1]
ARCH=ROOT/'论文-PaperA 写作/03_论文草稿/figures_paperbanana_v3/source_data'
OUT=HERE/'analysis';OUT.mkdir(exist_ok=True)
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def save(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    records=read(OLD/'analysis/completed_runs.csv');assert len(records)==162
    by={r['job']:r for r in records}
    def get(c,s,a):return by[f'dirlab_c{c:02d}_s{s}_{a}']
    archived=read(ARCH/'fig03_accuracy.csv');accuracy=[];protocol=[];repeats=[];fold=[]
    raw_table=json.loads((ROOT/'results/MAIN_TABLE_generated.json').read_text(encoding='utf-8'))['rows']
    for c in range(1,11):
        base=get(c,0,'final');row=dict(archived[c-1]);row['PMR']=base['tre_mm']
        # Aggregate archived source values, not their three-decimal display copies.
        row['Identity']=raw_table['Identity']['per_case'][str(c)]
        row['Elastix']=raw_table['Elastix 5.3.1 (multi5_g6_mi, 2 mm)']['per_case'][str(c)]
        for col,stem in [('VoxelMorph-class','voxelmorph'),('LapIRN-class','lapirn')]:
            item=json.loads((ROOT/f'results/dl_baselines/{stem}_fold{c}_2mm.json').read_text(encoding='utf-8'))
            row[col]=item['tre_total_STANDARD']
        accuracy.append(row)
        protocol.append(dict(case=c,minmax=np.mean([float(get(c,s,'minmax')['tre_mm']) for s in range(3)]),robust_lung_r1=np.mean([float(get(c,s,'control')['tre_mm']) for s in range(3)]),whole=np.mean([float(get(c,s,'whole')['tre_mm']) for s in range(3)]),robust_lung_r10=np.mean([float(get(c,s,'final')['tre_mm']) for s in range(3)])))
        values=np.array([float(get(c,s,'final')['tre_mm']) for s in range(3)])
        repeats.append(dict(case=c,seeds=3,mean_mm=values.mean(),sd_mm=values.std(ddof=1),min_mm=values.min(),max_mm=values.max(),relative_range_pct=100*np.ptp(values)/values.mean()))
        d=json.loads((OLD/f'runs/dirlab_c{c:02d}_s0_final/diagnostics.json').read_text(encoding='utf-8'));ep=next(p for p in d['phase_jacobian'] if p['phase']==5)
        fold.append(dict(case=c,endpoint_whole_pct=ep['pct_nonpositive'],endpoint_lung_pct=ep['pct_nonpositive_lung'],endpoint_min_jacobian=ep['det_min'],all_phase_lung_pct=d['mean_nonpositive_lung_pct']))
    save('fig03_accuracy.csv',accuracy);save('fig04_ablation.csv',protocol);save('repeats.csv',repeats);save('folding.csv',fold)
    cases=[r for r in records if r['dataset']=='creatis'];save('creatis.csv',cases)
    phases=read(OLD/'analysis/same_run_phasewise.csv');phase_summary=[]
    for phase in range(1,6):
        p=[r for r in phases if r['dataset']=='dirlab' and int(r['phase'])==phase];assert len(p)==10
        phase_summary.append(dict(phase=phase*10,cases=10,landmarks_per_case=75,identity_mm=np.mean([float(r['identity_mm']) for r in p]),tre_mm=np.mean([float(r['mean_tre_mm']) for r in p])))
    save('phase_summary.csv',phase_summary)
    seed0=[float(get(c,0,'final')['tre_mm']) for c in range(1,11)]
    cref=[r for r in cases if r['arm']=='final'];ce=[r for r in cases if r['arm']=='elastix']
    controls={k:float(np.mean([r[k] for r in protocol])) for k in ['minmax','robust_lung_r1','whole','robust_lung_r10']}
    meta=dict(primary_dirlab_seed0_mean=float(np.mean(seed0)),primary_dirlab_seed0_sd=float(np.std(seed0,ddof=1)),full_three_seed_case_mean=float(controls['robust_lung_r10']),creatis_pmr_mean=float(np.mean([float(r['tre_mm']) for r in cref])),creatis_elastix_continuous_mean=float(np.mean([float(r['exact_bspline_tre_mm']) for r in ce])),protocol_means=controls,
      norm_increase_pct=100*(controls['minmax']/controls['robust_lung_r1']-1),support_reduction_pct=100*(1-controls['robust_lung_r1']/controls['whole']),residual_scale_reduction_pct=100*(1-controls['robust_lung_r10']/controls['robust_lung_r1']),max_endpoint_identity_mm=max(json.loads((OLD/f'runs/dirlab_c{c:02d}_s0_final/result.json').read_text(encoding='utf-8'))['closure_2pi'] for c in range(1,11)))
    (OUT/'existing_evidence_summary.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
