"""Recompute the non-field evidence from the raw archived records."""
from pathlib import Path
import csv,json,re,hashlib,os,ast
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
DATA=Path(os.environ.get('PMR_DATA_ROOT',str(ROOT.parent/'reference_4/data')));A=HERE/'analysis'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def write(name,rows):
    with (A/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    table=load(ROOT/'results/MAIN_TABLE_generated.json')['rows'];base=[];metadata_notes=[]
    for case in range(1,11):
        r=dict(case=case,identity_mm=table['Identity']['per_case'][str(case)],elastix_mm=table['Elastix 5.3.1 (multi5_g6_mi, 2 mm)']['per_case'][str(case)])
        for stem in ['voxelmorph','lapirn']:
            p=ROOT/f'results/dl_baselines/{stem}_fold{case}_2mm.json'
            v=load(p)
            assert v['seed']==0 and v['down_mm']==2 and v['epochs']==600 and v['iters_per_epoch']==25
            assert v['patch']==96 and v['lr']==1e-4 and v['protocol']=='leave-one-out'
            if 'n_train_cases' in v:assert v['n_train_cases']==9
            else:
                log=(ROOT/f'results/dl_baselines/logs/{stem}_fold{case}.log').read_text(encoding='utf-8',errors='replace')
                ids=ast.literal_eval(re.search(r'训练例=(\[[^]]+\])',log).group(1))
                assert ids==[i for i in range(1,11) if i!=case]
                metadata_notes.append(dict(run=p.name,missing_fields=['n_train_cases','n_points'],nine_training_subjects_verified_from_log=True,endpoint_count_note='300-point loader source and matching initial TRE; no point-count field inserted into the historical JSON'))
            if 'n_points' in v:assert v['n_points']==300
            lm=DATA/f'DIRLAB/points/case{case}'
            x,y=[np.loadtxt(lm/f'case{case}_300_T{phase}_xyz_R.txt') for phase in ['00','50']]
            assert x.shape==y.shape==(300,3) and abs(np.linalg.norm(x-y,axis=1).mean()-v['init_tre'])<.00006
            r[stem+'_mm']=v['tre_total_STANDARD'];r[stem+'_source_sha256']=sha(p)
        base.append(r)
    write('archived_comparator_source_audit.csv',base)
    loops=load(ROOT/'results/closure_loop/closure_loop_2mm.json')['rows'];loop=[]
    for key,r in loops.items():loop.append(dict(job=key,method=r['method'],mean_abs_component_mm=r['loop_err_mean_mm'],maximum_abs_component_mm=r['loop_err_max_mm']))
    write('archived_pairwise_loops.csv',loop)
    closure={}
    for method in ['bspline','vxm']:
        rr=[r for r in loop if r['method']==method];assert len(rr)==5
        closure[method]=dict(mean_abs_component_mm=float(np.mean([r['mean_abs_component_mm'] for r in rr])),maximum_abs_component_mm=max(r['maximum_abs_component_mm'] for r in rr))
    memory=[]
    for case in range(1,6):
        p=ROOT/f'results/pmr_v2_1mm/v2_1mm_case{case}_1mm.json';r=load(p)
        memory.append(dict(case=case,voxels=r['n_vox'],peak_gpu_gib=r['peak_mem_gb'],status='measured',source=str(p.relative_to(ROOT)),sha256=sha(p)))
    x=np.array([r['voxels'] for r in memory],dtype=float);y=np.array([r['peak_gpu_gib'] for r in memory])
    slope=float(x@y/(x@x));projection=slope*79e6
    write('archived_1mm_memory.csv',memory)
    # These are explicitly historical capacity measurements, not the new fit protocol.
    arch_mem=[]
    for dataset,pattern in [('DIR-Lab / 2mm','results/pmr_v2/det10_best_case*_2mm.json'),('CREATIS / 2mm','results/pmr_creatis/cr_base_cr*_2mm.json')]:
        paths=sorted(ROOT.glob(pattern));assert len(paths)==(10 if dataset.startswith('DIR') else 6)
        for p in paths:
            r=load(p);arch_mem.append(dict(dataset_grid=dataset,run=p.stem,peak_gpu_gib=r['peak_mem_gb'],source=str(p.relative_to(ROOT)),sha256=sha(p)))
    write('archived_2mm_memory.csv',arch_mem)
    trajectories=[];theta=np.arange(10)*2*np.pi/10
    for case in range(3):
        pts=np.stack([np.loadtxt(DATA/f'CREATIS/{case}/{p*10:02d}.pts') for p in range(10)]);assert pts.shape==(10,100,3)
        y=(pts-pts[:1]).reshape(10,-1)
        for k,nyquist in [(1,False),(2,False),(3,False),(4,False),(4,True)]:
            cols=[v for j in range(1,k+1) for v in [np.cos(j*theta)-1,np.sin(j*theta)]]
            if nyquist:cols.append(np.cos(5*theta)-1)
            b=np.column_stack(cols);res=(y-b@np.linalg.lstsq(b,y,rcond=None)[0]).reshape(10,100,3)
            trajectories.append(dict(case=case,functions=b.shape[1],rank=int(np.linalg.matrix_rank(b)),condition_number=float(np.linalg.cond(b)),mean_temporal_rms_mm=float(np.sqrt((res**2).sum(axis=2).mean(axis=0)).mean())))
    write('independent_temporal_fitting.csv',trajectories)
    original=list(csv.DictReader((ROOT/'论文-PaperA 写作/03_论文草稿/paperA_bmc/figures/figS02_temporal_order_source.csv').open(encoding='utf-8-sig')))
    assert len(original)==15
    for r,o in zip(trajectories,original):assert abs(r['mean_temporal_rms_mm']-float(o['mean_landmark_temporal_rms_mm']))<1e-10
    ledger=(ROOT/'docs/37_缺陷台账与验证套件.md').read_text(encoding='utf-8')
    ledgerrows=[line.split('|')[1:4] for line in ledger.split('### 3.1')[0].splitlines() if re.match(r'\| [CE]\d+ \|',line)]
    assert len(ledgerrows)==14
    defects=[dict(id=r[0].strip(),silent='Yes' if '✅' in r[2] else 'No',recorded_defect=r[1].strip()) for r in ledgerrows]
    assert sum(r['silent']=='Yes' for r in defects)==10
    write('verification_ledger_rows.csv',defects)
    report=dict(status='pass',comparator_means={k:float(np.mean([r[k] for r in base])) for k in ['identity_mm','elastix_mm','voxelmorph_mm','lapirn_mm']},comparator_scope='Archived subject-level scalar scores reaggregated; learned/comparator fields were not rerun or independently re-sampled in this audit.',pairwise_loops=closure,one_mm_projection=dict(fit='through-origin least squares on five historical runs',slope_gib_per_voxel=slope,predicted_79_million_voxels_gib=projection,measured_range_voxels=[int(x.min()),int(x.max())],measured_range_gib=[float(y['peak_gpu_gib']) for y in [min(memory,key=lambda r:r['peak_gpu_gib']),max(memory,key=lambda r:r['peak_gpu_gib'])]]),tensor_storage=dict(voxels=14453440,channels=24,bytes_per_value=4,full_gib=14453440*24*4/2**30,coarse_mib=29*31*31*24*4/2**20,ratio=14453440/(29*31*31)),temporal_means={str(n):float(np.mean([r['mean_temporal_rms_mm'] for r in trajectories if r['functions']==n])) for n in [2,4,6,8,9]},ledger=dict(defects=14,silent_row_labels=10,aggregate_in_source=11,source_sha256=sha(ROOT/'docs/37_缺陷台账与验证套件.md')))
    report['archived_comparator_metadata_notes']=metadata_notes
    (HERE/'reports/other_evidence_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
