"""Independent CPU audit: original landmarks -> saved fields -> case statistics."""
from pathlib import Path
import csv,json,hashlib,sys,re,math,os
import numpy as np
from scipy.ndimage import map_coordinates
import SimpleITK as sitk
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=Path(os.environ.get('PMR_DATA_ROOT',str(ROOT.parent/'reference_4/data')))
STUDIES=[HERE.parent/'paperA_strengthening_20261021',HERE.parent/'paperA_components_20261022']
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save_csv(name,rows):
    with (HERE/'analysis'/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def independent_tre(root,folder,d):
    paths=[DATA/k for k in d['landmark_sha256']]
    assert len(paths)==2
    a,b=[np.loadtxt(p).reshape(-1,3) for p in paths]
    for p in paths:assert sha(p)==d['landmark_sha256'][str(p.relative_to(DATA))]
    geom=d['geometry'];origin=np.array(geom['origin']);spacing=np.array(geom['spacing'])
    assert np.allclose(geom['direction'],np.eye(3).ravel())
    index=(a-origin)/spacing
    assert np.all(index>=0) and np.all(index<=np.array(geom['size'])-1)
    field=np.load(root/d['field_file'],mmap_mode='r');endpoint=field[5] if field.ndim==5 else field
    disp=np.stack([map_coordinates(endpoint[k],index[:,::-1].T,order=1,mode='nearest',prefilter=False) for k in range(3)],axis=1)
    err=np.linalg.norm(a+disp-b,axis=1);del field
    saved=list(csv.DictReader((folder/'landmarks.csv').open(encoding='utf-8-sig')))
    original=np.array([float(r['tre_mm']) for r in saved]);assert len(err)==len(original)
    assert np.max(np.abs(err-original))<3e-5,(folder.name,np.max(np.abs(err-original)))
    assert abs(err.mean()-d['tre_mean_mm'])<2e-5
    assert abs(np.quantile(err,.95)-d['tre_p95_mm'])<3e-5
    return dict(tre_mm=float(err.mean()),p95_mm=float(np.quantile(err,.95)),max_point_discrepancy_mm=float(np.max(np.abs(err-original))),landmarks=len(err))
def physical_det(field,spacing,block=8):
    sx,sy,sz=spacing;depth=field.shape[1]
    for start in range(0,depth,block):
        stop=min(depth,start+block);lo=max(0,start-1);hi=min(depth,stop+1)
        grads=[np.gradient(np.asarray(field[k,lo:hi],dtype=np.float64),sz,sy,sx) for k in range(3)]
        a=1+grads[0][2];b=grads[0][1];c=grads[0][0]
        d=grads[1][2];e=1+grads[1][1];f=grads[1][0]
        g=grads[2][2];h=grads[2][1];i=1+grads[2][0]
        yield start,stop,(a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g))[start-lo:stop-lo]
def jacobian_checks():
    sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'frozen'))
    from jacobian_stats import det_jacobian_array
    from study import common_mask
    spacing=np.array([1.3,2.1,3.7]);z,y,x=np.meshgrid(np.arange(7)*spacing[2],np.arange(8)*spacing[1],np.arange(9)*spacing[0],indexing='ij')
    A=np.array([[.12,.03,-.02],[.02,-.08,.04],[-.01,.05,.07]])
    field=np.einsum('ij,jzyx->izyx',A,np.stack([x,y,z]));expected=np.linalg.det(np.eye(3)+A)
    reference=det_jacobian_array(field,spacing);ind=np.concatenate([v for _,_,v in physical_det(field,spacing)],axis=0)
    assert np.max(abs(reference-expected))<1e-12 and np.max(abs(ind-expected))<1e-12
    records=[];masks={}
    for dataset,case,seeds in [('dirlab',6,range(3)),('dirlab',8,range(3)),('creatis',0,[0])]:
        key=(dataset,case);masks[key]=common_mask(dataset,case)
        for seed in seeds:
            folder=STUDIES[0]/'runs'/f'{dataset}_c{case:02d}_s{seed}_final';d=load(folder/'diagnostics.json')
            field=np.load(STUDIES[0]/d['field_file'],mmap_mode='r')[5];mask=masks[key];count=0;total=0;minimum=math.inf;maxdiff=0
            for start,stop,det in physical_det(field,d['geometry']['spacing']):
                roi=mask[start:stop];v=det[roi];count+=int((v<=0).sum());total+=len(v);minimum=min(minimum,float(det.min()))
            pct=100*count/total;target=next(p for p in d['phase_jacobian'] if p['phase']==5)
            assert abs(pct-target['pct_nonpositive_lung'])<1e-10 and abs(minimum-target['det_min'])<2e-6
            records.append(dict(dataset=dataset,case=case,seed=seed,independent_roi_nonpositive_pct=pct,archived_roi_nonpositive_pct=target['pct_nonpositive_lung'],independent_min_jacobian=minimum,roi='unwarped T00/T50 lung union with 5-voxel cube dilation',method='direct physical-coordinate derivatives and explicit 3x3 determinant'))
    save_csv('independent_jacobian_checks.csv',records)
    return dict(analytic_anisotropic_test='pass',real_endpoint_fields_checked=len(records),canonical_derivative_definition='determinant invariant under physical/voxel coordinate conjugation; xyz channels and zyx array axes aligned',roi_correction_required='Main Methods previously called this the reference lung mask; actual ROI is the dilated unwarped T00/T50 lung union. Numbers unchanged.')
def difficult_cases():
    curves=[];rows=[]
    for case in [6,8]:
        for seed in range(3):
            folder=STUDIES[0]/'runs'/f'dirlab_c{case:02d}_s{seed}_final';r=load(folder/'result.json');d=load(folder/'diagnostics.json')
            lm=list(csv.DictReader((folder/'landmarks.csv').open(encoding='utf-8-sig')));motion=np.array([float(x['motion_mm']) for x in lm]);errors=np.array([float(x['tre_mm']) for x in lm]);tail=errors[motion>=25]
            log=STUDIES[0]/'logs'/(folder.name+'.log');text=log.read_text(encoding='utf-8',errors='replace') if log.exists() else ''
            values=re.findall(r'(S[12])\s*iter\s+(\d+)\s+loss\s+([+-]?[\d.eE+-]+)',text)
            assert values,'Encoder loss history missing: '+str(log)
            for stage,it,loss in values:curves.append(dict(case=case,seed=seed,stage=stage,iteration=int(it),loss=float(loss)))
            rows.append(dict(case=case,seed=seed,tre_mm=d['tre_mean_mm'],median_mm=d['tre_median_mm'],p95_mm=d['tre_p95_mm'],fold_pct=d['endpoint_nonpositive_lung_pct'],tail25_n=len(tail),tail25_mean_mm=float(tail.mean()) if len(tail) else None,tail25_over5=int((tail>5).sum()),nrms_mean=r.get('nrm_train_mean'),logged_losses=len(values),logged_nonfinite_tokens=bool(re.search(r'\b(?:nan|inf|infinity)\b',text,re.I)),gradient_norm_history='not recorded',residual_loss_history='not recorded; residual iteration progress only'))
    save_csv('difficult_case_seed_metrics.csv',rows);save_csv('difficult_case_encoder_loss_curves.csv',curves)
    return rows
def main():
    audit=[]
    for root in STUDIES:
        for stamp in sorted((root/'runs').glob('*/complete.json')):
            folder=stamp.parent;d=load(folder/'diagnostics.json');c=load(stamp);r=load(folder/'result.json')
            assert sha(folder/'result.json')==c['result_sha256'] and sha(folder/'diagnostics.json')==c['diagnostics_sha256']
            if d['job'].get('arm')=='elastix':continue
            row=independent_tre(root,folder,d)
            audit.append(dict(study=root.name,job=folder.name,dataset=d['job']['dataset'],case=d['job']['case'],seed=d['job']['seed'],arm=d['job']['arm'],**row,source_result_sha256=sha(folder/'result.json'),source_diagnostics_sha256=sha(folder/'diagnostics.json')))
    assert len(audit)==196,len(audit)
    save_csv('independent_existing_tre_audit.csv',audit)
    jac=jacobian_checks();hard=difficult_cases()
    primary=[r['tre_mm'] for r in audit if r['study']==STUDIES[0].name and r['dataset']=='dirlab' and r['arm']=='final' and r['seed']==0]
    creatis=[r['tre_mm'] for r in audit if r['dataset']=='creatis']
    assert len(primary)==10 and len(creatis)==6
    report=dict(status='pass',saved_fields_independently_sampled=len(audit),maximum_point_difference_mm=max(r['max_point_discrepancy_mm'] for r in audit),primary_dirlab_mean_mm=float(np.mean(primary)),creatis_mean_mm=float(np.mean(creatis)),jacobian=jac,hard_case_runs=len(hard),implementation_notes=['All acquired images are reoriented with identity direction; native physical origins retained.','Fold ROI is the dilated T00/T50 union, not the T00-only mask.','Finite-difference coefficient regularisation has a different grid scaling in coarse and residual stages; state both explicitly.','No retrospective gradient norms or residual loss curves are available; do not claim they were checked.'])
    (HERE/'reports/independent_existing_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
