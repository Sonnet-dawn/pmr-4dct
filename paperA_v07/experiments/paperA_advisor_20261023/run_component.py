"""Run a prespecified structural arm and independently evaluate its saved field."""
from pathlib import Path
import os,sys,json,time,inspect,functools,gc,argparse,platform
HERE=Path(__file__).resolve().parent
os.environ.setdefault('PMR_DATA_ROOT',str(HERE.parents[1].parent/'reference_4/data'))
os.environ.setdefault('PYTHONIOENCODING','utf-8')
sys.path.insert(0,str(HERE/'frozen'));sys.path.insert(1,str(HERE))
import torch,numpy as np,SimpleITK as sitk
from study import sha,write_json,evaluate
from checkpoint_export import export_checkpoint
from component_runtime import initialise_factory

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);a=ap.parse_args()
    job=next(x for x in json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))['jobs'] if x['id']==a.job)
    out=HERE/'runs'/job['id'];out.mkdir(exist_ok=True)
    assert not (out/'complete.json').exists()
    for name,h in json.loads((HERE/'frozen_hashes.json').read_text(encoding='utf-8')).items():assert sha(HERE/name)==h,name
    torch.set_num_threads(4);sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(4)
    import pmr_extended as model
    model.CoefNetFast=initialise_factory(model.CoefNetFast)
    original_stage=model.train_stage;original_residual=model.train_residual
    durations={};params={};stage_ckpt=out/'encoder_stage2.pt'
    @functools.wraps(original_stage)
    def stage(*args,**kwargs):
        b=inspect.signature(original_stage).bind(*args,**kwargs);label=b.arguments['label'].strip();net=b.arguments['net']
        params['network_weights']=sum(p.numel() for p in net.parameters())
        params['affine_head_weights']=sum(p.numel() for p in net.fc.parameters()) if net.affine else 0
        params['residual_coefficients']=0
        if stage_ckpt.exists():
            ck=torch.load(stage_ckpt,map_location='cpu',weights_only=True);assert ck['argv']==sys.argv and ck['model_hash']==sha(model.__file__)
            durations.update(ck['durations']);params.update(ck['params'])
            if label=='S2':
                net.load_state_dict(ck['net']);b.arguments['gen'].set_state(ck['gen']);torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
            print('RESUME stage-2 checkpoint: skip '+label,flush=True);return 0.
        torch.cuda.synchronize();start=time.perf_counter();v=original_stage(*args,**kwargs);torch.cuda.synchronize();durations[label]=time.perf_counter()-start
        if label=='S2':
            ck=dict(argv=list(sys.argv),model_hash=sha(model.__file__),net={k:v.detach().cpu() for k,v in net.state_dict().items()},gen=b.arguments['gen'].get_state(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),durations=durations,params=params)
            temp=stage_ckpt.with_suffix('.pt.partial')
            with temp.open('wb') as f:torch.save(ck,f)
            temp.replace(stage_ckpt)
        write_json(out/'stage_resources.json',dict(durations=durations,parameters=params))
        return v
    @functools.wraps(original_residual)
    def residual(*args,**kwargs):
        b=inspect.signature(original_residual).bind(*args,**kwargs)
        params['residual_coefficients']=b.arguments['coef_r'].numel()
        torch.cuda.synchronize();start=time.perf_counter();v=original_residual(*args,**kwargs);torch.cuda.synchronize();durations['R']=time.perf_counter()-start
        write_json(out/'stage_resources.json',dict(durations=durations,parameters=params));return v
    model.train_stage=stage;model.train_residual=residual
    checkpoint=out/'registration_checkpoint.pt';field=out/'field.npy';base=out/'base_field.npy'
    sys.argv=['pmr_extended.py','--dataset','dirlab','--case',str(job['case']),'--down','2','--mask','union','--metric','local','--res-metric','local','--norm','robust','--phases-per-step','2','--iters1','800','--iters2','800','--res-iters','300','--res-reg-scale','10','--cudnn-benchmark','0','--seed',str(job['seed']),'--save-dvf','0','--export-state',str(checkpoint),'--tag','component','--out',str(out)]
    if job['arm']=='noaffine':sys.argv+=['--affine','0']
    elif job['arm']=='noresidual':sys.argv[sys.argv.index('--res-iters')+1]='0';sys.argv+=['--disable-residual','1']
    elif job['arm']=='twoharmonics':sys.argv+=['--K','2','--res-K','2']
    write_json(out/'command.json',sys.argv)
    write_json(out/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),worker_sha256=sha(__file__),model_sha256=sha(model.__file__),seed=job['seed'],deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    start=time.perf_counter();resumed=checkpoint.exists()
    if resumed:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True);assert ck['argv']==sys.argv and ck['model_sha256']==sha(model.__file__);result=ck['result']
    else:result=model.main();ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    del model;gc.collect();torch.cuda.empty_cache()
    export_checkpoint(ck,field);del ck;gc.collect();torch.cuda.empty_cache()
    if (out/'stage_resources.json').exists():
        stage_data=json.loads((out/'stage_resources.json').read_text(encoding='utf-8'));durations=stage_data['durations'];params=stage_data['parameters']
    result.update(component_arm=job['arm'],actual_optimizer_steps=dict(encoder=1600,residual=0 if job['arm']=='noresidual' else 300),parameters=params,optimisation_stage_seconds=durations,compatible_registration_to_report_s=result['time_s']-result.get('base_snapshot_export_s',0),resumed_from_final_checkpoint=resumed)
    diagnostics=evaluate(field,job,result,out)
    result['total_job_wall_s']=time.perf_counter()-start
    write_json(out/'result.json',result)
    if job.get('capture_coarse',False):
        cj=dict(job,id=job['id'].replace('noaffine','coarseonly'),arm='coarseonly');co=HERE/'runs'/cj['id'];co.mkdir(exist_ok=True)
        cr=dict(result,tre_total_STANDARD=result['tre_manifold_STD'],component_arm='coarseonly',actual_optimizer_steps=dict(encoder=1600,residual=0),parameters=dict(params,residual_coefficients=0),optimisation_stage_seconds={k:v for k,v in durations.items() if k!='R'},compatible_registration_to_report_s=None,time_s=None,runtime_scope='Derived stage-2 snapshot; full pipeline runtime unavailable')
        cd=evaluate(base,cj,cr,co);write_json(co/'result.json',cr)
        write_json(co/'complete.json',dict(job=cj,derived_from=job['id'],result_sha256=sha(co/'result.json'),diagnostics_sha256=sha(co/'diagnostics.json'),tre_mm=cd['tre_mean_mm']))
    write_json(out/'complete.json',dict(job=job,result_sha256=sha(out/'result.json'),diagnostics_sha256=sha(out/'diagnostics.json'),tre_mm=diagnostics['tre_mean_mm']))
    print('COMPLETE '+job['id']+' TRE='+str(diagnostics['tre_mean_mm']),flush=True)

if __name__=='__main__':main()
