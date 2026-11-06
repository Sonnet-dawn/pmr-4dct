"""Serial, guarded and resumable component study; never skip a failed case."""
from pathlib import Path
import sys,os,time,json,subprocess,hashlib
import psutil
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from reliable_io import atomic_text

def write(p,d):atomic_text(p,json.dumps(d,indent=2,ensure_ascii=False),required=Path(p).name!='state.json')
def gpu():
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,creationflags=0x08000000)
    r.check_returncode();u,t=map(float,r.stdout.strip().splitlines()[0].split(','));return dict(used_mib=u,total_mib=t)
def stop(p):
    try:
        parent=psutil.Process(p.pid);children=parent.children(recursive=True)
        for child in children:child.terminate()
        parent.terminate();psutil.wait_procs(children+[parent],timeout=10)
    except psutil.NoSuchProcess:pass
def main():
    manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'));jobs=manifest['jobs']
    lock=HERE/'queue.lock'
    if lock.exists():assert not psutil.pid_exists(json.loads(lock.read_text())['pid']),'Queue running'
    write(lock,dict(pid=os.getpid()));state=dict(pid=os.getpid(),status='starting',total=len(jobs),completed=0,current=None,failed_jobs={})
    process=None
    try:
        for job in jobs:
            out=HERE/'runs'/job['id'];stamp=out/'complete.json'
            if stamp.exists():
                d=json.loads(stamp.read_text(encoding='utf-8'))
                for name,key in [('result.json','result_sha256'),('diagnostics.json','diagnostics_sha256')]:assert hashlib.sha256((out/name).read_bytes()).hexdigest()==d[key]
                state['completed']+=1;continue
            if (HERE/'STOP').exists():raise RuntimeError('Stopped by control file')
            resources=gpu()
            while resources['used_mib']/resources['total_mib']>.24 or psutil.virtual_memory().available<8*2**30:
                state.update(status='waiting_for_resources',current=job['id'],gpu=resources);write(HERE/'state.json',state);time.sleep(5);resources=gpu()
                if (HERE/'STOP').exists():raise RuntimeError('Stopped before dispatch')
            log=HERE/'logs'/(job['id']+'.log')
            if log.exists():log.rename(log.with_name(log.stem+'.previous_'+str(time.time_ns())+'.log'))
            with log.open('w',encoding='utf-8') as f:
                process=subprocess.Popen([sys.executable,'-u',str(HERE/'run_component.py'),'--job',job['id']],cwd=str(HERE),stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),creationflags=0x08000000)
                start=time.monotonic();over=0
                while process.poll() is None:
                    resources=gpu();available=psutil.virtual_memory().available/2**30;elapsed=time.monotonic()-start
                    over=over+1 if resources['used_mib']/resources['total_mib']>.75 else 0
                    state.update(status='running',current=job['id'],child_pid=process.pid,gpu=resources,available_ram_gib=round(available,2),elapsed_s=round(elapsed,1));write(HERE/'state.json',state)
                    if over>=2 or available<5 or elapsed>7200 or (HERE/'STOP').exists():
                        stop(process);raise RuntimeError('Resource/control guard: '+job['id'])
                    time.sleep(2)
            if process.returncode!=0 or not stamp.exists():
                state['failed_jobs'][job['id']]=dict(exit_code=process.returncode,log=str(log.relative_to(HERE)))
                state.update(status='continuing_after_job_failure');write(HERE/'state.json',state);continue
            state['completed']+=1;state.update(status='running',current=None);write(HERE/'state.json',state)
            print('DONE',job['id'],state['completed'],'/',len(jobs),flush=True)
        state.update(status='complete' if state['completed']==len(jobs) else 'incomplete_with_failures',current=None)
    except Exception as e:
        if process is not None and process.poll() is None:stop(process)
        state.update(status='stopped_on_error',error=str(e));raise
    finally:
        write(HERE/'state.json',state);lock.unlink(missing_ok=True)

if __name__=='__main__':main()
