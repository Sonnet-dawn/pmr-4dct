"""One child at a time; resource guards, verified resume and persistent state."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reliable_io import atomic_text
# Keep queue management lightweight: no torch or imaging imports here.


def write(path, value):
    # A busy status reader must never terminate registration or resource monitoring.
    return atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2),
                       required=path.name != 'state.json',
                       attempts=8 if path.name == 'state.json' else 40)


def gpu():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True,
                            timeout=10, creationflags=0x08000000 if os.name == 'nt' else 0)
    result.check_returncode()
    used, total = [float(v.strip()) for v in result.stdout.strip().splitlines()[0].split(',')]
    return dict(used_mib=used, total_mib=total)


def stop_tree(child):
    try:
        parent = psutil.Process(child.pid)
    except psutil.NoSuchProcess:
        return
    descendants = parent.children(recursive=True)
    for proc in descendants:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    parent.terminate()
    _, alive = psutil.wait_procs(descendants+[parent], timeout=5)
    for proc in alive:
        proc.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    # Capacity failures may be moved to the end, never omitted from the study.
    deferred = set(filter(None, os.environ.get('PAPER_A_DEFER_JOBS', '').split(',')))
    assert deferred <= {j['id'] for j in manifest['jobs']}, 'Unknown deferred job'
    jobs = sorted(manifest['jobs'], key=lambda j: j['id'] in deferred)
    assert json.loads((HERE/'validation.json').read_text(encoding='utf-8'))['status'] == 'passed'
    execution = json.loads((HERE/'execution_hashes.json').read_text(encoding='utf-8-sig'))
    for name, expected in execution.items():
        actual = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        assert actual == expected, f'Execution source changed: {name}'
    lock = HERE/'queue.lock'
    if lock.exists():
        previous = json.loads(lock.read_text(encoding='utf-8'))
        state_path = HERE/'state.json'
        recent_heartbeat = state_path.exists() and time.time()-state_path.stat().st_mtime < 30
        if psutil.pid_exists(previous['pid']) or recent_heartbeat:
            raise RuntimeError('Study queue is already running')
    write(lock, dict(pid=os.getpid()))
    completed_ids = {j['id'] for j in manifest['jobs']
                     if (HERE/'runs'/j['id']/'complete.json').exists()}
    state = dict(pid=os.getpid(), status='starting', total=len(manifest['jobs']),
                 completed=len(completed_ids), deferred_to_end=sorted(deferred))
    completed_now = 0
    attempts = {}
    failures = {}
    child = None
    logs = HERE/'logs'; logs.mkdir(exist_ok=True)
    try:
        # Two bounded passes: first keep healthy jobs moving, then retry failed jobs once.
        for job in jobs + jobs:
            out = HERE/'runs'/job['id']
            if (out/'complete.json').exists():
                stamp = json.loads((out/'complete.json').read_text(encoding='utf-8'))
                for name, key in [('result.json','result_sha256'), ('diagnostics.json','diagnostics_sha256')]:
                    assert hashlib.sha256((out/name).read_bytes()).hexdigest() == stamp[key]
                continue
            attempts[job['id']] = attempts.get(job['id'], 0) + 1
            if (HERE/'PAUSE_AFTER_CURRENT').exists() or (HERE/'STOP').exists():
                state.update(status='paused_by_control_file', current=None)
                break
            resources = gpu()
            needs_gpu = job.get('kind', 'pmr') == 'pmr'
            if (needs_gpu and resources['used_mib']/resources['total_mib'] > .24) or psutil.virtual_memory().available < 8*2**30:
                state.update(status='waiting_for_resources', current=job['id'], gpu=resources)
                write(HERE/'state.json', state)
                # Resource contention should wait, never discard the remaining queue.
                while (needs_gpu and resources['used_mib']/resources['total_mib'] > .24) or psutil.virtual_memory().available < 8*2**30:
                    if (HERE/'STOP').exists():
                        raise RuntimeError('Stopped by control file before dispatch')
                    time.sleep(5); resources = gpu()
                    state.update(status='waiting_for_resources', gpu=resources,
                                 available_ram_gib=round(psutil.virtual_memory().available/2**30, 2))
                    write(HERE/'state.json', state)
            print(f"START {job['id']} ({state['completed']}/{state['total']} complete)", flush=True)
            log_path = logs/f"{job['id']}.log"
            if log_path.exists():
                log_path.rename(logs/f"{job['id']}.previous_{time.time_ns()}.log")
            failure = None
            with log_path.open('w', encoding='utf-8') as log:
                env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
                           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                child = subprocess.Popen([sys.executable, str(HERE/'run_job.py'), '--job', job['id']],
                                         cwd=str(HERE), env=env, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=0x08000000 if os.name == 'nt' else 0)
                start = time.monotonic(); over = 0
                while child.poll() is None:
                    resources = gpu()
                    available = psutil.virtual_memory().available/2**30
                    over = over+1 if needs_gpu and resources['used_mib']/resources['total_mib'] > .75 else 0
                    state.update(status='running', current=job['id'], child_pid=child.pid,
                                 elapsed_s=round(time.monotonic()-start, 1), gpu=resources,
                                 available_ram_gib=round(available, 2))
                    write(HERE/'state.json', state)
                    if (HERE/'STOP').exists():
                        stop_tree(child)
                        raise RuntimeError('Stopped by control file during job')
                    if over >= 2 or available < 5 or state['elapsed_s'] > 7200:
                        failure = ('gpu_cap' if over >= 2 else 'ram_floor' if available < 5 else 'job_timeout')
                        stop_tree(child)
                        break
                    time.sleep(2)
                if child.returncode != 0 or not (out/'complete.json').exists():
                    failure = failure or f'worker_exit_{child.returncode}_or_missing_commit'
            if failure:
                failures[job['id']] = dict(reason=failure, attempt=attempts[job['id']],
                                          timestamp=time.time(), gpu=state.get('gpu'),
                                          available_ram_gib=state.get('available_ram_gib'))
                write(out/'last_failure.json', failures[job['id']])
                state.update(failed_jobs=failures.copy(), status='continuing_after_job_failure')
                write(HERE/'state.json', state)
                print(f"DEFER {job['id']}: {failure}; attempt {attempts[job['id']]}/2", flush=True)
                continue
            failures.pop(job['id'], None)
            state['failed_jobs'] = failures.copy()
            state['completed'] += 1; completed_now += 1
            print(f"DONE {job['id']}", flush=True)
            if args.limit and completed_now >= args.limit:
                state.update(status='batch_limit_reached', current=None)
                break
        else:
            state.update(status='complete' if state['completed'] == state['total'] else 'incomplete_with_failures',
                         current=None, failed_jobs=failures.copy())
    except Exception as error:
        state.update(status='stopped_on_error', error=str(error))
        if child is not None and child.poll() is None:
            stop_tree(child)
        raise
    finally:
        write(HERE/'state.json', state)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
