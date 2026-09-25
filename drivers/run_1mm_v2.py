"""
run_1mm_v2.py —— v2 的 1 mm 全分辨率运行（**严格串行 + 显存看门狗**）。
================================================================================
背景：现有的 1 mm 数据全部来自 v1（`results/probe_1mm/`），而 v2 才是当前代码。
Paper B 需要 v2 在 1 mm 上的规模/显存/耗时/精度。

🔴 资源纪律（见 `docs/29`）：
  * **一次只跑一个进程**（1 mm 的显存需求远大于 2 mm）
  * 每个病例启动前检查可用显存，不足 4 GB 则等待
  * 大病例（6–10，1 mm 下约 7900 万体素）**预计需要 ~20 GB，8 GB 卡装不下**
    ⇒ 默认只跑 case1–5，并在报告中如实说明这一限制

用法：
  python run_1mm_v2.py --cases 1,2,3,4,5
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
# --- end path shim ---
import os, sys, json, time, argparse, subprocess

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
PY = r'python'
OUTD = os.path.join(HERE, 'results', 'pmr_v2_1mm')
LOGD = os.path.join(HERE, 'results', 'pmr_v2', 'logs')
os.makedirs(OUTD, exist_ok=True)


def gpu_free_gb():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                            '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=15)
        u, t = [float(x) for x in r.stdout.strip().split(',')[:2]]
        return (t - u) / 1024.0
    except Exception:
        return 99.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=str, default='1,2,3,4,5')
    ap.add_argument('--need-gb', type=float, default=4.0,
                    help='每个病例启动前要求的最小可用显存')
    ap.add_argument('--tag', type=str, default='v2_1mm')
    args = ap.parse_args()

    cases = [int(c) for c in args.cases.split(',') if c.strip()]
    print(f'1 mm 串行运行: cases {cases}；每例前要求可用显存 ≥ {args.need_gb} GB', flush=True)

    for cn in cases:
        jf = os.path.join(OUTD, f'{args.tag}_case{cn}_1mm.json')
        if os.path.exists(jf):
            print(f'[skip] case{cn} 已完成', flush=True)
            continue
        # ---- 显存看门狗 ----
        for attempt in range(120):
            free = gpu_free_gb()
            if free >= args.need_gb:
                break
            print(f'  [gpu-watch] 可用 {free:.1f} GB < {args.need_gb} GB，等待 30 s', flush=True)
            time.sleep(30)
        else:
            print(f'[abort] case{cn}: 显存等待超时', flush=True)
            continue

        print(f'\n=== case{cn} @1mm（可用显存 {free:.1f} GB）===', flush=True)
        cmd = [PY, os.path.join(HERE, 'pmr_v2.py'),
               '--case', str(cn), '--down', '1',
               '--mask', 'union', '--metric', 'local',
               '--phases-per-step', '2', '--norm', 'robust',
               '--enc-down', '8',              # 1 mm 下的推荐值（见 docs/19）
               '--iters1', '800', '--iters2', '800', '--res-iters', '300',
               '--res-reg-scale', '10', '--cudnn-benchmark', '0',
               '--save-dvf', '0',
               '--tag', args.tag, '--out', OUTD]
        log = open(os.path.join(LOGD, f'{args.tag}_case{cn}.log'), 'w', encoding='utf-8')
        log.write('$ ' + ' '.join(cmd) + '\n'); log.flush()
        t0 = time.time()
        r = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONIOENCODING='utf-8',
                                    PYTHONUNBUFFERED='1'))
        dt = time.time() - t0
        log.write(f'\n[rc={r.returncode} {dt:.0f}s]\n'); log.close()
        if os.path.exists(jf):
            d = json.load(open(jf, encoding='utf-8'))
            print(f'  >> TRE {d["tre_total_STANDARD"]:.3f} | 峰值显存 '
                  f'{d["peak_mem_gb"]:.2f} GB | {d["time_s"]:.0f}s | '
                  f'体素 {d["n_vox"]/1e6:.1f} M', flush=True)
        else:
            print(f'  !! 失败 rc={r.returncode}（详见 {LOGD}）', flush=True)

    print('\n完成。汇总：python show_1mm.py')


if __name__ == '__main__':
    main()
