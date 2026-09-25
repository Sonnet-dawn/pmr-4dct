"""
run_v2_sweep.py —— PMR-v2 变体筛选（并行）。
================================================================================
目的：在代表性 3 例（1=易 / 5=中 / 8=难）上筛出最佳配置，再上 10 例全量。
所有变体共用同一算力预算，差别只在被检验的那一项。
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
import os, sys, json, subprocess, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r'python'
LOGD = os.path.join(HERE, 'results', 'pmr_v2', 'logs')
OUTD = os.path.join(HERE, 'results', 'pmr_v2')
os.makedirs(LOGD, exist_ok=True)

BUDGET = ['--iters1', '800', '--iters2', '800', '--res-iters', '300',
          '--phases-per-step', '2', '--save-dvf', '0',
          # 🔴 关闭 cuDNN 算法自动选择：实测把运行间波动从 ~6% 降到 **0.2%**
          #    （detA 1.13048 vs detB 1.13305，同 seed 同配置），见 docs/17 C56/C59
          '--cudnn-benchmark', '0']

# 变体名 -> 额外 CLI 参数（相对默认：mask=union metric=local reg-scale=1 norm=robust）
VARIANTS = {
    'base':        [],
    'nomask':      ['--mask', 'none'],
    't00':         ['--mask', 't00'],
    'global':      ['--metric', 'global', '--res-metric', 'global'],
    'mind':        ['--metric', 'mind', '--res-metric', 'mind'],
    'reg03':       ['--reg-scale', '0.3'],
    'reg3':        ['--reg-scale', '3.0'],
    'pps1':        ['--phases-per-step', '1'],
    'pps5':        ['--phases-per-step', '5'],
    'pps10':       ['--phases-per-step', '10'],
    'affine':      ['--affine-first-iters', '800'],
    'normminmax':  ['--norm', 'minmax'],
    'enc8':        ['--enc-down', '8'],
    'res800':      ['--res-iters', '800'],
    # --- 第二轮补充：优化预算 / 窗口尺度 / 残差网格 ---
    'lr3':         ['--lr', '3e-3'],
    'lr10':        ['--lr', '1e-2'],
    'win15':       ['--win-mm', '15'],
    'win45':       ['--win-mm', '45'],
    'res1':        ['--res-down', '1', '--res-iters', '800'],
    'iters3k':     ['--iters1', '3000', '--iters2', '3000', '--res-iters', '1000'],
    'maskwt':      ['--local-mask-mode', 'weight'],
    'maskwt_w15':  ['--local-mask-mode', 'weight', '--win-mm', '15'],
    # --- C2 跨领域移植：局部相似性属性（时移地震 → 4D-CT，Fomel & Jin 2009）---
    # 三级单变量阶梯（均在 res-reg-scale=1 下与 det1_base 对齐）：
    #   det1_base(盒窗+select) → maskwt(盒窗+掩膜占比加权) → lsg(+高斯 shaping 窗) → ls(+局部结构显著性权重)
    'lsg':         ['--metric', 'lsg', '--res-metric', 'lsg'],
    'ls':          ['--metric', 'ls', '--res-metric', 'ls'],
    'ls_res10':    ['--metric', 'ls', '--res-metric', 'ls', '--res-reg-scale', '10'],
    'lsg_res10':   ['--metric', 'lsg', '--res-metric', 'lsg', '--res-reg-scale', '10'],
    # --- 消融阶梯（加法式，从 v1 等价配置逐项加回）---
    # v1 等价：复现 v1 的行为，用于验证"v2 vs v1"这一前后对比是可比的基础
    'v1equiv':     ['--phases-per-step', '10', '--iters1', '400', '--iters2', '400',
                    '--res-iters', '200', '--norm', 'minmax', '--metric', 'global',
                    '--res-metric', 'global', '--mask', 'none'],
    # v1 配置但给 v2 的迭代预算 —— 隔离"训练预算"的贡献
    'budget':      ['--phases-per-step', '10', '--iters1', '800', '--iters2', '800',
                    '--res-iters', '300', '--norm', 'minmax', '--metric', 'global',
                    '--res-metric', 'global', '--mask', 'none'],
    # 存 DVF 以便做折叠/Jacobian 验证（体积大，仅少量病例）
    'folding':     ['--save-dvf', '1'],
    # 精度–折叠权衡：提高正则强度，看能否在很小精度代价下压低折叠
    'foldreg3':    ['--save-dvf', '1', '--reg-scale', '3'],
    'foldreg10':   ['--save-dvf', '1', '--reg-scale', '10'],
    # 残差阶段正则倍率（`--reg-scale` 管不到残差阶段，见 docs/26 §4.1）
    'resreg10':    ['--save-dvf', '1', '--res-reg-scale', '10'],
    'resreg100':   ['--save-dvf', '1', '--res-reg-scale', '100'],
    # det(J) hinge 惩罚：直接抑制折叠
    'jac100':      ['--save-dvf', '1', '--jac-weight', '100'],
    'jac1000':     ['--save-dvf', '1', '--jac-weight', '1000'],
    # 候选最优配置：残差阶段正则 ×10（实测同时改善 TRE 与折叠，见 docs/26 §4.2）
    'best':        ['--res-reg-scale', '10'],
}


def gpu_safe_workers(cases, requested, gpu_gb=8.0, headroom_gb=1.8):
    """按病例尺寸限制并发数，防止显存打满拖垮整机。

    🔴 教训（2026-09-22）：曾以 4 个并发跑 case6–10（每个实测峰值显存 2.5–2.8 GB），
    把 8 GB 卡顶到 7.4 GB，系统出现严重抖动直至**死机**。

    实测峰值显存（2 mm）：
      * 小病例 1–5（1.8–3.1 Mvox）：0.50–0.85 GB
      * 大病例 6–10（9.2–10.5 Mvox）：2.50–2.82 GB
    """
    per = 2.8 if any(c >= 6 for c in cases) else 1.0
    cap = max(1, int((gpu_gb - headroom_gb) / per))
    if requested > cap:
        print(f'  [gpu-guard] 请求 {requested} 个 worker -> 按 {per:.1f} GB/进程、'
              f'{gpu_gb:.0f} GB 卡（留 {headroom_gb:.1f} GB 余量）限制为 **{cap}**',
              flush=True)
    return min(requested, cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=str, default='1,5,8')
    ap.add_argument('--variants', type=str, default=','.join(VARIANTS))
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--tag', type=str, default='screen')
    ap.add_argument('--gpu-gb', type=float, default=8.0, help='显存预算，用于自动限制并发')
    ap.add_argument('--extra', type=str, default='',
                    help='附加参数（空格分隔），追加在变体参数**之后**，优先级最高。'
                         '用于不改动 VARIANTS 表的单变量实验，例如：'
                         '--extra "--lr-decay cosine"')
    args = ap.parse_args()
    extra = args.extra.split()
    if extra:
        print(f'[extra] 附加参数（优先级最高）: {extra}', flush=True)
    cases = [int(c) for c in args.cases.split(',') if c.strip()]
    vs = [v.strip() for v in args.variants.split(',') if v.strip()]
    bad = [v for v in vs if v not in VARIANTS]
    if bad:
        sys.exit(f'未知变体: {bad}\n可选: {list(VARIANTS)}')

    args.workers = gpu_safe_workers(cases, args.workers, args.gpu_gb)

    jobs = [(cn, v) for v in vs for cn in cases]
    # 跳过已完成的
    todo = []
    for cn, v in jobs:
        f = os.path.join(OUTD, f'{args.tag}_{v}_case{cn}_2mm.json')
        if os.path.exists(f):
            print(f'[skip] {v} case{cn}', flush=True)
        else:
            todo.append((cn, v))
    print(f'共 {len(jobs)} 个任务，待跑 {len(todo)} 个，{args.workers} 并行', flush=True)

    groups = [todo[i::args.workers] for i in range(args.workers)]
    procs = []
    for gi, g in enumerate(groups):
        if not g:
            continue
        script = os.path.join(LOGD, f'worker{gi}.py')
        with open(script, 'w', encoding='utf-8') as fh:
            fh.write('import subprocess, sys, os, time\n')
            fh.write(f'PY = r"{PY}"\n')
            fh.write(f'HERE = r"{HERE}"\n')
            fh.write(f'JOBS = {g!r}\n')
            fh.write(f'VAR = {VARIANTS!r}\n')
            fh.write(f'OUTD = r"{OUTD}"\n')
            fh.write(f'TAG = r"{args.tag}"\n')
            fh.write(f'NEED_GB = {2.8 if any(c >= 6 for c in cases) else 1.0}\n')
            # EXTRA：附加参数，**放在变体参数之后**，因此优先级最高。
            # 用途：不改动 VARIANTS 表就能做单变量实验（如 --lr-decay cosine，见 docs/38）。
            fh.write(f'EXTRA = {extra!r}\n')
            # ---- 运行时显存看门狗：可用显存不足则等待，绝不硬闯 ----
            fh.write('def gpu_free_gb():\n')
            fh.write('    try:\n')
            fh.write('        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",\n')
            fh.write('                           "--format=csv,noheader,nounits"],\n')
            fh.write('                          capture_output=True, text=True, timeout=15)\n')
            fh.write('        u, t = [float(x) for x in r.stdout.strip().split(",")[:2]]\n')
            fh.write('        return (t - u) / 1024.0\n')
            fh.write('    except Exception:\n        return 99.0\n')
            fh.write('def wait_gpu(need):\n')
            fh.write('    for _ in range(120):\n')
            fh.write('        free = gpu_free_gb()\n')
            fh.write('        if free >= need + 0.5:\n            return True\n')
            fh.write('        print(f"  [gpu-watch] 可用 {free:.1f} GB < 需要 {need+0.5:.1f} GB，等待 30 s", flush=True)\n')
            fh.write('        time.sleep(30)\n')
            fh.write('    return False\n')
            fh.write('for cn, v in JOBS:\n')
            fh.write('    f = os.path.join(OUTD, f"{TAG}_{v}_case{cn}_2mm.json")\n')
            fh.write('    if os.path.exists(f):\n        print("skip", v, cn, flush=True); continue\n')
            fh.write('    cmd = [PY, os.path.join(HERE, "pmr_v2.py"), "--case", str(cn),\n')
            fh.write('           "--down", "2", "--tag", f"{TAG}_{v}", "--out", OUTD]\n')
            fh.write(f'    cmd += {BUDGET!r}\n')
            fh.write('    cmd += VAR[v]          # 变体参数放在最后，可覆盖预算项\n')
            fh.write('    cmd += EXTRA           # 附加参数优先级最高\n')
            fh.write('    print("RUN", v, cn, flush=True)\n')
            fh.write('    if not wait_gpu(NEED_GB):\n')
            fh.write('        print("  [gpu-watch] 等待超时，跳过本任务", flush=True); continue\n')
            fh.write('    t = time.time()\n')
            fh.write('    r = subprocess.run(cmd, cwd=HERE)\n')
            fh.write('    print(f"DONE {v} case{cn} rc={r.returncode} {time.time()-t:.0f}s", flush=True)\n')
        log = open(os.path.join(LOGD, f'{args.tag}_w{gi}.log'), 'w', encoding='utf-8')
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        p = subprocess.Popen([PY, script], stdout=log, stderr=subprocess.STDOUT, cwd=HERE, env=env)
        procs.append((gi, p, log))
        print(f'[launch] worker{gi}: {len(g)} jobs pid={p.pid}', flush=True)
        time.sleep(4)

    while procs:
        alive = []
        for gi, p, log in procs:
            if p.poll() is None:
                alive.append((gi, p, log))
            else:
                log.close()
                print(f'[done] worker{gi} exit={p.returncode}', flush=True)
        procs = alive
        if procs:
            time.sleep(30)
    print('全部完成', flush=True)
    summarize(args.tag)


def summarize(tag):
    import collections
    by = collections.defaultdict(dict)
    for f in sorted(os.listdir(OUTD)):
        if not (f.startswith(tag) and f.endswith('.json')):
            continue
        parts = f[len(tag) + 1:].rsplit('_case', 1)
        if len(parts) != 2:
            continue
        v = parts[0]
        try:
            by[v][int(parts[1].split('_')[0])] = json.load(
                open(os.path.join(OUTD, f), encoding='utf-8'))
        except Exception as e:
            print(f'  [warn] {f}: {e}')
    if not by:
        print('无结果'); return
    cases = sorted({c for d in by.values() for c in d})
    print(f'\n=== {tag} 汇总（标准口径 TRE, mm）===')
    print(f'{"variant":>14} ' + ' '.join(f'{("c"+str(c)):>7}' for c in cases) + f'{"MEAN":>8}')
    ranked = []
    for v in sorted(by):
        tre = [by[v][c]['tre_total_STANDARD'] for c in cases if c in by[v]]
        if len(tre) == len(cases):
            ranked.append((sum(tre) / len(tre), v, tre))
    for v in sorted(by):                      # 先打印不完整的，避免丢失
        tre = [by[v].get(c, {}).get('tre_total_STANDARD') for c in cases]
        if any(t is None for t in tre):
            print(f'{v:>14} ' + ' '.join(f'{(f"{t:.3f}" if t else "-"):>7}' for t in tre) + f'{"(partial)":>8}')
    for m, v, tre in sorted(ranked):
        print(f'{v:>14} ' + ' '.join(f'{t:>7.3f}' for t in tre) + f'{m:>8.3f}')
    if ranked:
        best = min(ranked)
        print(f'\n>>> 最佳: {best[1]}  mean {best[0]:.3f} mm')
    print('\nv1(pmr_fast) 同 3 例标准口径参照 = case1 1.358 / case5 2.928 / case8 11.691')


if __name__ == '__main__':
    main()
