"""
run_creatis_v2.py —— 在 CREATIS 4D-CT 上运行 PMR-v2（**严格串行 + 显存看门狗**）。
================================================================================
为什么值得做（见对话记录与 `docs/30`）：

  * **第二个数据集** —— 直接消掉"只有一个数据集"这条最大的审稿风险。
  * **每个相位都有 landmark**（100–113 点 × 10 相位）。DIRLAB **只有 T00/T50**
    有真值，因此"留出相位可靠性"实验在 DIRLAB 上无法验证，在 CREATIS 上可以闭环。
  * **位移大得多** —— CREATIS T00→T50 平均位移 5.7–14.0 mm（最大 32 mm），
    DIRLAB 只有几毫米。这正好打在 P3（大位移捕获范围）上。
  * **体积更大** —— 1 mm 下 60–116 Mvox/相位（DIRLAB 14–79 M）。

🔴 已核实的两个数据集差异（用错就是静默 bug）：
  1. **HU 偏移不同**：DIRLAB 存 HU+1024；**CREATIS 存原生 HU（offset 0）**。
     → 已写入 `lung_mask.HU_OFFSETS`，实测依据 `tools/probe_creatis_hu.py`。
  2. **landmark 参考系不同**：CREATIS 的 `.pts` 是**绝对物理坐标**（origin 非零且有负值），
     DIRLAB origin=(0,0,0)。→ `pmr_v2.load_lm_dataset` 已减去 origin。

🔴 资源纪律（`docs/29`）：一次一个进程；每例启动前要求可用显存 ≥ --need-gb。

用法：
  python run_creatis_v2.py --cases 0,1,2,3,4,5
  python run_creatis_v2.py --cases 0 --variants base,aff
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import os
import sys
import json
import time
import argparse
import subprocess

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
PY = r'python'
OUTD = os.path.join(HERE, 'results', 'pmr_creatis')
LOGD = os.path.join(HERE, 'results', 'pmr_v2', 'logs')
os.makedirs(OUTD, exist_ok=True)
os.makedirs(LOGD, exist_ok=True)

BUDGET = ['--down', '2', '--mask', 'union', '--metric', 'local', '--res-metric', 'local',
          '--norm', 'robust', '--phases-per-step', '2',
          '--iters1', '800', '--iters2', '800', '--res-iters', '300',
          '--res-reg-scale', '10', '--cudnn-benchmark', '0']

VARIANTS = {
    'base': [],                                        # 与 DIRLAB 的 det10_best 同配置
    'aff':  ['--affine-first-iters', '800'],           # 大位移：先做仿射预对齐
    'ls':   ['--metric', 'ls', '--res-metric', 'ls'],  # C2 局部相似性属性（跨领域移植）
    'win45': ['--win-mm', '45'],                       # CREATIS 的 FOV 更大（500² vs 248²）
    # CREATIS 的肺只占 FOV 的 ~11%（DIRLAB 是 34%）⇒ select 模式会丢掉大量边界窗口
    'maskwt': ['--local-mask-mode', 'weight'],
}


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
    ap.add_argument('--cases', type=str, default='0,1,2,3,4,5')
    ap.add_argument('--variants', type=str, default='base')
    ap.add_argument('--need-gb', type=float, default=4.2,
                    help='每例启动前要求的最小可用显存（CREATIS@2mm 实测 case0 用 2.37 GB，'
                         '最大例外推约 3.9 GB ⇒ 4.2 足够）')
    ap.add_argument('--wait-min', type=float, default=150.0,
                    help='显存不足时最多等多少分钟（默认 150，足以等其他作业跑完）')
    ap.add_argument('--tag', type=str, default='cr')
    args = ap.parse_args()

    cases = [int(c) for c in args.cases.split(',') if c.strip()]
    vs = [v.strip() for v in args.variants.split(',') if v.strip()]
    bad = [v for v in vs if v not in VARIANTS]
    if bad:
        sys.exit(f'未知变体 {bad}；可选 {list(VARIANTS)}')
    print(f'CREATIS 运行: cases {cases}, variants {vs}；每例前要求可用显存 ≥ {args.need_gb} GB',
          flush=True)

    for v in vs:
        for cn in cases:
            jf = os.path.join(OUTD, f'{args.tag}_{v}_cr{cn}_2mm.json')
            if os.path.exists(jf):
                print(f'[skip] {v} cr{cn} 已完成', flush=True)
                continue
            for _ in range(int(args.wait_min * 60 / 30)):
                free = gpu_free_gb()
                if free >= args.need_gb:
                    break
                print(f'  [gpu-watch] 可用 {free:.1f} GB < {args.need_gb} GB，等待 30 s',
                      flush=True)
                time.sleep(30)
            else:
                print(f'[abort] {v} cr{cn}: 显存等待超时', flush=True)
                continue

            print(f'\n=== {v} cr{cn} @2mm（可用显存 {free:.1f} GB）===', flush=True)
            cmd = [PY, os.path.join(HERE, 'pmr_v2.py'),
                   '--dataset', 'creatis', '--case', str(cn)] + BUDGET + VARIANTS[v] + \
                  ['--tag', f'{args.tag}_{v}', '--out', OUTD]
            log = open(os.path.join(LOGD, f'{args.tag}_{v}_cr{cn}.log'), 'w', encoding='utf-8')
            log.write('$ ' + ' '.join(cmd) + '\n')
            log.flush()
            t0 = time.time()
            r = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONIOENCODING='utf-8',
                                        PYTHONUNBUFFERED='1'))
            dt = time.time() - t0
            log.write(f'\n[rc={r.returncode} {dt:.0f}s]\n')
            log.close()
            if os.path.exists(jf):
                d = json.load(open(jf, encoding='utf-8'))
                print(f'  >> 初始 {d["init_tre"]:.2f} → TRE {d["tre_total_STANDARD"]:.3f} mm | '
                      f'峰值显存 {d["peak_mem_gb"]:.2f} GB | {d["time_s"]:.0f}s | '
                      f'体素 {d["n_vox"]/1e6:.1f} M | 平均|d| {d["mean_disp_mm_t50"]:.2f} mm',
                      flush=True)
            else:
                print(f'  !! 失败 rc={r.returncode}（详见 {LOGD}）', flush=True)
    print('\n完成。')


if __name__ == '__main__':
    main()
