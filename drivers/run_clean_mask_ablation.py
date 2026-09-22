"""
run_clean_mask_ablation.py —— 消除混杂的**三路分解**（docs/25 要求的决定性实验）。
================================================================================
在同一参数化、同一优化器、同一迭代数下，只改**相似度定义**：

  global_full   全局 NCC，全图            （v1 的实际训练目标）
  local_full    局部 NCC，全图            （换度量，不动作用域）
  local_masked  局部 NCC，肺掩膜内        （换度量 + 换作用域）

于是：
  * local_full  vs global_full  = **"全局→局部"的效应**
  * local_masked vs local_full  = **"掩膜"的净效应**（这才是能归因给掩膜的量）
  * local_masked vs global_full = 原先 `test_mask_hypothesis` 报的**混合效应**（含混杂）

用法：
  python run_clean_mask_ablation.py --cases 1,3,5,8 --iters 1200
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import os, sys, json, time, argparse
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from diag_metric_probe import probe
from recompute_tre_standard import load_lm, sample_trilinear

OUTD = os.path.join(HERE, 'results', 'mask_ablation')
os.makedirs(OUTD, exist_ok=True)
MODES = ['global_full', 'local_full', 'local_masked']


def run(cn, down, iters, mode):
    d0, img0, img5, g, S, spacing, m, win, stride = probe(
        cn, down, iters=iters, sim_mode=mode)
    with torch.no_grad():
        d_mm = (d0.cpu().numpy()) * spacing[:, None, None, None]
    lm0, lm5 = load_lm(cn)
    tre = float(np.linalg.norm(lm0 + sample_trilinear(d_mm, lm0, spacing) - lm5, axis=1).mean())
    mag = np.linalg.norm(d_mm, axis=0)
    return {'case': cn, 'down_mm': down, 'iters': iters, 'sim_mode': mode,
            'init_tre': float(np.linalg.norm(lm0 - lm5, axis=1).mean()),
            'tre_std': tre, 'mean_disp_mm': float(mag.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=str, default='1,3,5,8')
    ap.add_argument('--down', type=int, default=2)
    ap.add_argument('--iters', type=int, default=1200)
    ap.add_argument('--modes', type=str, default=','.join(MODES))
    args = ap.parse_args()
    resf = os.path.join(OUTD, 'clean_ablation.json')
    res = json.load(open(resf, encoding='utf-8')) if os.path.exists(resf) else []
    done = {(r['case'], r['sim_mode']) for r in res}
    modes = [x.strip() for x in args.modes.split(',') if x.strip()]

    for cn in [int(x) for x in args.cases.split(',') if x.strip()]:
        for mode in modes:
            if (cn, mode) in done:
                print(f'[skip] case{cn} {mode}'); continue
            print(f'=== case{cn}  {mode} ===', flush=True)
            t0 = time.time()
            r = run(cn, args.down, args.iters, mode)
            r['time_s'] = round(time.time() - t0, 1)
            print(f'  >> TRE {r["tre_std"]:.3f} | 平均位移 {r["mean_disp_mm"]:.2f} mm '
                  f'(真值 {r["init_tre"]:.2f}) | {r["time_s"]:.0f}s', flush=True)
            res.append(r)
            json.dump(res, open(resf, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print('\n' + '=' * 84)
    print('三路分解')
    print('=' * 84)
    print(f'{"case":>4} {"真值":>7} {"global_full":>12} {"local_full":>11} '
          f'{"local_masked":>13} {"全局→局部":>10} {"掩膜净效应":>10} {"混合效应":>10}')
    for cn in sorted({r['case'] for r in res}):
        g = next((r for r in res if r['case'] == cn and r['sim_mode'] == 'global_full'), None)
        lf = next((r for r in res if r['case'] == cn and r['sim_mode'] == 'local_full'), None)
        lm = next((r for r in res if r['case'] == cn and r['sim_mode'] == 'local_masked'), None)
        if not (g and lf and lm):
            continue
        d1 = (lf['tre_std'] - g['tre_std']) / g['tre_std'] * 100
        d2 = (lm['tre_std'] - lf['tre_std']) / lf['tre_std'] * 100
        d3 = (lm['tre_std'] - g['tre_std']) / g['tre_std'] * 100
        print(f'{cn:>4} {g["init_tre"]:>7.2f} {g["tre_std"]:>12.3f} {lf["tre_std"]:>11.3f} '
              f'{lm["tre_std"]:>13.3f} {d1:>+9.1f}% {d2:>+9.1f}% {d3:>+9.1f}%')
    print('\n判读：')
    print('  · "掩膜净效应"才是能归因给掩膜的量')
    print('  · "混合效应" = 原先 test_mask_hypothesis 报的数字，**含混杂，不可单独归因**')
    print(f'\n已保存 {resf}')


if __name__ == '__main__':
    main()
