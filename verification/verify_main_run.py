"""verify_main_run.py —— 核验主运行结果的自洽性（防再次误报）。

用法: python verify_main_run.py [tag]     # 默认 main_base
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import os, sys, json, glob, argparse
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'results', 'pmr_v2')

# --- SUITE-SKIP guard (injected by hand; 见 tools/add_suite_skip_guard.py 的说明) -----
# 本项核对的是 `results/` 里的一组历史结果。公开仓库发行版不含 results/，
# 在新 clone 里无法执行 ⇒ **显式跳过**（入口记为 skip，**不计入通过**）。
if not os.path.isdir(D):
    print('SUITE-SKIP: 缺少 results/pmr_v2 —— 本项核对历史结果文件，'
          '仓库发行版不附带；在开发树（含 results/）中运行同一入口即可完整执行。',
          flush=True)
    sys.exit(0)
# --- end SUITE-SKIP guard ------------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument('tag', nargs='?', default='main_base')
args = ap.parse_args()

rows = []
for f in glob.glob(os.path.join(D, f'{args.tag}_case*_2mm.json')):
    rows.append(json.load(open(f, encoding='utf-8')))
rows.sort(key=lambda d: d['case'])
if not rows:
    sys.exit(f'未找到 {args.tag} 的结果')

print('=' * 104)
print(f'主运行 {args.tag} 逐例核验')
print('=' * 104)
hdr = (f'{"case":>4} {"init":>6} {"TRE":>7} {"流形TRE":>8} {"mean|d|":>8} {"p95|d|":>7} '
       f'{"mask%":>6} {"enc":>4} {"res":>4} {"pps":>4} {"mem":>5} {"time":>6} '
       f'{"closure0":>10} {"closure2pi":>11}')
print(hdr)
bad = []
for d in rows:
    print(f'{d["case"]:>4} {d["init_tre"]:>6.2f} {d["tre_total_STANDARD"]:>7.3f} '
          f'{d["tre_manifold_STD"]:>8.3f} {d["mean_disp_mm_t50"]:>8.2f} '
          f'{d["p95_disp_mm_t50"]:>7.2f} {d["mask_frac"]*100:>6.1f} {d["enc_down"]:>4} '
          f'{d["res_down"]:>4} {d["phases_per_step"]:>4} {d["peak_mem_gb"]:>5.2f} '
          f'{d["time_s"]:>6.0f} {d["closure_0"]:>10.2e} {d["closure_2pi"]:>11.2e}')
    # 自洽性检查
    if d['closure_2pi'] > 1e-3:
        bad.append(f'case{d["case"]}: 闭合误差 {d["closure_2pi"]:.2e} mm 偏大')
    if d['mean_disp_mm_t50'] <= 0:
        bad.append(f'case{d["case"]}: 平均位移非正')
    if d['tre_total_STANDARD'] > d['init_tre']:
        bad.append(f'case{d["case"]}: TRE 比 identity 更差！')

print()
print('=' * 104)
print('关键比值')
print('=' * 104)
print(f'{"case":>4} {"真值|d|":>8} {"模型|d|":>8} {"比值":>7}   判读')
for d in rows:
    r = d['mean_disp_mm_t50'] / d['init_tre']
    tag = '贴合理想' if 0.6 <= r <= 1.4 else ('偏小(欠形变)' if r < 0.6 else '偏大(过形变)')
    print(f'{d["case"]:>4} {d["init_tre"]:>8.2f} {d["mean_disp_mm_t50"]:>8.2f} {r:>7.2f}   {tag}')

v = [d['tre_total_STANDARD'] for d in rows]
print(f'\n10 例均值 = {np.mean(v):.3f} mm   (中位 {np.median(v):.3f}, 最大 {max(v):.3f})')

print()
print('=' * 104)
if bad:
    print('🔴 发现问题：')
    for b in bad:
        print('  -', b)
else:
    print('✅ 自洽性检查全部通过（闭合、位移、不劣于 identity）')
