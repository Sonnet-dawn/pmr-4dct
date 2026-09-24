"""tools/make_fig2.py —— 生成 Paper B 的 Fig. 2（显存/内存缩放）
================================================================================
两个面板，**刻意分开**，因为两者量的是**不同的资源**：

  (a) elastix 的**主机内存**随 B 样条控制网格细化的增长（1 mm、单分辨率，单变量）
      —— 实测 4 点 + 线性拟合 + 2 个"被中止"的下界，以及本机 31.4 GB 的容量线。
  (b) PMR 的**GPU 显存**随体素数的增长（2 mm 与 1 mm 两组实测），
      以及"朴素逐体素参数化"的**推导**需求线。

🔴 纪律：CPU RAM 与 GPU 显存不可放在同一坐标轴上比较（Paper B §3.3 的免责说明）。
   因此这里是两个独立面板，各自的纵轴单位与含义不同，不做跨面板的比值解读。

用法: python tools/make_fig2.py
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
import glob
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'paperB', 'figures')

plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8,
                     'font.family': 'DejaVu Sans'})

# ── 面板 (a)：elastix 主机内存 vs 控制点 ─────────────────────────────────────
rows = []
for p in glob.glob(os.path.join(HERE, 'results', 'elastix_memory',
                                'case1_grid*mm.json')):
    j = json.load(open(p, encoding='utf-8'))
    rows.append(j)
rows.sort(key=lambda r: -float(r['grid_mm']))
n8 = next(r['n_control_points'] for r in rows if float(r['grid_mm']) == 8.0)

ok = [r for r in rows if r['status'] == 'ok']
N = np.array([r['n_control_points'] for r in ok], float)
M = np.array([r['peak_rss_gb'] for r in ok], float)
b, a = np.polyfit(N, M, 1)
r2 = 1 - ((M - (a + b * N)) ** 2).sum() / ((M - M.mean()) ** 2).sum()

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))

axa = ax[0]
xs = np.logspace(np.log10(N.min() * 0.7), np.log10(2.6e7), 200)
axa.plot(xs, a + b * xs, '-', color='0.55', lw=1.2, zorder=1,
         label=f'linear fit (R² = {r2:.5f})')
axa.plot(N, M, 'o', ms=6, color='C0', zorder=3, label='measured (completed)')
for r in rows:
    if r['status'] != 'ok':
        g = float(r['grid_mm'])
        nc = n8 * (8.0 / g) ** 3
        axa.annotate('', xy=(nc, r['peak_rss_gb'] * 1.9),
                     xytext=(nc, r['peak_rss_gb']),
                     arrowprops=dict(arrowstyle='-|>', color='C3', lw=1.2))
        axa.plot([nc], [r['peak_rss_gb']], 'v', ms=6, color='C3', zorder=3)
axa.plot([], [], 'v', ms=6, color='C3',
         label='stopped: out of memory\n(lower bound, still rising)')
axa.axhline(31.4, ls='--', lw=1.0, color='k')
axa.text(4.2e4, 31.4 * 1.08, 'machine RAM 31.4 GB', fontsize=7.5)
axa.set_xscale('log'); axa.set_yscale('log')
axa.set_xlabel('B-spline control points')
axa.set_ylabel('peak host RAM (GiB)')
axa.set_title('(a) elastix at 1 mm: host memory vs control grid', fontsize=8.5)
axa.legend(fontsize=6.8, loc='upper left', frameon=False)
axa.grid(alpha=0.25, which='both', lw=0.4)

# ── 面板 (b)：PMR GPU 显存 vs 体素数 ─────────────────────────────────────────
series = {}
for pat, lbl, mk in ((os.path.join(HERE, 'results', 'pmr_v2',
                                   'det10_best_case*_2mm.json'), 'PMR, 2 mm', 'o'),
                     (os.path.join(HERE, 'results', 'pmr_v2_1mm',
                                   'v2_1mm_case*_1mm.json'), 'PMR, 1 mm', 's')):
    nv, pk = [], []
    for p in glob.glob(pat):
        j = json.load(open(p, encoding='utf-8'))
        if j.get('n_vox') and j.get('peak_mem_gb') is not None:
            nv.append(j['n_vox'] / 1e6)
            pk.append(j['peak_mem_gb'])
    series[lbl] = (np.array(nv), np.array(pk), mk)

axb = ax[1]
for lbl, (nv, pk, mk) in series.items():
    o = np.argsort(nv)
    axb.plot(nv[o], pk[o], mk + '-', ms=5, lw=1.1, label=f'{lbl} (measured)')

nvmax = 120.0
nv_lin = np.linspace(1, nvmax, 100)
naive = 5.17 * nv_lin / 14.45344      # 见 §2.1 的 tensor 算术（推导，非实测）
axb.plot(nv_lin, naive, '--', color='C3', lw=1.2,
         label='naive per-voxel (derived, not measured)')
axb.axhline(7.96, ls='--', lw=1.0, color='k')
axb.text(3, 7.96 * 1.06, '8 GiB card (7.96 GiB usable)', fontsize=7.5)
axb.set_xscale('log'); axb.set_yscale('log')
axb.set_xlabel('voxels per phase (millions)')
axb.set_ylabel('peak GPU memory (GiB)')
axb.set_title('(b) PMR: GPU memory vs volume', fontsize=8.5)
axb.legend(fontsize=6.8, loc='upper left', frameon=False)
axb.grid(alpha=0.25, which='both', lw=0.4)

fig.tight_layout()

# ── 程序化版式自检（替代肉眼检查） ───────────────────────────────────────────
# 动机：图必须**目视**检查才可靠，但视觉后端可能不可用（本轮实测被限流 429）。
# 于是直接用 matplotlib 的渲染器取包围盒，检查两类会毁掉期刊图的缺陷：
#   ① 文字之间/文字与图例互相重叠；② 数据点被轴范围裁掉。
fig.canvas.draw()
rend = fig.canvas.get_renderer()
problems = []
for i, axx in enumerate(ax):
    tags = [f'(a) elastix' if i == 0 else '(b) PMR']
    # ① 越界：所有数据点必须落在轴范围内
    x0, x1 = axx.get_xlim(); y0, y1 = axx.get_ylim()
    for ln in axx.get_lines():
        # 只检查**数据坐标系**下的线：`axhline`/`axvline` 的 x（或 y）用轴分数坐标，
        # 恒为 [0,1]，若一起检查会稳定误报"越界"。
        if ln.get_transform() is not axx.transData:
            continue
        xs_, ys_ = ln.get_xdata(), ln.get_ydata()
        if len(xs_) == 0:
            continue
        if (np.min(xs_) < x0 or np.max(xs_) > x1
                or np.min(ys_) < y0 or np.max(ys_) > y1):
            problems.append(f'{tags[0]}: 折线被轴范围裁掉 '
                            f'(x∈[{np.min(xs_):.3g},{np.max(xs_):.3g}] vs [{x0:.3g},{x1:.3g}], '
                            f'y∈[{np.min(ys_):.3g},{np.max(ys_):.3g}] vs [{y0:.3g},{y1:.3g}])')
    # ② 文字重叠
    texts = [t for t in axx.texts if t.get_text().strip()]
    boxes = [(t.get_text()[:28], t.get_window_extent(rend)) for t in texts]
    for j in range(len(boxes)):
        for k in range(j + 1, len(boxes)):
            if boxes[j][1].overlaps(boxes[k][1]):
                problems.append(f'{tags[0]}: 文字重叠 "{boxes[j][0]}" ↔ "{boxes[k][0]}"')
    lg = axx.get_legend()
    if lg is not None:
        lb = lg.get_window_extent(rend)
        for name, tb in boxes:
            if lb.overlaps(tb):
                problems.append(f'{tags[0]}: 图例与文字重叠 "{name}"')

for ext in ('png', 'pdf'):
    p = os.path.join(OUT, f'fig2_memory.{ext}')
    fig.savefig(p, dpi=300)
    print(f'已保存 {os.path.relpath(p, HERE)}')

print(f'\n版式自检：{"✅ 无重叠、无越界" if not problems else "❌ 发现问题"}')
for pr in problems:
    print('   -', pr)

print(f'\n(a) 拟合 peak = {a:.3f} + {b:.3e} × N_ctrl,  R² = {r2:.5f}（{len(ok)} 点）')
print(f'(b) PMR 实测点: ' + ', '.join(
    f'{k}: n={len(v[0])}, 峰值 {v[1].min():.2f}–{v[1].max():.2f} GiB'
    for k, v in series.items()))
if problems:
    sys.exit(1)
