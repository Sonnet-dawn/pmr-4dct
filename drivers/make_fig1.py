"""
make_fig1.py —— 生成 Paper B 的 Fig. 1（方法示意图）+ **程序化版面自检**。

为什么要自检：本机视觉后端不可用时无法"看图"，因此用 renderer 的包围盒
断言"没有文字溢出画布、没有文字互相重叠" —— 这能抓住绝大多数排版缺陷。

两块：
  (a) 相位流形：10 个相位 → 一条以 θ 为参数的周期曲线；锚定 d(x,0)=0、闭合 d(x,2π)=d(x,0)
  (b) 计算图：24 通道系数在**粗网格上**与相位权合成 → 只上采样 3 通道 → warp → 相似度
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
import itertools

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.stdout.reconfigure(encoding='utf-8')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'paperB', 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.size': 8.0, 'font.family': 'DejaVu Sans',
                     'axes.linewidth': 0.8, 'savefig.dpi': 400})

fig = plt.figure(figsize=(7.4, 3.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.30], wspace=0.30)

# ============================================================ (a)
ax = fig.add_subplot(gs[0, 0])
th = np.linspace(0, 2 * np.pi, 400)
d = 0.55 * (1 - np.cos(th)) - 0.22 * np.sin(2 * th)
ax.plot(th, d, color='#1f4e79', lw=1.9, zorder=3)
ax.axhline(0, color='0.78', lw=0.7, zorder=1)

tp = np.linspace(0, 2 * np.pi, 10, endpoint=False)
dp = 0.55 * (1 - np.cos(tp)) - 0.22 * np.sin(2 * tp)
ax.plot(tp, dp, 'o', ms=4.5, mfc='#c00000', mec='white', mew=0.8, zorder=4)

ax.plot([0], [0], 'o', ms=9, mfc='none', mec='#2e7d32', mew=1.8, zorder=5)
ax.plot([2 * np.pi], [0], 'o', ms=9, mfc='none', mec='#2e7d32', mew=1.8, zorder=5)
ax.axvline(0, color='#2e7d32', lw=0.8, ls=':', zorder=1)
ax.axvline(2 * np.pi, color='#2e7d32', lw=0.8, ls=':', zorder=1)

ax.annotate(r'$d(x,0)=0$', xy=(0.06, 0.03), xytext=(0.55, -0.78), fontsize=8,
            color='#2e7d32', arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.0))
ax.annotate(r'$d(x,2\pi)=d(x,0)$', xy=(2 * np.pi - 0.06, 0.03),
            xytext=(3.15, -0.78), fontsize=8, color='#2e7d32',
            arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.0))

ax.text(3.0, 1.12, r'$d(x,\theta)=\sum_k[a_k(\cos k\theta-1)+b_k\sin k\theta]$',
        fontsize=8, color='#1f4e79', ha='center')

h = [plt.Line2D([], [], color='#1f4e79', lw=1.9),
     plt.Line2D([], [], color='#c00000', marker='o', ls='', ms=4.5),
     plt.Line2D([], [], color='#2e7d32', marker='o', ls='', ms=8, mfc='none', mew=1.6)]
ax.legend(h, [r'$d(x,\theta)$', r'sampled phases ($N{=}10$)',
              'anchored & closed'], loc='lower center', bbox_to_anchor=(0.5, -0.42),
          fontsize=7.2, frameon=False, ncol=3, handletextpad=0.4, columnspacing=0.9)

ax.set_xlim(-0.35, 2 * np.pi + 0.35)
ax.set_ylim(-1.0, 1.30)
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
ax.set_xlabel(r'breathing phase $\theta$')
ax.set_ylabel(r'displacement $d$')
ax.set_title('(a) One periodic curve, not ten independent fields', fontsize=8.6, pad=7)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

# ============================================================ (b)
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('(b) Exact coarse-grid decomposition', fontsize=8.6, pad=7)


def box(x, y, w, hh, text, fc='#eef3fa', ec='#1f4e79', fs=7.2):
    ax2.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle='round,pad=0.09',
                                 fc=fc, ec=ec, lw=1.0, zorder=2))
    ax2.text(x + w / 2, y + hh / 2, text, ha='center', va='center',
             fontsize=fs, zorder=3, linespacing=1.3)


def arrow(x1, y1, x2, y2, color='#1f4e79', lw=1.1, ls='-'):
    ax2.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                  mutation_scale=9, color=color, lw=lw, ls=ls,
                                  zorder=4, shrinkA=1.5, shrinkB=1.5))


box(0.10, 7.30, 2.05, 1.55, 'reference\nimage $I_0$', fc='#f2f2f2', ec='0.45')
box(0.10, 4.60, 2.05, 1.55, 'sampled phase\n$I_{\\theta_i}$', fc='#f2f2f2', ec='0.45')

box(2.85, 7.30, 2.30, 1.55, 'coefficients on\ncoarse grid\n($3K{\\times}2$ ch.)',
    fc='#e8f2e8', ec='#2e7d32')
arrow(2.15, 8.07, 2.85, 8.07)

box(2.85, 4.60, 2.30, 1.75, 'compose with\nphase weights\non coarse grid',
    fc='#fff4e5', ec='#b26a00')
arrow(4.00, 7.30, 4.00, 6.35)
arrow(4.00, 6.15, 4.00, 6.35, color='0.45', ls='--')
arrow(2.15, 5.37, 2.85, 5.37, color='0.45', ls='--')

box(5.90, 4.60, 1.85, 1.75, 'upsample\n\n3 channels\n(not $3K{\\times}2$)',
    fc='#fdeaea', ec='#c00000')
arrow(5.15, 5.47, 5.90, 5.47)

box(5.90, 7.30, 1.85, 1.55, 'warp\n$I_{\\theta_i}$', fc='#f2f2f2', ec='0.45')
arrow(7.75, 5.47, 8.55, 5.47)
arrow(8.55, 5.47, 8.55, 8.07)
arrow(8.55, 8.07, 7.75, 8.07)

ax2.add_patch(FancyBboxPatch((8.75, 6.55), 1.15, 3.05, boxstyle='round,pad=0.09',
                             fc='#eef3fa', ec='#1f4e79', lw=1.2, zorder=2))
ax2.text(9.32, 8.07, 'masked\nsimilarity\nloss', ha='center', va='center',
         fontsize=7.2, zorder=3)
arrow(8.65, 8.07, 8.75, 8.07)

ax2.text(0.10, 3.15, r'$U\!\left(\sum_k c_k w_k(\theta)\right)\equiv\sum_k U(c_k)w_k(\theta)$',
         fontsize=10.0, color='#1f4e79')
ax2.text(0.10, 1.55, 'exact: trilinear interpolation is linear in its input,\n'
                     'and its weights do not depend on the channel index $k$.',
         fontsize=7.2, color='0.28', linespacing=1.4)
ax2.text(0.10, 0.75, 'coefficient tensor:  1.29 GiB  →  2.55 MiB   (519×)',
         fontsize=7.6, color='#2e7d32')
ax2.text(0.10, 0.42, '[derived: tensor-size arithmetic, not a measured allocation]',
         fontsize=6.4, color='0.35', style='italic')

# ============================================================ 版面自检
fig.canvas.draw()
r = fig.canvas.get_renderer()
W, H = fig.canvas.get_width_height()
figbb = fig.bbox

texts = []
for a in fig.axes:
    for t in a.texts:
        texts.append((a, t, t.get_window_extent(renderer=r)))
    if a.get_title():
        pass

print(f'画布 {W}x{H} px')
problems = []

# 1) 溢出画布
for a, t, bb in texts:
    if bb.x0 < -2 or bb.y0 < -2 or bb.x1 > W + 2 or bb.y1 > H + 2:
        problems.append(f'溢出画布: {t.get_text()[:38]!r} bbox=({bb.x0:.0f},{bb.y0:.0f},'
                        f'{bb.x1:.0f},{bb.y1:.0f})')

# 2) 文字互相重叠（同一 axes 内）
# 🔴 注意两件事：
#   (i) 每维重叠必须**下限取 0**。若直接相乘，两个"负重叠"会得到正数而被误判为重叠
#       （第一版就踩了这个坑：(-21)×(-47)=+996 被报成"重叠 996 px²"）。
#   (ii) Annotation 的 window_extent **包含箭头**，会把框撑得很大，故排除。
for a in fig.axes:
    ts = [t for t in a.texts if type(t).__name__ != 'Annotation']
    for t1, t2 in itertools.combinations(ts, 2):
        b1 = t1.get_window_extent(renderer=r)
        b2 = t2.get_window_extent(renderer=r)
        ox = max(0.0, min(b1.x1, b2.x1) - max(b1.x0, b2.x0))
        oy = max(0.0, min(b1.y1, b2.y1) - max(b1.y0, b2.y0))
        ov = ox * oy
        if ov > 4 and b1.width > 3 and b2.width > 3:
            problems.append(f'文字重叠: {t1.get_text()[:26]!r} × {t2.get_text()[:26]!r} '
                            f'重叠 {ox:.0f}×{oy:.0f} px')

# 3) 文字是否超出所属 box（只查 (b) 里的 box 与其中文字）
if problems:
    print(f'\n🔴 发现 {len(problems)} 个排版问题：')
    for p in problems:
        print(f'   - {p}')
else:
    print('\n✅ 版面自检通过：无溢出、无文字重叠')

fig.savefig(os.path.join(OUT, 'fig1_method.png'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig1_method.pdf'), bbox_inches='tight')
for ext in ('png', 'pdf'):
    p = os.path.join(OUT, f'fig1_method.{ext}')
    print(f'  saved {p} ({os.path.getsize(p)/1024:.0f} KB)')
plt.close(fig)
