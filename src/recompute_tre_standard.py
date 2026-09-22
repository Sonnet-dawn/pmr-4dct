"""
recompute_tre_standard.py —— 把 PMR 的 TRE 重算为 DIRLAB 标准约定
================================================================================
背景（A1 审计发现，见 lit/A1_DIRLAB协议与基线文献核查.md §4/H3）：

  DIRLAB 标准约定（Castillo 2009 p.11 原文）：
      把 **T00（源/fixed）** 标记点用配准变换**正向映射**过去，与 T50 点求距离
      TRE = |T(lm0) − lm5|

  本项目此前的做法（run_elastix_baseline.py / pmr_fast.py）：
      在 **T50 点**处采样形变场并**取负**
      TRE = |lm5 − d(lm5) − lm0|

  两者一阶等价、二阶不同。A1 实测偏差：小位移例约 +10%，**大位移例可达 +29%**
  （因为场定义在 T00 域，却在 T50 点采样）。

  对 PMR 的 backward 场 d（定义在 T00 网格，"采样 moving 于 x+d"），
  标准约定即：
      T(lm0) = lm0 + d(lm0)
      TRE_std = |lm0 + d(lm0) − lm5|

本脚本**不重跑配准**，直接从已保存的 DVF 离线重算，因此可以立即把历史数字换算到
标准口径。同时输出旧口径以便量化差异。

产出：results/tre_convention_migration.json（含逐例双口径对照）
用法： python recompute_tre_standard.py [--down 2] [--cases 1,2,...]
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('PMR_DATA_ROOT') or os.path.join(HERE, '..', 'reference_4', 'data')
RES = os.path.join(HERE, 'results')
FAST = os.path.join(RES, 'fast_phase2')
OUT = os.path.join(RES, 'tre_convention_migration.json')


def sample_trilinear(d_mm, pts_mm, spacing):
    """在 pts_mm 处三线性采样位移场 d_mm (3,D,H,W) mm。返回 (N,3) mm。"""
    D, H, W = d_mm.shape[1:]
    idx = np.asarray(pts_mm, dtype=np.float64) / np.asarray(spacing, dtype=np.float64)
    out = np.zeros((len(idx), 3))
    lo = np.floor(idx).astype(int)
    fr = idx - lo
    for j in range(len(idx)):
        x0, y0, z0 = lo[j]
        fx, fy, fz = fr[j]
        xs = [min(max(x0, 0), W - 1), min(max(x0 + 1, 0), W - 1)]
        ys = [min(max(y0, 0), H - 1), min(max(y0 + 1, 0), H - 1)]
        zs = [min(max(z0, 0), D - 1), min(max(z0 + 1, 0), D - 1)]
        wx = [1 - fx, fx]; wy = [1 - fy, fy]; wz = [1 - fz, fz]
        v = np.zeros(3)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    v += d_mm[:, zs[c], ys[b], xs[a]] * wz[c] * wy[b] * wx[a]
        out[j] = v
    return out


def sample_nearest(d_mm, pts_mm, spacing):
    D, H, W = d_mm.shape[1:]
    out = np.zeros((len(pts_mm), 3))
    for j, p in enumerate(pts_mm):
        x = int(round(p[0] / spacing[0])); y = int(round(p[1] / spacing[1]))
        z = int(round(p[2] / spacing[2]))
        z = min(max(z, 0), D - 1); y = min(max(y, 0), H - 1); x = min(max(x, 0), W - 1)
        out[j] = d_mm[:, z, y, x]
    return out


def load_lm(cn):
    d = os.path.join(DATA, 'DIRLAB', 'points', f'case{cn}')
    return (np.loadtxt(os.path.join(d, f'case{cn}_300_T00_xyz_R.txt')),
            np.loadtxt(os.path.join(d, f'case{cn}_300_T50_xyz_R.txt')))


def load_dvf(cn, down):
    for p in (os.path.join(FAST, f's1f_dirlab_case{cn}_{down}mm_dvf.npy'),
              os.path.join(RES, 'phase1_2mm', f's1abc_dvf_dirlab_case{cn}_{down}mm_dvf.npy')):
        if os.path.exists(p):
            return np.load(p), os.path.relpath(p, HERE)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--down', type=int, default=2)
    ap.add_argument('--cases', type=str, default='1,2,3,4,5,6,7,8,9,10')
    args = ap.parse_args()
    down = args.down
    cases = [int(c) for c in args.cases.split(',') if c.strip()]

    out = {'down_mm': down,
           'conventions': {
               'legacy': 'TRE = |lm5 - d(lm5) - lm0|  (在 T50 采样取负；本项目旧口径)',
               'standard': 'TRE = |lm0 + d(lm0) - lm5|  (DIRLAB 标准，Castillo 2009 p.11)'},
           'note': '不重跑配准，直接从已保存 DVF 离线重算', 'results': {}}

    print(f'{"case":>4} {"legacy_near":>12} {"legacy_lin":>11} {"STD_lin":>9} {"Δ%":>7}')
    deg, std_vals, leg_vals = [], [], []
    for cn in cases:
        dvf, src = load_dvf(cn, down)
        if dvf is None:
            print(f'{cn:>4}   （无 DVF，跳过）')
            continue
        lm0, lm5 = load_lm(cn)
        spacing = np.array([down] * 3, dtype=float)
        d50 = dvf[5]                      # θ=π -> T00→T50 的 backward 场（mm）
        # 旧口径
        leg_near = np.linalg.norm(lm5 - sample_nearest(d50, lm5, spacing) - lm0, axis=1).mean()
        leg_lin = np.linalg.norm(lm5 - sample_trilinear(d50, lm5, spacing) - lm0, axis=1).mean()
        # 标准口径
        std_lin = np.linalg.norm(lm0 + sample_trilinear(d50, lm0, spacing) - lm5, axis=1).mean()
        init = np.linalg.norm(lm0 - lm5, axis=1).mean()
        d_pct = 100 * (leg_lin - std_lin) / std_lin
        print(f'{cn:>4} {leg_near:12.3f} {leg_lin:11.3f} {std_lin:9.3f} {d_pct:+6.1f}%')
        out['results'][str(cn)] = {'init_tre': float(init),
                                   'legacy_nearest': float(leg_near),
                                   'legacy_linear': float(leg_lin),
                                   'standard_linear': float(std_lin),
                                   'delta_pct': float(d_pct), 'dvf_source': src}
        deg.append(d_pct); std_vals.append(std_lin); leg_vals.append(leg_lin)
        del dvf

    if std_vals:
        print()
        print(f'10 例均值:  旧口径(三线性) {np.mean(leg_vals):.3f} mm  ->  '
              f'标准口径 {np.mean(std_vals):.3f} mm   （平均偏差 {np.mean(deg):+.1f}%）')
        print(f'  => 论文主表数字应整体下降约 {100*(np.mean(leg_vals)-np.mean(std_vals))/np.mean(leg_vals):.1f}%')
        out['summary'] = {'n': len(std_vals),
                          'mean_legacy_linear': float(np.mean(leg_vals)),
                          'mean_standard': float(np.mean(std_vals)),
                          'mean_delta_pct': float(np.mean(deg))}
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n已保存: {OUT}')


if __name__ == '__main__':
    main()
