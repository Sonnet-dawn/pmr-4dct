"""
nrms_landmark_test.py — 用 300 个真实 landmark 检验「NRMS 可靠性图」是否真能预测误差
================================================================================
背景（见 `docs/30`）：把时移地震的 cross-equalization / NRMS 检测门槛**整合进相位流形**后，
我们得到一个**不需要真值**的逐体素可靠性图。但"不需要真值"恰恰意味着**它必须被真值检验**，
否则就是自说自话。

DIRLAB 每例给 300 个对应 landmark（仅 T00/T50），于是可以做这个检验：

    landmark 误差  e_j = |lm0_j + d(lm0_j) − lm5_j|          （标准 TRE 口径）
    局部残差      r_j = 局部 NRMS 图在 lm0_j 处的值

    → 若 Spearman(e, r) 显著为正，则**残差图确实携带误差信息**。

同时给两个**对照预测器**，用来证明 NRMS 不是"纹理多就误差大"的别名：
    * 局部强度方差 var_f   （= C2 的结构显著性权重所用的量）
    * 局部梯度幅值 |∇f|

🔴 本脚本只做**相关性**检验，不主张因果，也不主张 r_j 是误差的无偏估计。
🔴 误差在 T00 框架定义，故必须在 **lm0** 处采样残差图（不是 lm5）。

用法
----
    python nrms_landmark_test.py --tag fold_resreg10
    python nrms_landmark_test.py --tag fold_folding,fold_resreg10 --down 2
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
import os
import re
import sys
import glob
import json
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pmr_fast import grid_, warp                      # noqa: E402
from pmr_v2 import load_imgs_v2, gauss_smooth3d, nrms_local  # noqa: E402
from recompute_tre_standard import load_lm, sample_trilinear  # noqa: E402

RESD = os.path.join(HERE, 'results', 'pmr_v2')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def sample_scalar(field, pts_mm, spacing):
    """在 pts_mm (N,3) 处三线性采样标量场 field (D,H,W)。返回 (N,)。"""
    D, H, W = field.shape
    idx = np.asarray(pts_mm, dtype=np.float64) / np.asarray(spacing, dtype=np.float64)
    out = np.zeros(len(idx))
    lo = np.floor(idx).astype(int)
    fr = idx - lo
    for j in range(len(idx)):
        x0, y0, z0 = lo[j]
        fx, fy, fz = fr[j]
        xs = [min(max(x0, 0), W - 1), min(max(x0 + 1, 0), W - 1)]
        ys = [min(max(y0, 0), H - 1), min(max(y0 + 1, 0), H - 1)]
        zs = [min(max(z0, 0), D - 1), min(max(z0 + 1, 0), D - 1)]
        wx = [1 - fx, fx]; wy = [1 - fy, fy]; wz = [1 - fz, fz]
        v = 0.0
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    v += field[zs[c], ys[b], xs[a]] * wz[c] * wy[b] * wx[a]
        out[j] = v
    return out


def spearman(x, y):
    """Spearman ρ + 近似双侧 p（t 近似，n 大时足够）。"""
    from scipy import stats
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def run_one(cn, down, dvf_path, sigma_mm=4.0, verbose=True):
    imgs, sitk_imgs = load_imgs_v2(cn, down, 'robust')
    spacing = np.array(sitk_imgs[0].GetSpacing(), dtype=np.float64)
    shape = (1, 1) + tuple(imgs[0].shape[2:])
    D, H, W = shape[2], shape[3], shape[4]
    S = torch.tensor([W, H, D], device=DEV)
    g = grid_(shape)

    dvf = np.load(dvf_path)
    if dvf.ndim == 4:
        dvf = dvf[None]
    d5 = dvf[5]                                      # (3,D,H,W) mm
    d5_vox = d5 / spacing[:, None, None, None]

    f0 = imgs[0].to(DEV)
    w5 = warp(imgs[5].to(DEV), g,
              torch.as_tensor(d5_vox, dtype=torch.float32, device=DEV), S)
    sigma = max(1.0, sigma_mm / float(spacing[0]))
    nl = nrms_local(w5, f0, sigma=sigma)[0, 0]       # (D,H,W)
    # 对照量：局部强度方差与局部梯度幅值（都在参考图 T00 上）
    fp = gauss_smooth3d(f0, sigma)
    var_f = (gauss_smooth3d(f0 * f0, sigma) - fp * fp).clamp_(min=0.0)[0, 0]
    gz, gy, gx = torch.gradient(f0[0, 0])
    grad = torch.sqrt(gz ** 2 + gy ** 2 + gx ** 2)
    grad = gauss_smooth3d(grad[None, None], sigma)[0, 0]

    nl_np = nl.cpu().numpy().astype(np.float64)
    var_np = var_f.cpu().numpy().astype(np.float64)
    grad_np = grad.cpu().numpy().astype(np.float64)

    lm0, lm5 = load_lm(cn)
    d_lm = sample_trilinear(d5, lm0, spacing)        # (N,3) mm
    err = np.linalg.norm(lm0 + d_lm - lm5, axis=1)   # 标准 TRE 口径，逐点

    r_nrm = sample_scalar(nl_np, lm0, spacing)
    r_var = sample_scalar(var_np, lm0, spacing)
    r_grd = sample_scalar(grad_np, lm0, spacing)

    # 只统计残差图有定义的 landmark（掩膜内 / NRMS>0）
    sel = r_nrm > 1e-6
    n_out = len(sel) - int(sel.sum())

    out = {'case': cn, 'down_mm': down, 'dvf': os.path.basename(dvf_path),
           'n_lm': int(len(err)), 'n_lm_in_mask': int(sel.sum()), 'n_lm_outside': int(n_out),
           'tre_mean': round(float(err.mean()), 4),
           'tre_median': round(float(np.median(err)), 4)}
    if sel.sum() >= 30:
        e = err[sel]
        out['spearman_nrms'] = dict(zip(('rho', 'p'), spearman(r_nrm[sel], e)))
        out['spearman_varf'] = dict(zip(('rho', 'p'), spearman(r_var[sel], e)))
        out['spearman_grad'] = dict(zip(('rho', 'p'), spearman(r_grd[sel], e)))
        # 分位数分箱：把 landmark 按残差图分 4 组，看各组平均误差（单调性检查）
        q = np.quantile(r_nrm[sel], [0.25, 0.5, 0.75])
        bins = np.digitize(r_nrm[sel], q)
        out['err_by_nrms_quartile'] = [round(float(e[bins == b].mean()), 4) if (bins == b).sum() else None
                                       for b in range(4)]
        out['n_by_quartile'] = [int((bins == b).sum()) for b in range(4)]
    else:
        out['spearman_nrms'] = None
    if verbose:
        sn = out.get('spearman_nrms')
        print(f"  case{cn} {down}mm [{out['dvf']}] lm {out['n_lm_in_mask']}/{out['n_lm']} 在掩膜内 "
              f"| TRE {out['tre_mean']:.3f} mm | ρ(NRMS)="
              f"{sn['rho']:+.3f} (p={sn['p']:.2e})" if sn else
              f"  case{cn} 掩膜内 landmark 不足")
        if sn:
            print(f"     对照 ρ(var_f)={out['spearman_varf']['rho']:+.3f}  "
                  f"ρ(|∇f|)={out['spearman_grad']['rho']:+.3f}  | "
                  f"各分位平均误差 {out['err_by_nrms_quartile']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', type=str, default='fold_resreg10')
    ap.add_argument('--down', type=int, default=2)
    ap.add_argument('--sigma-mm', type=float, default=4.0)
    ap.add_argument('--out', type=str, default=os.path.join(RESD, 'nrms_landmark.json'))
    args = ap.parse_args()

    tags = [t.strip() for t in args.tag.split(',') if t.strip()]
    rows = []
    for tag in tags:
        for p in sorted(glob.glob(os.path.join(RESD, f'{tag}_case*_{args.down}mm_dvf.npy'))):
            m = re.search(r'_case(\d+)_', os.path.basename(p))
            if not m:
                continue
            try:
                rows.append(run_one(int(m.group(1)), args.down, p, args.sigma_mm))
            except Exception as e:
                print(f'  [FAIL] {os.path.basename(p)}: {type(e).__name__}: {e}')

    # 汇总：跨病例合并（注意 landmark 不独立，同病例内相关，故同时给按病例的 ρ）
    pooled = [r for r in rows if r.get('spearman_nrms')]
    if pooled:
        rhos = [r['spearman_nrms']['rho'] for r in pooled]
        print(f"\n=== 汇总 {len(pooled)} 例 ===")
        print(f"  ρ(NRMS, 误差) 逐例: " + ' '.join(f"{x:+.3f}" for x in rhos))
        print(f"  中位 ρ = {np.median(rhos):+.3f} | 全部为正: {all(x > 0 for x in rhos)}")
        print(f"  对照 中位 ρ(var_f) = {np.median([r['spearman_varf']['rho'] for r in pooled]):+.3f}"
              f" | 中位 ρ(|∇f|) = {np.median([r['spearman_grad']['rho'] for r in pooled]):+.3f}")
    json.dump(rows, open(args.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n已保存 {args.out}  （{len(rows)} 条）')


if __name__ == '__main__':
    main()
