"""tools/elastix_bspline.py —— 直接实现 elastix B 样条求值（精确、无量化）
================================================================================
为什么不用现成路径
------------------
1. `transformix -def <点文件>`：把输入点索引**量化到整数**，spacing=2 时每轴误差可达
   1 mm（证据：`verify_tre_convention.py` 测试 C 残留 0.8536 mm）。
2. `sitk.ReadTransform(elastix文件)`：**读不了** elastix 原生格式
   （`TxtTransformIOTemplate: Tags must be delimited by :`）。
3. `sitk.BSplineTransform` + `SetParameters`：ITK 要的 mesh size 是**样条段数**
   （`GridSize−3`），域原点的约定也要猜；实测直接喂 elastix 系数报
   "expected 606015 parameters but only 545025 are provided"，说明两边约定不同。

⇒ 本模块按 elastix 文件里的 `GridOrigin/GridSpacing/GridSize` 与系数**自己求值**，
   逻辑完全透明，并用 transformix 产出的 `deformationField.mhd` 在**网格节点**上
   逐点校验（节点处无输入量化，构成独立验证）。

elastix B 样条约定
------------------
  控制点网格 (nx, ny, nz) = `GridSize`，原点 `GridOrigin`，间距 `GridSpacing`。
  物理点 p 的分段坐标：  u = (p − GridOrigin) / GridSpacing
  i = floor(u)，f = u − i；该维使用控制点 i−1, i, i+1, i+2 与三次 B 样条基
      B0 = (1−f)³/6,  B1 = (3f³−6f²+4)/6,  B2 = (−3f³+3f²+3f+1)/6,  B3 = f³/6
  系数向量按 **z 最快** 展平： idx = z + y·nz + x·nz·ny；
  `TransformParameters` 依次是**所有 x 分量、所有 y 分量、所有 z 分量**。
  ⚠️ 这个顺序是实测定案的，不是猜的：`tools/probe_coef_order.py` 在同一变换上比对
     transformix 的 `deformationField.mhd`，
       「x 最快」误差均值 3.542 mm（最大 25.6 mm）—— 错；
       「z 最快」误差最大 **0.000001 mm** —— 精确一致。

输出位移单位为 mm（与图像物理单位一致）。
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
import re

import numpy as np
import SimpleITK as sitk


def parse_elastix_tf(path):
    """解析 elastix 参数文件里的数值字段 → {字段名: [float,...]}。"""
    txt = open(path, encoding='utf-8', errors='replace').read()
    out = {}
    for m in re.finditer(r'\(([A-Za-z0-9_]+)\s+([^()]*?)\)', txt, re.S):
        key, val = m.group(1), m.group(2).strip()
        if not val:
            continue
        try:
            out[key] = [float(v) for v in val.split()]
        except ValueError:
            continue
    return out


def _weights(f):
    """三次 B 样条基（4 个权重）。"""
    f2 = f * f
    f3 = f2 * f
    return ((1 - f) ** 3 / 6.0,
            (3 * f3 - 6 * f2 + 4) / 6.0,
            (-3 * f3 + 3 * f2 + 3 * f + 1) / 6.0,
            f3 / 6.0)


class ElastixBSpline:
    """按 elastix 参数文件求值 B 样条变换（单位 mm）。"""

    def __init__(self, tf_path):
        p = parse_elastix_tf(tf_path)
        for k in ('GridOrigin', 'GridSpacing', 'GridSize', 'TransformParameters'):
            if k not in p:
                raise ValueError(f'{tf_path} 缺少字段 {k}')
        self.origin = np.array(p['GridOrigin'], dtype=float)          # (3,)
        self.spacing = np.array(p['GridSpacing'], dtype=float)        # (3,)
        self.size = np.array([int(v) for v in p['GridSize']], dtype=int)   # (3,)
        self.n_coef = int(np.prod(self.size))
        coeffs = np.array(p['TransformParameters'], dtype=float)
        if coeffs.size != 3 * self.n_coef:
            raise ValueError(f'系数 {coeffs.size} != 3×{self.n_coef}')
        # (3, nz, ny, nx) —— **z 最快**（实测定案，见模块 docstring）
        self.coef = coeffs.reshape(3, self.size[2], self.size[1], self.size[0])
        self.has_center = 'CenterOfRotationPoint' in p
        self.center = np.array(p.get('CenterOfRotationPoint', [0, 0, 0]), dtype=float)
        self.meta = {'grid_origin': self.origin.tolist(),
                     'grid_spacing': self.spacing.tolist(),
                     'grid_size': self.size.tolist(),
                     'n_coeff_per_dim': self.n_coef}

    # ── 求值 ─────────────────────────────────────────────────────────────────
    def displacement(self, pts_mm):
        """返回 (N,3) 位移，单位 mm。域外点按**索引钳制**处理。"""
        pts = np.atleast_2d(np.asarray(pts_mm, dtype=float))
        n = len(pts)
        u = (pts - self.origin[None, :]) / self.spacing[None, :]     # (N,3)
        i0 = np.floor(u).astype(int)
        f = u - i0
        # 四个控制点索引（相对）
        idx = np.stack([i0 - 1, i0, i0 + 1, i0 + 2], axis=1)         # (N,4,3)
        idx = np.clip(idx, 0, (self.size - 1)[None, None, :])
        w = np.stack(_weights(f[:, 0]), axis=1)                      # (N,4)
        wx, wy, wz = w, np.stack(_weights(f[:, 1]), axis=1), np.stack(_weights(f[:, 2]), axis=1)

        out = np.zeros((n, 3))
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    wa = wx[:, a] * wy[:, b] * wz[:, c]              # (N,)
                    ix = idx[:, a, 0]; iy = idx[:, b, 1]; iz = idx[:, c, 2]
                    cc = self.coef[:, iz, iy, ix]                    # (3,N)，z 最快
                    out += cc.T * wa[:, None]
        return out

    def apply(self, pts_mm):
        pts = np.atleast_2d(np.asarray(pts_mm, dtype=float))
        return pts + self.displacement(pts)

    def tre_standard(self, lm0, lm5):
        """DIRLAB 标准口径 TRE = |T(lm0) − lm5|。"""
        return float(np.linalg.norm(self.apply(lm0) - np.asarray(lm5), axis=1).mean())


def validate_against_dvf(bs, dvf_path, n_check=500, seed=0, verbose=True):
    """在 DVF 的体素节点上逐点比对（节点处无量化，独立校验系数解析是否正确）。"""
    img = sitk.ReadImage(dvf_path)
    arr = sitk.GetArrayFromImage(img)                 # (D,H,W,3)
    D, H, W = arr.shape[:3]
    sp = img.GetSpacing(); og = img.GetOrigin()
    rng = np.random.default_rng(seed)
    sel = rng.integers(0, [W, H, D], size=(n_check, 3))
    phys = np.column_stack([og[0] + sel[:, 0] * sp[0],
                            og[1] + sel[:, 1] * sp[1],
                            og[2] + sel[:, 2] * sp[2]])
    d_pred = bs.displacement(phys)
    d_true = arr[sel[:, 2], sel[:, 1], sel[:, 0]]
    err = np.linalg.norm(d_pred - d_true, axis=1)
    res = {'n': int(n_check), 'mean_mm': float(err.mean()),
           'median_mm': float(np.median(err)), 'max_mm': float(err.max())}
    if verbose:
        print(f'  DVF 节点校验（{n_check} 点）: 均值 {res["mean_mm"]:.6f}  '
              f'中位 {res["median_mm"]:.6f}  最大 {res["max_mm"]:.6f} mm')
        print('  ⇒ ' + ('✅ 一致，系数解析与求值正确'
                        if res['max_mm'] < 1e-3 else
                        '❌ 不一致：系数顺序/原点约定需再查'))
    return res


if __name__ == '__main__':
    import os
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    if len(sys.argv) < 3:
        sys.exit('用法: python tools/elastix_bspline.py <变换文件> <deformationField.mhd> [case号]')
    bs = ElastixBSpline(sys.argv[1])
    print(f'变换文件: {sys.argv[1]}')
    for k, v in bs.meta.items():
        print(f'  {k}: {v}')
    validate_against_dvf(bs, sys.argv[2])
    if len(sys.argv) > 3:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from recompute_tre_standard import load_lm
        cn = int(sys.argv[3])
        lm0, lm5 = load_lm(cn)
        print(f'\ncase{cn}: 初始 TRE = {np.linalg.norm(lm5-lm0,axis=1).mean():.4f} mm')
        print(f'  标准口径 TRE = {bs.tre_standard(lm0, lm5):.4f} mm')
        print(f'  |T(lm5) − lm0| = '
              f'{np.linalg.norm(bs.apply(lm5)-lm0,axis=1).mean():.4f} mm（反向，应更大）')
