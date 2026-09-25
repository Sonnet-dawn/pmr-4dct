"""
jacobian_stats.py —— 形变场物理性（Jacobian 行列式）的**数值产物**
================================================================================
背景：论文声称 "PMR 全 10 相 %det(J)≤0 = 0%（无折叠）"，但此前**只有图（fig08/09），
      没有任何 JSON 数值产物**（见 docs/17 C7）。审稿人要求可追溯数字，本脚本补齐。

口径：
  φ(x) = x + d(x)，J = I + ∂d/∂x（中心差分，单位一致：先转成体素位移再求梯度）
  det(J) ≤ 0  -> 该体素发生折叠（定向反转）
  mean det(J) -> 体积保持性（理想 = 1）

用法：
  python jacobian_stats.py                       # 自动扫描已知 DVF 产物
  python jacobian_stats.py --glob "results/**/*dvf*.npy"
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
import os, json, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')


def det_jacobian_array(dvf_vox, spacing=None, chan_to_axis=(2, 1, 0)):
    """返回 det(J) 数组 (D,H,W)。**唯一的 Jacobian 实现** —— 其他脚本一律调用它，
    避免出现第二份实现而与已修正的通道/轴序配对产生分歧。

    ⚠️ 索引配对陷阱：本项目的位移张量通道序为 (x,y,z)（见 proto 中
    `(d/(S/2)).permute(1,2,3,0)` 与 S=[W,H,D]），而数组空间轴序为 (z,y,x)
    （SimpleITK GetArrayFromImage 输出）。Jacobian 必须让「第 i 个位移分量」
    与「第 i 个求导坐标」对齐，否则 det 会算错（近恒等场下仍近似为 1，
    但局部折叠被判错）。chan_to_axis[i] = 对应数组空间轴 i 的通道索引。

    spacing: 各向异性时用于把物理位移换算为体素位移（(x,y,z) mm）。"""
    d = dvf_vox.astype(np.float64)
    if spacing is not None:
        sp = np.asarray(spacing, dtype=np.float64).reshape(3, 1, 1, 1)
        d = d / sp                       # mm -> 体素
    d = d[list(chan_to_axis)]            # 通道重排，使分量 i 对齐空间轴 i
    g = np.stack([np.stack([np.gradient(d[i], axis=j) for j in range(3)])
                  for i in range(3)])    # g[i][j] = ∂d_i/∂axis_j
    J = g.transpose(2, 3, 4, 0, 1).copy()  # (3,3,D,H,W) -> (D,H,W,3,3): J[..., i, j]
    for i in range(3):
        J[..., i, i] += 1.0              # J = I + grad d
    return np.linalg.det(J)


def stats_from_det(det):
    """由 det(J) 数组汇总统计量。"""
    n = det.size
    return {
        'n_voxels': int(n),
        'pct_nonpositive': float(100.0 * np.count_nonzero(det <= 0) / n),
        'pct_negative': float(100.0 * np.count_nonzero(det < 0) / n),
        'pct_below_0.5': float(100.0 * np.count_nonzero(det < 0.5) / n),
        'mean_det': float(det.mean()),
        'std_det': float(det.std()),
        'min_det': float(det.min()),
        'max_det': float(det.max()),
        'p1_det': float(np.percentile(det, 1)),
        'p99_det': float(np.percentile(det, 99)),
    }


def det_stats(dvf_vox, spacing=None, chan_to_axis=(2, 1, 0)):
    """便捷入口：一次算完 det 并返回统计量。"""
    return stats_from_det(det_jacobian_array(dvf_vox, spacing, chan_to_axis))


# 兼容旧名（jacobian_stats.py 的 main 与其他脚本原先调用 det_jacobian_stats）
det_jacobian_stats = det_stats


def self_test():
    """用解析形变验证 Jacobian 实现（含通道/轴序配对）。

    在 (z,y,x) 数组轴序下：d_z 沿 axis0、(混合) d_y 沿 axis1、d_x 沿 axis2。
    """
    D, H, W = 12, 14, 16
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    a, b, c = 0.10, -0.05, 0.08
    # 位移通道序 (x,y,z)：x 分量沿 axis2、y 沿 axis1、z 沿 axis0
    d_x = a * xx
    d_y = b * yy
    d_z = c * zz
    d = np.stack([d_x, d_y, d_z]).astype(np.float64)
    st = det_jacobian_stats(d, spacing=None)
    expected = (1 + a) * (1 + b) * (1 + c)
    assert abs(st['mean_det'] - expected) < 1e-6, \
        f'self-test FAILED: got {st["mean_det"]:.6f}, expected {expected:.6f}'
    assert abs(st['min_det'] - expected) < 1e-6, 'self-test FAILED: min_det'

    # 纯剪切（非对角项）验证：d_x = 0.2*y  ->  det 仍为 1
    d2 = np.stack([0.2 * yy, np.zeros_like(yy), np.zeros_like(yy)]).astype(np.float64)
    st2 = det_jacobian_stats(d2, spacing=None)
    assert abs(st2['mean_det'] - 1.0) < 1e-6, \
        f'self-test FAILED (shear): got {st2["mean_det"]:.6f}, expected 1.0'

    # 折叠必须被检出：d_x = -1.5*x  ->  det = -0.5 < 0
    d3 = np.stack([-1.5 * xx, np.zeros_like(yy), np.zeros_like(yy)]).astype(np.float64)
    st3 = det_jacobian_stats(d3, spacing=None)
    assert abs(st3['mean_det'] - (-0.5)) < 1e-6, \
        f'self-test FAILED (folding): got {st3["mean_det"]:.6f}, expected -0.5'
    print('Jacobian self-test: PASS (scaling / shear / folding)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(R, 'jacobian_stats.json'))
    ap.add_argument('--spacing', type=float, default=None,
                    help='各向同性 spacing(mm)；默认按分辨率从文件名推断')
    args = ap.parse_args()

    self_test()

    report = {'protocol': 'phi(x)=x+d(x), J=I+grad d, central differences, '
                          'det(J)<=0 判定折叠; 通道序(x,y,z) 已对齐数组轴序(z,y,x)',
              'self_test': 'PASS', 'results': []}

    sources = []

    # 1) PMR 全相位 DVF
    for p in sorted(glob.glob(os.path.join(R, '**', '*dvf*.npy'), recursive=True)):
        try:
            sp = None
            if '_1mm' in p:
                sp = 1.0
            elif '_2mm' in p:
                sp = 2.0
            elif '_3mm' in p:
                sp = 3.0
            sources.append(('pmr', os.path.relpath(p, HERE), p, sp, '(10,3,D,H,W) mm'))
        except Exception:
            pass

    # 2) 逐对 B-spline / VoxelMorph 相位场（closure_loop npz，单位 mm）
    for p in sorted(glob.glob(os.path.join(R, 'closure_loop', '*_fields.npz'))):
        sources.append(('pairwise', os.path.relpath(p, HERE), p, 2.0, '(10,3,D,H,W) mm'))

    seen = set()
    for kind, rel, path, sp, note in sources:
        if rel in seen:
            continue
        seen.add(rel)
        try:
            if path.endswith('.npz'):
                arr = np.load(path)['fields']
            else:
                arr = np.load(path)
        except Exception as e:
            report['results'].append({'file': rel, 'error': str(e)})
            continue
        arr = np.asarray(arr)
        if arr.ndim == 4:                     # 单相位 (3,D,H,W)
            arr = arr[None]
        per_phase = []
        for i in range(arr.shape[0]):
            per_phase.append(det_jacobian_stats(arr[i], spacing=None if sp is None else
                                                np.array([sp, sp, sp])))
        agg = {k: float(np.mean([p[k] for p in per_phase])) for k in per_phase[0]
               if k != 'n_voxels'}
        agg['n_voxels'] = per_phase[0]['n_voxels']
        agg['worst_pct_nonpositive'] = float(max(p['pct_nonpositive'] for p in per_phase))
        agg['worst_min_det'] = float(min(p['min_det'] for p in per_phase))
        report['results'].append({'kind': kind, 'file': rel, 'spacing_mm': sp,
                                  'n_phases': int(arr.shape[0]), 'units': note,
                                  'aggregate_mean_over_phases': agg,
                                  'per_phase': per_phase})
        print(f'{kind:9s} {rel}')
        print(f'   %det(J)<=0 = {agg["pct_nonpositive"]:.4f}%  '
              f'(worst phase {agg["worst_pct_nonpositive"]:.4f}%)  '
              f'mean det = {agg["mean_det"]:.4f}  min det = {agg["min_det"]:.4f} '
              f'(worst {agg["worst_min_det"]:.4f})')

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'\n已保存: {args.out}')


if __name__ == '__main__':
    main()
