"""
lung_mask.py —— 从 DIRLAB _R.mha 生成肺掩膜。
================================================================================
已实测标定：**HU = 存储值 − 1024**（1024->0 软组织, 24->-1000 空气, -2048->-3072 FOV 外）。

做法（逐层填充体轮廓，避免气管把肺与体外空气连通导致 3D 填充失败）：
  1. solid = HU > -500                    组织/骨
  2. 对每个轴位层：取 solid 的最大连通域 -> binary_fill_holes -> 得到"体内"区域
     （气管/肺在层内是闭合的洞，因此能被填上）
  3. lung = 体内 & (-990 <= HU <= -250)    体内空气 = 肺
  4. 3D 上保留最大的两个连通域（左右肺），可选膨胀补齐血管
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
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('PMR_DATA_ROOT') or os.path.join(HERE, '..', 'reference_4', 'data')
# 🔴 HU 偏移**逐数据集不同**，已实测确认（tools/probe_creatis_hu.py）：
#    DIRLAB 的 _R.mha 存 HU+1024；CREATIS 的 _R.mha 存**原生 HU**（offset 0）。
#    用错 offset 会让掩膜**完全为空**且不报错——正是 docs/17 X-7 那一类静默 bug。
HU_OFFSETS = {'dirlab': 1024, 'creatis': 0}
HU_OFFSET = HU_OFFSETS['dirlab']          # 向后兼容
_cache = {}


def hu_offset(dataset='dirlab'):
    if dataset not in HU_OFFSETS:
        raise ValueError(f'未知数据集 {dataset!r}，可选 {list(HU_OFFSETS)}')
    return HU_OFFSETS[dataset]


def raw_case(cn, p=0, dataset='dirlab'):
    """返回原生网格上的图像。p 为相位索引 0..9。"""
    if dataset == 'creatis':
        return sitk.ReadImage(os.path.join(DATA, 'CREATIS', str(cn), f'{p * 10:02d}_R.mha'))
    return sitk.ReadImage(os.path.join(DATA, 'DIRLAB', 'mha', f'case{cn}',
                                       f'case{cn}_T{p}0_R.mha'))


def case_origin(cn, p=0, dataset='dirlab'):
    """图像原点（物理坐标，mm）。landmark 采样前必须减去它。"""
    return np.array(raw_case(cn, p, dataset).GetOrigin(), dtype=np.float64)


def _largest_cc_2d(m):
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def lung_mask_native(cn, p=0, keep_two=True, dilate=1, verbose=False, dataset='dirlab'):
    """返回原生网格上的 bool 掩膜 (z,y,x)。"""
    key = (dataset, cn, p, keep_two, dilate)
    if key in _cache:
        return _cache[key]
    im = raw_case(cn, p, dataset)
    a = sitk.GetArrayFromImage(im).astype(np.int32) - hu_offset(dataset)   # -> HU
    z, y, x = a.shape

    solid = a > -500
    inside = np.zeros_like(solid)
    for k in range(z):
        sl = _largest_cc_2d(solid[k])
        if sl.sum() < 0.02 * sl.size:            # 层内组织太少（颈部/腹部边缘）跳过
            continue
        inside[k] = ndi.binary_fill_holes(sl)

    lung = inside & (a >= -990) & (a <= -250)

    if keep_two:
        lab, n = ndi.label(lung)
        if n > 2:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            keep = np.argsort(sizes)[::-1][:2]
            lung = np.isin(lab, keep)
        elif n == 0:
            pass
    if dilate > 0:
        lung = ndi.binary_dilation(lung, iterations=dilate)

    if verbose:
        print(f'  [mask] case{cn:>2} T{p}0: solid {solid.mean()*100:5.1f}% -> inside '
              f'{inside.mean()*100:5.1f}% -> lung {lung.mean()*100:5.1f}%  '
              f'(z-slices with lung: {int((lung.sum(axis=(1,2))>50).sum())}/{z})')
    _cache[key] = lung
    return lung


def _geom_down(m, src, down, interp):
    """把 bool/uint8 掩膜按整数倍降采样，**带上源图像的几何并显式保留 origin**。

    🔴 见 `pmr_v2.resample_down` 的说明：ResampleImageFilter 不设 origin 就会回落到
    (0,0,0)。掩膜与图像必须用**同一套约定**，否则非零 origin 数据集（CREATIS）上
    掩膜会与图像错位。这里通过 `CopyInformation(src)` + 显式 SetOutputOrigin 保证一致。
    """
    mt = sitk.GetImageFromArray(np.asarray(m).astype(np.uint8))
    mt.CopyInformation(src)
    if down == 1:
        return mt
    rs = sitk.ResampleImageFilter()
    rs.SetSize([max(1, s // down) for s in mt.GetSize()])
    rs.SetOutputSpacing([sp * down for sp in mt.GetSpacing()])
    rs.SetOutputOrigin(mt.GetOrigin())
    rs.SetOutputDirection(mt.GetDirection())
    rs.SetInterpolator(interp)
    return rs.Execute(mt)


def mask_at_down(cn, down, p=0, verbose=False, dataset='dirlab'):
    """返回与 `pmr_v2.load_imgs_v2(cn, down, dataset=...)` **逐体素对齐**的 float32 掩膜 (D,H,W)。"""
    m = lung_mask_native(cn, p, verbose=verbose, dataset=dataset)
    mt = _geom_down(m, raw_case(cn, p, dataset), down, sitk.sitkNearestNeighbor)
    return sitk.GetArrayFromImage(mt).astype(np.float32)


def mask_torch(cn, down, shape=None, p=0, device='cuda', verbose=False):
    import torch
    m = mask_at_down(cn, down, p, verbose)
    if shape is not None and tuple(m.shape) != tuple(shape[2:]):
        mt = torch.from_numpy(m)[None, None].float()
        m = torch.nn.functional.interpolate(mt, size=tuple(shape[2:]), mode='nearest')[0, 0].numpy()
    return torch.from_numpy(np.ascontiguousarray(m))[None, None].to(device)


# ==================== 对照实验用的掩膜变体 ====================
def _shift_zero(m, shift):
    """沿 (z,y,x) 平移掩膜，缺的地方补 0（不用 np.roll，避免环绕）。"""
    out = np.zeros_like(m)
    sz, sy, sx = shift
    z0, z1 = max(0, sz), min(m.shape[0], m.shape[0] + sz)
    y0, y1 = max(0, sy), min(m.shape[1], m.shape[1] + sy)
    x0, x1 = max(0, sx), min(m.shape[2], m.shape[2] + sx)
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return out
    out[z0:z1, y0:y1, x0:x1] = m[z0 - sz:z1 - sz, y0 - sy:y1 - sy, x0 - sx:x1 - sx]
    return out


def mask_variant_native(cn, mode='lung', dilate=0, shift=(0, 0, 0), p=0, verbose=False):
    """对照实验用的掩膜变体。

    mode:
      'lung'      —— 肺（默认）
      'eroded'    —— 肺向内侵蚀 |dilate| 体素（排除胸膜/肋骨滑动界面）
      'shifted'   —— 把肺掩膜整体平移到准静止软组织（心/纵隔/肝）区域，
                     用于**反向对照**：若任意等体积区域都能改善，则机制声称被削弱
    dilate: 追加的膨胀(正)/侵蚀(负)体素数
    """
    base = lung_mask_native(cn, p, dilate=0)          # 不预膨胀，便于精确控制
    m = base
    if dilate > 0:
        m = ndi.binary_dilation(m, iterations=dilate)
    elif dilate < 0:
        m = ndi.binary_erosion(m, iterations=-dilate)
    if tuple(shift) != (0, 0, 0):
        m = _shift_zero(m, shift)
    if verbose:
        print(f'  [mask:{mode}] case{cn} dilate={dilate} shift={shift} '
              f'-> 占比 {m.mean()*100:.1f}%（原 %{base.mean()*100:.1f}）')
    return m


def mask_variant_at_down(cn, down, mode='lung', dilate=0, shift=(0, 0, 0), p=0, verbose=False,
                         dataset='dirlab'):
    m = mask_variant_native(cn, mode, dilate, shift, p, verbose)
    mt = _geom_down(m, raw_case(cn, p, dataset), down, sitk.sitkNearestNeighbor)
    return sitk.GetArrayFromImage(mt).astype(np.float32)


def auto_shift_for_volume(cn, target_frac, p=0, min_shift=20, verbose=False):
    """找一个**实质性**平移量，使平移后的掩膜体积≈target_frac（等体积错位对照）。

    从肺区移到心/纵隔/肝等准静止软组织区。要求 |dz|+|dy| >= min_shift 体素，
    以免退化为零平移。
    """
    base = lung_mask_native(cn, p, dilate=0)
    shape = base.shape
    best, bestdiff = None, float('inf')
    for dz in range(-shape[0] // 3, shape[0] // 3, 6):
        for dy in range(-shape[1] // 4, shape[1] // 4, 8):
            if abs(dz) + abs(dy) < min_shift:
                continue
            mv = _shift_zero(base, (dz, dy, 0))
            d = abs(mv.mean() - target_frac)
            if d < bestdiff:
                bestdiff, best = d, (dz, dy, 0)
    if best is None:
        best = (min_shift, 0, 0)
    if verbose:
        mv = _shift_zero(base, best)
        print(f'  [auto_shift] case{cn} shift={best} -> 占比 {mv.mean()*100:.1f}%'
              f'（目标 {target_frac*100:.1f}%，肺 {base.mean()*100:.1f}%）')
    return best


if __name__ == '__main__':
    tot = []
    for cn in range(1, 11):
        m = lung_mask_native(cn, 0, verbose=True)
        tot.append(m.mean() * 100)
    print(f'\n10 例肺掩膜体积占比: 均值 {np.mean(tot):.1f}%  '
          f'范围 {min(tot):.1f}–{max(tot):.1f}%  （文献期望 ~15–30%）')
