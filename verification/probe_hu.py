"""probe_hu.py —— 确认 DIRLAB _R.mha 的强度标定，渲染正交切片供目视核查。"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import os, sys
import numpy as np
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'reference_4', 'data', 'DIRLAB', 'mha', 'case1')
im = sitk.ReadImage(os.path.join(D, 'case1_T00_R.mha'))
a = sitk.GetArrayFromImage(im).astype(np.float32)      # (z,y,x)
z, y, x = a.shape
print('shape', a.shape, 'spacing', im.GetSpacing(), 'origin', im.GetOrigin())
print('corner values:', a[0, 0, 0], a[-1, -1, -1], a[z // 2, 0, 0], a[0, y // 2, x // 2])
print('center value:', a[z // 2, y // 2, x // 2])
print('percentiles:', {p: float(np.percentile(a, p)) for p in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)})
for lo, hi, tag in [(-2048, -1900, 'min-valued'), (-1050, -950, 'HU=-1000'),
                    (-1024, -900, 'near -1000'), (-500, -400, 'HU=-450'),
                    (0, 50, 'near 0'), (24, 40, 'HU=-1000 if +1024'),
                    (524, 560, 'HU=-500 if +1024'), (1024, 1100, 'HU=0 if +1024')]:
    frac = ((a >= lo) & (a < hi)).mean() * 100
    print(f'  frac[{lo:6d},{hi:6d}) = {frac:6.3f}%   ({tag})')

# 保存正交中切片
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = os.path.join(HERE, 'results', 'diag_pairwise', 'hu_probe_case1.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    for j, zz in enumerate([z // 4, z // 2, 3 * z // 4]):
        ax[0, j].imshow(a[zz], cmap='gray', vmin=-1000, vmax=1000)
        ax[0, j].set_title(f'axial z={zz}')
    ax[0, 3].imshow(a[z // 2], cmap='gray', vmin=float(a[z // 2].min()), vmax=float(a[z // 2].max()))
    ax[0, 3].set_title('axial z=mid (full range)')
    ax[1, 0].imshow(a[:, y // 2, :], cmap='gray', vmin=-1000, vmax=1000); ax[1, 0].set_title('coronal')
    ax[1, 1].imshow(a[:, :, x // 2], cmap='gray', vmin=-1000, vmax=1000); ax[1, 1].set_title('sagittal')
    h, e = np.histogram(a, bins=200)
    ax[1, 2].plot(e[:-1], h); ax[1, 2].set_yscale('log'); ax[1, 2].set_title('histogram (log)')
    m = (a < -400) if a.min() < -400 else (a < np.percentile(a, 20))
    ax[1, 3].imshow(m[z // 2], cmap='gray'); ax[1, 3].set_title(f'mask preview (< -400) frac={m.mean()*100:.1f}%')
    plt.tight_layout(); plt.savefig(out, dpi=70)
    print('已保存', out)
except Exception as ex:
    print('渲染失败:', ex)
