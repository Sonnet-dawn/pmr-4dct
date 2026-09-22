"""check_mask_volume.py —— 用绝对肺体积(mL)验证掩膜是否合理（文献 ~2500-6000 mL）。"""

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
sys.path.insert(0, HERE)
from lung_mask import lung_mask_native, raw_case

print(f'{"case":>4} {"size":>18} {"spacing":>16} {"FOV(L)":>8} {"lung(mL)":>9} {"lung/FOV":>9}')
vols = []
for cn in range(1, 11):
    im = raw_case(cn, 0)
    sz = im.GetSize(); sp = im.GetSpacing()
    fov_L = np.prod(sz) * np.prod(sp) / 1e6
    m = lung_mask_native(cn, 0)
    v = m.sum() * np.prod(sp) / 1000.0        # mL
    vols.append(v)
    print(f'{cn:>4} {str(sz):>18} {str(tuple(round(s,2) for s in sp)):>16} '
          f'{fov_L:>8.1f} {v:>9.0f} {m.mean()*100:>8.1f}%')
print(f'\n肺体积: 均值 {np.mean(vols):.0f} mL, 范围 {min(vols):.0f}–{max(vols):.0f} mL')
print('文献参考: 成人双肺 T00(吸气末) 约 4000-6500 mL; 若在此区间则掩膜可信')
