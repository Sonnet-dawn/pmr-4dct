"""
verify_coarse_tre.py —— 检验"4mm 粗尺度结果更好"是不是**评估分辨率的假象**。
================================================================================
动机：`run_v2_multiscale.py` 的 4mm 粗尺度结果 TRE 比 2mm 细尺度更低（case8: 4.022 vs 7.207），
      但 4mm 的位移场更平滑，关键点采样可能"虚低"。

做法：把保存的 **4mm 周期系数**重建为 **2mm 网格上的位移场**，再算一次标准口径 TRE。
      · 物理场不变，只是评估网格更细；
      · 换算：coef 是体素单位，物理位移 = coef × spacing
        => 2mm 网格上的系数 = 4mm 系数 × (4/2) = ×2

若 TRE 基本不变 -> 4mm 的优势是**真实的**；
若 TRE 明显上升 -> 是评估假象，必须以 2mm 评估为准。

用法：
  python verify_coarse_tre.py --cases 1,2,3,4,5,8
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
import os, sys, json, argparse, glob
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))

_NEED_DIR = os.path.join(HERE, *'results/pmr_v2'.split("/"))

# --- SUITE-SKIP guard (injected by tools/add_suite_skip_guard.py) ---------------
# 本检查需要 `results/` 里的历史产物。仓库发行版**不附带**结果文件，因此在新 clone 里
# 它无法执行。这里**显式声明"跳过"而不是悄悄通过**：`run_verification_suite.py`
# 见到 `SUITE-SKIP:` 会记为 skip 状态，且**不计入通过数**。
if not os.path.isdir(_NEED_DIR):
    print('SUITE-SKIP: 缺少 results/pmr_v2 —— 本项需要历史结果文件，仓库发行版不附带；'
          '在开发树（含 results/）中运行同一入口即可完整执行。', flush=True)
    sys.exit(0)
# --- end SUITE-SKIP guard ------------------------------------------------------

sys.path.insert(0, HERE)
from recompute_tre_standard import load_lm, sample_trilinear
from pmr_fast import disp_coarse

D = os.path.join(HERE, 'results', 'pmr_v2')


def rebuild(coef_np, target_shape, from_mm, to_mm, theta=np.pi, K=4):
    """把 from_mm 网格上的**周期系数** (1,3*K*2,dc,hc,wc) 重建为 to_mm 网格上的
    位移场（to_mm 体素单位）。

    步骤：① 在 θ 处合成时间基（得到 from_mm 体素单位的位移）
          ② ×(from_mm/to_mm) 换算到 to_mm 体素单位
          ③ 三线性上采样到目标网格
    """
    c = torch.from_numpy(coef_np).float()
    d = disp_coarse(c, theta, K)                    # (1,3,dc,hc,wc)，from_mm 体素单位
    d = d * (from_mm / to_mm)                       # -> to_mm 体素单位
    d = F.interpolate(d, size=target_shape, mode='trilinear', align_corners=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=str, default='1,2,3,4,5,8')
    ap.add_argument('--from-mm', type=int, default=4)
    ap.add_argument('--to-mm', type=int, default=2)
    args = ap.parse_args()

    print(f'{"case":>4} {"4mm评测":>9} {"2mm网格重评":>12} {"差":>8}   {"细尺度(2mm)":>11}  来源')
    rows = []
    for cn in [int(x) for x in args.cases.split(',') if x.strip()]:
        cf = os.path.join(D, f'ms{args.from_mm}to{args.to_mm}_base_case{cn}_{args.from_mm}mm_coef.npy')
        jf = os.path.join(D, f'ms{args.from_mm}to{args.to_mm}_base_coarse_case{cn}_{args.from_mm}mm.json')
        ff = os.path.join(D, f'ms{args.from_mm}to{args.to_mm}_base_case{cn}_{args.to_mm}mm.json')
        if not os.path.exists(cf):
            print(f'{cn:>4}  [缺 {os.path.basename(cf)}]')
            continue
        coef = np.load(cf)
        # 🔴 目标必须是 **to_mm 下的体积尺寸**（不是系数网格！系数网格 = 体积/8）
        im = sitk.ReadImage(os.path.join(
            HERE, '..', 'reference_4', 'data', 'DIRLAB', 'mha', f'case{cn}',
            f'case{cn}_T00_R.mha'))
        sz = im.GetSize()                                    # (x,y,z)
        vol2 = (max(1, sz[2] // args.to_mm), max(1, sz[1] // args.to_mm),
                max(1, sz[0] // args.to_mm))                 # (D,H,W)
        c2 = rebuild(coef, vol2, args.from_mm, args.to_mm)

        lm0, lm5 = load_lm(cn)
        sp = np.array([float(args.to_mm)] * 3)
        d_mm = (c2[0].numpy()) * args.to_mm            # 体素 -> mm
        tre2 = float(np.linalg.norm(lm0 + sample_trilinear(d_mm, lm0, sp) - lm5, axis=1).mean())

        j4 = json.load(open(jf, encoding='utf-8')) if os.path.exists(jf) else {}
        tre4 = j4.get('tre_total_STANDARD', float('nan'))
        fine = json.load(open(ff, encoding='utf-8'))['tre_total_STANDARD'] if os.path.exists(ff) else float('nan')
        rows.append((cn, tre4, tre2, fine))
        print(f'{cn:>4} {tre4:>9.3f} {tre2:>12.3f} {tre2-tre4:>+8.3f}   {fine:>11.3f}  '
              f'coef{tuple(coef.shape[2:])} -> vol{vol2}')

    if rows:
        a = np.array([[r[1], r[2], r[3]] for r in rows])
        print(f'\n{"均值":>4} {np.nanmean(a[:,0]):>9.3f} {np.nanmean(a[:,1]):>12.3f} '
              f'{np.nanmean(a[:,1])-np.nanmean(a[:,0]):>+8.3f}   {np.nanmean(a[:,2]):>11.3f}')
        print('\n判读：')
        print('  · 若"2mm网格重评"与"4mm评测"接近 -> 4mm 的优势是真实的（不是平滑假象）')
        print('  · 若明显上升 -> 是评估假象，最终表必须用 2mm 评估')
        print('  · 与"细尺度(2mm)"比较：若粗尺度（重评后）仍更好 -> 细尺度阶段在破坏解')


if __name__ == '__main__':
    main()
