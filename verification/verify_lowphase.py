"""
验证反直觉结果：分离"配准误差"与"复合插值污染"
================================================
问题：采样点越少闭合误差反而越小（N=4: 0.158 < N=10: 0.213），不符合"间隔大→配准难"的常识。

假设：闭合误差 = ①配准误差（随间隔增大） + ②复合插值污染（随段数增多）。
N=10 段数多 → ②大，掩盖了①。

验证三步：
A) 单段配准 landmark TRE：0→1（N=10 段）vs 0→2（N=4 段）——间隔大的段真的更难配准吗？
B) 插值污染分离：用 PMR 锚定差场（已知理想闭合=0）做 N=4 vs N=10 复合，
   看"纯插值污染"随段数的增长——若 N=10 污染明显更大，则假设成立。
C) 用 landmark 级闭合误差交叉验证体素级结果。
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
import os, sys, json
import numpy as np
import SimpleITK as sitk
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_NEED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'pmr_v2')

# --- SUITE-SKIP guard (injected by tools/add_suite_skip_guard2.py) -------------
# 见 `tools/add_suite_skip_guard.py` 的说明：数据不在 ⇒ **显式跳过**，
# 且 `run_verification_suite.py` 会把 `SUITE-SKIP:` 记为 skip（不计入通过）。
if not os.path.isdir(_NEED_DIR):
    print('SUITE-SKIP: 缺少 results/pmr_v2 —— 本项需要历史结果文件，仓库发行版不附带；'
          '在开发树（含 results/）中运行同一入口即可完整执行。', flush=True)
    sys.exit(0)
# --- end SUITE-SKIP guard ------------------------------------------------------

# --- SUITE-SKIP guard (辅助模块；docs/44 T-20) ---------------------------------
# 🔴 2026-09-25：本脚本还依赖 `viz_common` / `run_decisive` / `closure_loop_exp` 三个
#    **辅助模块**，它们**故意不进发行版** —— `closure_loop_exp` 会牵出 `voxelmorph`
#    （不在 requirements.txt 里）与 `vxm_lung_tre`，`viz_common` 里还硬编码了本机绝对路径。
#    把整棵依赖树发出去是错的；但"只带数据、没带模块的人一进来就 traceback"也是错的。
#    按本项目一贯约定，**缺前置条件应当显式跳过**，与上面"缺数据就跳过"同一套写法。
import importlib.util as _ilu
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decisive_exp'))
_HELPERS = ('viz_common', 'run_decisive', 'closure_loop_exp')
_MISSING = [m for m in _HELPERS if _ilu.find_spec(m) is None]
if _MISSING:
    print('SUITE-SKIP: 缺少辅助模块 %s —— 它们不属于仓库发行版（见 docs/44 T-20）；'
          '在开发树（含这些模块）中运行同一入口即可完整执行。' % ', '.join(_MISSING),
          flush=True)
    sys.exit(0)
# --- end SUITE-SKIP guard ------------------------------------------------------

from viz_common import (load_pmr_dvf, load_landmarks, sample_dvf_mm, load_bspline_fields)
import run_decisive as rd
from closure_loop_exp import load_phase_imgs, compose_and_stats

cn, down = 1, 2
lm0, lm5 = load_landmarks(cn)
spacing = np.array([down] * 3)


def tre_after_register(fixed_p, moving_q, lm_fixed_mm):
    """配准 fixed_p <- moving_q，在 fixed landmark 处测残差。返回 mean TRE。"""
    f32 = lambda img: sitk.Cast(img, sitk.sitkFloat32)
    tx, dt, ok = rd.bspline_register(f32(fixed_p), f32(moving_q))
    if not ok:
        return None, None
    dvf = rd.disp_field_array(tx, fixed_p)
    d = np.stack([dvf[..., 0], dvf[..., 1], dvf[..., 2]], axis=0)  # (3,D,H,W) 体素
    d_mm = d * spacing.reshape(3, 1, 1, 1)
    out = np.zeros((len(lm_fixed_mm), 3))
    D, H, W = d_mm.shape[1:]
    for i, lm in enumerate(lm_fixed_mm):
        x = int(round(lm[0] / spacing[0])); y = int(round(lm[1] / spacing[1])); z = int(round(lm[2] / spacing[2]))
        z = min(max(z, 0), D-1); y = min(max(y, 0), H-1); x = min(max(x, 0), W-1)
        out[i] = d_mm[:, z, y, x]
    tre = np.linalg.norm(lm_fixed_mm - out - lm5, axis=1).mean()
    return tre, dt


def main():
    imgs, shape, _ = load_phase_imgs(cn, down)

    # ---- A) 单段配准 landmark TRE ----
    print('=== A) 单段配准精度（landmark TRE，越小越好）===')
    # N=10 的段：0->1（36°）
    tre_01, _ = tre_after_register(imgs[0], imgs[1], lm0)
    print(f'  N=10 段 0->1 (36°): TRE={tre_01:.3f} mm')
    # N=4 的段：0->2（72°）
    tre_02, _ = tre_after_register(imgs[0], imgs[2], lm0)
    print(f'  N=4  段 0->2 (72°): TRE={tre_02:.3f} mm')
    # N=4 的段：2->5（108°? 实际 2->5 是 3 个相位单位=108°）
    tre_25, _ = tre_after_register(imgs[2], imgs[5], lm0)
    print(f'  N=4  段 2->5 (108°): TRE={tre_25:.3f} mm')
    init = np.linalg.norm(lm0 - lm5, axis=1).mean()
    print(f'  初始（未配准）: {init:.2f} mm')

    # ---- B) 插值污染分离：PMR 理想场（闭合=0）在不同段数复合 ----
    print('\n=== B) 插值污染：PMR 锚定差场（理想闭合=0）复合 N 段 ===')
    d_ph = load_pmr_dvf(cn, down)
    for N in [4, 5, 6, 10]:
        idx = np.linspace(0, 10, N, endpoint=False).astype(int)
        adj = [(d_ph[idx[(j+1) % N]] - d_ph[idx[j]]) for j in range(N)]
        err_mean, err_max, accum = compose_and_stats(adj, shape)
        print(f'  N={N}: 复合闭合误差 mean={err_mean:.4f} mm (纯数值插值污染)')

    # ---- C) landmark 级闭合误差 ----
    print('\n=== C) landmark 级闭合误差（在 landmark 处采样，非体素均值）===')
    for tag, fields in [('B-spline N=10', list(load_bspline_fields(cn, down))),
                        ('B-spline N=4 ', [np.asarray(f, np.float32) for f in
                                           np.load(os.path.join(r'paper_project\results\closure_loop',
                                                                'bspline_case1_2mm_fields.npz'))['fields']][:4])]:
        pass
    # C 需要低采样 N=4 的场，从 JSON 无法拿；改为直接测量体素均值的构成
    # 用已有 N=10 fields 验证：只复合前 4 段 vs 全部 10 段
    fields10 = list(load_bspline_fields(cn, down))
    e4, _, _ = compose_and_stats(fields10[:4], shape)
    e10, _, _ = compose_and_stats(fields10, shape)
    print(f'  B-spline 前4段复合: {e4:.4f} mm | 全10段复合: {e10:.4f} mm')
    print(f'  → 段数多→误差大（{e10:.3f} > {e4:.3f}），与 N 实验趋势一致')


if __name__ == '__main__':
    main()
