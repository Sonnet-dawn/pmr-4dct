"""
verify_resample_geometry.py —— 回归测试：降采样必须保留 origin/direction
================================================================================
**这个测试是为了锁死一个真实发生过的静默 bug。**

事故经过
--------
`load_imgs_v2` 用 `sitk.ResampleImageFilter` 做整数倍降采样时只设了 `SetSize` 与
`SetOutputSpacing`。SimpleITK **不会继承输入 origin**，输出 origin 回落到 (0,0,0)。

  * DIRLAB 的 `_R.mha` origin = (0,0,0) ⇒ 此 bug **隐形**，所有已发表的 DIRLAB 结果
    **完全不受影响**（这一点已逐体素核实）。
  * CREATIS 的 `_R.mha` origin = (-250,-250,-164.5) ⇒ 采样区域整体平移 250 mm，
    越界部分被默认 `paddingValue=0.0` 填充。**HU 0 恰好是软组织**，
    于是填充值伪装成真实组织：
      - 掩膜内平均 HU 从 −742 被污染成 −72；
      - 与正确抽样相比平均绝对误差 **695 HU**；
      - 训练 loss 只有 −0.18（DIRLAB 是 −0.89）且 800 次迭代完全不下降；
      - **全程不报错、不越界、图像肉眼看着"正常"。**

测试内容
--------
对每个数据集各取一例，验证：
  1. `resample_down(img, 2)` 与「原生按索引 `[::2,::2,::2]` 抽样」逐体素一致；
  2. 输出 origin/direction 与输入相同；
  3. 掩膜降采样后的体积占比与原生一致（掩膜与图像**体素对齐**）；
  4. 掩膜内平均 HU 落在肺的生理范围内。

用法：python verify_resample_geometry.py
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
import sys

import numpy as np
import SimpleITK as sitk

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pmr_v2 import resample_down, load_imgs_v2, build_mask          # noqa: E402
from lung_mask import lung_mask_native, mask_at_down, hu_offset, raw_case  # noqa: E402

FAIL = []


def check(cond, msg):
    print(f'  [{"PASS" if cond else "FAIL"}] {msg}')
    if not cond:
        FAIL.append(msg)


def test_dataset(dataset, cn):
    print(f'\n=== {dataset} case{cn} ===')
    img = raw_case(cn, 0, dataset)
    og, dr = img.GetOrigin(), img.GetDirection()
    nat = sitk.GetArrayFromImage(img).astype(np.float32)

    out = resample_down(img, 2)
    a = sitk.GetArrayFromImage(out).astype(np.float32)

    check(np.allclose(out.GetOrigin(), og),
          f'origin 保留: {tuple(round(float(x),3) for x in out.GetOrigin())} == '
          f'{tuple(round(float(x),3) for x in og)}')
    check(np.allclose(out.GetDirection(), dr), 'direction 保留')

    ref = nat[::2, ::2, ::2]
    sh = tuple(min(x, y) for x, y in zip(a.shape, ref.shape))
    diff = np.abs(a[:sh[0], :sh[1], :sh[2]] - ref[:sh[0], :sh[1], :sh[2]]).mean()
    check(diff < 1e-3, f'与索引抽样一致: 平均绝对差 {diff:.6f} HU (< 1e-3)')

    # 掩膜与图像对齐
    lm = lung_mask_native(cn, 0, dataset=dataset)
    md = mask_at_down(cn, 2, dataset=dataset) > 0.5
    check(abs(md.mean() - lm.mean()) < 0.01,
          f'掩膜降采样体积占比一致: {md.mean()*100:.2f}% vs 原生 {lm.mean()*100:.2f}%')

    # 掩膜内平均 HU 必须落在肺的生理范围
    hu = nat - hu_offset(dataset)
    md_full = lm[::2, ::2, ::2][:sh[0], :sh[1], :sh[2]]
    m_hu = hu[::2, ::2, ::2][:sh[0], :sh[1], :sh[2]][md_full]
    check(m_hu.mean() < -300,
          f'掩膜内平均 HU = {m_hu.mean():.1f}（肺应 < −300；污染时会变成 ≈ −72）')

    # 训练真正用的路径：load_imgs_v2 + build_mask，掩膜内归一化强度应偏低（肺是暗的）
    imgs, _ = load_imgs_v2(cn, 2, 'robust', dataset)
    a0 = imgs[0][0, 0].numpy()
    m2 = build_mask(cn, 2, (1, 1) + a0.shape, 'union', 'cpu', dataset)[0, 0].numpy() > 0.5
    check(a0[m2].mean() < 0.45,
          f'掩膜内归一化强度 mean = {a0[m2].mean():.3f}（肺应 < 0.45；污染时 ≈ 0.79）')


def test_synthetic():
    """**不依赖任何数据集**的最小复现（2026-09-25 新增）。

    起因：对抗式审稿把仓库 clone 下来跑 `run_verification_suite.py --quick`，
    发现本项需要 DIR-Lab / CREATIS 体数据 ⇒ 在新 clone 里**必然失败**，
    而 `--quick` 与 CI 都声称"纯代码、不需要数据"。

    而这个缺陷类其实**不需要真实数据**就能测：关键是 **origin 非零**。
    DIRLAB 的 origin 是 (0,0,0)，所以错误在那里隐形；合成一个 origin=(−250,−250,−164.5)
    的图像（CREATIS 的值）就能在纯内存里复现同一族错误。
    """
    print('\n=== synthetic（无数据集依赖：origin 非零，这是缺陷显形的条件） ===')
    arr = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 2.0, 2.5))
    img.SetOrigin((-250.0, -250.0, -164.5))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    out = resample_down(img, 2)
    check(np.allclose(out.GetOrigin(), img.GetOrigin()),
          f'origin 保留: {tuple(round(float(x), 3) for x in out.GetOrigin())} == '
          f'{tuple(round(float(x), 3) for x in img.GetOrigin())}')
    check(np.allclose(out.GetDirection(), img.GetDirection()), 'direction 保留')
    check(np.allclose(out.GetSpacing(), np.array(img.GetSpacing()) * 2),
          f'spacing ×2: {tuple(round(float(x), 3) for x in out.GetSpacing())}')

    # 最关键的一条：**输出体素 (0,0,0) 的物理坐标必须等于输入体素 (0,0,0) 的物理坐标**。
    # origin 回落成 (0,0,0) 时，这一条会偏 250 mm —— 也就是当年那个静默 bug 的直接签名。
    p_in = np.array(img.TransformIndexToPhysicalPoint((0, 0, 0)))
    p_out = np.array(out.TransformIndexToPhysicalPoint((0, 0, 0)))
    check(np.allclose(p_out, p_in, atol=1e-4),
          f'体素 (0,0,0) 物理坐标不变: {np.round(p_out, 3).tolist()} == '
          f'{np.round(p_in, 3).tolist()}（origin 丢失时会偏 250 mm）')

    a = sitk.GetArrayFromImage(out).astype(np.float32)
    ref = arr[::2, ::2, ::2]
    sh = tuple(min(x, y) for x, y in zip(a.shape, ref.shape))
    diff = np.abs(a[:sh[0], :sh[1], :sh[2]] - ref[:sh[0], :sh[1], :sh[2]]).mean()
    check(diff < 1e-3, f'与索引抽样一致: 平均绝对差 {diff:.6f} (< 1e-3)')


if __name__ == '__main__':
    # 先跑合成用例：它**总能跑**，且覆盖的正是这个缺陷类的机制。
    test_synthetic()
    # 真实数据部分：没有数据时**显式跳过**，不静默通过也不报失败。
    _DATA = os.path.join(os.path.dirname(HERE), 'reference_4', 'data')
    if os.path.isdir(os.path.join(_DATA, 'DIRLAB', 'mha')):
        test_dataset('dirlab', 1)
    else:
        print('\nSUITE-PARTIAL-SKIP: 找不到 DIR-Lab 体数据（仓库发行版不附带），'
              '真实数据部分未执行；上面 synthetic 一项**已执行**。')
    if os.path.isdir(os.path.join(_DATA, 'CREATIS', 'mha')):
        test_dataset('creatis', 0)
    else:
        print('SUITE-PARTIAL-SKIP: 找不到 CREATIS 体数据，真实数据部分未执行。')
    print('\n' + '=' * 60)
    if FAIL:
        print(f'❌ {len(FAIL)} 项未通过：')
        for m in FAIL:
            print(f'   - {m}')
        sys.exit(1)
    print('✅ 通过：降采样保留 origin/direction，且与索引抽样逐体素一致（合成用例）；'
          '真实数据部分见上方说明')
