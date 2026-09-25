"""verify_lapirn_shapes.py —— 两个 DL 基线模型的**纯代码**形状/前向验证
================================================================================
**为什么要有这一项**：`LapIRN.forward` 曾有一个**只在多级金字塔下才触发**的尺寸错配 ——
第 `lv` 级的 `cur`（来自更粗一级）被直接拿去做 `_warp(m, cur)`，而 `_warp` 的基准网格
按**当前更细**的 `m` 建 ⇒

    RuntimeError: The size of tensor a (48) must match the size of tensor b (24)

现场后果：**20 折 lapirn 全部立刻失败**（每折约 120 s、退出码 1），而驱动的
`if os.path.exists(jf)` 只把它记成一行 `!! 失败`，于是**整批实验看起来"跑完了"**
（`run_dl_baselines.py` 退出码 0）。这是本项目"静默失败"家族的又一例。

本脚本不需要数据、不需要 GPU、秒级完成，因此可以进 CI —— 在花掉 20 分钟 GPU 之前
就把这类错误拦住。

覆盖：
  1. `voxelmorph` 前向形状正确（输出 = 输入空间尺寸、3 通道）；
  2. `lapirn` 前向形状正确，且**多级之间不报尺寸错**；
  3. **不能被 2^levels 整除的尺寸**也要能跑（这是错配最容易暴露的地方）；
  4. `LapIRN(return_all=True)` 返回 levels 个张量，且**逐级空间尺寸递增**、
     最细一级与输入一致（说明"由粗到细"真的是这个方向）；
  5. 初始位移为 0（`out` 层零初始化）—— 若这条挂了，说明初始化被改坏，
     训练会从一个非零位移起步。

用法: python verify_lapirn_shapes.py
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

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

try:
    from dl_baseline import UNet3D, LapIRN, predict, _smooth   # noqa: E402
except Exception as e:                        # pragma: no cover
    print(f'❌ 无法 import dl_baseline：{type(e).__name__}: {e}')
    sys.exit(1)

n_pass = n_fail = 0


def check(name, cond, extra=''):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f'  ✅ {name}{("  " + extra) if extra else ""}')
    else:
        n_fail += 1
        print(f'  ❌ {name}{("  " + extra) if extra else ""}')


print('=' * 88)
print('1) voxelmorph（单尺度 U-Net）')
print('=' * 88)
torch.manual_seed(0)
vm = UNet3D(3, base=4, depth=3).eval()
for shp in [(1, 2, 32, 32, 32), (1, 2, 30, 34, 26)]:
    with torch.no_grad():
        y = vm(torch.randn(*shp))
    check(f'输入 {tuple(shp[2:])} → 输出 {tuple(y.shape[2:])}、通道 {y.shape[1]}',
          tuple(y.shape[2:]) == tuple(shp[2:]) and y.shape[1] == 3)

print('\n' + '=' * 88)
print('2) lapirn（3 级金字塔）—— 这条曾经直接抛尺寸错')
print('=' * 88)
torch.manual_seed(0)
lp = LapIRN(levels=3, base=4).eval()
ok_any = False
# 用**真实会遇到的**尺寸形状：DIRLAB 2 mm 图像的每一维都 ≥ 100，patch 是 96³。
# 最粗一级是 1/4 分辨率 ⇒ 最小需要 32；下面这些尺寸都满足。
for shp in [(1, 1, 32, 32, 32), (1, 1, 96, 96, 96), (1, 1, 40, 48, 44)]:
    try:
        with torch.no_grad():
            y = lp(torch.randn(*shp), torch.randn(*shp))
        good = tuple(y.shape[2:]) == tuple(shp[2:]) and y.shape[1] == 3
        ok_any |= good
        check(f'输入 {tuple(shp[2:])} → 输出 {tuple(y.shape[2:])}', good)
    except Exception as e:
        check(f'输入 {tuple(shp[2:])} 前向不抛错', False,
              f'→ {type(e).__name__}: {str(e)[:90]}')
check('至少一个尺寸能跑通（防空跑）', ok_any)

print('\n' + '=' * 88)
print('2b) 尺寸过小时必须给出**可读**的报错，而不是 avg_pool 的晦涩信息')
print('=' * 88)
try:
    with torch.no_grad():
        lp(torch.randn(1, 1, 20, 20, 20), torch.randn(1, 1, 20, 20, 20))
    check('过小尺寸应报错', False, '→ 竟然没报错')
except ValueError as e:
    msg = str(e)
    check('抛的是 ValueError 且说明"最小维 / 需要多大 / 怎么修"',
          '最小维' in msg and '要求每个空间维' in msg and 'patch' in msg,
          f'→ {msg[:100]}')
except Exception as e:
    check('抛的是 ValueError', False, f'→ {type(e).__name__}: {str(e)[:90]}')

print('\n' + '=' * 88)
print('3) lapirn return_all：逐级空间尺寸应**递增**，最细一级 = 输入')
print('=' * 88)
with torch.no_grad():
    outs = lp(torch.randn(1, 1, 32, 32, 32), torch.randn(1, 1, 32, 32, 32), return_all=True)
sizes = [tuple(o.shape[2:]) for o in outs]
check(f'返回 {len(outs)} 个张量（levels=3）', len(outs) == 3, f'→ {sizes}')
inc = all(sizes[i][0] <= sizes[i + 1][0] for i in range(len(sizes) - 1))
check('空间尺寸由粗到细递增', inc, f'→ {sizes}')
check('最细一级与输入一致', sizes[-1] == (32, 32, 32), f'→ {sizes[-1]}')

print('\n' + '=' * 88)
print('4) 初始化：输出层零初始化 ⇒ 初始位移应为 0')
print('=' * 88)
torch.manual_seed(1)
vm0 = UNet3D(3, base=4, depth=3).eval()
with torch.no_grad():
    y0 = vm0(torch.randn(1, 2, 16, 16, 16))
check('voxelmorph 初始位移 max|d| == 0', float(y0.abs().max()) == 0.0,
      f'→ {float(y0.abs().max()):.3e}')

print('\n' + '=' * 88)
print('5) 🔴 `predict()` 统一入口 —— 这条锁的是缺陷 C10')
print('=' * 88)
print('   背景：原代码两处写 `d = model(f, mv)[-1]`，意图是"取金字塔最后一级"，')
print('   但 `LapIRN.forward` 默认返回**张量**，于是 `[-1]` 变成了**最后一维的切片**，')
print('   位移从 (N,3,D,H,W) 静默变成 (N,3,D,H,W-1)，随后 `_smooth()` 抛 IndexError。')
torch.manual_seed(2)
vm_p = UNet3D(3, base=4, depth=3).eval()
lp_p = LapIRN(levels=3, base=4).eval()
for name, mdl in (('voxelmorph', vm_p), ('lapirn', lp_p)):
    a = torch.randn(1, 1, 32, 32, 32)
    b = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        d = predict(name, mdl, a, b)
    check(f'predict({name}) 返回 5 维 (N,3,D,H,W) 且与输入同尺寸',
          d.dim() == 5 and tuple(d.shape[2:]) == (32, 32, 32),
          f'→ {tuple(d.shape)}')
    # 这条是"陷阱本身"的回归：对 predict 的返回值再取 [-1] 必须**变得不像位移**
    with torch.no_grad():
        bad = d[-1]
    check(f'  ↳ 对张量取 [-1] 会退化成 4 维（这正是 C10 的形状，应被 assert 拦住）',
          bad.dim() == 4, f'→ {tuple(bad.shape)}')
    # `_smooth` 必须能吃 predict 的输出（C10 就是死在这一步）
    try:
        _smooth(d)
        check(f'  ↳ `_smooth(predict(...))` 不抛错', True)
    except Exception as e:
        check(f'  ↳ `_smooth(predict(...))` 不抛错', False,
              f'→ {type(e).__name__}: {str(e)[:70]}')

# 静态检查：源码里不允许再出现 `model(f, mv)[-1]` 式写法。
# ⚠️ 必须**先剥掉 docstring**：`predict()` 的文档里正是用这一行作为反例 ——
#    第一版没剥，于是把文档里的反例当成了真实代码（验证器自己错了，见 docs/44）。
# ⚠️ 路径必须取 **import 进来的模块路径**，不能写"同目录"：仓库布局是 src/ + verification/，
#    同目录下没有 dl_baseline.py（第一版这么写，CI 直接 FileNotFoundError）。
import dl_baseline as _dlb   # noqa: E402
src = open(_dlb.__file__, encoding='utf-8').read()
src_nc = re.sub(r'"""(?:.|\n)*?"""', '', src)      # 去掉三引号块（docstring）
src_nc = re.sub(r'#[^\n]*', '', src_nc)            # 去掉 # 注释
bad_pat = re.findall(r'model\(f,\s*mv\)\s*\[\s*-1\s*\]', src_nc)
check('dl_baseline.py 中不再出现 `model(f, mv)[-1]`（C10 的写法）',
      not bad_pat, f'→ 命中 {len(bad_pat)} 处')

print('\n' + '=' * 88)
print(f'结果：{n_pass} 通过 / {n_fail} 失败')
print('=' * 88)
if n_fail == 0:
    print('结论：两个模型的前向在多级、非整除尺寸下均成立，且**调用点的写法**也被钉死；')
    print('      "lapirn 一跑就崩"与"[-1] 静默切片"这两类错误都无法再静默通过。')
sys.exit(1 if n_fail else 0)
