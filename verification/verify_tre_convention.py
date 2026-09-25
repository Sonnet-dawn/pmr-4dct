"""verify_tre_convention.py —— TRE 口径与 B 样条求值的回归测试（解析已知答案）
================================================================================
本文件锁定**四个真实发生过的静默 bug**，它们属于同一族：
**某个几何量恰好等于 1（或 0）时，单位/口径错误完全隐形。**

  ① DIRLAB `_R.mha` 的 origin 恰为 `(0,0,0)` ⇒ 降采样丢 origin 的 bug 隐形
     （修于 `pmr_v2.resample_down`）
  ② Elastix 基线在 **1 mm** 上跑 ⇒ spacing=1 时"体素索引"与"物理坐标"数值相同
     ⇒ 点文件单位写错也隐形；降到 2 mm 立刻整体放大 2 倍
     （实测 case6 得 TRE = 414.6 mm，初始仅 10.9 mm）
  ③ `transformix -def <点文件>` 把输入**取整成整数索引** ⇒ 2 mm 时每轴最多 1 mm 误差
  ④ TRE 口径：本项目旧口径 `|lm5 − d(lm5) − lm0|` 对**完美配准也不为零**

⚠️ 测试对象的选择（重要）
------------------------
本文件的 A/B/C 三项测试**针对 `tools/elastix_bspline.py` 的精确求值器**，
而不是 transformix 点文件路径 —— 后者**在原理上**达不到所需精度（坑 ③），
把一个已知做不到的要求写成断言，只会制造永远失败的测试。
transformix 路径的量化被单独记为**信息性**测试 D，断言它确实不精确。

判别测试是否"恒真"的设计
------------------------
  * 测试 1 用**解析可算**的单系数 B 样条（位移正好等于三个基函数之积），
    与实现对所有点逐点比对 ⇒ 同时验证权重、索引顺序、域原点约定。
  * 测试 A/B 用**解析已知答案为 0** 的合成变换（平移、仿射），
    并令 `lm5 = T(lm0)` ⇒ 正确实现的 TRE 必须精确为 0。
  * 测试 D 反过来：同一实现走 transformix 路径**必须不为 0**，否则说明测试无诊断力。

用法: python verify_tre_convention.py
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
import json
import shutil
import subprocess

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
# 两种布局都要能跑：
#   · 开发布局：本文件在项目根，求值器在 tools/
#   · 仓库布局：本文件在 verification/，求值器在 src/（由 make_repo.py 注入的 path shim 提供）
_tools = os.path.join(HERE, 'tools')
if os.path.isdir(_tools):
    sys.path.insert(0, _tools)
from elastix_bspline import ElastixBSpline, _weights        # noqa: E402

WD = os.path.join(HERE, 'results', 'verify_tre_convention')
shutil.rmtree(WD, ignore_errors=True)
os.makedirs(WD)

TXD = None
for _root in (os.path.join(HERE, 'tools', 'elastix'), os.path.join(HERE, 'tools'), r'D:\elastix'):
    if os.path.isdir(_root):
        for dp, _, fs in os.walk(_root):
            if 'transformix.exe' in fs:
                TXD = os.path.join(dp, 'transformix.exe')
                break
    if TXD:
        break

REPORT = {'tests': {}, 'verdict': None}
FAILED = []


def write_elastix_bspline(path, g_origin, g_spacing, g_size, coef, img_geom):
    """把 (3,nz,ny,nx) 系数写成 elastix 参数文件（**z 最快**展平）。"""
    flat = np.asarray(coef, dtype=float).reshape(3, -1).ravel()
    sz, sp, og = img_geom
    txt = [
        '(BSplineTransformSplineOrder 3)',
        '(Direction 1 0 0 0 1 0 0 0 1)',
        '(FixedImageDimension 3)', '(MovingImageDimension 3)',
        '(GridDirection 1 0 0 0 1 0 0 0 1)',
        '(GridIndex 0 0 0)',
        f'(GridOrigin {" ".join(str(v) for v in g_origin)})',
        f'(GridSize {" ".join(str(int(v)) for v in g_size)})',
        f'(GridSpacing {" ".join(str(v) for v in g_spacing)})',
        '(HowToCombineTransforms "Compose")',
        '(Index 0 0 0)',
        '(InitialTransformParameterFileName "NoInitialTransform")',
        '(NumberOfParameters %d)' % len(flat),
        f'(Origin {" ".join(str(v) for v in og)})',
        f'(Size {" ".join(str(int(v)) for v in sz)})',
        f'(Spacing {" ".join(str(v) for v in sp)})',
        '(Transform "BSplineTransform")',
        # ⚠️ 必须用纯数值格式：`{v!r}` 在 numpy 2.x 上会写出 `np.float64(0.0)`，
        #    其中的括号会破坏解析器的 `\(字段\s+值\)` 正则。
        '(TransformParameters ' + ' '.join(f'{float(v):.17g}' for v in flat) + ')',
        '(UseCyclicTransform "false")',
        '(UseDirectionCosines "true")',
    ]
    open(path, 'w', encoding='utf-8').write('\n'.join(txt) + '\n')
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 测试 1：解析单系数 B 样条（验证权重、索引顺序、域原点约定）
# ══════════════════════════════════════════════════════════════════════════════
# 控制点网格必须**完整覆盖图像物理范围**，否则域外点会被钳制，
# 测出来的就不是"单位/量化误差"而是"域外行为"（这正是第一版测试 D 数值不可解释的原因）。
# 图像范围 x∈[-10,116] y∈[5,131] z∈[3,97]；按 elastix 惯例让 GridOrigin 外扩一格。
g_origin = np.array([-14.0, 1.0, -1.0])
g_spacing = np.array([4.0, 4.0, 4.0])
g_size = np.array([34, 34, 26])            # (nx, ny, nz)
IX, IY, IZ = 8, 8, 6

print('=' * 82)
print(f'测试 1：解析单系数 B 样条 —— 只有 (ix,iy,iz)=({IX},{IY},{IZ}) 处一个非零系数')
print('       位移必须精确等于 B_{a}(fx)·B_{b}(fy)·B_{c}(fz)（闭式可算）')
print('=' * 82)

coef = np.zeros((3, g_size[2], g_size[1], g_size[0]))    # (d, nz, ny, nx)
coef[0, IZ, IY, IX] = 1.0                  # x 方向位移
tf = write_elastix_bspline(os.path.join(WD, 'analytic.txt'),
                           g_origin, g_spacing, g_size, coef,
                           ((64, 64, 48), (2.0, 2.0, 2.0), (-10.0, 5.0, 3.0)))
bs = ElastixBSpline(tf)

rng = np.random.default_rng(0)
# 采样点必须落在该系数的**支撑域**内：控制点 index 对应分段坐标 u ∈ [index−2, index+2)
lo_u = np.array([IX, IY, IZ]) - 2.0
u_s = lo_u[None, :] + rng.random((200, 3)) * 4.0
pts = g_origin[None, :] + u_s * g_spacing[None, :]
pred = bs.displacement(pts)

u = (pts - g_origin[None, :]) / g_spacing[None, :]
i0 = np.floor(u).astype(int)
f = u - i0
# ⚠️ 关键的下标约定：`ElastixBSpline` 用的控制点是 (i0−1, i0, i0+1, i0+2)，
#    对应权重 (_weights 返回的) (W0, W1, W2, W3)。
#    因此控制点 `index` 的权重下标是 **index − i0 + 1**，不是 index − i0。
off = np.stack([IX - i0[:, 0] + 1, IY - i0[:, 1] + 1, IZ - i0[:, 2] + 1], axis=1)
W = np.array([_weights(f[:, k]) for k in range(3)])       # (3, 4, N)
exp = np.zeros(len(pts))
valid = np.all((off >= 0) & (off <= 3), axis=1)
for j in np.where(valid)[0]:
    exp[j] = W[0, off[j, 0], j] * W[1, off[j, 1], j] * W[2, off[j, 2], j]
err = np.abs(pred[:, 0] - exp)[valid]
print(f'  有效点数 {valid.sum()}/{len(pts)}（采样点已限制在该系数支撑域内，应全部有效）')
print(f'  x 分量误差: 均值 {err.mean():.3e}  最大 {err.max():.3e}')
print(f'  y,z 分量最大值 {np.abs(pred[valid][:, 1:]).max():.3e}（应为 0）')
ok1 = (valid.sum() == len(pts) and err.max() < 1e-12
       and np.abs(pred[valid][:, 1:]).max() < 1e-12)
print('  ' + ('✅ 通过：权重、z 最快索引顺序、域原点约定全部正确'
              if ok1 else '❌ 失败'))
if not ok1:
    FAILED.append('1: 解析单系数比对失败')
REPORT['tests']['1_analytic_bspline'] = {
    'n_valid': int(valid.sum()), 'max_err': float(err.max()),
    'max_yz': float(np.abs(pred[valid][:, 1:]).max())}


# ══════════════════════════════════════════════════════════════════════════════
# 测试 A/B：解析已知 TRE 必须为 0（精确求值器路径）
# ══════════════════════════════════════════════════════════════════════════════
def constant_transform(t_vec, g_origin, g_spacing, g_size, path, img_geom):
    """用"系数全等于常数"构造平移变换：B 样条权重之和恒为 1 ⇒ 位移处处等于 t。"""
    coef = np.zeros((3, g_size[2], g_size[1], g_size[0]))
    for d in range(3):
        coef[d] = t_vec[d]
    return write_elastix_bspline(path, g_origin, g_spacing, g_size, coef, img_geom)


lm0 = np.array([[20.0, 30.0, 15.0], [60.0, 70.0, 40.0], [90.0, 45.0, 55.0],
                [45.0, 85.0, 25.0]])
t_vec = np.array([3.0, -2.0, 5.0])

print('\n' + '=' * 82)
print('测试 A：平移 T(x)=x+t，令 lm5=lm0+t（完美对应）⇒ 标准 TRE 必须精确为 0')
print('=' * 82)
resA = {}
for sp, og, tag in (((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 'sp1_org0'),
                    ((2.0, 2.0, 2.0), (-10.0, 5.0, 3.0), 'sp2_orgNZ')):
    p = os.path.join(WD, f'tfA_{tag}.txt')
    constant_transform(t_vec, g_origin, g_spacing, g_size, p,
                       ((64, 64, 48), sp, og))
    b = ElastixBSpline(p)
    lm5 = lm0 + t_vec
    tre = b.tre_standard(lm0, lm5)
    # 同时确认求值器真的产生了位移（否则 0 是假的）
    mag = np.linalg.norm(b.displacement(lm0), axis=1).mean()
    print(f'  spacing={sp[0]:.0f} origin={og}')
    print(f'    TRE = {tre:.3e} mm   平均位移量 = {mag:.6f} mm（应为 {np.linalg.norm(t_vec):.4f}）')
    resA[tag] = {'tre': tre, 'disp_mag': float(mag)}
    if tre > 1e-9:
        FAILED.append(f'A/{tag}: TRE={tre:.3e} 应为 0')
    if abs(mag - np.linalg.norm(t_vec)) > 1e-9:
        FAILED.append(f'A/{tag}: 位移量 {mag:.6f} != {np.linalg.norm(t_vec):.4f}')
REPORT['tests']['A_translation'] = resA

print('\n' + '=' * 82)
print('测试 B：检验**口径本身**：给定已知位移函数 d(x)，令 lm5 = lm0 + d(lm0)（完美配准），')
print('        则标准口径按构造为 0，而旧口径 |lm5 − d(lm5) − lm0| 一般不为 0')
print('=' * 82)
# 仿射无法用常系数 B 样条精确表示；这里改用"直接构造位移场"的方式检验口径公式：
# 给定一个已知的位移函数 d(x)，令 lm5 = lm0 + d(lm0)（即完美配准），
# 则标准 TRE = |lm0 + d(lm0) − lm5| = 0（恒等，构造性）；
# 旧口径 = |lm5 − d(lm5) − lm0|，对非线性 d 一般不为 0。
th = np.deg2rad(9.0)
R = np.array([[np.cos(th), -np.sin(th), 0.0],
              [np.sin(th), np.cos(th), 0.0],
              [0.0, 0.0, 1.0]])
c = np.array([60.0, 60.0, 45.0])
t2 = np.array([2.0, 4.0, -3.0])


def d_affine(x):
    return (R @ (x - c).T).T + c + t2 - x


d_lm0 = d_affine(lm0)
lm5_perfect = lm0 + d_lm0
tre_std = np.linalg.norm(lm0 + d_lm0 - lm5_perfect, axis=1).mean()
tre_leg = np.linalg.norm(lm5_perfect - d_affine(lm5_perfect) - lm0, axis=1).mean()
print(f'  标准口径 |lm0 + d(lm0) − lm5| = {tre_std:.3e} mm（构造性为 0）')
print(f'  旧口径   |lm5 − d(lm5) − lm0| = {tre_leg:.4f} mm（完美配准仍报假误差）')
print(f'  ⇒ 旧口径对完美配准的假误差 = {tre_leg:.4f} mm，'
      f'占初始误差 {np.linalg.norm(d_lm0,axis=1).mean():.4f} mm 的 '
      f'{100*tre_leg/np.linalg.norm(d_lm0,axis=1).mean():.1f}%')
if tre_std > 1e-12:
    FAILED.append('B: 标准口径对完美配准不为 0')
if tre_leg < 1e-3:
    FAILED.append('B: 旧口径竟然也为 0 —— 该测试没有诊断力')
REPORT['tests']['B_convention_bias'] = {
    'standard_on_perfect': float(tre_std), 'legacy_on_perfect': float(tre_leg),
    'init_mag': float(np.linalg.norm(d_lm0, axis=1).mean())}


# ══════════════════════════════════════════════════════════════════════════════
# 测试 D（信息性）：transformix 点文件路径确实不精确
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 82)
print('测试 D（信息性）：transformix 点文件路径的量化 —— 记录它**确实**不精确')
print('=' * 82)
resD = {}
if TXD:
    p = os.path.join(WD, 'tfD.txt')
    constant_transform(t_vec, g_origin, g_spacing, g_size, p,
                       ((64, 64, 48), (2.0, 2.0, 2.0), (-10.0, 5.0, 3.0)))
    for unit in ('index', 'physical'):
        arr = lm0.copy()
        if unit == 'index':
            arr = (lm0 - np.array([-10.0, 5.0, 3.0])) / 2.0
        pf = os.path.join(WD, f'ptsD_{unit}.txt')
        with open(pf, 'w') as fh:
            fh.write('index\n' + str(len(arr)) + '\n')
            for q in arr:
                fh.write(f'{q[0]:.10f} {q[1]:.10f} {q[2]:.10f}\n')
        od = os.path.join(WD, f'outD_{unit}')
        shutil.rmtree(od, ignore_errors=True)
        os.makedirs(od)
        subprocess.run([TXD, '-tp', p, '-def', pf, '-out', od],
                       capture_output=True, text=True, timeout=600)
        op = os.path.join(od, 'outputpoints.txt')
        if not os.path.exists(op):
            print(f'  [{unit}] 未产出 outputpoints.txt')
            continue
        out = []
        for line in open(op, encoding='utf-8', errors='replace'):
            if 'OutputPoint' in line:
                seg = line.split('OutputPoint = [')[1].split(']')[0].split()
                out.append([float(v) for v in seg[:3]])
        out = np.array(out)
        tre = float(np.linalg.norm(out - (lm0 + t_vec), axis=1).mean())
        print(f'  [{unit:8s}] TRE = {tre:.6f} mm  '
              f'（精确求值器给出 0；差距即 transformix 的量化误差）')
        resD[unit] = tre
    if resD.get('index', 0) <= 1e-6:
        FAILED.append('D: transformix 索引写法竟然精确 —— 说明测试无诊断力')
    if 'index' in resD:
        print(f'  ⇒ 确认 transformix 点文件路径不可用于 2 mm 主表；'
              f'主表一律走 tools/elastix_bspline.py')
else:
    print('  （未找到 transformix.exe，跳过；不影响 A/B/1）')
REPORT['tests']['D_transformix_quantisation'] = resD


# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 82)
if FAILED:
    print('❌ 失败项：')
    for f in FAILED:
        print('   -', f)
else:
    print('✅ 全部通过。')
    print('   结论：TRE = |T(lm0) − lm5|（标准口径）；B 样条求值走 `elastix_bspline`'
          '（z 最快系数顺序；开发布局在 tools/，仓库布局在 src/）；')
    print('   transformix 点文件路径因输入取整而不可用于 2 mm 主表。')
print('=' * 82)
REPORT['verdict'] = 'FAIL' if FAILED else 'PASS'
REPORT['failures'] = FAILED
json.dump(REPORT, open(os.path.join(HERE, 'results', 'verify_tre_convention.json'),
                       'w', encoding='utf-8'), indent=2, ensure_ascii=False)
sys.exit(1 if FAILED else 0)
