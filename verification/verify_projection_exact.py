"""
verify_projection_exact.py —— 证明：**9 项周期基对锚定的 10 相数据是完备的，投影无损**。
================================================================================
待证命题
--------
设 N=10 个相位 θ_i = 2πi/N，锚定在 θ_0=0（即 d(x,0)=0）。则每个体素每个分量的
锚定场数据 D = (D_0,...,D_9) ∈ R^10 满足 D_0 = 0，其所在子空间维数为 N−1 = 9。

取周期基 B = { cos kθ − 1 }_{k=1..K} ∪ { sin kθ }_{k=1..K} ∪ { cos((K+1)θ) − 1 }，
在 N = 2(K+1)（此处 K=4, N=10）时共 **N−1 = 9** 个函数。

证明
----
1. 每个基函数在 θ=0 处取值 0（cos0−1=0, sin0=0）
   => 设计矩阵 M ∈ R^{N×(N−1)} 的**每一列**第 0 个元素都是 0
   => range(M) ⊆ { v ∈ R^N : v_0 = 0 }
2. dim{ v_0 = 0 } = N−1 = 9
3. 数值上 rank(M) = 9（见输出）
   => range(M) 是 9 维且含于 9 维空间 => **range(M) = { v_0 = 0 }**（精确相等）
4. 锚定数据 D 满足 D_0 = 0 => D ∈ { v_0=0 } = range(M)
   => 存在唯一系数 c 使 M c = D => **最小二乘投影残差恒为 0**

推论
----
* **精确闭合 + 连续相位 + 零表示损失 可以同时达到**，不存在取舍。
* 若只用 {cos kθ−1, sin kθ}_{k=1..4}（8 项），range 只有 8 维 ⊊ {v_0=0}，
  丢掉的是 **Nyquist 模态**（θ 交替分量）。这正是"8 vs 9"问题的确切含义。
* 第 9 项**必须**是 cos5θ−1；`sin5θ` 在 10 个采样点上恒为 0（秩仍为 8）。

用法： python verify_projection_exact.py
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import numpy as np

N = 10
TH = 2 * np.pi * np.arange(N) / N


def design(K=4, nyquist=True, thetas=TH):
    cols = []
    for k in range(1, K + 1):
        cols.append(np.cos(k * thetas) - 1.0)
        cols.append(np.sin(k * thetas))
    if nyquist:
        cols.append(np.cos((K + 1) * thetas) - 1.0)
    return np.stack(cols, axis=1)


def report(tag, M, D):
    """D: (N, M_pts) 锚定数据（D[0] 应为 0）。返回投影残差范数。

    🔴 返回的是 `max|D − M pinv(M) D|` —— **逐元素绝对残差在全部 (N × n_pts) 个条目上的最大值**，
    **不是**归一化残差、也不是范数。因此它**可以 > 1**（对标准正态的随机数据，
    20000 个条目的最大值约 1.3）。稿件中引用该数时必须写明这一定义，否则读者会
    误以为它是"投影算子对单位向量的残差"（那个上界才是 1）。
    """
    P = np.linalg.pinv(M)
    Drec = M @ (P @ D)
    r = np.abs(D - Drec)
    print(f'  {tag:<38} rank={np.linalg.matrix_rank(M)}  cond={np.linalg.cond(M):8.3f}  '
          f'max|逐元素残差|={r.max():.3e}  相对={r.max()/max(np.abs(D).max(),1e-30):.3e}')
    return r.max()


def main():
    print('=' * 90)
    print('一、设计矩阵的结构')
    print('=' * 90)
    for K, ny in ((4, False), (4, True), (5, False), (5, True), (6, False)):
        M = design(K, ny)
        print(f'  K={K} nyquist={str(ny):<5} 列数={M.shape[1]:>2}  '
              f'M[0,:] 最大绝对值={np.abs(M[0]).max():.3e}  rank={np.linalg.matrix_rank(M)}')

    print()
    print('=' * 90)
    print('二、核心命题：锚定数据落在 range(M) 内 (D[0]=0)')
    print('=' * 90)
    rng = np.random.default_rng(0)
    n_pts = 2000
    D = rng.standard_normal((N, n_pts))          # 任意"独立"的 10 相场
    D[0] = 0.0                                    # 锚定：θ=0 处恒等
    print(f'  随机锚定数据 D: shape={D.shape}, D[0] 全零={np.allclose(D[0],0)}')
    e9 = report('K=4 + Nyquist (9 项, 完备)', design(4, True), D)
    e8 = report('K=4 无 Nyquist (8 项, 欠完备)', design(4, False), D)
    M5 = design(5, False)
    e5 = report('K=5 无显式 Nyquist (数值奇异)', M5, D)
    c5 = np.linalg.cond(M5)
    print(f'  {"":<38} ^ 注意：残差看似很小，但 cond={c5:.3e} —— 矩阵数值上奇异，')
    print(f'  {"":<38}   sin5θ 在 10 个采样点上恒为 0，系数不可辨识，**不可用**')

    print()
    print('=' * 90)
    print('三、精确重构（9 项）：系数 -> 场 -> 回代')
    print('=' * 90)
    M = design(4, True); P = np.linalg.pinv(M)
    c = P @ D
    D2 = M @ c
    print(f'  最大重构误差 = {np.abs(D-D2).max():.3e}   (机器精度量级应 <1e-12)')

    print()
    print('=' * 90)
    print('四、8 项丢掉的到底是什么？—— Nyquist 模态')
    print('=' * 90)
    print('  ⚠️ 必须区分两件**不同**的东西（早先稿件把二者混为一谈，见 docs/33）：')
    print('     (A) **缺失方向 m** —— 锚定空间（v_0=0, 9 维）里未被 8 项基张成的那 1 维。')
    print('         它就是"参考相位处为 0、之后交替"的向量：m ∝ (0, 1, −1, 1, −1, …)。')
    print('     (B) **补偿向量 cos5θ−1** —— 一个**可选**的补齐项，**不等于** m；')
    print('         它在 m 方向上有非零投影，所以加它能把秩补满。')
    print('         注意 cos5θ−1 = (0,−2,0,−2,…) 与锚定交替向量夹角余弦仅 0.7071。')
    print('     下面测的是 (A)：残差的主方向是否就是锚定交替向量。')
    # 8 项基的零空间在 {v_0=0} 中的补方向
    M8 = design(4, False)
    res8 = D - M8 @ (np.linalg.pinv(M8) @ D)
    # 对残差做主成分，看它与"锚定后的交替模态"的关系
    u, s, vt = np.linalg.svd(res8 - res8.mean(axis=1, keepdims=True), full_matrices=False)
    alt = np.array([(-1.0) ** i for i in range(N)]); alt[0] = 0.0   # ← 置零即锚定，得到 (A)
    alt = alt / np.linalg.norm(alt)
    v1 = u[:, 0]                                  # 左奇异向量，长度 N（=相位数）
    v1 = v1 / np.linalg.norm(v1)
    print(f'  (A) 残差第 1 主方向 与 **锚定交替向量** (0,±1,∓1,…) 的 |内积| = '
          f'{abs(float(v1 @ alt)):.6f}')
    _nq = np.cos(5 * TH) - 1.0
    _a0 = np.array([(-1.0) ** i for i in range(N)]); _a0 = _a0 / np.linalg.norm(_a0)
    print(f'  (B) 补偿向量 cos5θ−1 与**未锚定**的 (−1)^i 的夹角余弦 = '
          f'{abs(float(_nq @ _a0)) / np.linalg.norm(_nq):.6f}   ← = 1/√2，**不是 1**')
    print(f'  8 项残差占数据能量比例 = {np.linalg.norm(res8)/np.linalg.norm(D)*100:.2f}%')
    print(f'  (该比例对**随机**数据成立；真实呼吸轨迹平滑，Nyquist 分量远小于此 ——')
    print(f'   真实数据上的实测见 tools/nyquist_content_real.py：8 项残差 0.24–0.32 mm)')

    print()
    print('=' * 90)
    print('结论')
    print('=' * 90)
    ok = e9 < 1e-12 and e8 > 1e-3
    print(f'  [{"PASS" if ok else "FAIL"}] 9 项周期基对锚定 10 相数据**无损**（残差 {e9:.2e}）')
    print(f'  [{"PASS" if e8 > 1e-3 else "FAIL"}] 8 项基**有损**（残差 {e8:.2e}）—— 缺的是 Nyquist 模态')
    print(f'  [{"PASS" if c5 > 1e10 else "FAIL"}] K=5 无显式 Nyquist 项时 cond={c5:.2e} '
          f'—— 数值奇异，系数不可辨识')
    print()
    print('  => 精确闭合 + 连续相位 + 零表示损失 可同时达到，无取舍。')
    print('  => 因此 PMR 的精度瓶颈**不可能**来自流形表示；只能来自相似度 / 优化。')
    print('     （实测印证：见 docs/21 —— 换用肺掩膜后最简成对配置 TRE 改善 41–46%）')


if __name__ == '__main__':
    main()
