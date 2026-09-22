"""
verify_exact_basis.py —— 验证"锚定完备基"能否精确表示任意锚定相位场族
================================================================================
核心问题：原实现用 K=4 的 8 个基函数 {cos kθ-1, sin kθ}_{k=1..4}，
          但要表示的独立数据是 9 个（10 个相位，锚定 f0=0 后剩 f1..f9）。
          => 8 基表示 9 自由度，**必然有残差**。这才是"结构性天花板"的真正来源，
             而不是"共享系数的哲学代价"。

若把 Nyquist 项 cos(5θ)-1 加进来，得到 9 个基函数，正好与 9 个自由度匹配
=> 可以**精确插值**任意锚定场族，采样相位上零误差。

同时验证：
  (a) 精确重构（采样相位残差 ~1e-15）
  (b) 锚定 d(0)=0（代数恒等）
  (c) 闭合 d(2π)=d(0)（代数恒等）
  (d) 绕环复合恒等（相邻场取锚定场之差，望远镜求和为 0）
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
TH = np.array([2 * np.pi * i / N for i in range(N)])


def anchored_basis(theta, K, include_nyquist=True):
    """返回 (len(theta), n_basis) 的锚定周期基矩阵，以及基函数名。"""
    B, names = [], []
    for k in range(1, K + 1):
        B.append(np.cos(k * theta) - 1); names.append('cos%d-1' % k)
    for k in range(1, K + 1):
        B.append(np.sin(k * theta)); names.append('sin%d' % k)
    if include_nyquist and N % 2 == 0:
        B.append(np.cos((N // 2) * theta) - 1); names.append('cos%d-1' % (N // 2))
    return np.array(B).T, names


def main():
    print('=' * 78)
    print('1) 基函数个数 vs 自由度')
    print('=' * 78)
    print('  独立数据 = N-1 = %d 个场（锚定后 f0=0）' % (N - 1))
    for K in (2, 3, 4, 5):
        for ny in (False, True):
            B, names = anchored_basis(TH, K, ny)
            tag = 'K=%d%s' % (K, '+Nyq' if ny else '    ')
            shape = B[1:].shape
            r = np.linalg.matrix_rank(B[1:])
            cond = np.linalg.cond(B[1:]) if shape[0] == shape[1] else float('inf')
            ok = '方阵满秩 -> 可精确插值' if (shape[0] == shape[1] and r == shape[0]) else ''
            print('  %-9s 基函数 %2d 个  矩阵 %-8s rank=%d cond=%7.3g  %s'
                  % (tag, len(names), str(shape), r, cond, ok))

    print()
    print('=' * 78)
    print('2) 决定性检验：K=4+Nyquist（9 基）能否精确重构任意锚定场族')
    print('=' * 78)
    B, names = anchored_basis(TH, 4, True)
    P = B[1:]                                  # 只用 i=1..9 行
    print('  基函数: %s' % ', '.join(names))
    print('  矩阵 9x9 rank=%d cond=%.3g' % (np.linalg.matrix_rank(P), np.linalg.cond(P)))

    rng = np.random.default_rng(0)
    f = rng.normal(size=(N - 1, 7))            # 9 个相位 x 7 个"体素"的任意锚定场
    c = np.linalg.solve(P, f)
    rec = P @ c
    err = np.abs(rec - f).max()
    print('  重构 9 个任意场: max 残差 = %.3e  -> %s'
          % (err, '精确 (零损失)' if err < 1e-10 else '不精确'))

    # 锚定与闭合
    d0 = B[0] @ c                                                    # theta = 0
    th2 = np.array([2 * np.pi])
    B2, _ = anchored_basis(th2, 4, True)
    d2pi = (B2 @ c)[0]
    print('  d(0)    max = %.3e  -> %s' % (np.abs(d0).max(),
          '锚定成立' if np.abs(d0).max() < 1e-12 else '锚定不成立'))
    print('  d(2pi)  max = %.3e  -> %s' % (np.abs(d2pi).max(),
          '闭合成立' if np.abs(d2pi).max() < 1e-12 else '闭合不成立'))

    # 绕环复合（相邻场 = 锚定场之差，望远镜求和）
    F = np.vstack([np.zeros((1, f.shape[1])), f])                    # 含 f0=0 的完整 10 相
    adj = np.array([F[(i + 1) % N] - F[i] for i in range(N)])
    print('  绕环复合 max|sum| = %.3e  -> %s'
          % (np.abs(adj.sum(axis=0)).max(),
             '恒为零（望远镜求和）' if np.abs(adj.sum(axis=0)).max() < 1e-14 else '非零'))
    print('  （注意：这与周期性无关，任何锚定场族都成立）')

    print()
    print('=' * 78)
    print('3) 对照：当前实现的 8 基（K=4，无 Nyquist）')
    print('=' * 78)
    B8, names8 = anchored_basis(TH, 4, False)
    P8 = B8[1:]
    print('  基函数 %d 个: %s' % (len(names8), ', '.join(names8)))
    c8, *_ = np.linalg.lstsq(P8, f, rcond=None)
    print('  最小二乘拟合 9 个自由度 -> max 残差 = %.4f （不可能为零）'
          % np.abs(P8 @ c8 - f).max())
    print()
    print('  >>> 结论：把 K=4 的 8 基改为 9 基（+Nyquist 项），')
    print('      采样相位上的表示误差归零 —— 这是"结构性天花板"的算术根源。')


if __name__ == '__main__':
    main()
