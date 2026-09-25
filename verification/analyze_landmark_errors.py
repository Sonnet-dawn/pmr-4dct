"""
analyze_landmark_errors.py —— 逐关键点误差分布分析。
================================================================================
动机：case8 的 Elastix TRE 均值 9.84 但 SD 6.21 —— 高度怀疑是**少数关键点灾难性失败**
把均值拉高（DIRLAB 的关键点集是已知存在离群点的）。若中位数远小于均值，
则"均值 9.84"这个数字本身需要附加说明，且可能解释我们与已发表值（4.3）的差距。

用法：
  python analyze_landmark_errors.py --case 8 --tag kanai2
  python analyze_landmark_errors.py --case 1 --tag kanai2
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
import os, sys, argparse
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
POINTS = os.path.join(HERE, '..', 'reference_4', 'data', 'DIRLAB', 'points')
LIT = os.path.join(HERE, 'results', 'elastix_lit')


def load_outpoints(path):
    pts = []
    for line in open(path, encoding='utf-8', errors='replace'):
        if 'OutputPoint' not in line:
            continue
        seg = line.split('OutputPoint = [')[1].split(']')[0].split()
        pts.append([float(v) for v in seg[:3]])
    return np.array(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', type=int, default=8)
    ap.add_argument('--tag', type=str, default='kanai2')
    ap.add_argument('--all', action='store_true', help='分析所有已有 outputpoints')
    args = ap.parse_args()

    cn = args.case
    lm0 = np.loadtxt(os.path.join(POINTS, f'case{cn}', f'case{cn}_300_T00_xyz_R.txt'))
    lm5 = np.loadtxt(os.path.join(POINTS, f'case{cn}', f'case{cn}_300_T50_xyz_R.txt'))
    init = np.linalg.norm(lm0 - lm5, axis=1)

    cands = []
    if args.all:
        for d in sorted(os.listdir(LIT)):
            p = os.path.join(LIT, d, 'outputpoints.txt')
            if os.path.exists(p):
                cands.append((d, p))
    else:
        cands = [(args.tag, os.path.join(LIT, f'{args.tag}_case{cn}', 'outputpoints.txt'))]

    for name, p in cands:
        if not os.path.exists(p):
            print(f'[跳过] {name}: 无 {p}')
            continue
        out = load_outpoints(p)
        if len(out) != len(lm5):
            print(f'[跳过] {name}: 点数 {len(out)} != {len(lm5)}')
            continue
        err = np.linalg.norm(out - lm5, axis=1)
        print('=' * 76)
        print(f'{name}  case{cn}  n={len(err)}')
        print(f'  初始误差:  均值 {init.mean():6.3f}  中位 {np.median(init):6.3f}')
        print(f'  配准后:    均值 {err.mean():6.3f}  中位 {np.median(err):6.3f}  '
              f'SD {err.std():6.3f}')
        for q in (50, 75, 90, 95, 99, 100):
            print(f'    p{q:<3}: {np.percentile(err, q):8.3f}')
        for th in (2, 5, 10, 20, 30):
            print(f'    误差 > {th:>2} mm 的关键点数: {int((err > th).sum()):>3} '
                  f'({(err > th).mean()*100:5.1f}%)  它们贡献了 '
                  f'{err[err > th].sum()/err.sum()*100:5.1f}% 的总误差')
        # 去掉最差的 k 个后的均值
        s = np.sort(err)[::-1]
        for k in (0, 1, 3, 5, 10, 20):
            print(f'    去掉最差 {k:>2} 个后的均值: {s[k:].mean():7.3f}  (k=0 即原始均值)')
        # 与初始误差的相关性：失败的 landmark 是不是初始位移大的那些？
        if init.std() > 0:
            r = float(np.corrcoef(init, err)[0, 1])
            print(f'  配准误差 与 初始位移 的相关系数: r = {r:.3f}')
            big = init > np.percentile(init, 90)
            print(f'    初始位移最大 10% 的点: 配准后均值 {err[big].mean():6.3f} '
                  f'(n={int(big.sum())})')
            print(f'    其余 90% 的点:          配准后均值 {err[~big].mean():6.3f}')
        print()


if __name__ == '__main__':
    main()
