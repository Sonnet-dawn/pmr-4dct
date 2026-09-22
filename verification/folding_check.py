"""
folding_check.py —— 折叠检查（**分肺内 / 肺外**）。
================================================================================
动机：v2 把相似度限制在**肺掩膜内**，掩膜外的位移只由粗网格模型与平滑正则外推，
没有数据约束。因此必须回答一个审稿人一定会问的问题：

    **掩膜外的位移会不会折叠（det(J) <= 0）？**

本脚本对任意已保存的 DVF 报告：
  * 全图 / 肺内 / 肺外的 det(J) 最小值与负值体素比例
  * 每个相位分别报告（折叠常出现在大位移相位）

⚠️ 论文红线（`docs/17` C7a）：**不得**写"PMR 无折叠 / B 样条有折叠"。
   实测是 PMR 0.0000–0.0512%、B 样条与 VoxelMorph 0.0000% —— 方向相反。
   本脚本只用于**如实报告**，不得用于任何优越性声称。

用法：
  python folding_check.py --dvf results/pmr_v3/real_case1_2mm_proj_dvf.npy --case 1 --down 2
  python folding_check.py --glob "results/fast_phase2/*_2mm_dvf.npy" --down 2
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import os, sys, json, glob, argparse
import numpy as np

try:                                    # Windows 控制台默认 GBK，emoji 会炸
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from jacobian_stats import det_jacobian_array
from lung_mask import mask_at_down


def analyse(dvf_mm, spacing, mask=None):
    """dvf_mm: (P,3,D,H,W) mm。返回逐相位与汇总的折叠统计。"""
    if dvf_mm.ndim == 4:
        dvf_mm = dvf_mm[None]
    P = dvf_mm.shape[0]
    dvf_vox = dvf_mm / np.asarray(spacing)[None, :, None, None, None]
    out = []
    for p in range(P):
        det = det_jacobian_array(dvf_vox[p])
        rec = {'phase': p, 'det_min': float(det.min()),
               'det_p001': float(np.percentile(det, 0.1)),
               'pct_nonpositive': float((det <= 0).mean() * 100)}
        if mask is not None:
            m = mask if mask.shape == det.shape else None
            if m is not None:
                inn, outm = m > 0.5, m <= 0.5
                rec['pct_nonpositive_in'] = float((det[inn] <= 0).mean() * 100) if inn.any() else None
                rec['pct_nonpositive_out'] = float((det[outm] <= 0).mean() * 100) if outm.any() else None
                rec['det_min_in'] = float(det[inn].min()) if inn.any() else None
                rec['det_min_out'] = float(det[outm].min()) if outm.any() else None
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dvf', type=str, default='')
    ap.add_argument('--glob', type=str, default='')
    ap.add_argument('--case', type=int, default=0)
    ap.add_argument('--down', type=int, default=2)
    ap.add_argument('--norm', type=str, default='robust')
    ap.add_argument('--out', type=str, default=os.path.join(HERE, 'results', 'folding_check.json'))
    args = ap.parse_args()

    files = []
    if args.dvf:
        files = [(args.dvf, args.case)]
    elif args.glob:
        for f in sorted(glob.glob(os.path.join(HERE, args.glob))):
            base = os.path.basename(f)
            cn = 0
            for tok in base.replace('_', ' ').replace('-', ' ').split():
                if tok.startswith('case') and tok[4:].isdigit():
                    cn = int(tok[4:])
            files.append((f, cn))
    if not files:
        sys.exit('请给 --dvf 或 --glob')

    res = []
    for f, cn in files:
        dvf = np.load(f)
        sp = float(args.down)
        spacing = np.array([sp, sp, sp])
        mask = None
        if cn:
            try:
                mask = mask_at_down(cn, args.down)
            except Exception as e:
                print(f'  [warn] 掩膜不可用 case{cn}: {e}')
        recs = analyse(dvf, spacing, mask)
        agg = {
            'file': os.path.relpath(f, HERE), 'case': cn, 'down_mm': args.down,
            'n_phases': int(dvf.shape[0]), 'shape': list(dvf.shape[2:]),
            'det_min': float(np.min([r['det_min'] for r in recs])),
            'pct_nonpositive_mean': float(np.mean([r['pct_nonpositive'] for r in recs])),
            'pct_nonpositive_max': float(np.max([r['pct_nonpositive'] for r in recs])),
            'per_phase': recs,
        }
        if mask is not None:
            agg['mask_frac'] = float(mask.mean())
            agg['pct_nonpositive_in_mean'] = float(np.mean(
                [r['pct_nonpositive_in'] for r in recs if r.get('pct_nonpositive_in') is not None]))
            agg['pct_nonpositive_out_mean'] = float(np.mean(
                [r['pct_nonpositive_out'] for r in recs if r.get('pct_nonpositive_out') is not None]))
            agg['det_min_in'] = float(np.min(
                [r['det_min_in'] for r in recs if r.get('det_min_in') is not None]))
            agg['det_min_out'] = float(np.min(
                [r['det_min_out'] for r in recs if r.get('det_min_out') is not None]))
        res.append(agg)
        print(f'\n{agg["file"]}')
        print(f'  相位 {agg["n_phases"]}, 网格 {agg["shape"]}, 掩膜占比 '
              f'{agg.get("mask_frac", float("nan"))*100:.1f}%')
        print(f'  全图  det_min={agg["det_min"]:.4f}  %det<=0: 均值 '
              f'{agg["pct_nonpositive_mean"]:.4f}% 最大 {agg["pct_nonpositive_max"]:.4f}%')
        if 'det_min_in' in agg:
            print(f'  肺内  det_min={agg["det_min_in"]:.4f}  %det<=0: '
                  f'{agg["pct_nonpositive_in_mean"]:.4f}%')
            print(f'  肺外  det_min={agg["det_min_out"]:.4f}  %det<=0: '
                  f'{agg["pct_nonpositive_out_mean"]:.4f}%')

    json.dump(res, open(args.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n已保存 {args.out}')
    print('\n⚠️ 论文红线：不得用本结果做"PMR 不易折叠"的优越性声称（实测方向相反）。')


if __name__ == '__main__':
    main()
