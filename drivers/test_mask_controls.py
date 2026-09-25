"""
test_mask_controls.py —— 掩膜机制的**对照实验**（拆 A4 指出的审稿人子弹）。
================================================================================
动机（`docs/24` 第三节）：我们的掩膜同时改变了两件事 ——
  (a) 去掉准静止多数； (b) 把**胸膜/肋骨滑动界面**从数据项切掉。
后者在文献中已有充分解释（Staring 2010 / Rühaak 2017 / Hering 2021），
审稿人可以说"增益全部来自已知的滑动界面效应"。三组对照用于把机制区分开：

  ① **侵蚀掩膜**（向内侵蚀 5/10/15 体素）
     把胸膜界面排除在外。**若增益仍在 ⇒ 支持"不动多数"机制**（不是滑动界面）。
  ② **等体积错位掩膜**（把肺掩膜整体平移到心/纵隔/肝等**准静止软组织**）
     **反向对照**：若任意等体积区域都能改善 ⇒ 机制声称被大幅削弱
     （说明效应只是"改变了优化动力学/数据量"，而非"选对了区域"）。
  ③ **膨胀扫描**（0/5/10/20/40 体素）
     画 **TRE vs 相似度作用域体积** 曲线。单调曲线是最难反驳的证据。

统一使用**最小成对配置**（无 CNN、无时间耦合、无残差）+ 固定 1200 迭代，
只改相似度的空间作用域 —— 与其他变量完全隔离。评价用标准口径。

用法：
  python test_mask_controls.py --cases 1,5,8 --iters 1200
  python test_mask_controls.py --cases 1,5,8 --which erosion,shift,dilation
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
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pmr_fast import (load_imgs, grid_, ncc, pool_stats, reg_coarse, warp, device)
from pmr_v2 import load_imgs_v2, masked_local_ncc, LOCAL_MASK_MODE
from lung_mask import (lung_mask_native, mask_variant_native, _shift_zero,
                       auto_shift_for_volume)
from recompute_tre_standard import load_lm, sample_trilinear

OUTD = os.path.join(HERE, 'results', 'mask_controls')
os.makedirs(OUTD, exist_ok=True)
TH = float(np.pi)


def run_pair(cn, down, mask_np, iters=1200, lr=1e-3, grid_div=8, res_div=2,
             win_mm=30.0, norm='robust', seed=0, log=False):
    """最小成对配准（T00<->T50），相似度只在 mask_np 内。返回标准口径 TRE。

    mask_np: 原生网格 bool；内部按 down 降采样。
    """
    import SimpleITK as sitk
    torch.manual_seed(seed)
    imgs, sitk_imgs = load_imgs_v2(cn, down, norm)
    spacing = np.array(sitk_imgs[0].GetSpacing())
    img0 = imgs[0].to(device); img5 = imgs[5].to(device)
    D, H, W = img0.shape[2], img0.shape[3], img0.shape[4]
    S = torch.tensor([W, H, D], device=device)
    g = grid_(img0.shape)
    win = max(5, int(round(win_mm / spacing[0]))); stride = max(1, win // 2)

    mt = sitk.GetImageFromArray(mask_np.astype(np.uint8))
    if down != 1:
        rs = sitk.ResampleImageFilter()
        rs.SetSize([max(1, s // down) for s in mt.GetSize()])
        rs.SetOutputSpacing([float(down)] * 3)
        rs.SetInterpolator(sitk.sitkNearestNeighbor)
        mt = rs.Execute(mt)
    mnp = sitk.GetArrayFromImage(mt).astype(np.float32)
    m = torch.from_numpy(np.ascontiguousarray(mnp))[None, None].to(device)
    if tuple(m.shape[2:]) != (D, H, W):
        m = F.interpolate(m, size=(D, H, W), mode='nearest')
    frac = float(m.mean())
    if frac < 1e-4:
        return None, 0.0, 'empty mask'

    cs = (max(1, D // grid_div), max(1, H // grid_div), max(1, W // grid_div))
    rs_ = (max(1, D // res_div), max(1, H // res_div), max(1, W // res_div))
    coef_c = torch.zeros(1, 3, *cs, device=device, requires_grad=True)
    coef_r = torch.zeros(1, 3, *rs_, device=device, requires_grad=True)
    opt = torch.optim.Adam([coef_c], lr=lr)
    f_stats = pool_stats(img0, win, stride)

    def field():
        return (F.interpolate(coef_c, size=(D, H, W), mode='trilinear', align_corners=True)[0]
                + F.interpolate(coef_r, size=(D, H, W), mode='trilinear', align_corners=True)[0])

    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = (masked_local_ncc(warp(img5, g, field(), S), img0, f_stats, m, win, stride)
                + reg_coarse(coef_c, 1.0, 0.1, grid_div))
        loss.backward(); opt.step()
        if log and (it + 1) % 400 == 0:
            print(f'      it {it+1:4d} loss {float(loss):+.5f}', flush=True)
    opt2 = torch.optim.Adam([coef_r], lr=1e-2)
    for it in range(200):
        opt2.zero_grad(set_to_none=True)
        loss = (masked_local_ncc(warp(img5, g, field(), S), img0, f_stats, m, win, stride)
                + reg_coarse(coef_r, 0.3, 0.1, res_div))
        loss.backward(); opt2.step()

    with torch.no_grad():
        d = field().cpu().numpy() * spacing[:, None, None, None]
    lm0, lm5 = load_lm(cn)
    tre = float(np.linalg.norm(lm0 + sample_trilinear(d, lm0, spacing) - lm5, axis=1).mean())
    return tre, frac, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=str, default='1,5,8')
    ap.add_argument('--down', type=int, default=2)
    ap.add_argument('--iters', type=int, default=1200)
    ap.add_argument('--which', type=str, default='erosion,shift,dilation')
    ap.add_argument('--which-cases', type=str, default='')
    args = ap.parse_args()
    global LOCAL_MASK_MODE
    LOCAL_MASK_MODE = 'weight'          # 小掩膜（侵蚀后）用加权更稳

    # 文件名含病例列表，避免并行进程互相覆盖（竞态）
    cases_tag = ''.join(c for c in args.cases if c.isdigit() or c == '-')
    resf = os.path.join(OUTD, f'controls_{cases_tag}_{args.down}mm.json')
    res = json.load(open(resf, encoding='utf-8')) if os.path.exists(resf) else []
    done = {(r['case'], r['kind'], r['param']) for r in res}

    for cn in [int(x) for x in args.cases.split(',') if x.strip()]:
        print(f'\n########## case{cn} ##########', flush=True)
        lm0, lm5 = load_lm(cn)
        init = float(np.linalg.norm(lm0 - lm5, axis=1).mean())
        base = lung_mask_native(cn, 0, dilate=0)
        base_frac = float(base.mean())
        variants = []
        if 'none' in args.which:
            variants.append(('nomask', 0, np.ones_like(base)))
        if 'lung' in args.which:
            variants.append(('lung', 0, base))
        if 'erosion' in args.which:
            for k in (5, 10, 15):
                variants.append(('erosion', k, mask_variant_native(cn, dilate=-k)))
        if 'dilation' in args.which:
            for k in (5, 10, 20, 40):
                variants.append(('dilation', k, mask_variant_native(cn, dilate=k)))
        if 'shift' in args.which:
            sh = auto_shift_for_volume(cn, base_frac, verbose=True)
            variants.append(('shift', 0, mask_variant_native(cn, shift=sh)))
            variants.append(('shift_half', 0, mask_variant_native(
                cn, shift=tuple(int(v * 0.5) for v in sh))))

        for kind, param, mnp in variants:
            if (cn, kind, param) in done:
                print(f'  [skip] {kind}{param}'); continue
            t0 = time.time()
            tre, frac, err = run_pair(cn, args.down, mnp, iters=args.iters)
            if tre is None:
                print(f'  {kind}{param}: {err}'); continue
            rec = {'case': cn, 'kind': kind, 'param': param, 'init_tre': init,
                   'tre_std': tre, 'mask_frac': frac,
                   'improve_vs_nomask': None,
                   'time_s': round(time.time() - t0, 1)}
            res.append(rec)
            print(f'  {kind:<10}{param:<4} 作用域 {frac*100:5.1f}%  ->  TRE {tre:7.3f}  '
                  f'({time.time()-t0:.0f}s)', flush=True)
            json.dump(res, open(resf, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    # ---- 汇总 ----
    print('\n' + '=' * 78)
    print('汇总（同一病例内互比）')
    print('=' * 78)
    for cn in sorted({r['case'] for r in res}):
        rows = sorted([r for r in res if r['case'] == cn], key=lambda r: r['mask_frac'])
        nm = [r for r in rows if r['kind'] == 'nomask']
        ref = nm[0]['tre_std'] if nm else float('nan')
        print(f'\ncase{cn}（初始 {rows[0]["init_tre"]:.2f} mm；未掩膜对照 {ref:.3f}）')
        print(f'  {"变体":<14}{"作用域%":>9}{"TRE":>9}{"vs 未掩膜":>11}')
        for r in rows:
            imp = f'{(ref-r["tre_std"])/ref*100:+6.1f}%' if ref == ref else '   —'
            print(f'  {r["kind"]+str(r["param"]):<14}{r["mask_frac"]*100:>9.1f}'
                  f'{r["tre_std"]:>9.3f}{imp:>11}')
    print(f'\n已保存 {resf}')
    print('\n判读：')
    print('  · 侵蚀 10-15 mm 后增益仍在  -> 支持"去掉不动多数"机制')
    print('  · 等体积错位掩膜也能改善    -> 机制声称被削弱（效应只是改变优化动力学）')
    print('  · TRE vs 作用域体积单调     -> 最强证据')


if __name__ == '__main__':
    main()
