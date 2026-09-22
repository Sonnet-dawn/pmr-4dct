"""
run_elastix_lit.py —— 重建**文献背书**的 Elastix 基线（Kanai 2014 参数 2）。
================================================================================
背景（`docs/19` §7.2 / `lit/A1_...md` §5.1）：
  本项目原有的 Elastix 配置是**弱配置**（默认 8mm 控制网格、仅 3 级、无掩膜、
  无预对齐、无弯曲能），10 例均值 3.348（旧口径），比已发表的 1.83（Med Phys 2025）
  / 1.28（Kanai 2014 参数 2）弱约 2 倍。**弱基线使对照失去意义。**

本脚本照抄 Kanai 2014 Table 2 的"参数 2"（已知在 DIRLAB 10 例 × 300 点上得 1.28 mm）：

  σ(体素)  迭代  最终网格(mm)
    16     1000    80
     8     1000    80
     4     1000    40
     2     1000    20
     1     1000    10
     4     2000    80
     3     2000    40
     2     2000    20
     1     2000    10
     1     2000     5     <-- 最终 5 mm

  ASGD + NormalizedCorrelation (+ 可选弯曲能) + RandomCoordinate 2000 采样/迭代
  每级一个 -p 参数文件，由 elastix 串联。

配套（A1 明确列为本项目缺失的三项）：
  * **肺掩膜** -fMask/-mMask（HU −990…−250）
  * 标准 TRE 口径：transformix 变换 **T00** 点集后与 T50 求距离
  * 仿射预对齐（可选，默认开）

用法：
  python run_elastix_lit.py --probe                 # 只查 elastix 可用性
  python run_elastix_lit.py --case 1                # 单例
  python run_elastix_lit.py --cases 1,2,3 --workers 2
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import os, sys, json, time, argparse, shutil, subprocess
import numpy as np
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lung_mask import lung_mask_native, raw_case

DATA = os.path.join(os.environ.get('PMR_DATA_ROOT')
                    or os.path.join(HERE, '..', 'reference_4', 'data'), 'DIRLAB')
OUTD = os.path.join(HERE, 'results', 'elastix_lit')
LOGD = os.path.join(OUTD, 'logs')

# Kanai 2014 Table 2「参数 2」
STAGES = [(16, 1000, 80), (8, 1000, 80), (4, 1000, 40), (2, 1000, 20), (1, 1000, 10),
          (4, 2000, 80), (3, 2000, 40), (2, 2000, 20), (1, 2000, 10), (1, 2000, 5)]


def find_elastix():
    for name in ('elastix', 'elastix.exe'):
        p = shutil.which(name)
        if p:
            return p
    for root in (r'C:\Program Files', r'C:\Program Files (x86)', r'C:\elastix',
                 os.path.join(HERE, 'tools'), r'D:\elastix'):
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            if 'elastix.exe' in files:
                return os.path.join(dirpath, 'elastix.exe')
            if dirpath.count(os.sep) - root.count(os.sep) > 3:
                continue
    return None


def param_text(sigma, iters, grid_mm, bending=0.0, samples=2000, metric_name='AdvancedNormalizedCorrelation'):
    """metric_name：elastix 的 NCC 叫 **AdvancedNormalizedCorrelation**
    （`NormalizedCorrelation` 在此构建中未安装 —— 已实测报错 "component is not installed"）。"""
    metric = (f'(Metric "{metric_name}" "TransformBendingEnergyPenalty")'
              if bending > 0 else f'(Metric "{metric_name}")')
    mw = f'(MetricWeight 1.0 {bending})' if bending > 0 else '(MetricWeight 1.0)'
    return f"""// Kanai 2014 (J Radiat Res 55(6):1163) Table 2 -- parameter set 2
(Transform "BSplineTransform")
(NumberOfResolutions 1)
(FinalGridSpacingInPhysicalUnits {grid_mm} {grid_mm} {grid_mm})
(GridSpacingSchedule 1.0 1.0 1.0)
(HowToCombineTransforms "Compose")
{metric}
{mw}
(Optimizer "AdaptiveStochasticGradientDescent")
(ASGDParameterEstimationMethod "Default")
(MaximumNumberOfIterations {iters})
(NumberOfSpatialSamples {samples})
(ImageSampler "RandomCoordinate")
(NewSamplesEveryIteration "true")
(UseRandomSampleRegion "false")
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(ImagePyramidSchedule {sigma} {sigma} {sigma})
(Interpolator "BSplineInterpolator")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(WriteResultImage "false")
(ErodeMask "false")
(DefaultPixelValue -1000)
"""


AFFINE_TXT = """(Transform "AffineTransform")
(NumberOfResolutions 3)
(MaximumNumberOfIterations 500 500 250)
(Metric "AdvancedMattesMutualInformation")
(MetricNumberOfHistogramBins 32)
(NumberOfSpatialSamples 4096)
(ImageSampler "RandomCoordinate")
(NewSamplesEveryIteration "true")
(Optimizer "AdaptiveStochasticGradientDescent")
(WriteResultImage "false")
(DefaultPixelValue -1000)
"""

# 本项目原有的 Elastix 配置（`run_elastix_baseline.py`）：**已知能在 10 例上全部跑通**。
# 作为 kanai2 配置在部分病例抛 "Error in metric" 时的可用回退。
LEGACY_TXT = """(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingImageDimension 3)
(Registration "MultiResolutionRegistration")
(Interpolator "LinearInterpolator")
(ResampleInterpolator "FinalLinearInterpolator")
(Resampler "DefaultResampler")
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(Optimizer "AdaptiveStochasticGradientDescent")
(MaximumNumberOfIterations 1500)
(NumberOfSpatialSamples 4096)
(NewSamplesEveryIteration "true")
(ImageSampler "RandomCoordinate")
(Metric "AdvancedMattesMutualInformation")
(MetricNumberOfHistogramBins 64)
(Transform "BSplineTransform")
(NumberOfResolutions 3)
(HowToCombineTransforms "Compose")
(DefaultPixelValue 0)
(WriteResultImage "false")
"""


def legacy_text(samples=4096, iterations=1500):
    """legacy 配置文本（本项目原配置），采样数与迭代数可调。

    🔴 动机：`NumberOfSpatialSamples` 固定 4096 时，在 1 mm 的 case8（7900 万体素）上
    采样覆盖率仅 **0.005%** —— 这被怀疑是 case8 配准失败（TRE ≈ 10）的主因。
    """
    return LEGACY_TXT.replace('(NumberOfSpatialSamples 4096)',
                              f'(NumberOfSpatialSamples {samples})').replace(
        '(MaximumNumberOfIterations 1500)',
        f'(MaximumNumberOfIterations {iterations})')


def write_mask(cn, p, path, dilate=1):
    """把肺掩膜写成与源图同几何的 .mha（elastix -fMask/-mMask 用）。"""
    im = raw_case(cn, p)
    m = lung_mask_native(cn, p, dilate=dilate).astype(np.uint8) * 255
    mi = sitk.GetImageFromArray(m)
    mi.CopyInformation(im)
    sitk.WriteImage(mi, path)
    return path


def tre_standard(transformix, tp_file, cn, workdir):
    """标准口径 TRE：transformix 变换 T00 点集，与 T50 求欧氏距离。"""
    pts_in = os.path.join(workdir, 'points_T00.txt')
    lm0 = np.loadtxt(os.path.join(DATA, 'points', f'case{cn}', f'case{cn}_300_T00_xyz_R.txt'))
    lm5 = np.loadtxt(os.path.join(DATA, 'points', f'case{cn}', f'case{cn}_300_T50_xyz_R.txt'))
    with open(pts_in, 'w') as f:
        f.write('index\n' + str(len(lm0)) + '\n')
        for p in lm0:
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
    r = subprocess.run([transformix, '-tp', tp_file, '-def', pts_in, '-out', workdir],
                       capture_output=True, text=True)
    outp = os.path.join(workdir, 'outputpoints.txt')
    if not os.path.exists(outp):
        return None, f'transformix 失败: {r.stdout[-500:]} {r.stderr[-500:]}'
    pts_out = []
    for line in open(outp):
        if 'OutputPoint' not in line:
            continue
        seg = line.split('OutputPoint = [')[1].split(']')[0].split()
        pts_out.append([float(v) for v in seg[:3]])
    pts_out = np.array(pts_out)
    if len(pts_out) != len(lm0):
        return None, f'点数不匹配 {len(pts_out)} vs {len(lm0)}'
    d = np.linalg.norm(pts_out - lm5, axis=1)
    return {'tre_mean': float(d.mean()), 'tre_sd': float(d.std()),
            'tre_median': float(np.median(d)), 'n': int(len(d))}, None


def run_case(cn, exe, transformix, affine=True, bending=0.0, samples=2000, keep=False,
             metric_name='AdvancedNormalizedCorrelation', use_mask=True, mask_dilate=1,
             tag=None, config='kanai2', legacy_iters=1500):
    tag = tag or f'case{cn}'
    wd = os.path.join(OUTD, tag)
    os.makedirs(wd, exist_ok=True)
    t00 = os.path.join(DATA, 'mha', f'case{cn}', f'case{cn}_T00_R.mha')
    t50 = os.path.join(DATA, 'mha', f'case{cn}', f'case{cn}_T50_R.mha')
    m0 = write_mask(cn, 0, os.path.join(wd, 'mask_T00.mha'), mask_dilate)
    m5 = write_mask(cn, 5, os.path.join(wd, 'mask_T50.mha'), mask_dilate)

    plist = []
    if config == 'legacy':
        # 单次多分辨率调用（本项目原配置，10 例全部跑通）
        p = os.path.join(wd, 'p00_legacy.txt')
        open(p, 'w').write(legacy_text(samples, legacy_iters)); plist.append(p)
    else:
        if affine:
            pa = os.path.join(wd, 'p00_affine.txt')
            open(pa, 'w').write(AFFINE_TXT); plist.append(pa)
        for i, (sg, it, gs) in enumerate(STAGES):
            p = os.path.join(wd, f'p{i+1:02d}_s{sg}_i{it}_g{gs}.txt')
            open(p, 'w').write(param_text(sg, it, gs, bending, samples, metric_name))
            plist.append(p)

    out = os.path.join(wd, 'out')
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)          # elastix 要求输出目录必须已存在
    cmd = [exe, '-f', t00, '-m', t50]
    if use_mask:
        cmd += ['-fMask', m0, '-mMask', m5]
    for p in plist:
        cmd += ['-p', p]
    cmd += ['-out', out, '-threads', '8']
    log = open(os.path.join(LOGD, f'{tag}.log'), 'w', encoding='utf-8')
    log.write('$ ' + ' '.join(cmd) + '\n'); log.flush()
    t0 = time.time()
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    log.write(f'\n[rc={r.returncode} {dt:.0f}s]\n'); log.close()
    if r.returncode != 0:
        return {'case': cn, 'error': f'elastix rc={r.returncode}', 'time_s': dt}

    tps = sorted(f for f in os.listdir(out) if f.startswith('TransformParameters.')
                 and f.endswith('.txt'))
    if not tps:
        return {'case': cn, 'error': '无 TransformParameters', 'time_s': dt}
    tp_last = os.path.join(out, tps[-1])
    tre, err = tre_standard(transformix, tp_last, cn, wd)
    res = {'case': cn, 'impl': 'elastix_lit', 'config': config,
           'affine': affine, 'bending': bending, 'samples': samples,
           'metric': metric_name, 'use_mask': use_mask, 'mask_dilate': mask_dilate,
           'n_stages': len(STAGES), 'time_s': round(dt, 1),
           'tp_last': os.path.relpath(tp_last, HERE)}
    if tre:
        res.update(tre)
        init = float(np.linalg.norm(
            np.loadtxt(os.path.join(DATA, 'points', f'case{cn}', f'case{cn}_300_T00_xyz_R.txt'))
            - np.loadtxt(os.path.join(DATA, 'points', f'case{cn}', f'case{cn}_300_T50_xyz_R.txt')),
            axis=1).mean())
        res['init_tre'] = init
    else:
        res['error'] = err
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--case', type=int, default=None)
    ap.add_argument('--cases', type=str, default='')
    ap.add_argument('--no-affine', action='store_true')
    ap.add_argument('--bending', type=float, default=0.0,
                    help='弯曲能权重；0=Kanai 参数4（无弯曲能，1.36mm），>0=参数2')
    ap.add_argument('--samples', type=int, default=2000)
    ap.add_argument('--metric', type=str, default='AdvancedNormalizedCorrelation',
                    help='elastix 度量名；NCC=AdvancedNormalizedCorrelation，'
                         'MI=AdvancedMattesMutualInformation')
    ap.add_argument('--nomask', action='store_true', help='不使用肺掩膜（隔离掩膜影响）')
    ap.add_argument('--mask-dilate', type=int, default=1, help='掩膜膨胀体素数')
    ap.add_argument('--tag', type=str, default='', help='结果子目录/汇总键前缀')
    ap.add_argument('--config', choices=['kanai2', 'legacy'], default='kanai2',
                    help='kanai2=Kanai2014 参数2 的 10 级串联；legacy=本项目原配置（10 例均可跑通）')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--retries', type=int, default=3,
                    help='"Error in metric" 是随机采样造成的偶发失败，重试可解')
    args = ap.parse_args()
    os.makedirs(LOGD, exist_ok=True)

    exe = find_elastix()
    if not exe:
        print('!! 未找到 elastix。A1 报告称本机曾用 5.3.1；请安装或把路径加入 PATH。')
        print('   下载: https://elastix.lumc.nl/  (或 conda install -c conda-forge elastix)')
        return
    bindir = os.path.dirname(exe)
    transformix = shutil.which('transformix') or os.path.join(bindir, 'transformix.exe')
    print(f'elastix   : {exe}')
    print(f'transformix: {transformix} (存在: {os.path.exists(transformix)})')
    if args.probe:
        r = subprocess.run([exe, '--version'], capture_output=True, text=True)
        print((r.stdout or r.stderr)[:400])
        return

    cases = ([args.case] if args.case else
             [int(c) for c in args.cases.split(',') if c.strip()])
    if not cases:
        cases = list(range(1, 11))
    print(f'配置: affine={not args.no_affine} bending={args.bending} samples={args.samples}')
    print(f'阶段表(σ,iter,grid_mm): {STAGES}\n')

    key = args.tag or 'default'
    resf = os.path.join(OUTD, f'elastix_lit_summary_{key}.json')
    res = json.load(open(resf, encoding='utf-8')) if os.path.exists(resf) else []
    done = {r['case'] for r in res if 'tre_mean' in r}
    for cn in cases:
        if cn in done:
            print(f'[skip] case{cn} 已有 TRE'); continue
        print(f'=== case{cn} [{key}] ===', flush=True)
        r = None
        for attempt in range(1, args.retries + 1):
            r = run_case(cn, exe, transformix, affine=not args.no_affine,
                         bending=args.bending, samples=args.samples, metric_name=args.metric,
                         use_mask=not args.nomask, mask_dilate=args.mask_dilate,
                         config=args.config,
                         tag=(f'{args.tag}_case{cn}' if args.tag else f'case{cn}'))
            if 'tre_mean' in r:
                break
            # "Error in metric" 是 RandomCoordinate 采样落到零方差区域导致的**随机**失败，
            # 重试即可（已实测：同一配置重跑可通过）。非随机错误不会在重试中变好。
            print(f'  [重试 {attempt}/{args.retries}] {r.get("error")}', flush=True)
            time.sleep(2)
        r['tag'] = args.tag or 'default'
        r['attempts'] = attempt
        if 'tre_mean' in r:
            print(f'  >> 标准 TRE {r["tre_mean"]:.3f} (SD {r["tre_sd"]:.3f}) '
                  f'| 初始 {r["init_tre"]:.2f} | {r["time_s"]:.0f}s', flush=True)
        else:
            print(f'  !! {r.get("error")}', flush=True)
        res = [x for x in res if x['case'] != cn] + [r]
        json.dump(sorted(res, key=lambda x: x['case']), open(resf, 'w', encoding='utf-8'),
                  indent=2, ensure_ascii=False)
    ok = [r['tre_mean'] for r in res if 'tre_mean' in r]
    if ok:
        print(f'\n已完成 {len(ok)} 例, 均值标准 TRE = {np.mean(ok):.3f} mm')
    print(f'汇总: {resf}')


if __name__ == '__main__':
    main()
