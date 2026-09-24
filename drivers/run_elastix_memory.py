"""
run_elastix_memory.py —— 实测 Elastix 的**峰值内存 / 耗时 / 控制点数量**随 B 样条网格细化的变化
================================================================================
为什么必须做（见 `paperB/07_known_gaps.md` 硬伤 H1）
----------------------------------------------------
Paper B 的核心主张是「主流成对工具按粗网格调优，**所以在 8 GiB 卡上做不到 1 mm 全场配准**」。
前半句有实测（Elastix 默认 `GridSpacing 8 8 8`），**后半句一次都没测过**。
本脚本把后半句从**断言**变成**实测曲线**。

测什么
------
对固定的 B 样条网格间距 g ∈ {8, 4, 2, 1} mm（`FinalGridSpacingInPhysicalUnits`）：
  * **峰值常驻内存 RSS**（psutil 轮询整个进程树，含子进程）
  * **墙钟时间**
  * **退出码 / 是否跑完**
  * **变换文件体积**与**B 样条控制点数量**（直接解析 `TransformParameters.0.txt`）

控制点数量是关键证据：它随 1/g³ 增长，是"内存为什么爆"的机制性解释，
    8 mm: ~31×31×29  ≈ 2.8e4 个控制点
    1 mm: ~248×248×235 ≈ 1.45e7 个控制点   （≈519× 更多）

🔴 资源纪律（`docs/29`）
  * **一次只跑一个 elastix**（它会吃很多 RAM）
  * 每个组合启动前检查可用 RAM，不足 `--need-gb` 则等待；等待超时则跳过并如实记为"未测"
  * 运行中若可用 RAM 跌破 `--abort-gb`，**立刻杀掉**，记为 `aborted_low_memory`
    —— 绝不让它把机器拖死

用法
----
    python run_elastix_memory.py --cases 1 --grids 8,4,2
    python run_elastix_memory.py --cases 1,5 --grids 8,4,2,1
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import os
import re
import sys
import json
import time
import argparse
import subprocess
import threading

import psutil

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.environ.get('PMR_DATA_ROOT')
                    or os.path.join(HERE, '..', 'reference_4', 'data'), 'DIRLAB')
OUTD = os.path.join(HERE, 'results', 'elastix_memory')
LOGD = os.path.join(OUTD, 'logs')
os.makedirs(LOGD, exist_ok=True)


def find_elastix():
    for root in (os.path.join(HERE, 'tools', 'elastix'), os.path.join(HERE, 'tools'), r'D:\elastix'):
        if not os.path.isdir(root):
            continue
        for dp, _, files in os.walk(root):
            if 'elastix.exe' in files:
                return os.path.join(dp, 'elastix.exe')
    return None


def param_text(grid_mm, iters=1500, samples=4096,
               metric_name='AdvancedNormalizedCorrelation'):
    """以「显式指定 B 样条网格间距」为核心的配置，用于网格细化扫描。

    🔴 与 `run_elastix_lit.py` 的 kanai2 配置不同：这里 **NumberOfResolutions=1** 且
    显式给出 `FinalGridSpacingInPhysicalUnits`，因此网格间距**完全由我们控制**，
    不受金字塔调度影响 —— 这是做"网格细化"实验的必要条件。
    """
    return f"""// Grid-refinement sweep: explicit final B-spline grid spacing
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingImageDimension 3)
// 🔴 此构建**未安装** `SingleResolutionRegistration`（实测报
//    "This component is not installed"）。用 MultiResolutionRegistration +
//    NumberOfResolutions 1 达到同样效果：只有一层，且该层网格由我们显式指定。
(Registration "MultiResolutionRegistration")
(Interpolator "LinearInterpolator")
(ResampleInterpolator "FinalLinearInterpolator")
(Resampler "DefaultResampler")
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(Optimizer "AdaptiveStochasticGradientDescent")
(MaximumNumberOfIterations {iters})
(NumberOfSpatialSamples {samples})
(NewSamplesEveryIteration "true")
(ImageSampler "RandomCoordinate")
(Metric "{metric_name}")
(Transform "BSplineTransform")
(NumberOfResolutions 1)
(FinalGridSpacingInPhysicalUnits {grid_mm} {grid_mm} {grid_mm})
(GridSpacingSchedule 1.0 1.0 1.0)
(HowToCombineTransforms "Compose")
(DefaultPixelValue 0)
(WriteResultImage "false")
"""


class RssWatcher(threading.Thread):
    """轮询**整棵进程树**的 RSS，记录峰值；可用内存过低时置 abort 标志。"""

    def __init__(self, pid, abort_gb, interval=0.25):
        super().__init__(daemon=True)
        self.pid = pid
        self.abort_gb = abort_gb
        self.interval = interval
        self.peak = 0
        self.peak_avail_drop = None
        self.abort = False
        self._stop = False

    def run(self):
        try:
            proc = psutil.Process(self.pid)
        except Exception:
            return
        base_avail = psutil.virtual_memory().available
        while not self._stop:
            try:
                total = proc.memory_info().rss
                for ch in proc.children(recursive=True):
                    try:
                        total += ch.memory_info().rss
                    except Exception:
                        pass
                self.peak = max(self.peak, total)
                avail = psutil.virtual_memory().available
                drop = (base_avail - avail) / 1024 ** 3
                if self.peak_avail_drop is None or drop > self.peak_avail_drop:
                    self.peak_avail_drop = drop
                if avail / 1024 ** 3 < self.abort_gb:
                    self.abort = True
                    return
            except Exception:
                return
            time.sleep(self.interval)

    def stop(self):
        self._stop = True


def parse_transform(tf_path):
    """从 elastix 的 TransformParameters txt 里读出**控制网格尺寸**与**参数总数**。

    🔴 实测（tools/diag_tf.py）：elastix 5.3.1 写出的变换文件里**没有**
    `BSplineTransformCoefficients` 段（那个名字存在于别的版本/格式）。
    可靠可用的字段是：
      * `(GridSize  a b c)`        ← B 样条控制网格的维度，这正是"控制点爆炸"的直接度量
      * `(NumberOfParameters  n)`  ← 参数总数（= 3 × 控制点数）
    两者都远比"数系数"稳健。
    """
    try:
        s = open(tf_path, encoding='utf-8', errors='replace').read()
    except Exception:
        return {}
    out = {}
    m = re.search(r'\(GridSize\s+(\d+)\s+(\d+)\s+(\d+)\s*\)', s)
    if m:
        g = [int(x) for x in m.groups()]
        out['grid_size'] = g
        out['n_control_points'] = g[0] * g[1] * g[2]
    m = re.search(r'\(NumberOfParameters\s+(\d+)\s*\)', s)
    if m:
        out['n_parameters'] = int(m.group(1))
    m = re.search(r'\(GridSpacing\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\)', s)
    if m:
        out['grid_spacing_actual_mm'] = [round(float(x), 4) for x in m.groups()]
    out['transform_file_bytes'] = os.path.getsize(tf_path)
    return out


def wait_ram(need_gb, minutes):
    for i in range(int(minutes * 60 / 15)):
        avail = psutil.virtual_memory().available / 1024 ** 3
        if avail >= need_gb:
            return avail
        print(f'    [ram-watch] 可用 {avail:.1f} GiB < 需要 {need_gb} GiB，等待 15 s', flush=True)
        time.sleep(15)
    return None


def run_one(exe, cn, grid_mm, iters, samples, need_gb, abort_gb, keep, max_min,
            suffix='', wait_min=60):
    tag = f'case{cn}_grid{grid_mm}mm{suffix}'
    jf = os.path.join(OUTD, f'{tag}.json')
    if os.path.exists(jf):
        print(f'  [skip] {tag} 已完成', flush=True)
        return json.load(open(jf, encoding='utf-8'))

    avail = wait_ram(need_gb, minutes=wait_min)
    if avail is None:
        # 🔴 刻意**不写**结果文件：这一行只是"这次没等到内存"，
        #    不是一次测量。若写成 json，下次运行会被上面的 skip 当成"已完成"，
        #    从此再也测不到 —— 而"等不到内存"和"测出来内存不够"是两件完全不同的事。
        print(f'  [skip] {tag}: 等待内存超时（如实记为"未测"）', flush=True)
        return {'case': cn, 'grid_mm': grid_mm, 'status': 'not_measured_low_memory'}

    run_dir = os.path.join(OUTD, tag)
    os.makedirs(run_dir, exist_ok=True)
    pf = os.path.join(run_dir, 'params.txt')
    open(pf, 'w', encoding='utf-8').write(param_text(grid_mm, iters, samples))

    f = os.path.join(DATA, 'mha', f'case{cn}', f'case{cn}_T00_R.mha')
    m = os.path.join(DATA, 'mha', f'case{cn}', f'case{cn}_T50_R.mha')

    print(f'\n=== {tag}  (网格 {grid_mm} mm, iter {iters}, samples {samples})'
          f' 可用内存 {avail:.1f} GiB ===', flush=True)
    log = open(os.path.join(LOGD, f'{tag}.log'), 'w', encoding='utf-8')
    t0 = time.time()
    p = subprocess.Popen([exe, '-f', f, '-m', m, '-p', pf, '-out', run_dir],
                         stdout=log, stderr=subprocess.STDOUT)
    w = RssWatcher(p.pid, abort_gb)
    w.start()
    status = 'ok'
    rc = None
    t_cap = max_min * 60 if max_min > 0 else None
    while True:
        try:
            rc = p.wait(timeout=1)
            break
        except subprocess.TimeoutExpired:
            if w.abort:
                print(f'    🔴 可用内存跌破 {abort_gb} GiB ⇒ 杀进程（避免拖死机器）', flush=True)
                status = 'aborted_low_memory'
                break
            if t_cap is not None and (time.time() - t0) > t_cap:
                print(f'    ⏱  超过墙钟上限 {max_min} min ⇒ 终止（记为未在限时内完成）', flush=True)
                status = 'timeout'
                break
    if status in ('aborted_low_memory', 'timeout'):
        try:
            for ch in psutil.Process(p.pid).children(recursive=True):
                ch.kill()
        except Exception:
            pass
        p.kill()
        rc = p.wait()
    w.stop()
    dt = time.time() - t0
    log.close()

    tf = os.path.join(run_dir, 'TransformParameters.0.txt')
    tinfo = parse_transform(tf) if os.path.exists(tf) else {}
    if rc != 0 and status == 'ok':
        status = 'failed'

    res = {'case': cn, 'grid_mm': grid_mm, 'iters': iters, 'samples': samples,
           'status': status, 'rc': rc,
           'peak_rss_gb': round(w.peak / 1024 ** 3, 3),
           'peak_ram_drop_gb': round(w.peak_avail_drop or 0, 3),
           'time_s': round(dt, 1)}
    res.update(tinfo)
    json.dump(res, open(jf, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    if not keep:
        import shutil
        shutil.rmtree(run_dir, ignore_errors=True)

    cp = res.get('n_control_points')
    gs = res.get('grid_size')
    print(f'    → {status}  rc={rc}  峰值 RSS {res["peak_rss_gb"]:.2f} GiB  '
          f'{dt:.0f} s  控制网格 {gs}  控制点 {cp if cp is None else format(cp, ",")}'
          f'  参数 {res.get("n_parameters", "-")}', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=str, default='1')
    ap.add_argument('--grids', type=str, default='8,4,2',
                    help='B 样条网格间距(mm)，逗号分隔。1 mm ≈ 每体素一个控制点，很吃内存')
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--samples', type=int, default=4096)
    ap.add_argument('--need-gb', type=float, default=6.0, help='启动前要求的最小可用内存')
    ap.add_argument('--abort-gb', type=float, default=1.5, help='运行中可用内存跌破此值就杀进程')
    ap.add_argument('--keep', type=int, default=0, help='1=保留变换文件')
    ap.add_argument('--max-min', type=float, default=90.0,
                    help='单个组合的墙钟上限(分钟)。超时即终止并记为 timeout '
                         '—— "在限时内跑不完"本身就是有用证据')
    ap.add_argument('--tag', type=str, default='')
    ap.add_argument('--suffix', type=str, default='',
                    help='逐例结果文件名后缀。用途：**在更空闲的机器上重测**而不覆盖旧记录。'
                         '旧记录本身就是"当时机器状态"的证据，不能删。'
                         '例：--suffix _recheck ⇒ case1_grid2.0mm_recheck.json')
    ap.add_argument('--wait-min', type=float, default=60.0,
                    help='启动前等内存的最长等待（分钟）。默认 60；'
                         '**排在别的长任务后面时把它调大**（例如 600），'
                         '否则会在一台"整晚都忙"的机器上白白把每个组合都判成"未测"。')
    args = ap.parse_args()

    exe = find_elastix()
    if not exe:
        sys.exit('未找到 elastix.exe')
    print(f'elastix: {exe}')
    r = subprocess.run([exe, '--version'], capture_output=True, text=True, timeout=120)
    print(f'  {(r.stdout + r.stderr).strip().splitlines()[0]}')

    cases = [int(c) for c in args.cases.split(',') if c.strip()]
    grids = [float(g) for g in args.grids.split(',') if g.strip()]
    print(f'病例 {cases} | 网格 {grids} mm | 启动需 ≥{args.need_gb} GiB | '
          f'跌破 {args.abort_gb} GiB 就杀')

    rows = []
    for cn in cases:
        for g in grids:
            rows.append(run_one(exe, cn, g, args.iters, args.samples,
                                args.need_gb, args.abort_gb, bool(args.keep),
                                args.max_min, args.suffix, args.wait_min))

    sf = os.path.join(OUTD, f'summary{("_" + args.tag) if args.tag else ""}.json')
    json.dump(rows, open(sf, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n已保存 {sf}')

    print('\n' + '=' * 74)
    print(f'{"case":>5s}{"grid_mm":>9s}{"状态":>22s}{"峰值RSS(GiB)":>14s}'
          f'{"时间(s)":>9s}{"控制点":>14s}')
    for r in rows:
        cp = r.get('n_control_points')
        print(f'{r["case"]:>5d}{r["grid_mm"]:>9.1f}{r.get("status",""):>22s}'
              f'{r.get("peak_rss_gb", 0):>14.2f}{r.get("time_s", 0):>9.0f}'
              f'{(format(cp, ",") if cp else "-"):>14s}')
    print('=' * 74)


if __name__ == '__main__':
    main()
