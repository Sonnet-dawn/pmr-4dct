"""
run_dl_baselines.py —— 批量跑深度学习基线（留一法 10 折 × 2 个模型）
================================================================================
调 `dl_baseline.py`，串行执行，带 GPU 显存看门狗。

为什么是留一法
--------------
DIRLAB 只有 10 例、CREATIS 只有 6 例。把某例留作测试、其余例训练，是这两个数据集上
深度学习配准的标准做法，也是唯一能给出**逐例可比 TRE** 的协议（与 PMR 的口径一致）。
**绝不能"在全部病例上训练再在同样病例上评测"。**

两个数据集
----------
`--dataset dirlab`（病例 1–10，训 9）与 `--dataset creatis`（病例 0–5，训 5）。
CREATIS 每例更大（最大 14.5 Mvox @2 mm vs DIRLAB 约 10），更吃内存。
**文件名模式不同**（`fold{k}` vs `cr{k}`），以免两个数据集的折混进主表同一行。

用法
----
    python run_dl_baselines.py --models voxelmorph,lapirn --down 2 --epochs 200
    python run_dl_baselines.py --models voxelmorph --folds 1,2       # 先试折
    python run_dl_baselines.py --dataset creatis --models voxelmorph,lapirn
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import os
import sys
import json
import time
import argparse
import subprocess

import psutil

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
PY = r'python'
OUTD = os.path.join(HERE, 'results', 'dl_baselines')
LOGD = os.path.join(OUTD, 'logs')
os.makedirs(LOGD, exist_ok=True)


def gpu_free_gb():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                            '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=20)
        u, t = [float(x) for x in r.stdout.strip().split(',')[:2]]
        return (t - u) / 1024.0
    except Exception:
        return 99.0


def wait_gpu(need_gb, minutes):
    for _ in range(int(minutes * 60 / 20)):
        f = gpu_free_gb()
        if f >= need_gb:
            return f
        print(f'  [gpu-watch] 可用 {f:.1f} < {need_gb} GiB，等 20 s', flush=True)
        time.sleep(20)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['dirlab', 'creatis'], default='dirlab')
    ap.add_argument('--models', type=str, default='voxelmorph,lapirn')
    ap.add_argument('--folds', type=str, default='',
                    help='留空则按数据集取默认：dirlab=1..10，creatis=0..5')
    ap.add_argument('--down', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--iters-per-epoch', type=int, default=25,
                    help='每个 epoch 的迭代数；总迭代 = epochs × iters-per-epoch')
    ap.add_argument('--patch', type=int, default=96)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--need-gb', type=float, default=3.0,
                    help='每折启动前要求的最小可用显存')
    ap.add_argument('--wait-min', type=float, default=180.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default='',
                    help='结果文件名后缀（如 ep2400），用于**不覆盖**地比较训练预算')
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    default_folds = list(range(1, 11)) if args.dataset == 'dirlab' else list(range(0, 6))
    folds = ([int(f) for f in args.folds.split(',') if f.strip()]
             if args.folds.strip() else default_folds)
    # 🔴 防静默：病例号必须落在该数据集的编号空间内，否则会跑去读不存在的文件
    bad = [f for f in folds if f not in default_folds]
    if bad:
        sys.exit(f'🔴 {args.dataset} 的病例编号是 {default_folds}，收到越界的 {bad}')
    stem = (lambda k: f'fold{k}') if args.dataset == 'dirlab' else (lambda k: f'cr{k}')
    print(f'DL 基线：数据集 {args.dataset} | 模型 {models} | 折 {folds} | {args.down}mm | '
          f'{args.epochs} epochs × {args.iters_per_epoch} iters '
          f'= {args.epochs * args.iters_per_epoch} 次迭代 | patch {args.patch}³ | λ={args.lam}')
    print(f'每折启动前需 ≥{args.need_gb} GiB 显存；最多等 {args.wait_min} min\n')

    rows = []
    for model in models:
        for fold in folds:
            jf = os.path.join(OUTD, f'{model}_{stem(fold)}_{args.down}mm'
                                     f'{("_" + args.tag) if args.tag else ""}.json')
            if os.path.exists(jf):
                d = json.load(open(jf, encoding='utf-8'))
                print(f'[skip] {args.dataset} {model} {stem(fold)}: '
                      f'已有 TRE {d.get("tre_total_STANDARD")}', flush=True)
                rows.append(d)
                continue

            free = wait_gpu(args.need_gb, args.wait_min)
            if free is None:
                print(f'[abort] {model} {stem(fold)}: 显存等待超时', flush=True)
                continue

            print(f'=== {args.dataset} {model} {stem(fold)}'
                  f'（可用显存 {free:.1f} GiB）===', flush=True)
            cmd = [PY, os.path.join(HERE, 'dl_baseline.py'),
                   '--model', model, '--dataset', args.dataset, '--fold', str(fold),
                   '--down', str(args.down),
                   '--epochs', str(args.epochs), '--patch', str(args.patch),
                   '--iters-per-epoch', str(args.iters_per_epoch),
                   '--lr', str(args.lr), '--lam', str(args.lam),
                   '--seed', str(args.seed), '--log-every', '25']
            if args.tag:
                cmd += ['--tag', args.tag]
            log = open(os.path.join(LOGD, f'{model}_{args.dataset}_{stem(fold)}.log'),
                       'w', encoding='utf-8')
            log.write('$ ' + ' '.join(cmd) + '\n')
            log.flush()
            t0 = time.time()
            r = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONIOENCODING='utf-8',
                                        PYTHONUNBUFFERED='1'))
            log.write(f'\n[rc={r.returncode} {time.time()-t0:.0f}s]\n')
            log.close()
            if os.path.exists(jf):
                d = json.load(open(jf, encoding='utf-8'))
                rows.append(d)
                # ⚠️ 键名必须容错：本行原先写死 `d["time_s"]`，而 `dl_baseline.py` 写的是
                # `train_time_s` ⇒ fold1 训练成功、JSON 已落盘，**驱动却在打印时崩溃**，
                # 20 折只跑完 1 折且退出码为 1。读结果文件的代码不允许因缺字段而中断整批实验。
                _t = d.get('train_time_s', d.get('time_s'))
                print(f'  >> {stem(fold)} 初始 {d["init_tre"]:.2f} → '
                      f'TRE {d["tre_total_STANDARD"]:.3f} mm | '
                      f'{(f"{_t:.0f}s" if _t is not None else "time?")} | '
                      f'峰值 {d.get("train_peak_gb", "?")} GiB', flush=True)
            else:
                print(f'  !! 失败 rc={r.returncode}（见 {LOGD}）', flush=True)

    # 汇总（按模型分组）。
    # ⚠️ 这里打印的是**本批已完成**的均值，**不构成可引用的整体结果**：
    #    `tools/summarize_dl_baselines.py` 才负责"只在 N/N 折齐全时给均值"。
    print('\n' + '=' * 66)
    for model in models:
        v = [d['tre_total_STANDARD'] for d in rows
             if d.get('model') == model and d.get('down_mm') == args.down
             and d.get('dataset', args.dataset) == args.dataset]
        if v:
            print(f'  [{args.dataset}] {model:12s} n={len(v):2d}  本批均值 '
                  f'{sum(v)/len(v):.4f} mm  逐例 {[round(x,3) for x in sorted(v)]}')
    print('=' * 66)
    json.dump(rows, open(os.path.join(OUTD, f'summary_{args.dataset}_{args.down}mm.json'),
                         'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n已保存 {os.path.join(OUTD, f"summary_{args.dataset}_{args.down}mm.json")}')
    # 老名字（DIRLAB）保留一份，免得既有引用断掉
    if args.dataset == 'dirlab':
        json.dump(rows, open(os.path.join(OUTD, f'summary_{args.down}mm.json'), 'w',
                             encoding='utf-8'), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
