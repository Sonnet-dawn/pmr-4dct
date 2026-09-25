"""tools/verify_determinism_repeats.py —— 量化**确定性口径下的**运行间波动
================================================================================
**为什么必须做这件事**（2026-09-24 发现）：

`docs/17` C59 说"`--cudnn-benchmark 0` 把运行间波动从 ~6% 降到 **0.2%**"，
但那个 0.2% 只来自 **case1 的一次配对**（`detA` 1.13048 vs `detB` 1.13305）。

随后 `det10_best` 与 `ph6_best` 出现了**同一配置、同一确定性开关**的两次运行：
  * 逐字段比对：**超参数一个都没差**（只差 schema 新加的溯源字段）；
  * 两边日志都写了 `[cudnn] benchmark = False`；
  * 但 **case8 是 3.511 vs 3.331（差 5.1%）**，十例均值 1.5975 vs 1.5899。

⇒ "0.2%"**不能外推到所有病例**。本脚本把同配置、确定性口径的运行**分组**，
逐例给出均值/SD/极差，并据此判定：哪些差异**小到不可解释**。

分组规则（**只比较超参数**，schema 缺失不算差异 —— 见 `docs/44` T-6）：
  * 同一 `res_reg_scale`；同一 `mask`/`metric`/`res_metric`/`norm`/`reg_scale`；
  * 同一 `iters1/iters2/res_iters`、`lr1`、`phases_per_step`、`enc_down`、`res_down`、
    `win`、`ncc_stride`、`K`、`local_mask_mode`、`affine_first_iters`、`jac_weight`；
  * **且**确定性可由结果字段或日志坐实。

用法:
  python tools/verify_determinism_repeats.py
  python tools/verify_determinism_repeats.py --group           # 只看分组
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
import glob
import json
import argparse
import statistics as st

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_NEED_DIR = os.path.join(HERE, *'results/pmr_v2'.split("/"))

# --- SUITE-SKIP guard (injected by tools/add_suite_skip_guard.py) ---------------
# 本检查需要 `results/` 里的历史产物。仓库发行版**不附带**结果文件，因此在新 clone 里
# 它无法执行。这里**显式声明"跳过"而不是悄悄通过**：`run_verification_suite.py`
# 见到 `SUITE-SKIP:` 会记为 skip 状态，且**不计入通过数**。
if not os.path.isdir(_NEED_DIR):
    print('SUITE-SKIP: 缺少 results/pmr_v2 —— 本项需要历史结果文件，仓库发行版不附带；'
          '在开发树（含 results/）中运行同一入口即可完整执行。', flush=True)
    sys.exit(0)
# --- end SUITE-SKIP guard ------------------------------------------------------

R = os.path.join(HERE, 'results', 'pmr_v2')
LOGD = os.path.join(R, 'logs')
CASES = list(range(1, 11))

# 判定"同一配置"要比的超参（顺序无关）
HYP = ['dataset', 'down_mm', 'mask', 'metric', 'res_metric', 'norm', 'reg_scale',
       'iters1', 'iters2', 'res_iters', 'lr1', 'lr_decay', 'phases_per_step',
       'enc_down', 'res_down', 'win', 'ncc_stride', 'K', 'local_mask_mode',
       'affine_first_iters', 'res_reg_scale', 'jac_weight', 'init_coef']

# 旧结果文件里"字段不存在"（get → None）与"取某个默认值"在**下列字段上等价**。
# 每一条都给理由 —— 这不是猜测：
NONE_EQ = {
    # `--lr-decay` 2026-09-23 才加入，默认 'none'。更早的运行**不可能**用过
    # cosine/linear（当时参数不存在）⇒ 缺字段 ≡ 'none'。
    'lr_decay': 'none',
    # `--dataset` 默认 'dirlab'；CREATIS 的结果写在 `results/pmr_creatis/` 且
    # caseid 是 `cr{n}`，**不会**出现在 `results/pmr_v2/*_case{k}_2mm.json` 里。
    # 所以本目录下缺该字段的旧文件必然是 DIR-LAB。
    'dataset': 'dirlab',
}
# 缺字段**无法**归一的字段：出现在"未记录"清单里，分组时用占位符，
# 并在输出中明确标注"这一项无法核验"。
UNREC = '<未记录>'


def signature(run):
    """超参签名 + 未记录字段清单（用于事后如实标注）。"""
    j = run[1]
    sig, unrec = {}, []
    for k in HYP:
        v = j.get(k)
        if v is None and k in NONE_EQ:
            v = NONE_EQ[k]
        if v is None:
            sig[k] = UNREC
            unrec.append(k)
        else:
            sig[k] = v
    return json.dumps(sig, sort_keys=True), unrec


def load_runs():
    """返回 {tag: {case: json}}，只保留 DIR-LAB 2 mm。"""
    tags = {}
    for p in sorted(glob.glob(os.path.join(R, '*_case*_2mm.json'))):
        b = os.path.basename(p)
        m = re.match(r'(.+)_case(\d+)_2mm\.json$', b)
        if not m:
            continue
        tag, c = m.group(1), int(m.group(2))
        j = json.load(open(p, encoding='utf-8'))
        if j.get('dataset', 'dirlab') != 'dirlab':
            continue
        tags.setdefault(tag, {})[c] = j
    return tags


def log_deterministic(tag):
    """从日志里找 `[cudnn] benchmark = ...`；找不到返回 None。

    🔴 日志文件名用的是 **sweep 的 tag**，不是"tag_variant"。例如结果文件
    `det10_best_case1_2mm.json` 对应日志 `det10_w0.log`（`det10` 是 sweep tag，
    `best` 是 variant）。第一版只按结果文件名去找，于是把 `det10_best` 判成
    "无法核验"——而它的确定性其实写在日志里。所以这里要**由长到短**试前缀。
    """
    cands = [tag]
    parts = tag.split('_')
    for i in range(len(parts) - 1, 0, -1):
        cands.append('_'.join(parts[:i]))
    for pre in cands:
        for pat in (f'{pre}_w*.log', f'{pre}.log', f'{pre}_*.log'):
            for fn in sorted(glob.glob(os.path.join(LOGD, pat))):
                try:
                    txt = open(fn, encoding='utf-8', errors='replace').read()
                except Exception:
                    continue
                m = re.search(r'\[cudnn\] benchmark = (\w+)', txt)
                if m:
                    return m.group(1) == 'True'
    return None


def is_deterministic(tag, run):
    j = run.get(1, {})
    if 'cudnn_benchmark' in j:
        return bool(j['cudnn_benchmark']) is False, '结果字段'
    b = log_deterministic(tag)
    if b is None:
        return None, '无法核验'
    return (b is False), '日志'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', action='store_true', help='只看分组')
    args = ap.parse_args()

    tags = load_runs()
    print('=' * 96)
    print(f'扫描到 {len(tags)} 个 tag（DIR-LAB 2 mm）')
    print('=' * 96)

    # 1) 确定性核验
    det, undet, unknown = [], [], []
    for tag, run in sorted(tags.items()):
        d, how = is_deterministic(tag, run)
        (det if d is True else (undet if d is False else unknown)).append((tag, how))
    print(f'  确定性可核验（benchmark=False）：{len(det)} 个')
    print(f'  明确非确定性（benchmark=True）：{len(undet)} 个 → '
          f'{[t for t, _ in undet][:12]}')
    print(f'  无法核验：{len(unknown)} 个 → {[t for t, _ in unknown][:12]}')

    # 2) 按超参签名分组（只保留确定性的，且 10 例齐全的）
    groups, unrec_of = {}, {}
    for tag, how in det:
        run = tags[tag]
        if len(run) != 10:
            continue
        key, unrec = signature(run)
        groups.setdefault(key, []).append(tag)
        unrec_of[tag] = unrec

    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f'\n  确定性 + 10 例齐全的运行：{sum(len(v) for v in groups.values())} 个，'
          f'分成 {len(groups)} 组配置')
    print(f'  其中**有重复**的配置组：{len(multi)} 组')

    if args.group or not multi:
        for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            cfg = json.loads(k)
            print(f'    n={len(v):2d}  {v}')
        if not multi:
            print('\n⚠️ 目前还没有"同一配置的确定性重复运行"，无法量化波动。')
        return

    print('\n' + '=' * 96)
    print('确定性口径下、**同一配置**重复运行的逐例波动')
    print('=' * 96)
    worst = []
    for k, v in sorted(multi.items(), key=lambda kv: -len(kv[1])):
        cfg = json.loads(k)
        print(f'\n▶ 配置（关键项）：res_reg={cfg.get("res_reg_scale")} '
              f'mask={cfg.get("mask")} metric={cfg.get("metric")} '
              f'init_coef={cfg.get("init_coef")}  |  运行：{v}')
        for t in v:
            if unrec_of.get(t):
                print(f'   ⚠️ {t} 有 {len(unrec_of[t])} 个超参**未记录**'
                      f'（{unrec_of[t]}）—— 分组基于其余字段，这些项无法核验')
        print(f'  {"case":>4} ' + ' '.join(f'{t[:10]:>10}' for t in v)
              + f' {"均值":>9} {"SD":>8} {"极差%":>8}')
        per_case_range = {}
        for c in CASES:
            vals = [tags[t][c]['tre_total_STANDARD'] for t in v]
            mean = st.mean(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            rng = max(vals) - min(vals)
            pct = 100 * rng / mean
            per_case_range[c] = pct
            print(f'  {c:>4} ' + ' '.join(f'{x:>10.4f}' for x in vals)
                  + f' {mean:>9.4f} {sd:>8.4f} {pct:>7.1f}%')
        means = [st.mean([tags[t][c]['tre_total_STANDARD'] for t in v]) for c in CASES]
        print(f'  十例均值：' + ' / '.join(
            f'{st.mean([tags[t][c]["tre_total_STANDARD"] for c in CASES]):.4f}' for t in v)
            + f'   ⇒ 运行间极差 {100*abs(st.mean([tags[t][c]["tre_total_STANDARD"] for c in CASES for t in [v[0]]]) - st.mean([tags[t][c]["tre_total_STANDARD"] for c in CASES for t in [v[-1]]]))/st.mean(means):.2f}%')
        worst.append((max(per_case_range.values()), v, per_case_range))

    print('\n' + '=' * 96)
    print('结论')
    print('=' * 96)
    for mx, v, pcr in worst:
        top = sorted(pcr.items(), key=lambda kv: -kv[1])[:3]
        print(f'  {v}: 最大逐例极差 **{mx:.1f}%**（'
              + ', '.join(f'case{c} {p:.1f}%' for c, p in top) + '）')
    print('\n  ⇒ **确定性口径不等于"逐例可复现到 0.2%"**：')
    print('     · 小位移病例（case1 等）确实在 ~0.2%；')
    print('     · 但最难的 case8 在两次**完全相同**的确定性运行之间可差 **5%**。')
    print('     ⇒ 因此："确定性口径下 <X% 的差异是否可判定"必须**逐例**说，')
    print('        不能用一个全局阈值；case8 上低于 ~5% 的差异**不可作为结论依据**。')

    # ------------------------------------------------------------------ 非 10 例齐全的运行
    # 🔴 2026-09-25 新增：`repdA/repdB/repdC`（队列 2 的确定性重复运行）只跑了 case 1/5/8，
    #    上面那段因"必须 10 例齐全"把它们**整批排除**了 —— 于是最有价值的一批重复数据
    #    没被用上。"例数不全"不等于"不能比"：**在交集病例上比**是合法的，只要写明是交集。
    sub = {}
    # 🔴 **必须按 tag 排除多尺度管线**：`ms4to2_*` 的结果 JSON **没有**任何字段区分
    #    它与单尺度主配置（`tools/probe_signature_gap.py` 实测：键集相同、取值只是结果不同），
    #    于是按签名分组会把"多尺度"和"单尺度"当成同一配置，算出 **86.55%** 的假"运行间极差"。
    #    这是"检查器自己出错"的又一例：**签名不完整 ⇒ 分组错误 ⇒ 数字荒谬**。
    #    处置：显式按 tag 排除，并把原因写在这里而不是让读者去猜。
    EXCLUDE = re.compile(r'^(ms|.*coarse)', re.I)
    for tag, how in det:
        run = tags[tag]
        if EXCLUDE.match(tag):
            continue
        if 3 <= len(run) < 10:
            key, _ = signature(run)
            sub.setdefault(key, []).append(tag)
    sub_multi = {k: v for k, v in sub.items() if len(v) >= 2}

    print('\n' + '=' * 96)
    print('确定性口径下、**病例子集**上的重复运行（例如只跑了 case 1/5/8 的那几次）')
    print('=' * 96)
    if not sub_multi:
        print('  （还没有这样的运行）')
    else:
        for k, v in sorted(sub_multi.items(), key=lambda kv: -len(kv[1])):
            shared = sorted(set.intersection(*[set(tags[t]) for t in v]))
            if len(shared) < 2:
                continue
            print(f'\n▶ 运行：{v}   共同病例：{shared}')
            print(f'  {"case":>4} ' + ' '.join(f'{t[:12]:>12}' for t in v)
                  + f' {"极差%":>8}')
            mx = 0.0
            for c in shared:
                vals = [tags[t][c]['tre_total_STANDARD'] for t in v]
                pct = 100 * (max(vals) - min(vals)) / st.mean(vals)
                mx = max(mx, pct)
                print(f'  {c:>4} ' + ' '.join(f'{x:>12.4f}' for x in vals)
                      + f' {pct:>7.2f}%')
            print(f'  ⇒ 交集上的最大逐例极差 **{mx:.2f}%**'
                  f'（⚠️ 只在 {shared} 上可比，不得外推到十例）')


if __name__ == '__main__':
    main()
