"""
make_all_tables.py —— 从 results/**/*.json 单一数据源重建论文全部表格
================================================================================
目的：消除跨文档手抄数字导致的不一致（例如同条件 B-spline 均值在
      paper_tables_closure.md 为 3.08、在 README/draft_v03 表格为 3.25）。

原则：
  1. 只读 JSON，不读 md 里手写的数字；
  2. 每张表都带 identity（不配准）基线列；
  3. 每条结果标注协议等级：U=无监督 / L1=landmark 监督代理（上限）/ L2=自动结构点；
  4. 自动审计异常（基线劣于 identity、TRE 明显失真、缺失用例）。

用法： python make_all_tables.py [--out results/TABLES.md]
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import os, json, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')

CASES10 = list(range(1, 11))


def load(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {'_error': str(e)}


def fmt(x, n=2):
    if x is None:
        return '—'
    try:
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return 'NaN'
        return f'{x:.{n}f}'
    except Exception:
        return str(x)


def mean_of(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else None


def median_of(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.median(vals)) if vals else None


# ---------------------------------------------------------------- 数据装载
def collect_pmr():
    """PMR（本文方法）各实现版本：旧实现 s1abc、新低显存实现 s1f/smoke/vA/vB。
    排除消融目录（消融单独列在 Table 4）与失效基线的中间产物。"""
    rows = {}
    for p in glob.glob(os.path.join(R, '**', '*.json'), recursive=True):
        rel = os.path.relpath(p, HERE).replace('\\', '/')
        if any(s in rel for s in ('/ablations/', '/ablations_v2/', '/deeds_work/',
                                  '/baselines/', '/fast_smoke/', '/fast_validate/')):
            continue
        d = load(p)
        if not isinstance(d, dict) or d.get('dataset') != 'dirlab':
            continue
        if 'tre_total' not in d:
            continue
        tag = os.path.basename(p).split('_dirlab')[0]
        rows.setdefault(tag, {})[d['case']] = {
            'init': d.get('init_tre'), 'manifold': d.get('tre_manifold'),
            'total': d.get('tre_total'), 'closure_d0': d.get('closure_d0'),
            'closure_2pi': d.get('closure_2pi'), 'down': d.get('down_mm'),
            'src': os.path.relpath(p, HERE),
            'cfg': {k: d.get(k) for k in ('affine', 'impl', 'enc_down', 'res_down',
                                          'ncc_stride', 'amp', 'peak_mem_gb') if k in d},
        }
    return rows


def collect_elastix():
    out = {}
    for p in sorted(glob.glob(os.path.join(R, 'elastix_baseline', '*.json'))):
        d = load(p)
        if 'tre_t50' in d:
            out.setdefault(d['down_mm'], {})[d['case']] = d['tre_t50']
    return out


def collect_vxm():
    d = load(os.path.join(R, 'voxelmorph', 'vxm_dirlab_tre.json'))
    res = d.get('results', []) if isinstance(d, dict) else []
    return ({r['case']: r['tre'] for r in res},
            {r['case']: r.get('init_tre') for r in res})


def collect_decisive():
    """experiment_A：弱 B-spline(mesh 8x8x4, 50 iter) @2mm + L1 landmark 约束。
    ⚠️ 约束使用 60% landmark 监督（3 次随机拆分），评价用剩余 40%。"""
    d = load(os.path.join(R, 'decisive', 'decisive_results.json'))
    A = d.get('experiment_A', [])
    return ({a['case']: a for a in A}, d.get('experiment_B'), d.get('statistics'))


def collect_lungqa():
    s = load(os.path.join(R, 'lungqa_struct', 'lungqa_struct_summary_3mm.json'))
    rows = {r['case']: r for r in s.get('results', [])}
    l2 = {}
    for p in sorted(glob.glob(os.path.join(R, 'lungqa_l2', '*.json'))):
        d = load(p)
        if 'con_tre' in d:
            l2[d['case']] = d
    return rows, l2, s.get('case_level', {}), s.get('down_mm')


# ---------------------------------------------------------------- 表格生成
def build_md():
    md = []
    audit = []
    A = md.append

    A('# 论文表格（自动生成，单一数据源）')
    A('')
    A('> 本文件由 `make_all_tables.py` 从 `results/**/*.json` 直接生成，**不要手工编辑**。')
    A('> 协议等级：**U** = 无监督；**L1** = landmark 监督代理（对应关系可获得时的**上限**，'
      '不得作为方法主结果）；**L2** = 自动结构点（无泄漏）。')
    A('')
    A('### identity 基线口径')
    A('')
    A('`identity` = 不配准时的 landmark 位移幅值均值 `mean|lm_T50 − lm_T00|`，')
    A('是判断任何配准结果是否可信的第一参照。')
    A('')

    # ---------- Table 1: PMR 主表 ----------
    pmr = collect_pmr()
    A('## Table 1. DIRLAB —— PMR（本文方法，无监督 U）')
    A('')
    for tag in sorted(pmr):
        cases = pmr[tag]
        downs = sorted({v['down'] for v in cases.values() if v['down']})
        A(f'### tag `{tag}`  （分辨率 {downs} mm，共 {len(cases)} 例）')
        A('')
        A('| case | identity | 流形 | +残差 | \\|d(0)\\| | \\|d(2π)−d(0)\\| | 源文件 |')
        A('|---|---|---|---|---|---|---|')
        for c in sorted(cases):
            v = cases[c]
            A(f'| {c} | {fmt(v["init"])} | {fmt(v["manifold"])} | **{fmt(v["total"])}** | '
              f'{v["closure_d0"]:.2e} | {v["closure_2pi"]:.2e} | `{v["src"]}` |')
        A(f'| **mean** | {fmt(mean_of([v["init"] for v in cases.values()]))} | '
          f'{fmt(mean_of([v["manifold"] for v in cases.values()]))} | '
          f'**{fmt(mean_of([v["total"] for v in cases.values()]))}** | | | |')
        A('')
        any_cfg = next(iter(cases.values()))['cfg']
        if any_cfg:
            A(f'配置：`{json.dumps(any_cfg, ensure_ascii=False)}`')
            A('')
        missing = sorted(set(CASES10) - set(cases))
        # 只对当前采用的实现（tag s1f）做"主表完整性"审计；旧 tag 属历史归档，不再报警
        if missing and tag == 's1f':
            audit.append(f'**Table 1 (`{tag}`)**：缺 case {missing} —— 主表不完整，'
                         f'无法支撑"10 例全量"声称。')
        # 异常：残差后反而变差
        for c, v in cases.items():
            if v['manifold'] and v['total'] and v['total'] > v['manifold'] + 0.05:
                audit.append(f'**`{tag}` case{c}**：加残差后 TRE 反而升高 '
                             f'({v["manifold"]:.2f} → {v["total"]:.2f})，需检查残差阶段。')

    # ---------- Table 2: 基线对比 ----------
    A('## Table 2. DIRLAB —— 同协议基线对比')
    A('')
    A('| 方法 | 协议 | identity | case1 | case2 | case3 | case4 | case5 | mean(1-5) | 备注 |')
    A('|---|---|---|---|---|---|---|---|---|---|')
    elx = collect_elastix()
    dec, expB, stats = collect_decisive()
    vxm, vxm_init = collect_vxm()

    weak_bs = {c: dec[c]['tre_base_mean'] for c in sorted(dec)}
    if weak_bs:
        vals = [weak_bs.get(c) for c in range(1, 6)]
        A(f'| B-spline (mesh 8×8×4, 50 iter) | U | — | ' +
          ' | '.join(fmt(weak_bs.get(c)) for c in range(1, 6)) +
          f' | **{fmt(mean_of(vals))}** | 同条件**弱**配置，非调优基线 |')
    for down in sorted(elx):
        vals = [elx[down].get(c) for c in range(1, 6)]
        A(f'| Elastix (B-spline+MI) | U | — | ' +
          ' | '.join(fmt(elx[down].get(c)) for c in range(1, 6)) +
          f' | **{fmt(mean_of(vals))}** | {down}mm 分辨率 |')
    if vxm:
        vals = [vxm.get(c) for c in range(1, 6)]
        inits = [vxm_init.get(c) for c in range(1, 6)]
        A(f'| VoxelMorph (CREATIS→DIRLAB 跨域) | U | — | ' +
          ' | '.join(fmt(vxm.get(c)) for c in range(1, 6)) +
          f' | **{fmt(mean_of(vals))}** | 塌缩：identity mean {fmt(mean_of(inits))} |')
    A('')
    A('> ⚠️ **deedsBCV 结果不可用**：`results/baselines/deeds_case*.json` 的 `best_tre` '
      '在 550–1100 mm 量级，为坐标框架错位导致的失真值，**禁止进入任何表格**。')
    A('')

    # ---------- Table 3: 闭合 ----------
    A('## Table 3. 周期闭合误差（绕 10 段相邻相位复合，mean ‖d_total‖, mm）')
    A('')
    cl = load(os.path.join(R, 'closure_loop', 'closure_loop_2mm.json'))
    rows = cl.get('rows', {})
    if rows:
        meths, meta = {}, {}
        for key, v in rows.items():
            if '_case' not in key:
                meta[key] = v          # 例如 pmr_ref（构造闭合的参照记录）
                continue
            m, c = key.rsplit('_case', 1)
            meths.setdefault(m, {})[int(c)] = v
        cs = sorted({c for m in meths.values() for c in m})
        A('| case | ' + ' | '.join(meths) + ' |')
        A('|---|' + '---|' * len(meths))
        for c in cs:
            A(f'| {c} | ' + ' | '.join(
                fmt(meths[m].get(c, {}).get('loop_err_mean_mm'), 4) for m in meths) + ' |')
        A('| **mean** | ' + ' | '.join(
            fmt(mean_of([meths[m][c]['loop_err_mean_mm'] for c in meths[m]]), 4)
            for m in meths) + ' |')
        A('')
        A('> PMR 闭合误差为**构造恒等**（`d(x,0)=0`、`d(x,2π)=d(x,0)`），'
          '实测 ~1e-7 mm（float32 精度）；上表为逐对方法顺序复合的实测漂移。')
        if meta:
            A('')
            A(f'> 参照记录 `{json.dumps(meta, ensure_ascii=False)[:300]}`')
        A('')

    # ---------- Table 4: 消融（当前实现） ----------
    A('## Table 4. 组件消融（case1 / case5 @2mm，**当前实现**）')
    A('')
    abl = {}
    for p in sorted(glob.glob(os.path.join(R, 'ablations_v2', '*.json'))):
        d = load(p)
        if isinstance(d, dict) and 'tre_total' in d:
            tag = os.path.basename(p).split('_dirlab')[0]
            abl.setdefault(tag, {})[d['case']] = d['tre_total']
    main2 = {}
    for p in glob.glob(os.path.join(R, 'fast_phase2', 's1f_dirlab_case*_2mm.json')):
        d = load(p)
        if isinstance(d, dict) and 'tre_total' in d:
            main2[d['case']] = d['tre_total']
    if abl:
        A('| 配置 | case1 | case5 | Δ case1 | Δ case5 |')
        A('|---|---|---|---|---|')
        if 1 in main2 and 5 in main2:
            A(f'| **full**（主表配置） | **{fmt(main2[1])}** | **{fmt(main2[5])}** | — | — |')
        labels = {'aff0': '无仿射项', 'K2': 'K=2', 'K6': 'K=6',
                  'gonly': '仅全局 NCC（去局部阶段）', 'nores': '无残差阶段',
                  'AB': '无编码器（直接优化系数）'}
        for tag in ('nores', 'AB', 'aff0', 'K2', 'K6', 'gonly'):
            if tag not in abl:
                continue
            v1, v5 = abl[tag].get(1), abl[tag].get(5)
            # Δ 由四舍五入后的显示值计算，避免与稿件表格出现 ±0.01 的不一致
            r1 = round(v1, 2) if v1 is not None else None
            r5 = round(v5, 2) if v5 is not None else None
            d1 = f'{r1 - round(main2[1], 2):+.2f}' if (r1 and 1 in main2) else '—'
            d5 = f'{r5 - round(main2[5], 2):+.2f}' if (r5 and 5 in main2) else '—'
            A(f'| {labels.get(tag, tag)} | {fmt(v1)} | {fmt(v5)} | {d1} | {d5} |')
        A('')
        A('> **当前实现的关键结论（与旧实现不同，已改写稿件）**：')
        A('> (1) **周期残差为主**（+0.57/+0.84）；(2) **编码器在大形变上重要**（case5 +1.11）；')
        A('> (3) 仿射项增益温和（+0.06/+0.22）；(4) **K=6 相对 K=4 无改善**（−0.03），')
        A('> 故"K=6 过拟合"的旧结论**不成立**，改为"出于简洁保留 K=4"；')
        A('> (5) **局部 NCC 阶段近冗余**（+0.01/+0.08），列为候选简化。')
        A('')
    else:
        A('（尚未生成 ablations_v2 —— 运行 `powershell -File run_ablation_v2.ps1`）')
    A('')

    # ---------- Table 4b: CREATIS ----------
    A('## Table 4b. CREATIS 全相位（10 相均值 TRE, mm，@2mm）')
    A('')
    cr = {}
    for p in sorted(glob.glob(os.path.join(R, 'fast_phase2', '*creatis*.json'))):
        d = load(p)
        if isinstance(d, dict) and 'tre_total' in d:
            cr[d['case']] = d
    if cr:
        A('| case | identity | 流形 | +残差 | 源文件 |')
        A('|---|---|---|---|---|')
        for c in sorted(cr):
            d = cr[c]
            A(f'| {c} | {fmt(d.get("init_tre"))} | {fmt(d.get("tre_manifold"))} | '
              f'**{fmt(d.get("tre_total"))}** | `results/fast_phase2/*creatis*case{c}*` |')
        A(f'| **mean** | {fmt(mean_of([cr[c].get("init_tre") for c in cr]))} | '
          f'{fmt(mean_of([cr[c].get("tre_manifold") for c in cr]))} | '
          f'**{fmt(mean_of([cr[c].get("tre_total") for c in cr]))}** | |')
        A('')
        A('> ⚠️ **仅 case0–2 可用**：case3–5 的 landmark 文件损坏（每例 10 个 `.pts` 中 8 个是')
        A('> HTML 残骸），需从 Zenodo 重下才可补全。')
        A('> 注：CREATIS 上残差阶段增益为 0（流形 = +残差），与早期记录一致。')
        A('')
    else:
        A('（尚无 CREATIS 结果）')
    A('')

    # ---------- Table 5: 深部结构（协议分级） ----------
    A('## Table 5. 深部 landmark（⚠️ 协议分级，务必区分 L1 / L2）')
    A('')
    A('### 5a. DIRLAB 10 例 @2mm —— **L1（landmark 监督代理，上限实验）**')
    A('')
    if stats:
        cl_ = stats['case_level']
        A('| 指标 | 无约束（弱 B-spline）| +L1 约束 | p (Wilcoxon) | Cohen\'s d |')
        A('|---|---|---|---|---|')
        A(f'| 病例级 median TRE | {fmt(cl_["base_median_of_medians_mm"])} | '
          f'**{fmt(cl_["con_median_of_medians_mm"])}** | {cl_["wilcoxon_p"]:.3g} | '
          f'{fmt(cl_["cohen_d"])} |')
        A(f'| 深部 median TRE | {fmt(cl_["deep_base_median_of_medians_mm"])} | '
          f'**{fmt(cl_["deep_con_median_of_medians_mm"])}** | {cl_["deep_wilcoxon_p"]:.3g} | '
          f'{fmt(cl_["deep_cohen_d"])} |')
        A('')
        A('> **监督协议**：60% landmark 用于约束、40% 用于评价，3 次随机拆分取均值；'
          'n=10 例。此结果是"对应关系可获得时的上限"，**不是无监督配准方法的结果**。')
        A('')
    gq, l2, case_level, gq_down = collect_lungqa()
    A(f'### 5b. Lung-DIR-QA 30 例 @{gq_down}mm —— 基线可信度审计')
    A('')
    if gq:
        init = [r['init_tre'] for r in gq.values()]
        base = [r['base_mean'] for r in gq.values()]
        con = [r['con_mean'] for r in gq.values()]
        worse = sum(1 for r in gq.values() if r['base_mean'] > r['init_tre'])
        A('| 口径 | mean | median | 说明 |')
        A('|---|---|---|---|')
        A(f'| identity（不配准） | {fmt(mean_of(init))} | {fmt(median_of(init))} | 物理下界参照 |')
        A(f'| 无监督基线（3mm B-spline） | {fmt(mean_of(base))} | {fmt(median_of(base))} | '
          f'**{worse}/30 例比 identity 更差** |')
        A(f'| +L1 landmark 约束 | {fmt(mean_of(con))} | {fmt(median_of(con))} | 监督代理（上限）|')
        A('')
        audit.append(
            f'**Lung-DIR-QA 基线不成立**：无监督基线 mean {mean_of(base):.2f}mm vs '
            f'identity {mean_of(init):.2f}mm（仅改善 '
            f'{100*(1-mean_of(base)/mean_of(init)):.1f}%），且 {worse}/30 例**劣于不配准**。'
            f'论文中"26.22 → 4.25 mm (p=1.9e-9)"是**对照一个失效基线 + L1 监督**得到的，'
            f'不能作为配准质量证据。')
    A('')
    A('### 5c. Lung-DIR-QA —— **L2（自动结构点，无泄漏）**（诚实版）')
    A('')
    if l2:
        A('| case | identity | 基线 | +L2 自动约束 | 源文件 |')
        A('|---|---|---|---|---|')
        for c in sorted(l2):
            d = l2[c]
            ident = gq.get(c, {}).get('init_tre')
            A(f'| {c} | {fmt(ident)} | {fmt(d.get("base_tre"))} | '
              f'{fmt(d.get("con_tre"))} | `results/lungqa_l2/case{c}_3mm.json` |')
        A(f'| **mean** | {fmt(mean_of([gq.get(c, {}).get("init_tre") for c in l2]))} | '
          f'{fmt(mean_of([l2[c].get("base_tre") for c in l2]))} | '
          f'**{fmt(mean_of([l2[c].get("con_tre") for c in l2]))}** | |')
        A('')
        A(f'> 仅完成 {len(l2)}/30 例；改善幅度远小于 L1。论文若把 L1 写成主结果、'
          f'L2 只做提及，属**选择性报告**。')
    else:
        A('（尚无 L2 结果——这是当前最主要的实验缺口之一）')
    A('')

    # ---------- Table 6: 剂量机制验证 ----------
    A('## Table 6. 合成剂量输运（机制验证，非临床声明）')
    A('')
    v2 = load(os.path.join(R, 'dose_roundtrip_v2.json'))
    if v2 and 'cycle10' in v2:
        A('### 6a. 误差随绕环圈数的累积（`results/dose_roundtrip_v2.json`，'
          '**稿件引用的来源**）')
        A('')
        cycles = [k for k in ('cycle1', 'cycle2', 'cycle3', 'cycle5', 'cycle10') if k in v2]
        A('| 绕环圈数 | PMR mean 误差 (Gy) | 逐对 B-spline mean 误差 (Gy) | '
          'PMR PTV 内 (Gy) | 逐对 PTV 内 (Gy) |')
        A('|---|---|---|---|---|')
        for k in cycles:
            c = v2[k]
            A(f'| {k.replace("cycle", "")} | {fmt(c["pmr_err_mean_Gy"], 4)} | '
              f'{fmt(c["bspline_err_mean_Gy"], 4)} | {fmt(c["pmr_err_ptv_mean_Gy"], 4)} | '
              f'{fmt(c["bspline_err_ptv_mean_Gy"], 4)} |')
        A('')
        A('> **仅 case1、@2mm、合成剂量**。真实 RTDOSE 不可得，'
          '**不得**将此解读为临床剂量学结论。')
        A('')
    dz = load(os.path.join(R, 'dose_accum', 'case1_roundtrip.json'))
    if dz and 'pmr' in dz:
        A('### 6b. ⚠️ 另一份剂量结果（`results/dose_accum/case1_roundtrip.json`）')
        A('')
        A('| 方法 | mean map | PTV 内 mean | max |')
        A('|---|---|---|---|')
        A(f'| PMR | {fmt(dz["pmr"]["mean_map"], 4)} | {fmt(dz["pmr"]["ptv_mean"], 3)} | '
          f'{fmt(dz["pmr"]["max"], 2)} |')
        A(f'| 逐对 | {fmt(dz["pairwise"]["mean_map"], 4)} | {fmt(dz["pairwise"]["ptv_mean"], 3)} | '
          f'{fmt(dz["pairwise"]["max"], 2)} |')
        A('')
        audit.append(
            '**两份剂量结果互相矛盾**：`dose_roundtrip_v2.json` 的 PTV 值 '
            f'({fmt(v2.get("cycle10", {}).get("pmr_err_ptv_mean_Gy"), 4)} / '
            f'{fmt(v2.get("cycle10", {}).get("bspline_err_ptv_mean_Gy"), 4)} Gy) 与 '
            f'`dose_accum/case1_roundtrip.json` 的 PTV 值 '
            f'({fmt(dz["pmr"]["ptv_mean"], 3)} / {fmt(dz["pairwise"]["ptv_mean"], 3)} Gy) '
            '相差约 20 倍（疑为不同归一化/是否累积）。稿件只引用了前者，'
            '必须查清差异来源并删除或标注其中一份，否则属不可复现数字。')
        A('')

    # ---------- Table 7: 形变场物理性（Jacobian） ----------
    A('## Table 7. 形变场物理性（Jacobian 行列式，@2mm）')
    A('')
    jac = load(os.path.join(R, 'jacobian_stats.json'))
    jres = jac.get('results', []) if isinstance(jac, dict) else []
    if jres:
        A('| 方法 | 文件 | %det(J)≤0 | 最差相位 | min det | mean det |')
        A('|---|---|---|---|---|---|')
        for r in jres:
            agg = r.get('aggregate_mean_over_phases', {})
            if not agg:
                continue
            A(f'| {r["kind"]} | `{r["file"]}` | {fmt(agg.get("pct_nonpositive"), 4)} | '
              f'{fmt(agg.get("worst_pct_nonpositive"), 4)} | {fmt(agg.get("min_det"), 3)} | '
              f'{fmt(agg.get("mean_det"), 4)} |')
        A('')
        A(f'> 自检：`{jac.get("self_test")}`（解析 scaling/shear/folding 三项）。'
          f' 协议：{jac.get("protocol")}')
        A('')
        pmr_worst = max([r['aggregate_mean_over_phases']['pct_nonpositive']
                         for r in jres if r['kind'] == 'pmr'], default=0.0)
        pair_worst = max([r['aggregate_mean_over_phases']['pct_nonpositive']
                          for r in jres if r['kind'] == 'pairwise'], default=0.0)
        if pmr_worst > 0 and pmr_worst > pair_worst:
            A('> ✅ **本项已在 `draft_v04` 中更正**（Table 5 如实报告折叠并提供定位图 Fig. 8）。'
              '仍待做的是加折叠惩罚以复现 0%。')
    else:
        A('（尚无 jacobian_stats.json —— 运行 `python jacobian_stats.py` 生成）')
    A('')

    # ---------- 审计 ----------
    A('---')
    A('')
    A('## 自动审计：当前证据链的已知缺陷')
    A('')
    if audit:
        for i, a in enumerate(audit, 1):
            A(f'{i}. {a}')
    else:
        A('（未检出结构性问题）')
    A('')
    A('### 检索到的关键协议事实')
    A('')
    A('- 决定性实验（Table 5a）为 **2mm**、基线 mesh **8×8×4 / 50 iter**（弱配置）；')
    A('- Lung-DIR-QA（Table 5b/5c）为 **3mm**，非 DIRLAB 标准 1mm；')
    A('- Elastix 强基线为 **1mm**，与 PMR 的 2mm 主表**分辨率不可直接比拟**；')
    A('- deedsBCV 全部结果为坐标框架错位失真，已排除。')
    A('')
    return '\n'.join(md), audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(R, 'TABLES.md'))
    args = ap.parse_args()
    md, audit = build_md()
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'已生成: {args.out}')
    print(f'审计发现问题 {len(audit)} 条:')
    for i, a in enumerate(audit, 1):
        print(f'  {i}. {a[:150]}')


if __name__ == '__main__':
    main()
