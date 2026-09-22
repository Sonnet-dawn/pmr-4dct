"""
make_main_table.py —— 生成论文主表（标准 TRE 口径，多方法对照）。
================================================================================
规则（沿用 docs/17 台账纪律）：
  * **每个数字必须能追溯到文件**；找不到就留空并标注，不得手填
  * 标准口径 = |lm0 + d(lm0) - lm5|（把 T00 点正向映射）；旧口径一律换算或弃用
  * identity 列强制存在（防止"基线比自己还差"的假象）
  * 缺例的方法不参与均值比较（给出 n 与均值，而非补零）

数据来源
  PMR v1   results/fast_phase2/s1f_dirlab_case*_2mm.json（tre_total_linear，旧口径）
           标准口径见 results/tre_convention_migration.json
  PMR v2   results/pmr_v2/<tag>_case*_2mm.json（tre_total_STANDARD，已标准口径）
  Elastix  results/elastix_lit/elastix_lit_summary_*.json（tre_mean，标准口径）
           文献值（不可比，仅参考）见 docs/19 §7.2

用法：
  python make_main_table.py
  python make_main_table.py --v2-tag best --out results/MAIN_TABLE.md
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import os, sys, json, glob, argparse

try:                                    # Windows 控制台默认 GBK
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')

# 已发表、**不同协议**的参考值（仅作参照，不得进入同口径对比）
LIT = {
    'elastix_medphys2025': {'mean': 1.83, 'per_case': [1.0, 1.0, 1.3, 1.5, 1.7, 1.5, 2.3, 4.3, 1.8, 1.9],
                            'note': 'Med Phys 2025, 逐例值原文给出，均值为本审计计算'},
    'elastix_kanai2014_p2': {'mean': 1.28, 'per_case': None,
                             'note': 'Kanai 2014 J Radiat Res 55(6):1163, DOI 10.1093/jrr/rru062'},
    'elastix_kanai2014_p1': {'mean': 1.78, 'per_case': None, 'note': 'Kanai 2014 参数 1'},
}


def load(f):
    try:
        return json.load(open(f, encoding='utf-8'))
    except Exception:
        return None


def collect_v2(tag="", outdir='pmr_v2'):
    """收集 pmr_v2 结果；tag 为空则返回 {tag: {case: rec}} 全部。"""
    by = {}
    for f in glob.glob(os.path.join(R, outdir, '*_case*_2mm.json')):
        d = load(f)
        if not d or 'tre_total_STANDARD' not in d:
            continue
        base = os.path.basename(f)
        t = base.rsplit('_case', 1)[0]
        cn = int(base.rsplit('_case', 1)[1].split('_')[0])
        by.setdefault(t, {})[cn] = d
    if tag:
        return {tag: by.get(tag, {})}
    return by


def collect_elastix():
    by = {}
    for f in glob.glob(os.path.join(R, 'elastix_lit', 'elastix_lit_summary_*.json')):
        key = os.path.basename(f).replace('elastix_lit_summary_', '').replace('.json', '')
        recs = load(f) or []
        by[key] = {r['case']: r for r in recs if 'tre_mean' in r}
    return by


def collect_v1_standard():
    d = load(os.path.join(R, 'tre_convention_migration.json'))
    m = {}
    if d:
        rows = d.get('rows') or d.get('per_case') or d
        if isinstance(rows, dict):
            for k, v in rows.items():
                if isinstance(v, dict) and 'standard' in v:
                    try:
                        m[int(k)] = float(v['standard'])
                    except Exception:
                        pass
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2-tag', type=str, default='')
    ap.add_argument('--out', type=str, default=os.path.join(R, 'MAIN_TABLE.md'))
    args = ap.parse_args()

    cases = list(range(1, 11))
    lines = []
    W = lines.append

    W('# 主表 · DIRLAB 10 例 · **标准 TRE 口径**\n')
    W(f'> 由 `make_main_table.py` 自动生成。标准口径 = `|lm0 + d(lm0) − lm5|`（正向映射 T00 点）。')
    W('> **每个单元格都来自 results/ 下的 JSON；缺失者留空，不补零、不手填。**\n')

    # ---- identity ----
    lm_root = os.path.join(HERE, '..', 'reference_4', 'data', 'DIRLAB', 'points')
    ident = {}
    try:
        import numpy as np
        for cn in cases:
            a = np.loadtxt(os.path.join(lm_root, f'case{cn}', f'case{cn}_300_T00_xyz_R.txt'))
            b = np.loadtxt(os.path.join(lm_root, f'case{cn}', f'case{cn}_300_T50_xyz_R.txt'))
            ident[cn] = float(np.linalg.norm(a - b, axis=1).mean())
    except Exception as e:
        W(f'\n> ⚠️ identity 计算失败: {e}\n')

    # ---- v1 ----
    v1 = collect_v1_standard()

    # ---- v2 ----
    v2 = collect_v2(args.v2_tag)

    # ---- elastix ----
    ex = collect_elastix()

    def row(name, vals, note=''):
        cells = []
        for cn in cases:
            v = vals.get(cn)
            cells.append(f'{v:.3f}' if isinstance(v, (int, float)) else '—')
        got = [v for v in vals.values() if isinstance(v, (int, float))]
        n = len(got)
        mean = f'{sum(got)/n:.3f}' if n else '—'
        return f'| {name} | ' + ' | '.join(cells) + f' | **{mean}** | {n}/10 | {note} |'

    hdr = ('| 方法 | ' + ' | '.join(f'c{cn}' for cn in cases)
           + ' | **均值** | n | 备注 |')
    sep = '|' + '---|' * (len(cases) + 4)

    W('## 1. 同口径对比（本机、同一数据、同一 TRE 代码）\n')
    W(hdr); W(sep)
    if ident:
        W(row('identity（下界参照）', ident))
    if v1:
        W(row('PMR v1（`fast_phase2`）', v1, '标准口径，由 `recompute_tre_standard.py` 换算'))
    for t in sorted(v2):
        vals = {cn: r['tre_total_STANDARD'] for cn, r in v2[t].items()}
        W(row(f'**PMR v2 · {t}**', vals, 'v2 直接输出标准口径'))
    for k in sorted(ex):
        vals = {cn: r['tre_mean'] for cn, r in ex[k].items()}
        W(row(f'Elastix · {k}', vals, '`run_elastix_lit.py`'))

    W('')
    W('## 2. 已发表参考值（**协议不同，不得与上表直接比较**）\n')
    W('| 来源 | 均值 | 逐例 | 备注 |')
    W('|---|---|---|---|')
    for k, v in LIT.items():
        pc = ' '.join(f'{x}' for x in v['per_case']) if v['per_case'] else '—'
        W(f'| {k} | {v["mean"]} | {pc} | {v["note"]} |')

    W('')
    W('## 3. 数据完整性警告\n')
    W('* **PMR v1 的 cases 6–10 来自早一版实现**（`audit_pmr_results.py` 证明其 JSON 缺 '
      '`coef_scale`/`l2_phys`/`K` 字段）。上表中 v1 行**仅供趋势参考**，'
      '最终表必须等 v2 全量重跑。')
    W('* DIRLAB 标准关键点集**只有 T00 与 T50**，故所有 TRE 均为 T00→T50。')
    W('* 本项目使用 1 mm 各向同性重采样数据（`_R.mha`），而 Kanai 2014 用原生分辨率'
      '（层厚 2.5 mm）——**这是需要如实声明的协议偏离**。')

    txt = '\n'.join(lines) + '\n'
    open(args.out, 'w', encoding='utf-8').write(txt)
    print(txt)
    print(f'已保存 {args.out}')


if __name__ == '__main__':
    main()
