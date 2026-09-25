"""Paper B 稿件数字回查：把稿件里的每个数字与 results/*.json 实际值对照。

任何对不上的都要么改稿件、要么改来源，绝不放任。
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
import glob
import json
import os
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')

import sys
_NEED_DIR = R

# --- SUITE-SKIP guard (injected by tools/add_suite_skip_guard2.py) -------------
# 见 `tools/add_suite_skip_guard.py` 的说明：数据不在 ⇒ **显式跳过**，
# 且 `run_verification_suite.py` 会把 `SUITE-SKIP:` 记为 skip（不计入通过）。
if not os.path.isdir(_NEED_DIR):
    print('SUITE-SKIP: 缺少 results —— 本项需要历史结果文件，仓库发行版不附带；'
          '在开发树（含 results/）中运行同一入口即可完整执行。', flush=True)
    sys.exit(0)
# --- end SUITE-SKIP guard ------------------------------------------------------

FAIL = []


def chk(label, claimed, actual, tol=0.02, note=''):
    if actual is None:
        print(f'  ❓ {label}: 稿件={claimed} | 来源缺失 {note}')
        FAIL.append(label)
        return
    ok = abs(claimed - actual) <= tol * max(abs(claimed), 1e-9)
    print(f'  {"✅" if ok else "🔴"} {label}: 稿件={claimed} 实测={actual} {note}')
    if not ok:
        FAIL.append(label)


def load(p):
    try:
        return json.load(open(os.path.join(R, p), encoding='utf-8'))
    except Exception:
        return None


print('=== A. 1 mm DIR-Lab（表 B1b）===')
one = {}
for c in range(1, 6):
    d = load(f'pmr_v2_1mm/v2_1mm_case{c}_1mm.json')
    if d:
        one[c] = d
for c, d in one.items():
    print(f'  case{c}: n_vox={d["n_vox"]:,}  peak={d["peak_mem_gb"]} GB  '
          f'{d["time_s"]}s  TRE={d["tre_total_STANDARD"]:.3f}')

print('\n--- 稿件声称 ---')
d1 = one.get(1)
if d1:
    chk('case1 n_vox = 14.45e6', 14_453_440, d1['n_vox'], tol=1e-6)
    chk('case1 十相合计 = 144.5e6', 144_534_400, d1['n_vox'] * 10, tol=1e-6)
    chk('case1 峰值 3.25 GB', 3.25, d1['peak_mem_gb'], tol=0.01)
    chk('case1 耗时 306 s', 306, d1['time_s'], tol=0.01)
    chk('case1 TRE 1.080', 1.080, d1['tre_total_STANDARD'], tol=0.01)
if one:
    pm = [d['peak_mem_gb'] for d in one.values()]
    tm = [d['time_s'] for d in one.values()]
    chk('1mm 显存下界 3.25', 3.25, min(pm), tol=0.01)
    chk('1mm 显存上界 5.82', 5.82, max(pm), tol=0.01)
    chk('1mm 耗时下界 306', 306, min(tm), tol=0.01)
    chk('1mm 耗时上界 494', 494, max(tm), tol=0.01)
d2 = one.get(2)
if d2:
    chk('case2 n_vox = 24.7e6', 24.7e6, d2['n_vox'], tol=0.01)
    chk('case2 峰值 5.82 GB', 5.82, d2['peak_mem_gb'], tol=0.01)

print('\n=== B. 2 mm DIR-Lab 十例（表 B1）===')
two = {}
for c in range(1, 11):
    d = load(f'pmr_v2/det10_best_case{c}_2mm.json')
    if d:
        two[c] = d['tre_total_STANDARD']
        print(f'  case{c}: TRE={d["tre_total_STANDARD"]:.3f}  peak={d["peak_mem_gb"]}  '
              f'{d["time_s"]}s')
if len(two) == 10:
    mean = sum(two.values()) / 10
    chk('10 例均值 1.597 mm', 1.597, round(mean, 3), tol=0.005)
pm2 = [load(f'pmr_v2/det10_best_case{c}_2mm.json')['peak_mem_gb'] for c in range(1, 11)
       if load(f'pmr_v2/det10_best_case{c}_2mm.json')]
tm2 = [load(f'pmr_v2/det10_best_case{c}_2mm.json')['time_s'] for c in range(1, 11)
       if load(f'pmr_v2/det10_best_case{c}_2mm.json')]
if pm2:
    chk('2mm 显存下界 0.50', 0.50, min(pm2), tol=0.02)
    chk('2mm 显存上界 2.82', 2.82, max(pm2), tol=0.02)
    chk('2mm 耗时下界 272', 272, min(tm2), tol=0.01)
    chk('2mm 耗时上界 652', 652, max(tm2), tol=0.01)

print('\n=== C. 运行间波动（`detA`/`detB` 是 **case1** 的两次确定性运行）===')
# 🔴 修正（2026-09-24）：本段原先去读 `detA_case8_2mm.json`，而 detA/detB **只有 case1**，
#    于是两个值恒为 None、三条 chk 一条都没执行 —— **一段看起来在检查、其实什么都没查的代码**。
det = {}
for tag in ('detA', 'detB'):
    d = load(f'pmr_v2/{tag}_case1_2mm.json')
    if d:
        det[tag] = d['tre_total_STANDARD']
print(f'  detA={det.get("detA")}  detB={det.get("detB")}')
if len(det) == 2:
    chk('detA 1.13048', 1.13048, det['detA'], tol=1e-4)
    chk('detB 1.13305', 1.13305, det['detB'], tol=1e-4)
    chk('确定性波动 0.2%（case1）', 0.2,
        abs(det['detA'] - det['detB']) / det['detA'] * 100, tol=0.3)
else:
    print('  ⚠️ 缺 detA/detB，本段未实际检查')

print('\n=== D. 未开 benchmark 的 5 次（稿件的 6.2% / 0.20 mm / 17.0%）===')
# 🔴 修正（2026-09-24）：原先的 tags 列表**缺 `fold_resreg10` 与 `main2_best`**，
#    而那两次正是稿件列出的五次里的两次 ⇒ 下面按五次算的检查一条也没执行。
tags = ['fold_resreg10', 'main2_best', 'rep2', 'rep3', 'rep4',
        'base_r2', 'base_r7', 'rep7a', 'main_base']
vals = []
for t in tags:
    for f in glob.glob(os.path.join(R, 'pmr_v2', f'{t}_case8_2mm.json')):
        d = json.load(open(f, encoding='utf-8'))
        vals.append((t, d['tre_total_STANDARD']))
if vals:
    print('  找到的 case8 重复运行:')
    for t, v in vals:
        print(f'    {t}: {v}')
    chk('稿件列的 3.584 max', 3.584, max(v for _, v in vals), tol=0.01)
    # 稿件 §3.4 的三个数字（旧标签写的是 5.8%，那是**错的口径**）
    v5 = sorted(v for t, v in vals
                if t in ('fold_resreg10', 'main2_best', 'rep2', 'rep3', 'rep4'))
    if len(v5) == 5:
        import statistics as _st
        mean5 = _st.mean(v5)
        chk('五次样本 SD = 0.205 mm（稿件）', 0.205, _st.stdev(v5), tol=0.001)
        chk('SD / 均值 = 6.2%（稿件）', 6.2, 100 * _st.stdev(v5) / mean5, tol=0.05)
        chk('极差 / 均值 = 17.0%（稿件）', 17.0,
            100 * (max(v5) - min(v5)) / mean5, tol=0.05)
    else:
        print(f'  ⚠️ 稿件列的五次只找到 {len(v5)} 次，SD/极差检查未执行')

print('\n=== D2. 🔴 确定性口径的**逐例**真实波动（2026-09-24 新增）===')
# `det10_best` 与 `ph6_best` 是同一配置、同一 `--cudnn-benchmark 0` 的两次完整运行。
# 稿件 §3.4 现在写的是"九例 0.5–1.6%、两个最难例 5.3%/5.4%、十例均值 0.47%"。
_a, _b = load('pmr_v2/det10_best_case8_2mm.json'), load('pmr_v2/ph6_best_case8_2mm.json')
if _a and _b:
    chk('case8 逐例极差 5.3%（稿件）', 5.3,
        100 * abs(_a['tre_total_STANDARD'] - _b['tre_total_STANDARD'])
        / min(_a['tre_total_STANDARD'], _b['tre_total_STANDARD']), tol=0.2)
_mA = [load(f'pmr_v2/det10_best_case{c}_2mm.json') for c in range(1, 11)]
_mB = [load(f'pmr_v2/ph6_best_case{c}_2mm.json') for c in range(1, 11)]
if all(_mA) and all(_mB):
    import statistics as _st2
    ma = _st2.mean(x['tre_total_STANDARD'] for x in _mA)
    mb = _st2.mean(x['tre_total_STANDARD'] for x in _mB)
    chk('十例均值 det10 = 1.5975', 1.5975, ma, tol=5e-4)
    chk('十例均值 ph6  = 1.5899', 1.5899, mb, tol=5e-4)
    chk('两次运行的均值极差 0.47%（稿件）', 0.47, 100 * abs(ma - mb) / ma, tol=0.05)

print('\n=== D3. 缺陷台账与验证脚本数（对外数字）===')
# 这两个数字被两篇论文与公开仓库同时引用，必须有机器核对。
_src37 = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'docs', '37_缺陷台账与验证套件.md'), encoding='utf-8').read()
chk('docs/37 声明 14 个缺陷', True,
    ('＝ 14 个缺陷' in _src37) or ('= 14 个缺陷' in _src37))
chk('docs/37 声明 11 个静默', True, '11 个完全静默' in _src37)
_draft = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'paperB', 'draft_softwarex.md'), encoding='utf-8').read()
chk('稿件写 fourteen', True, 'fourteen documented defects' in _draft)
chk('稿件写 eleven', True, 'eleven of them completely silent' in _draft)
_verif = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'repo', 'verification')
if os.path.isdir(_verif):
    # 🔴 2026-09-25 更正：原来比对的是 `repo/verification/` 里的 **文件数**（那会把
    #    辅助脚本 probe_hu.py / analyze_landmark_errors.py 等也算进去），而稿件里的
    #    "N checks" 指的是 **套件条目数**（`run_verification_suite.py` 的 SUITE）。
    #    两个口径混用 ⇒ 检查器报"对不上"，而真正该对的是 SUITE 的长度。
    _suite = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'run_verification_suite.py')
    _stxt = open(_suite, encoding='utf-8').read()
    # ⚠️ key 里有大写（`paperA_assembly`），第一版正则写 `[a-z_]+` 把两条漏了 ⇒ 数出 11。
    _entries = re.findall(r"\('([A-Za-z_0-9]+)',\s*'(math|geometry|convention|data)',"
                          r"\s*'([^']+)'", _stxt)
    _n_suite = len(_entries)
    _n_files = len([f for f in os.listdir(_verif) if f.endswith('.py')])
    chk(f'套件条目数 = {_n_suite}，稿件写 14', 14, _n_suite, tol=0)
    print(f'  · 参考：repo/verification/ 下共 {_n_files} 个 .py（含辅助脚本，'
          f'不等于套件条目数；稿件写的是套件条目数）')
    # 套件清单里的每一项都必须在发行版里**真的存在** —— 这正是审稿人踩到的坑：
    # 清单一列 14 项、磁盘上只有 7 个脚本，一键入口第一屏就是红的。
    _missing = []
    for _key, _cat, _script in _entries:
        _base = os.path.basename(_script)
        if not any(os.path.exists(os.path.join(_d, _base))
                   for _d in (_verif, os.path.join(_verif, '..', 'src'),
                              os.path.join(_verif, '..'), os.path.join(_verif, '..', 'tools'))):
            _missing.append(_script)
    chk(f'套件里每个脚本都在发行版中（缺 {_missing}）', 0, len(_missing), tol=0)

print('\n=== E. CREATIS 1mm 体素数（60–116 M）===')
import SimpleITK as sitk
DATA = os.environ.get('PMR_DATA_ROOT') or os.path.join(ROOT, '..', 'reference_4', 'data')
mm = []
for c in range(6):
    p = os.path.join(DATA, 'CREATIS', str(c), '00_R.mha')
    if os.path.exists(p):
        sz = sitk.ReadImage(p).GetSize()
        mm.append((c, sz[0] * sz[1] * sz[2]))
for c, v in mm:
    print(f'  case{c}: {v/1e6:.2f} Mvox/phase')
if mm:
    chk('CREATIS 下界 60 M', 60e6, min(v for _, v in mm), tol=0.02)
    chk('CREATIS 上界 116 M', 116e6, max(v for _, v in mm), tol=0.01)

print('\n=== F. 派生值（算术可核；单位 = GiB/MiB，与 torch 的 /1024**3 口径一致）===')
n_vox = 14_453_440
naive_coef = n_vox * 24 * 4
chk('朴素系数场 1.29 GiB', 1.29, naive_coef / 1024**3, tol=0.01)
chk('含 Adam+grad 5.17 GiB', 5.17, naive_coef * 4 / 1024**3, tol=0.01)
chk('朴素优化器状态 3.88 GiB', 3.88, naive_coef * 3 / 1024**3, tol=0.01)
coarse = 29 * 31 * 31 * 24 * 4
chk('粗网格系数 2.55 MiB', 2.55, coarse / 1024**2, tol=0.01)
chk('缩减比 519x', 519, naive_coef / coarse, tol=0.01)
print('  参考：实测峰值 case1@1mm = 3.25 GiB，显卡容量 8151 MiB = 7.96 GiB')
print(f'  ⇒ 朴素替换后约 {3.25 + naive_coef*4/1024**3:.2f} GiB > 7.96 GiB ⇒ 装不下')

print('\n' + '=' * 60)
if FAIL:
    print(f'🔴 {len(FAIL)} 项对不上，需修正：')
    for f in FAIL:
        print(f'   - {f}')
    sys.exit(1)
print('✅ 全部对得上')
