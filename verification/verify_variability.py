"""tools/verify_variability.py —— 核对"运行间变异 6.2%"这一头条可复现性结果
================================================================================
两篇论文现在都引用**统一口径**的版本：

  "repeated runs of an identical configuration on case 8 gave 3.023, 3.216, 3.320,
   3.354 and 3.584 mm — a sample standard deviation of 0.205 mm (6.2% of the mean) and
   a range of 0.56 mm (17.0% of the mean)."

要核四件事：
  ① 五个值确实来自同一配置（见 `tools/check_repeat_configs.py`：✅ 19 个字段全同）；
  ② SD（样本，n−1）与其百分比的算法；
  ③ **百分比的分母一致** —— SD 与极差**都**对均值，不再出现"SD 对均值、极差对最小值"
     的混用（旧稿写 "range of 18.6%"，分母是最小值 3.023）；
  ④ 稿件与文档里**不再有**把两者混写的残留（扫 "18.6%" 的出现处，只允许出现在
     明确标注为"若以最小值为基"的说明句里）。

用法: python tools/verify_variability.py
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
import statistics as st

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_NEED_DIR = os.path.join(HERE, *'results'.split("/"))

# --- SUITE-SKIP guard (injected by tools/add_suite_skip_guard.py) ---------------
# 本检查需要 `results/` 里的历史产物。仓库发行版**不附带**结果文件，因此在新 clone 里
# 它无法执行。这里**显式声明"跳过"而不是悄悄通过**：`run_verification_suite.py`
# 见到 `SUITE-SKIP:` 会记为 skip 状态，且**不计入通过数**。
if not os.path.isdir(_NEED_DIR):
    print('SUITE-SKIP: 缺少 results —— 本项需要历史结果文件，仓库发行版不附带；'
          '在开发树（含 results/）中运行同一入口即可完整执行。', flush=True)
    sys.exit(0)
# --- end SUITE-SKIP guard ------------------------------------------------------

R = os.path.join(HERE, 'results', 'pmr_v2')

FILES = ['fold_resreg10_case8_2mm.json', 'main2_best_case8_2mm.json',
         'rep2_case8_2mm.json', 'rep3_case8_2mm.json', 'rep4_case8_2mm.json']
v = [json.load(open(os.path.join(R, f), encoding='utf-8'))['tre_total_STANDARD']
     for f in FILES]

print('五次运行（case 8，配置已核验为完全相同）：')
for f, x in zip(FILES, v):
    print(f'  {x:.4f}   {f}')

mean = st.mean(v)
sd = st.stdev(v)          # 样本 SD (n-1)
rng = max(v) - min(v)
pct_sd = 100 * sd / mean
pct_rng = 100 * rng / mean
print(f'\n均值        {mean:.4f}')
print(f'样本 SD     {sd:.4f} mm  = {pct_sd:.2f}% of mean   ← 稿件 0.205 mm / 6.2%')
print(f'总体 SD     {st.pstdev(v):.4f} mm  = {100*st.pstdev(v)/mean:.2f}% of mean')
print(f'极差        {rng:.4f} mm  = {pct_rng:.2f}% of mean   ← 稿件 0.56 mm / 17.0%')
print(f'  （旧口径，已废弃）相对最小值 {100*rng/min(v):.2f}%  相对最大值 {100*rng/max(v):.2f}%')

# ---- ④ 扫描残留：同一句里混用两种基数 ----
SCAN = ['manuscript/draft_v05_results.md', 'manuscript/draft_v05_discussion.md',
        'paperB/draft_softwarex.md']
TOL_OK = re.compile(r'(以最小值为基|against the lowest|against the minimum|'
                    r'relative to the lowest|expressed against the lowest|用最小值)')
resid = []
for rel in SCAN:
    p = os.path.join(HERE, rel.replace('/', os.sep))
    if not os.path.exists(p):
        resid.append((rel, 0, '文件不存在'))
        continue
    for i, line in enumerate(open(p, encoding='utf-8'), 1):
        if '18.6' in line and not TOL_OK.search(line):
            resid.append((rel, i, line.strip()[:110]))

print('\n' + '=' * 84)
ok_sd = abs(sd - 0.205) < 0.006 and abs(pct_sd - 6.2) < 0.15
ok_rng = abs(rng - 0.56) < 0.01 and abs(pct_rng - 17.0) < 0.15
print(f'  {"✅" if ok_sd else "❌"} SD = {sd:.4f} mm（{pct_sd:.2f}% of mean）与稿件的 0.205 mm / 6.2% 一致')
print(f'  {"✅" if ok_rng else "❌"} 极差 = {rng:.4f} mm（{pct_rng:.2f}% of mean）与稿件的 0.56 mm / 17.0% 一致')
print(f'  {"✅" if not resid else "❌"} 口径残留扫描：{"未发现混用" if not resid else str(len(resid)) + " 处"}'
      '（分母不同 = 同一句里两个百分比不可比）')
for rel, ln, txt in resid:
    print(f'      {rel}:{ln}  {txt}')
print('\n⇒ ' + ('两者分母一致、稿件与文档同步 ✅' if (ok_sd and ok_rng and not resid)
                else '仍有不一致，见上 ❌'))
