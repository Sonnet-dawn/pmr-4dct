"""tools/check_xrefs.py —— 核对 Paper A 正文里的**章节交叉引用**是否都指向真实存在的节
================================================================================
**为什么**：这轮做了大量结构改动（Discussion 从 §6 改到 §5、删掉 Results 里的重复章节、
把 Limitations 搬走）。**引用编号不会自动跟着走** —— 一个指向已删除小节的 `§5.6`
在读者看来就是"作者自己都没校对"。这类错误人工很难发现，但机器一查就出来。

做法：
  1. 从各分稿里收集**实际存在的**章节号（`## N.` 与 `### N.M`）
  2. 从正文里提取所有 `§N.M` 形式的引用
  3. 报告"指向不存在节"的引用，以及"存在但从未被引用"的长小节（后者只提示）

用法: python tools/check_xrefs.py
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
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
MS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'manuscript')

# --- SUITE-SKIP guard (injected by hand; 见 tools/add_suite_skip_guard.py 的说明) -----
# 本项检查的是**手稿**内部的交叉引用。公开仓库发行版不含 manuscript/，
# 因此在新 clone 里它无法执行 —— **显式跳过**，不静默通过也不报失败。
if not os.path.isdir(MS):
    print('SUITE-SKIP: 缺少 manuscript/ —— 本项检查手稿的交叉引用，'
          '仓库发行版不附带手稿；在开发树中运行同一入口即可完整执行。', flush=True)
    sys.exit(0)
# --- end SUITE-SKIP guard ------------------------------------------------------------
FILES = ['draft_v05_title_abstract.md', 'draft_v05_intro.md', 'draft_v05_methods.md',
         'draft_v05_results.md', 'draft_v05_discussion.md']

texts = {}
for fn in FILES:
    p = os.path.join(MS, fn)
    if os.path.exists(p):
        texts[fn] = open(p, encoding='utf-8').read()

# ---- 实际存在的节
# ⚠️ 必须同时抓 `### 4.1` 与 `#### 4.1.1` 两种层级 —— 第一版只抓了 `###`，
#    于是把**真实存在**的 §4.1.1 / §4.1.2 报成"指向不存在的节"（假阳性）。
#    这是本项目第五次"验证器自己出错"（前四次见 `docs/41`、`docs/44`）。
exists_top, exists_sub, where = set(), set(), {}
for fn, t in texts.items():
    for m in re.finditer(r'^## (\d+)\.\s*(.+)$', t, re.M):
        exists_top.add(m.group(1))
        where[m.group(1)] = (fn, m.group(2).strip())
    for m in re.finditer(r'^#{3,4} (\d+\.\d+(?:\.\d+)?)', t, re.M):
        exists_sub.add(m.group(1))
        where[m.group(1)] = (fn, '')

print('=' * 88)
print('实际存在的节')
print('=' * 88)
for k in sorted(exists_top, key=int):
    fn, title = where[k]
    print(f'  §{k:4s} {title:52s} ({fn})')
print('  子节：' + ', '.join('§' + s for s in sorted(exists_sub, key=lambda x: [int(y) for y in x.split('.')])))

# ---- 正文里的引用
refs = Counter()
detail = []
for fn, t in texts.items():
    for m in re.finditer(r'§(\d+(?:\.\d+)*)', t):
        r = m.group(1)
        refs[r] += 1
        ln = t[:m.start()].count('\n') + 1
        detail.append((fn, ln, r))

print('\n' + '=' * 88)
print('引用统计与校验')
print('=' * 88)
bad = []
for r in sorted(refs, key=lambda x: [int(y) for y in x.split('.')]):
    if r in exists_top or r in exists_sub:
        ok = True
    elif r.split('.')[0] in exists_top and len(r.split('.')) == 2:
        # 形如 §4.1 但只存在 §4.1.1/§4.1.2 —— 常见且可接受（指整节），提示一下
        ok = '~'
    else:
        ok = False
    mark = {True: '✅', '~': '⚠️', False: '❌'}[ok]
    print(f'  {mark} §{r:6s} 被引用 {refs[r]:>3d} 次')
    if ok is False:
        bad.append(r)
    if ok == '~':
        subs = [s for s in exists_sub if s.startswith(r + '.')]
        print(f'        （只存在子节 {"、".join("§"+s for s in sorted(subs))}，'
              f'引用整节通常是可接受的）')

print('\n' + '=' * 88)
if bad:
    print(f'🔴 **{len(bad)} 个引用指向不存在的节**：{bad}')
    print('   出现位置：')
    for fn, ln, r in detail:
        if r in bad:
            print(f'     {fn}:{ln}   §{r}')
else:
    print('✅ 所有交叉引用都指向真实存在的节。')

# ---- 未被引用的子节（仅提示）
uncited = [s for s in sorted(exists_sub, key=lambda x: [int(y) for y in x.split('.')])
           if refs.get(s, 0) == 0]
if uncited:
    print(f'\n（提示：{len(uncited)} 个子节从未被正文交叉引用 —— 不一定有问题，'
          f'但值得确认它们确实被"就近阅读"到）')
    print('   ' + ', '.join('§' + s for s in uncited))
sys.exit(1 if bad else 0)
