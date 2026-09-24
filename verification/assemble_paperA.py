"""tools/assemble_paperA.py —— 把 Paper A 的分稿组装成一份**完整稿件**
================================================================================
**为什么需要**：Paper A 一直是四份分稿（intro / methods / results / discussion）+
单独的文件表 + 参考文献。这对写作方便，但**不是一份能投的稿子**：
没有题名页、章节编号跨文件不连续、内部修订标记与正文混在一起。

本脚本**可复算地**组装出两份：

  * `manuscript/draft_v05_full.md`           —— **投稿版**：只留正文与学术性的更正说明
  * `manuscript/draft_v05_full_internal.md`  —— **内部版**：保留全部内部标记与审计线索

**如何区分"内部标记"与"学术性更正"**（这条规则很重要，不能一刀切）：
  * `<!-- ... -->` HTML 注释                                              → 内部，去掉
  * `> 🔴 ...` 引用块（以 🔴 开头）                                        → 内部，去掉
  * 含 `docs/`、`red line`、`红线`、`本项目的唯一报告口径`、`待补`、`TODO`  → 内部，去掉
  * **其余引用块一律保留** —— 例如 §4.2 那段 "A note on two earlier claims,
    both of which we withdrew"。**主动更正不是内部事务，是学术诚实**，
    投稿版必须留着（`docs/17` §7.2 的纪律）。
  * 每个文件开头的 `> 本文件是 ...` 导语                                  → 内部，去掉

组装后做**一致性自检**：章节编号连续、必备件齐全（题名/摘要/参考文献）、
正文里不残留 `docs/` 引用、全文字数。

用法:
  python tools/assemble_paperA.py
  python tools/assemble_paperA.py --stats     # 只看统计与自检
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
import argparse

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(HERE, 'manuscript')

# --- SUITE-SKIP guard (injected by hand; 见 tools/add_suite_skip_guard.py 的说明) -----
# 本项组装的是**手稿**。公开仓库发行版不含 manuscript/，因此在新 clone 里无法执行 ——
# **显式跳过**，不静默通过也不报失败（否则"一键验证"在新 clone 里第一屏就是红的）。
if not os.path.isdir(MS):
    print('SUITE-SKIP: 缺少 manuscript/ —— 本项组装 Paper A 手稿，'
          '仓库发行版不附带手稿；在开发树中运行同一入口即可完整执行。', flush=True)
    sys.exit(0)
# --- end SUITE-SKIP guard ------------------------------------------------------------

# 组装顺序：题名/摘要 → 各章 → 文献
PARTS = [
    ('draft_v05_title_abstract.md', '题名与摘要', True),
    ('draft_v05_intro.md', '1. Introduction', True),
    ('draft_v05_methods.md', '2. Method', True),
    ('draft_v05_results.md', '3–4. Experiments and Results', True),
    ('draft_v05_discussion.md', '5. Discussion', True),
    ('draft_v05_backmatter.md', '投稿必需件（Data/Code availability 等）', True),
    ('draft_v05_appendix.md', 'Appendix A（验证记录）', True),
    ('draft_v05_references.md', 'References', True),
]

INTERNAL_MARKERS = ('docs/', '红线', '本项目的唯一报告口径', '待补', 'TODO',
                    'red line', 'INTERNAL')

# 🔴 2026-09-25 新增：**投稿版是英文论文，出现任何中日韩字符都只能是内部说明。**
#    起因见 `docs/44` T-12：对抗式审稿发现组装后的 `draft_v05_full.md` 里仍有
#    `> ⚠️ **内部说明：这一节需要作者确认后保留或删改。**` 与
#    `# draft_v05 · Introduction 章节草稿` 这类行 —— 而**组装自检当时报的是"无内部标记残留"**。
#    根因是上面那张词表**靠列举**，而内部说明的措辞是无穷的（"内部说明"就不在表里）。
#    ⇒ 改用**判定规则**而不是词表：区块里有汉字 ⇒ 内部；整行有汉字 ⇒ 丢掉。
CJK = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]')


def is_internal_block(block_lines):
    """判断一个引用块（连续的 `>` 行）是内部标记还是学术性更正。"""
    head = block_lines[0]
    joined = ' '.join(block_lines)
    if head.lstrip().startswith('> 🔴'):
        return True
    if CJK.search(joined):        # 英文论文里出现汉字 ⇒ 一定是内部说明
        return True
    return any(m in joined for m in INTERNAL_MARKERS)


def clean(text, keep_internal):
    """去掉内部标记；`keep_internal=True` 时原样返回。"""
    if keep_internal:
        return text
    # HTML 注释
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    out, block = [], []
    for line in text.splitlines():
        if line.lstrip().startswith('>'):
            block.append(line)
            continue
        if block:
            if not is_internal_block(block):
                out += block
            block = []
        out.append(line)
    if block and not is_internal_block(block):
        out += block
    text = '\n'.join(out)
    # 分稿自己的标题行（`# draft_v05 · … 章节草稿`、`# Paper A · …`）不是论文内容。
    # 它们此前**整行漏进投稿版**，而且会让"顶级章节编号"自检看起来仍然正常。
    text = re.sub(r'(?m)^#\s*(?:draft_v05|Paper A)\b[^\n]*\n?', '', text)
    # 兜底：任何仍含汉字的行一律丢掉（投稿版是英文）。
    text = '\n'.join(l for l in text.splitlines() if not CJK.search(l))
    # 连续的分隔线 `---` 合成一条（删掉内部块后会留下成对的 `---`）。
    text = re.sub(r'(?m)^---[ \t]*\n(?:[ \t]*\n)*^---[ \t]*$', '---', text)
    # 每个文件开头的导语（`> 本文件是 ...` 已被上面处理；这里再清一次空引用）
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', action='store_true', help='只做统计与自检，不写文件')
    args = ap.parse_args()

    pieces, missing = {}, []
    for fn, label, req in PARTS:
        p = os.path.join(MS, fn)
        if not os.path.exists(p):
            missing.append(fn)
            continue
        pieces[fn] = open(p, encoding='utf-8').read()

    if missing:
        sys.exit(f'🔴 缺文件：{missing}')

    pub = []
    for fn, label, req in PARTS:
        t = clean(pieces[fn], keep_internal=False)
        if fn == 'draft_v05_references.md':
            # 参考文献文件是一份**独立文档**，标题是 H1；组装进全文时必须是 H2，
            # 否则它与论文题名同级 ⇒ 章节编号自检与目录都会错。
            t = re.sub(r'^#\s+References.*$', '## References', t, count=1, flags=re.M)
        pub.append(t)
    internal = [pieces[fn] for fn, _, _ in PARTS]

    pub_text = '\n\n---\n\n'.join(pub) + '\n'
    int_text = '\n\n---\n\n'.join(internal) + '\n'

    print('=' * 92)
    print('组装结果')
    print('=' * 92)
    print(f'{"来源文件":34s} {"字符":>8} {"投稿版字符":>10} {"去掉":>7}')
    for (fn, label, _), p in zip(PARTS, pub):
        raw = len(pieces[fn]); cl = len(p)
        print(f'{fn:34s} {raw:>8} {cl:>10} {100*(raw-cl)/max(1,raw):>6.1f}%')

    # ---------------- 自检 ----------------
    print('\n' + '=' * 92)
    print('一致性自检')
    print('=' * 92)
    issues = []

    def wc(s):
        return len(re.findall(r"\b[A-Za-z][A-Za-z\-\']*\b", s))

    # 1) 必备件
    has_title = '# ' in pub_text[:400]
    has_abs = '## 2. Abstract' in pub_text or 'Abstract' in pub_text[:6000]
    has_ref = 'References' in pub_text[-20000:]
    for cond, name in ((has_title, '题名'), (has_abs, '摘要'), (has_ref, '参考文献')):
        print(f'  {"✅" if cond else "❌"} 含{name}')
        if not cond:
            issues.append(f'缺{name}')

    # 2) 章节编号连续性（只看顶级 `## N.`）
    nums = [int(m.group(1)) for m in re.finditer(r'^## (\d+)\.', pub_text, re.M)]
    seq_ok = nums == sorted(nums) and len(nums) == len(set(nums))
    print(f'  {"✅" if seq_ok else "⚠️"} 顶级章节编号：{nums}'
          + ('' if seq_ok else '  ← 有重复或乱序，需人工确认'))
    if not seq_ok:
        issues.append(f'章节编号 {nums}')

    # 3) 投稿版里不得残留内部引用
    #    ⚠️ 例外：作者/单位/通讯作者、以及 CRediT/Acknowledgements 里"待作者名单确定"的
    #    `TODO`，都是**有意的占位符**（按计划最后填）。把它们算成"泄漏"会让自检永远失败、
    #    进而被忽略 —— 那比不检查更糟。
    PLACEHOLDER_OK = ('**Authors:**', '**Affiliations:**', '**Corresponding author:**',
                      'author list', 'AUTHORS_TODO')
    leaked = []
    for m in INTERNAL_MARKERS:
        if m == 'TODO':
            for mm in re.finditer(r'TODO', pub_text):
                ln = pub_text[:mm.start()].count('\n') + 1
                line = pub_text.splitlines()[ln - 1] if ln - 1 < len(pub_text.splitlines()) else ''
                if any(p in line for p in PLACEHOLDER_OK):
                    continue
                leaked.append((m, ln))
            continue
        if m in pub_text:
            for mm in re.finditer(re.escape(m), pub_text):
                ln = pub_text[:mm.start()].count('\n') + 1
                leaked.append((m, ln))
    print(f'  {"✅" if not leaked else "❌"} 投稿版无内部标记残留'
          + (f'（{len(leaked)} 处：{leaked[:6]}）' if leaked else
             '（作者信息占位符已按规则豁免）'))
    if leaked:
        issues.append(f'{len(leaked)} 处内部标记泄漏')

    # 3b) 🔴 硬检查：投稿版里**一个汉字都不能有**（2026-09-25 新增，见 `docs/44` T-12）
    #     上面那张词表是"列举"，而内部说明的措辞列举不完（"内部说明"当时就不在表里）。
    #     这条规则不依赖任何词表：论文是英文，汉字出现即内部。
    cjk_lines = [(i, l.strip()[:90]) for i, l in enumerate(pub_text.splitlines(), 1)
                 if CJK.search(l)]
    print(f'  {"✅" if not cjk_lines else "❌"} 投稿版无非英文（CJK）残留'
          + (f'（{len(cjk_lines)} 行：{cjk_lines[:4]}）' if cjk_lines else ''))
    if cjk_lines:
        issues.append(f'{len(cjk_lines)} 行 CJK 残留')

    # 3c) 🔴 红线措辞扫描（2026-09-25 新增）
    #     这些是本项目**反复约定不得出现**的写法。它们此前靠人工记忆维持，
    #     其中"0.000 mm / exact"确实一度写进了 §2.1.1 的表里（对抗式审稿才发现）。
    #     能机械拦住的就不要靠记性。
    RED_LINES = [
        ('0.000 mm', '把浮点零写成精确零 —— 应写 ≈2×10⁻¹⁵ mm 并注明是机器精度'),
        ('condition number ∞', '把有限的坏条件数写成 ∞ —— 应写 2.9×10¹⁶'),
        ('condition number inf', '同上'),
        ('we prove', '不得声称证明（经典结果须引 Austin 2023）'),
        ('we are the first', '不得声称首创'),
        ('§4.1.2', '该节不存在（曾有过 5 处空引用，正确指向 §3.2.1）'),
        ('sections 6.', '本文只有 5 节 + 附录'),
        ('Sections 6.', '同上'),
        ('PMR folds less', '红线：不得写"PMR 折叠更少"'),
        ('the mask is the root cause', '红线：掩膜不是根因'),
        ('Lung-DIR-QA', '红线：该数据集的数字不得出现在本文'),
    ]
    red_hits = []
    for pat, why in RED_LINES:
        for mm in re.finditer(re.escape(pat), pub_text):
            ln = pub_text[:mm.start()].count('\n') + 1
            red_hits.append((pat, ln, why))
    print(f'  {"✅" if not red_hits else "❌"} 红线措辞扫描'
          + (f'（{len(red_hits)} 处）' if red_hits else '（0 处）'))
    for pat, ln, why in red_hits[:6]:
        print(f'      L{ln}: 「{pat}」 —— {why}')
    if red_hits:
        issues.append(f'{len(red_hits)} 处红线措辞')

    # 4) 字数
    def body_of(t):
        # 切掉参考文献：标题可能是 `## References`（组装后）或 `# References`（分稿）
        m = re.search(r'^#{1,2}\s+References\b', t, re.M)
        return t[:m.start()] if m else t

    full_words = wc(pub_text)
    body_words = wc(body_of(pub_text))
    print(f'  · 投稿版词数：全文 {full_words}，正文（去参考文献）{body_words}')
    # 与已发表同类论文的篇幅对照（若 `lit/softwarex_lengths.csv` 在，给出 SoftwareX 分布；
    # 那只是**经验分布**，不是目标期刊的规定 —— 见 docs/42）
    print(f'  · ⚠️ 正文 {body_words} 词：多数期刊正文在 5,000–8,000 词；'
          f'**目标期刊未定，压缩幅度待定**')

    # 5) 图编号必须与**出现顺序**一致
    #    🔴 2026-09-25 实际踩过：Fig. 1 在 §2.1、**Fig. 3 在 §2.1.1**、Fig. 2 在 §4.4
    #    ⇒ 全文顺序成了 1 → 3 → 2。图注与文件都对，只有**编号**错了。
    figs = [(int(m.group(1)), pub_text[:m.start()].count('\n') + 1)
            for m in re.finditer(r'!\[\*\*Fig\. (\d+)\.', pub_text)]
    nums = [n for n, _ in figs]
    fig_ok = nums == sorted(nums) and nums == list(range(1, len(nums) + 1))
    print(f'  {"✅" if fig_ok else "❌"} 图编号按出现顺序：'
          + '、'.join(f'Fig.{n}(行{l})' for n, l in figs))
    if not fig_ok:
        issues.append(f'图编号顺序错（出现顺序 {nums}）')

    # 6) 图文件必须存在
    missing_fig = [f for f in re.findall(r'\]\((figures/[^)]+)\)', pub_text)
                   if not os.path.exists(os.path.join(MS, f.replace('/', os.sep)))]
    print(f'  {"✅" if not missing_fig else "❌"} 图文件齐全'
          + (f'（缺 {missing_fig}）' if missing_fig else ''))
    if missing_fig:
        issues.append(f'缺图文件 {missing_fig}')

    # 7) 数字守护脚本
    print('  · 数字守护：`tools/verify_paperA_numbers.py` 需在每次改动后重跑')

    if not args.stats:
        for name, txt in (('draft_v05_full.md', pub_text),
                          ('draft_v05_full_internal.md', int_text)):
            p = os.path.join(MS, name)
            open(p, 'w', encoding='utf-8').write(txt)
            print(f'\n已写出 {p}')

    print('\n' + '=' * 92)
    if issues:
        print('🔴 自检发现问题：')
        for x in issues:
            print(f'   · {x}')
    else:
        print('✅ 自检通过。')
    print('=' * 92)
    return 1 if issues else 0


if __name__ == '__main__':
    sys.exit(main())
