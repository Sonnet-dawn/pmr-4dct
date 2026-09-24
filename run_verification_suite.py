"""run_verification_suite.py —— 一键运行全部验证脚本，输出统一报告
================================================================================
论文里主张"一个**可执行**的验证套件"，那它就得真的能一键跑、并且**输出可归档的结果**。
此前 8 个验证脚本散落在项目根目录与 `tools/`，没有任何统一入口，
既无法证明"跑过"，也无法在改动后一键回归。

本脚本：
  * 顺序运行全部验证脚本（各带超时保护），记录**返回码 / 耗时 / 末尾输出**；
  * 汇总为 `results/verification_report.json` + `results/VERIFICATION.md`；
  * 任一脚本失败 ⇒ 进程以非零码退出（可直接接 CI）。

分类
----
  `math`  / `geometry` / `convention` —— **纯代码**测试，不需要训练结果，秒级到分钟级
  `data`  —— 需要 `results/` 里的历史产物，用于核对报告数字与来源是否一致

用法
----
    python run_verification_suite.py            # 全部
    python run_verification_suite.py --quick     # 只跑纯代码类（math/geometry/convention）
    python run_verification_suite.py --only tre_convention
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import os
import io
import re
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# (key, 类别, 脚本, 参数, 超时秒, 说明)
SUITE = [
    ('exact_basis', 'math', 'verify_exact_basis.py', [],
     300, '锚定完备基：{cos kθ−1, sin kθ} 能否精确表示任意锚定相位场'),
    ('projection_exact', 'math', 'verify_projection_exact.py', [],
     300, '9 项周期基对锚定 10 相数据无损（投影残差、条件数）'),
    ('resample_geometry', 'geometry', 'verify_resample_geometry.py', [],
     300, '降采样必须保留 origin/direction（DIRLAB origin=0 曾让此 bug 隐形）'),
    ('tre_convention', 'convention', 'verify_tre_convention.py', [],
     900, 'TRE 约定 + transformix 点文件单位/量化（解析已知答案的合成变换）'),
    ('phase_selection', 'convention', 'verify_phase_selection.py', [],
     300, '相位参数与 landmark 集的一致性（300 点集无中间相位 ⇒ 必须报错，不许静默）'),
    ('dl_model_shapes', 'geometry', 'verify_lapirn_shapes.py', [],
     300, 'DL 基线两个模型的前向形状（lapirn 多级金字塔曾整批静默失败）'),
    ('main_run', 'data', 'verify_main_run.py', ['main_base'],
     600, '主运行结果自洽：闭合恒等式、与 identity 的比较'),
    ('coarse_tre', 'data', 'verify_coarse_tre.py', [],
     600, '4 mm 粗尺度"更好"是否只是评估分辨率的假象'),
    ('lowphase', 'data', 'verify_lowphase.py', [],
     900, '低相位下"闭合误差更小"是否只是复合插值污染'),
    ('paperB_numbers', 'data', 'tools/verify_paperB_numbers.py', [],
     600, 'Paper B 报告数字与 results/ 来源逐条核对'),
]

OK_MARK = re.compile(r'(✅|PASS|通过|一致)')


def resolve(script):
    """两种布局都要能找到脚本：
      · 开发布局：全部在项目根
      · 仓库布局：`verify_*.py` 在 verification/，`tools/` 下的在 tools/"""
    for cand in (script,
                 os.path.join('verification', os.path.basename(script)),
                 os.path.join('tools', os.path.basename(script))):
        p = os.path.join(HERE, cand)
        if os.path.exists(p):
            return p
    return None


def run_one(key, cat, script, argv, timeout):
    path = resolve(script)
    if path is None:
        return {'key': key, 'script': script, 'status': 'missing',
                'rc': None, 'seconds': 0.0, 'tail': f'找不到 {script}'}
    t0 = time.time()
    try:
        r = subprocess.run([PY, path] + argv, cwd=HERE, capture_output=True,
                           timeout=timeout,
                           env=dict(os.environ, PYTHONIOENCODING='utf-8',
                                    PYTHONUNBUFFERED='1'))
        out = (r.stdout or b'').decode('utf-8', 'replace')
        err = (r.stderr or b'').decode('utf-8', 'replace')
        rc, status = r.returncode, None
    except subprocess.TimeoutExpired:
        out = err = ''
        rc, status = None, 'timeout'
    dt = time.time() - t0
    if status is None:
        status = 'pass' if rc == 0 else 'fail'
    tail = [ln for ln in (out + '\n' + err).strip().splitlines() if ln.strip()][-12:]
    return {'key': key, 'script': script, 'status': status, 'rc': rc,
            'seconds': round(dt, 1), 'tail': tail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='只跑纯代码类（math / geometry / convention）')
    ap.add_argument('--only', type=str, default='',
                    help='只跑指定 key（逗号分隔）')
    ap.add_argument('--out', type=str, default='results')
    args = ap.parse_args()

    want = {k.strip() for k in args.only.split(',') if k.strip()}
    todo = []
    for item in SUITE:
        key, cat = item[0], item[1]
        if want and key not in want:
            continue
        if args.quick and cat == 'data':
            continue
        todo.append(item)

    print('=' * 84)
    print(f'验证套件：共 {len(todo)} 项'
          f'{"（--quick：仅纯代码类）" if args.quick else ""}')
    print('=' * 84)

    rows = []
    for key, cat, script, argv, timeout, desc in todo:
        print(f'\n▶ [{cat}] {key}  —— {desc}')
        print(f'   $ python {script} {" ".join(argv)}'.rstrip())
        res = run_one(key, cat, script, argv, timeout)
        res.update({'category': cat, 'description': desc})
        rows.append(res)
        icon = {'pass': '✅', 'fail': '❌', 'timeout': '⏱️', 'missing': '⚠️'}[res['status']]
        print(f'   {icon} {res["status"].upper()}  rc={res["rc"]}  {res["seconds"]:.1f}s')
        for ln in res['tail'][-6:]:
            print(f'      | {ln[:150]}')

    npass = sum(r['status'] == 'pass' for r in rows)
    nfail = sum(r['status'] in ('fail', 'timeout', 'missing') for r in rows)
    print('\n' + '=' * 84)
    print(f'结果：{npass}/{len(rows)} 通过' + (f'，{nfail} 项未通过' if nfail else ''))
    for r in rows:
        if r['status'] != 'pass':
            print(f'  ❌ {r["key"]} ({r["status"]}, rc={r["rc"]})')
    print('=' * 84)

    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE,
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        commit = ''
    report = {'generated_at': datetime.now().isoformat(timespec='seconds'),
              'python': PY, 'git_commit': commit,
              'quick': args.quick, 'n_pass': npass, 'n_total': len(rows),
              'tests': rows}
    os.makedirs(os.path.join(HERE, args.out), exist_ok=True)
    jf = os.path.join(HERE, args.out, 'verification_report.json')
    json.dump(report, open(jf, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    md = [f'# 验证套件报告', '',
          f'* 生成时间：`{report["generated_at"]}`',
          f'* git commit：`{commit or "(未取到)"}`',
          f'* 结果：**{npass}/{len(rows)} 通过**', '',
          '| 类别 | 验证项 | 状态 | 返回码 | 耗时(s) | 说明 |',
          '|---|---|:--:|---:|---:|---|']
    ic = {'pass': '✅ 通过', 'fail': '❌ 失败', 'timeout': '⏱️ 超时', 'missing': '⚠️ 缺失'}
    for r in rows:
        md.append(f'| {r["category"]} | `{r["key"]}` | {ic[r["status"]]} | '
                  f'{r["rc"]} | {r["seconds"]} | {r["description"]} |')
    md += ['', '## 各项末尾输出', '']
    for r in rows:
        md += [f'### `{r["key"]}` — {ic[r["status"]]}', '', '```']
        md += r['tail'] or ['(无输出)']
        md += ['```', '']
    mf = os.path.join(HERE, args.out, 'VERIFICATION.md')
    open(mf, 'w', encoding='utf-8').write('\n'.join(md) + '\n')

    print(f'\n已保存 {jf}\n已保存 {mf}')
    sys.exit(1 if nfail else 0)


if __name__ == '__main__':
    main()
