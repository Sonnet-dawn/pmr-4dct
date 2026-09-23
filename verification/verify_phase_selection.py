"""verify_phase_selection.py —— 相位参数 `--tre-phases` / `--lm-set` 的**纯代码**验证
================================================================================
**为什么要有这一项**（`docs/43`）：我们给 `pmr_v2.py` 加了"多相位 TRE"能力。它有一个
**静默失败面**：300 点标注集**只有 T00/T50**，如果有人用 `--lm-set 300 --tre-phases 3`，
最坏情况是**悄悄拿不到中间相位的点**、或者拿错点，而结果文件里看不出异常。
所以这条规则必须被**锁死**在验证套件里，而不是只写在注释里。

本脚本**不需要数据、不需要 GPU、不需要训练**（秒级），因此可以进 CI。

覆盖内容
--------
1. 空 `--tre-phases` ⇒ **不评**（保证历史结果文件与主结果数字不受影响）；
2. 解析正确：去重、排序；
3. 越界相位（6 / −1 / 99）**必须报错**；
4. **`lm_set='300'` 请求中间相位必须报错**（核心防护）；
5. `lm_set='4d75'` 允许 0–5；
6. `load_lm_4d75` 对越界相位必须报错（不许静默取到别的文件）；
7. CLI 默认值确实是"不评 + 300"，即**附加功能默认关闭**。

用法: python verify_phase_selection.py
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

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import pmr_v2
except Exception as e:                                    # pragma: no cover
    print(f'❌ 无法 import pmr_v2：{type(e).__name__}: {e}')
    print('   （本项需要 pmr_v2.py 可导入；它只依赖 torch / SimpleITK / numpy，不需要数据）')
    sys.exit(1)

n_pass = n_fail = 0


def check(name, cond, extra=''):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f'  ✅ {name}{("  " + extra) if extra else ""}')
    else:
        n_fail += 1
        print(f'  ❌ {name}{("  " + extra) if extra else ""}')


def raises(fn, *a):
    try:
        fn(*a)
        return None
    except Exception as e:
        return e


print('=' * 88)
print('1) 空 `--tre-phases` ⇒ 不评（主结果不受影响）')
print('=' * 88)
check("resolve_tre_phases('', '300') == []", pmr_v2.resolve_tre_phases('', '300') == [])
check("resolve_tre_phases('   ', '4d75') == []", pmr_v2.resolve_tre_phases('   ', '4d75') == [])

print('\n' + '=' * 88)
print('2) 解析：去重 + 排序')
print('=' * 88)
check("'1,3,2' + 4d75 → [1,2,3]", pmr_v2.resolve_tre_phases('1,3,2', '4d75') == [1, 2, 3])
check("'1,1,2, 2' + 4d75 → [1,2]", pmr_v2.resolve_tre_phases('1,1,2, 2', '4d75') == [1, 2])
check("'5' + 300 → [5]", pmr_v2.resolve_tre_phases('5', '300') == [5])

print('\n' + '=' * 88)
print('3) 越界相位必须报错（不许静默）')
print('=' * 88)
for spec in ('6', '-1', '99', '0,7'):
    e = raises(pmr_v2.resolve_tre_phases, spec, '4d75')
    check(f"'{spec}' 被拒绝", isinstance(e, ValueError))

print('\n' + '=' * 88)
print('4) 🔴 核心防护：300 点集只有 T00/T50，请求中间相位必须报错')
print('=' * 88)
for spec in ('1', '4', '0,1,2,3,4'):
    e = raises(pmr_v2.resolve_tre_phases, spec, '300')
    check(f"lm_set=300 + tre_phases='{spec}' 被拒绝", isinstance(e, ValueError),
          f'→ {type(e).__name__}' if e else '→ 未报错（会被静默接受！）')
check("lm_set=300 + tre_phases='5' 允许（T50 本来就有）",
      pmr_v2.resolve_tre_phases('5', '300') == [5])

print('\n' + '=' * 88)
print('5) lm_set=4d75 允许 0–5 全部')
print('=' * 88)
check("'0,1,2,3,4,5' → [0..5]",
      pmr_v2.resolve_tre_phases('0,1,2,3,4,5', '4d75') == [0, 1, 2, 3, 4, 5])

print('\n' + '=' * 88)
print('6) load_lm_4d75 对越界相位必须报错')
print('=' * 88)
e = raises(pmr_v2.load_lm_4d75, 1, 9)
check('load_lm_4d75(1, 9) 被拒绝', isinstance(e, ValueError),
      f'→ {type(e).__name__}: {e}' if e else '→ 未报错')

print('\n' + '=' * 88)
print('7) CLI 默认值 = 附加功能默认关闭')
print('=' * 88)
src = open(os.path.join(os.path.dirname(os.path.abspath(pmr_v2.__file__)), 'pmr_v2.py'),
           encoding='utf-8').read()
check("--tre-phases 默认 ''",
      bool(re.search(r"add_argument\('--tre-phases'.*?default=''", src, re.S)))
check("--lm-set choices=['300','4d75'] 且默认 '300'",
      "choices=['300', '4d75']" in src and re.search(r"--lm-set.*?default='300'", src, re.S)
      is not None)
check("tre_by_phase 写入结果且为空时为 None（不污染历史字段语义）",
      "'tre_by_phase': tre_by_phase or None" in src)

print('\n' + '=' * 88)
print(f'结果：{n_pass} 通过 / {n_fail} 失败')
print('=' * 88)
if n_fail == 0:
    print('结论：相位选择与 landmark 集的一致性**被锁死**；'
          '"用 300 点集评中间相位"这类错误无法静默通过。')
sys.exit(1 if n_fail else 0)
