"""The single definition of "a placeholder that is deliberately deferred".

WHY THIS FILE EXISTS
--------------------
Two independent checkers have to agree on which `TODO` markers are *deliberate*
and which are genuine leaks:

  * `tools/assemble_paperA.py`        — refuses to emit a submission version that
                                        still contains an internal marker
  * `tools/audit_completeness.py`     — reports intentional vs unintended TODOs

They previously each carried their own copy of the exemption rule, and a change to
one silently desynchronised them. That is the same root cause the project has now
hit four times: T-2 (ablation numbers), T-5 (defect counts), T-14 (archive
identifiers), T-16 (word counts). The fix every time is the same: one definition,
imported, not copied.

A placeholder is deferred when it needs an **outside input**:

  * author / affiliation / corresponding author  — the author list
  * CRediT, Acknowledgements                     — the author list
  * Declaration of competing interests           — the authors' own disclosure
  * Funding                                      — only the authors know whether a
                                                   grant supported the work; a
                                                   "no specific funding" statement
                                                   is required even when there is none
  * Supporting information                       — couples to which tables move to
                                                   supplementary material, which
                                                   depends on the target journal
"""

from __future__ import annotations

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

import re

# Line-content markers that make a `TODO` deferred regardless of section.
PLACEHOLDER_OK = (
    '**Authors:**', '**Affiliations:**', '**Corresponding author:**',
    'author list', 'AUTHORS_TODO',
)

# Section headings under which a bare `TODO` is deferred.
DEFERRED_SECTIONS = (
    'CRediT',
    'Declaration of competing',
    'Acknowledgements',
    'Funding',
    'Supporting information',
)

DEFERRED_SECTION_RE = re.compile(r'##\s+(?:' + '|'.join(map(re.escape, DEFERRED_SECTIONS)) + r')')

# How many preceding lines define "which section a line is in".
CONTEXT_LINES = 30


def is_deferred_todo(lines, lineno_1based: int) -> bool:
    """True when the `TODO` on this 1-based line is a deliberate placeholder.

    `lines` is the full list of lines of the text the line number refers to.
    """
    i = lineno_1based - 1
    if not (0 <= i < len(lines)):
        return False
    if any(p in lines[i] for p in PLACEHOLDER_OK):
        return True
    ctx = '\n'.join(lines[max(0, i - CONTEXT_LINES):i])
    return bool(DEFERRED_SECTION_RE.search(ctx))
