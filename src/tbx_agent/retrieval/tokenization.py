from __future__ import annotations

import re

_UNIT_RE = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*|[\u3400-\u9fff]")


def tokenize(text: str) -> tuple[str, ...]:
    """Deterministic dependency-free tokenizer for the sparse safety baseline."""

    units = [item.lower() for item in _UNIT_RE.findall(text)]
    chinese_runs: list[str] = []
    current = ""
    for item in units:
        if len(item) == 1 and "\u3400" <= item <= "\u9fff":
            current += item
        elif current:
            chinese_runs.append(current)
            current = ""
    if current:
        chinese_runs.append(current)
    ngrams = [
        run[index : index + size]
        for run in chinese_runs
        for size in (2, 3)
        for index in range(max(0, len(run) - size + 1))
    ]
    return tuple([*units, *ngrams])
