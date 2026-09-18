"""A narrow Chinese permission guard for explicit prohibitions on new execution.

This is deliberately not an intent router: an empty result grants nothing and
never selects a tool. Conditional requests and questions about not executing
remain the model's responsibility. The caller must apply the returned deny set
at the execution boundary as well as when presenting available actions.
"""

from __future__ import annotations

import re

from .llm.tool_calling import HighLevelToolName

_CLAUSE_BOUNDARY = re.compile(r"[，,。.!！?？;；\n]+|但是|然后|但(?=不要|别|先|暂|请|要)")
_NEGATION = re.compile(r"不要|(?<![分特识辨区])别|禁止|先不|暂不|不用|不需要")
_QUESTION = re.compile(r"是否|为什么|为何|为啥|是不是|要不要|需不需要")
_CONDITION = re.compile(r"如果|假如|除非|只有|若|(?:正常|异常|健康|阳性|阴性|可疑).*?(?:就|才)")
_CONDITIONAL_CONTINUATION = re.compile(r"^(?:那就|那么|就|则)")
_MODIFIERS = re.compile(r"^(?:(?:重新|重复|继续|立即|马上|现在|暂时|帮我|给我|为我|再|先)|\s)*")
_RUN = re.compile(r"^(?:运行|执行|启动|调用|分析|跑|做|进行)")
_DIRECT_TOOL_ACTION = re.compile(
    r"^(?:(?:胸片)?分类|(?:候选区域)?(?:定位|检测)|(?:肺野|肺部)?分割|"
    r"(?:知识|指南)?(?:检索|搜索)|(?:标|画)框)"
)
_RESULT_NOUN = re.compile(r"^(?:结果|分数|概率|报告|结论|标签)")
_OBJECTS = {
    HighLevelToolName.CLASSIFY_CXR: re.compile(r"分类"),
    HighLevelToolName.LOCALIZE_CXR: re.compile(r"定位|检测|(?:标|画)框"),
    HighLevelToolName.ANALYZE_LUNG_ANATOMY: re.compile(r"分割|肺野"),
    HighLevelToolName.SEARCH_TB_KNOWLEDGE: re.compile(r"检索|搜索"),
}


def prohibited_tools(query: str) -> set[HighLevelToolName]:
    """Return tools explicitly prohibited by the current user message.

    A bare imperative such as ``现在不要运行`` prohibits every new tool call.
    An explicit object such as ``不用重新分类`` prohibits only that tool.
    Punctuation separates scopes; conditional wording is intentionally ignored
    here so, for example, a healthy-only branch cannot disable all localization.

    This small grammar covers direct Chinese imperatives, not arbitrary
    paraphrases, historical/quoted commands, or negation with distant scope.
    It does not override a prohibition with a later positive instruction.
    """

    denied: set[HighLevelToolName] = set()
    previous_conditional = False
    for raw_clause in _CLAUSE_BOUNDARY.split(query):
        clause = raw_clause.strip()
        if not clause:
            continue
        conditional = bool(_CONDITION.search(clause))
        continuation = previous_conditional and bool(_CONDITIONAL_CONTINUATION.match(clause))
        previous_conditional = conditional
        if continuation:
            continue
        negations = list(_NEGATION.finditer(clause))
        for index, negation in enumerate(negations):
            prefix = clause[:negation.start()]
            if _QUESTION.search(clause[:negation.end()]) or _CONDITION.search(prefix):
                continue
            end = negations[index + 1].start() if index + 1 < len(negations) else len(clause)
            tail = clause[negation.end():end].strip()
            tail = _MODIFIERS.sub("", tail)
            direct = _DIRECT_TOOL_ACTION.match(tail)
            if not _RUN.match(tail) and direct is None:
                continue
            if direct is not None and _RESULT_NOUN.match(tail[direct.end():]):
                continue
            # Include a preceding subject: "肺野分割不要运行". Do not carry
            # objects across punctuation into a bare global prohibition.
            scope = prefix + tail
            objects = {tool for tool, pattern in _OBJECTS.items() if pattern.search(scope)}
            denied.update(objects or set(HighLevelToolName))
    return denied


__all__ = ["prohibited_tools"]
