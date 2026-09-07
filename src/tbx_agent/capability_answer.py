"""Code-owned TBX-Agent capability description for tool-free help turns."""

import re

TBX_CAPABILITY_ANSWER = (
    "我是 TBX-Agent，主要用于肺结核胸片辅助筛查和结核知识问答。我可以对上传的"
    "胸片进行健康、非结核异常或 TB 三分类，按需标出主要候选区域，并在肺野分析"
    "可用时说明左右侧及上、中、下二维肺野位置；还可以检索受审核指南，回答主动"
    "筛查、进一步检查和一般治疗教育问题。我不会把二维肺野当作肺叶。本系统不用于"
    "确诊或排除肺结核，也不提供个体化处方。"
)

_CAPABILITY_QUESTION = re.compile(
    r"(?:你|系统|tbx[- ]?agent).{0,10}(?:会|能|可以).{0,8}(?:干|做|帮)|"
    r"(?:你|系统|tbx[- ]?agent).{0,10}(?:功能|能力)|"
    r"(?:介绍|说明|列出).{0,8}(?:功能|能力|工具)|"
    r"(?:有什么|有哪些).{0,6}(?:功能|能力|工具)|"
    r"what\s+can\s+you\s+do|capabilit(?:y|ies)",
    re.IGNORECASE,
)


def is_tbx_capability_question(query: str) -> bool:
    """Recognize a request for this product's abilities, not a domain task."""

    return bool(_CAPABILITY_QUESTION.search(query.strip()))
