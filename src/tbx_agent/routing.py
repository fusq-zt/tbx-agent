from __future__ import annotations

from .safety import assess_red_flags
from .schemas import Urgency
from .tools.contracts import ToolName

ROUTER_POLICY_ID = "tbx-router-zh-context-v6"

_CASE_TERMS = (
    "结果",
    "胸片",
    "模型",
    "病灶",
    "候选框",
    "概率",
    "分数",
    "这张",
    "这个结果",
    "刚才",
    "case",
)
_CASE_LOCALIZATION_TARGETS = (
    "病灶",
    "候选框",
    "异常区",
    "异常区域",
    "阴影",
    "位置",
    "部位",
)
_CASE_LOCALIZATION_QUESTIONS = ("在哪", "哪里", "何处", "定位", "位于", "位置")
_CASE_RATIONALE_QUESTIONS = (
    "为什么",
    "为何",
    "依据",
    "理由",
    "凭什么",
    "怎么看出",
)
_CASE_RATIONALE_SUBJECTS = (
    "tb",
    "结核",
    "模型",
    "结果",
    "识别",
    "判断",
    "认为",
    "这张",
    "刚才",
)
_CASE_COMPARISON_REFERENCES = (
    "半年前",
    "以前",
    "之前",
    "上次",
    "上一张",
    "既往",
    "历史片",
    "旧片",
)
_CASE_COMPARISON_ACTIONS = (
    "相比",
    "比较",
    "对比",
    "变化",
    "恶化",
    "好转",
    "进展",
    "加重",
    "减轻",
)
_CASE_QUALITY_TARGETS = ("图像", "胸片", "片子", "影像", "画面", "这张片")
_CASE_QUALITY_TERMS = (
    "质量",
    "清晰",
    "模糊",
    "曝光",
    "旋转",
    "伪影",
    "太黑",
    "太白",
)
_TREATMENT_TERMS = (
    "治疗",
    "用药",
    "服药",
    "药物",
    "耐药",
    "停药",
    "停掉",
    "停用",
    "换药",
    "加药",
    "减量",
    "加量",
    "方案",
    "剂量",
    "疗程",
    "住院",
    "入院",
    "出院",
    "门诊",
    "社区治疗",
    "社区照护",
    "居家治疗",
    "在家治疗",
    "去中心化照护",
    "流动照护",
)
_DIAGNOSTIC_TERMS = (
    "诊疗",
    "检查",
    "诊断",
    "确诊",
    "排除",
    "阴性",
    "阳性",
    "痰",
    "涂片",
    "培养",
    "naat",
    "xpert",
    "药敏",
    "tst",
    "igra",
    "结核抗体",
    "影像",
    "指南",
    "引用",
    "主动筛查",
    "筛查人群",
    "高风险人群",
    "高危人群",
    "重点人群",
    "口罩",
    "感染控制",
    "隔离",
)

_CONTEXTUAL_FOLLOW_UPS = frozenset(
    {
        "给出",
        "继续",
        "可以",
        "同意",
        "好的",
        "好",
        "请说",
        "展开",
        "具体呢",
        "下一步",
    }
)
_MEMORABLE_TOOLS = frozenset(
    {
        ToolName.GET_EXACT_CASE_AND_EXPLAIN,
        ToolName.LOCALIZE_CURRENT_CXR,
        ToolName.INSPECT_IMAGE_QUALITY,
        ToolName.RETRIEVE_GUIDELINE,
    }
)


def case_question_focus(message: str) -> str | None:
    """Return the case-evidence focus for high-confidence follow-up wording.

    This recognizer intentionally requires both a localization target and a
    location question, or both a rationale question and a model/result subject.
    It therefore does not steal generic questions such as ``在哪里检查`` or
    ``为什么要做痰检查`` from the guideline tool.
    """

    lowered = message.casefold()
    if any(term in lowered for term in _CASE_COMPARISON_REFERENCES) and any(
        term in lowered for term in _CASE_COMPARISON_ACTIONS
    ):
        return "comparison"
    if any(term in lowered for term in _CASE_QUALITY_TARGETS) and any(
        term in lowered for term in _CASE_QUALITY_TERMS
    ):
        return "quality"
    if any(term in lowered for term in _CASE_LOCALIZATION_TARGETS) and any(
        term in lowered for term in _CASE_LOCALIZATION_QUESTIONS
    ):
        return "localization"
    if any(term in lowered for term in _CASE_RATIONALE_QUESTIONS) and any(
        term in lowered for term in _CASE_RATIONALE_SUBJECTS
    ):
        return "rationale"
    if any(term in lowered for term in _CASE_TERMS):
        return "general"
    return None


def route_tool(
    message: str,
    case_id: str | None,
    *,
    active_intent: str | ToolName | None = None,
) -> ToolName:
    """Select one allowlisted tool, resolving short follow-ups from thread state.

    ``active_intent`` is a structured tool identifier stored by the service.  No
    free-text conversation content is persisted or replayed into routing.
    """

    lowered = message.casefold()
    if assess_red_flags(message).urgency == Urgency.EMERGENCY:
        return ToolName.EMERGENCY_TRIAGE
    if any(word in lowered for word in _TREATMENT_TERMS):
        return ToolName.RETRIEVE_GUIDELINE
    case_focus = case_question_focus(message) if case_id else None
    if case_focus == "localization":
        return ToolName.LOCALIZE_CURRENT_CXR
    if case_focus == "comparison":
        return ToolName.COMPARE_WITH_PRIOR_CXR
    if case_focus == "quality":
        return ToolName.INSPECT_IMAGE_QUALITY
    if case_focus == "rationale":
        return ToolName.GET_EXACT_CASE_AND_EXPLAIN
    if any(word in lowered for word in _DIAGNOSTIC_TERMS):
        return ToolName.RETRIEVE_GUIDELINE
    if case_id and case_focus is not None:
        return ToolName.GET_EXACT_CASE_AND_EXPLAIN
    normalized = lowered.strip().rstrip("。！!？?")
    if normalized in _CONTEXTUAL_FOLLOW_UPS and active_intent:
        try:
            remembered = ToolName(str(active_intent))
        except ValueError:
            remembered = None
        if remembered in _MEMORABLE_TOOLS:
            return remembered
    return ToolName.DESCRIBE_AGENT_CAPABILITIES
