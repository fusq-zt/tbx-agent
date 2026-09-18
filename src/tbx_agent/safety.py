from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .capability_answer import TBX_CAPABILITY_ANSWER
from .schemas import AgentResponse, ResponseKind, ReviewStatus, Urgency, VisualResult


class SafetyViolationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RedFlagAssessment:
    urgency: Urgency
    matched: tuple[str, ...]


_EMERGENCY_PATTERNS = {
    "large_or_persistent_hemoptysis": re.compile(
        r"大量(?:咯血|咳血)|(?:咳|咯)(?:出|了)?(?:很多|大量)血|"
        r"咯血.*(?:不止|持续|增多)|吐血.*窒息"
    ),
    "severe_breathlessness": re.compile(r"严重呼吸困难|喘不上气|无法完整说话|窒息"),
    "cyanosis": re.compile(r"发绀|嘴唇.*(?:紫|青)"),
    "confusion_or_fainting": re.compile(r"意识不清|意识异常|晕厥|昏倒|晕过去|抽搐"),
    "severe_chest_pain": re.compile(r"(?:突发|突然).{0,8}(?:剧烈|严重)胸痛|胸痛.*快速加重"),
    "oxygen_saturation_below_90": re.compile(r"(?:血氧|spo2|氧饱和度).{0,8}(?:低于|<|＜)\s*90"),
}

_MEDICATION_CHANGE_REQUEST_PATTERNS = (
    re.compile(
        r"(?:自行|自己|擅自).{0,12}"
        r"(?:开始|停(?:药|掉|用|服|异烟肼|利福平)|停止|换药|加药|减量|加量|调整)"
    ),
    re.compile(
        r"(?:是否|能否|能不能|可否|可不可以|是不是应该|要不要).{0,16}"
        r"(?:开始|停(?:药|掉|用|服|异烟肼|利福平)|停止|换药|加药|减量|加量|调整)"
    ),
    re.compile(
        r"(?:开始|停(?:药|掉|用|服)|停止|换药|加药|减量|加量|调整)"
        r".{0,8}(?:可以吗|行吗|好吗|是否合适)"
    ),
)

_DISEASE_CONFIRMATION_PATTERNS = (
    re.compile(r"(?:已经|已|可以)?确诊(?:为|是)?肺?结核"),
    re.compile(r"肯定是肺?结核"),
)

_DISEASE_EXCLUSION_PATTERNS = (
    re.compile(r"(?:完全|已经|已|可以|能够|能)排除(?:了)?肺?结核"),
    re.compile(r"排除了肺?结核"),
    re.compile(r"肯定不是肺?结核"),
)

_MEDICATION_CHANGE_PATTERNS = (
    re.compile(r"(?:建议|应该|请|可以).{0,8}(?:自行)?(?:开始|停用|停药|换药|加药|减量|加量)"),
)

_PERSONALIZED_REGIMEN_PATTERNS = (
    re.compile(r"(?:每日|每天|每次)\s*\d+(?:\.\d+)?\s*(?:mg|g|片|毫克|克)"),
    re.compile(r"(?:方案|疗程)[:：].{0,40}(?:个月|周)"),
)

_MEDICATION_CONTEXT_PATTERN = re.compile(
    r"药|服用|口服|注射|片剂|胶囊|处方|剂量|抗生素|抗结核|"
    r"异烟肼|利福平|利福喷丁|吡嗪酰胺|乙胺丁醇|贝达喹啉|德拉马尼|"
    r"左氧氟沙星|莫西沙星|利奈唑胺"
)

_UNSUPPORTED_EXECUTION_CLAIM_PATTERN = re.compile(
    r"(?:我|已|已经|刚刚|正在|即将|开始).{0,8}(?:调用|执行|运行|删除|修改|上传|读取|分析)"
    r".{0,12}(?:工具|命令|记录|文件|病例|胸片|图像)"
)

# Free chat carries no image evidence. Both positive and negative patient-level
# findings must go through the typed evidence compositor, even if they avoid
# words such as "diagnosed". These are output checks, not user-intent routing.
_UNSUPPORTED_VISUAL_CLAIM_PATTERNS = (
    re.compile(
        r"(?:这张|该|当前|你的|您的).{0,8}(?:胸片|影像|图像|片子)"
        r".{0,12}(?:显示|提示|可见|存在|发现|正常|异常|未见|没有)"
    ),
    re.compile(
        r"(?:[左右双两](?:侧)?(?:上|中|下)?肺(?:野|叶)?|肺[尖门部野]|胸膜|心影)"
        r".{0,18}(?:可见|存在|发现|显示|未见|没有|增大|增厚|正常|异常|清晰|模糊)"
    ),
    re.compile(
        r"(?:可见|发现|检出|未见|没有|存在).{0,18}"
        r"(?:结节|空洞|浸润|实变|积液|气胸|钙化|斑片影|病灶)"
    ),
    re.compile(
        r"(?:结节|空洞|浸润|实变|积液|病灶|候选区域).{0,12}"
        r"(?:位于|在[左右]|直径|大小(?:为|约)|\d+(?:\.\d+)?\s*(?:厘米|毫米|cm|mm))"
    ),
    re.compile(
        r"\b(?:this|your|the current)\s+(?:chest\s+)?(?:x[- ]?ray|image|radiograph|scan)"
        r".{0,32}\b(?:shows?|reveals?|demonstrates?|contains?|is normal|is abnormal)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:left|right|bilateral)\s+(?:(?:upper|middle|lower)\s+)?"
        r"(?:lung|lobe|lung field).{0,32}"
        r"\b(?:has|shows?|contains?|nodule|opacity|cavity|effusion|clear|normal|abnormal)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:no|there is|there are|detected|identified)\s+(?:evidence of\s+)?"
        r".{0,24}\b(?:nodule|nodules|cavity|cavities|effusion|pneumothorax|lesion|lesions)\b",
        re.IGNORECASE,
    ),
)


def unsupported_general_visual_claim(text: str) -> bool:
    """Reject common patient-finding assertions without claiming exhaustive NLP.

    A conditional example or an explicit inability to assess is not an observed
    finding. The check is clause-scoped so a disclaimer in another sentence
    cannot authorize an invented result.
    """
    for clause in re.split(r"[。！？!?；;\n]+|(?<!\d)\.(?!\d)", text):
        for pattern in _UNSUPPORTED_VISUAL_CLAIM_PATTERNS:
            for match in pattern.finditer(clause):
                prefix = clause[:match.start()]
                if re.search(
                    r"(?:无法|不能|尚不能|不应|不能据此)(?:判断|确定|推断|声称).{0,8}$"
                    r"|(?:如果|假设|例如).{0,8}$"
                    r"|\b(?:cannot|can't|unable to)\s+(?:determine|say|assess)\s*"
                    r"(?:whether\s+)?(?:the\s+)?$"
                    r"|\b(?:if|suppose|for example)\s*$",
                    prefix,
                    re.IGNORECASE,
                ):
                    continue
                return True
    return False

_NON_DIAGNOSTIC_SCOPE_PATTERNS = (
    re.compile(r"不用于确诊(?:或排除)?肺?结核"),
    re.compile(r"不能(?:用于)?确诊(?:或排除)?肺?结核"),
    re.compile(r"不能确诊.*不能排除肺?结核"),
    re.compile(r"辅助筛查.{0,20}不(?:能|用于)确诊"),
)

_CLAIM_NEGATION_BEFORE = re.compile(
    r"(?:不|未|无|无法|不能|不可|不得|不用于|不能用于|尚未|尚不能|难以)$"
)

_IMAGE_RESPONSE_KINDS = {
    ResponseKind.VISUAL_SCREENING_RESULT,
    ResponseKind.LOCALIZATION_RESULT,
    ResponseKind.CASE_EXPLANATION,
    ResponseKind.ACTIVE_SCREENING_QUESTION,
    ResponseKind.ACTIVE_SCREENING_SUMMARY,
}
_CHINESE_LOBE_LOCATION = re.compile(
    r"(?:(?P<side>[左右])肺?|肺)?(?P<zone>[上中下])叶"
)
_ENGLISH_LOBE_LOCATION = re.compile(
    r"\b(?:(?P<side>left|right)\s+)?(?P<zone>upper|middle|lower)\s+lobe\b",
    re.IGNORECASE,
)


def _lung_field_terminology(value: str) -> str:
    """Project unsupported lobe wording back onto validated 2-D lung fields."""

    def chinese(match: re.Match[str]) -> str:
        return f"{match.group('side') or ''}{match.group('zone')}肺野"

    def english(match: re.Match[str]) -> str:
        side = f"{match.group('side')} " if match.group("side") else ""
        return f"{side}{match.group('zone')} lung field"

    normalized = _CHINESE_LOBE_LOCATION.sub(chinese, value)
    return _ENGLISH_LOBE_LOCATION.sub(english, normalized)


def _normalize_code_owned_terminology(response: AgentResponse) -> AgentResponse:
    # The general-chat recovery path does not carry the parsed user intent.
    # Promote the exact code-owned capability answer before execution-claim
    # checks so phrases such as "对上传的胸片进行三分类" describe a product
    # ability rather than being mistaken for an unreceipted completed action.
    if response.summary.strip() == TBX_CAPABILITY_ANSWER:
        return response.model_copy(
            update={
                "response_kind": ResponseKind.CAPABILITY_STATEMENT,
                "summary": TBX_CAPABILITY_ANSWER,
            }
        )
    if response.response_kind == ResponseKind.CAPABILITY_STATEMENT:
        return response.model_copy(update={"summary": TBX_CAPABILITY_ANSWER})
    if response.response_kind not in _IMAGE_RESPONSE_KINDS:
        return response
    summary = _lung_field_terminology(response.summary)
    notes = [_lung_field_terminology(item) for item in response.visual_evidence_notes]
    if summary == response.summary and notes == response.visual_evidence_notes:
        return response
    return response.model_copy(
        update={
            "summary": summary,
            "visual_evidence_notes": notes,
        }
    )


def assess_red_flags(text: str) -> RedFlagAssessment:
    matched = tuple(name for name, pattern in _EMERGENCY_PATTERNS.items() if pattern.search(text))
    return RedFlagAssessment(
        urgency=Urgency.EMERGENCY if matched else Urgency.ROUTINE_INFORMATION,
        matched=matched,
    )


def is_medication_change_request(text: str) -> bool:
    """Detect a request to personally change a medication regimen.

    This recognizes intent only. It never decides whether a change is medically
    appropriate; callers use it to enforce the clinician-review boundary.
    """

    return any(pattern.search(text) for pattern in _MEDICATION_CHANGE_REQUEST_PATTERNS)


def _all_text(response: AgentResponse) -> str:
    values: Iterable[str] = (
        [response.summary]
        + response.visual_evidence_notes
        + response.diagnostic_information
        + response.next_step_information
        + response.treatment_education
        + response.limitations
    )
    return "\n".join(values)


def _contains_unnegated_claim(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    """Return true only for affirmative claim matches.

    Regex engines can otherwise find ``能排除`` inside ``不能排除`` or a bare
    ``确诊`` inside ``不用于确诊``.  Inspecting the short prefix prevents a
    safety disclaimer from being mistaken for the prohibited claim it negates.
    """

    for pattern in patterns:
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 6) : match.start()]
            if not _CLAIM_NEGATION_BEFORE.search(prefix):
                return True
    return False


class SafetyVerifier:
    def __init__(self, policy: dict):
        self.policy = policy
        self.policy_id = str(policy["policy_id"])

    def verify(self, response: AgentResponse) -> AgentResponse:
        response = _normalize_code_owned_terminology(response)
        text = _all_text(response)
        if (
            response.response_kind == ResponseKind.GENERAL_ANSWER
            and unsupported_general_visual_claim(response.summary)
        ):
            raise SafetyViolationError("general answer contains ungrounded image findings")
        if (
            response.response_kind == ResponseKind.GENERAL_ANSWER
            and _UNSUPPORTED_EXECUTION_CLAIM_PATTERN.search(text)
        ):
            raise SafetyViolationError(
                "general answer claims an execution that has no tool receipt"
            )
        if response.response_kind == ResponseKind.SAFE_ABSTENTION and (
            response.diagnostic_information
            or response.next_step_information
            or response.treatment_education
        ):
            raise SafetyViolationError("safe abstention contains domain guidance fields")
        visual_screening_kinds = {
            ResponseKind.ACTIVE_SCREENING_QUESTION,
            ResponseKind.ACTIVE_SCREENING_SUMMARY,
        }
        if response.response_kind == ResponseKind.VISUAL_SCREENING_RESULT:
            if response.visual_result is None:
                raise SafetyViolationError("visual response lacks a visual routing result")
        elif response.response_kind == ResponseKind.LOCALIZATION_RESULT:
            if response.case_id is None:
                raise SafetyViolationError("localization response lacks case provenance")
            if (
                response.visual_result is not None
                or response.predicted_class is not None
                or response.review_status is not None
            ):
                raise SafetyViolationError(
                    "localization response cannot carry classification routing fields"
                )
        elif (
            response.visual_result is not None
            or response.predicted_class is not None
            or response.review_status is not None
            or response.visual_evidence_notes
        ):
            if response.response_kind not in visual_screening_kinds:
                raise SafetyViolationError("non-visual response contains visual routing fields")
            if response.case_id is None or response.visual_result is None:
                raise SafetyViolationError(
                    "case-bound active screening visual fields lack case provenance"
                )
        if response.urgency == Urgency.EMERGENCY:
            if response.diagnostic_information or response.treatment_education:
                raise SafetyViolationError("emergency response contains non-emergency guidance")
            if response.citations:
                raise SafetyViolationError(
                    "local emergency handoff must not borrow guideline citations"
                )
            if not re.search(r"急救|急诊|120|emergency", text, re.IGNORECASE):
                raise SafetyViolationError("emergency response lacks an immediate handoff")
        if self.policy.get(
            "forbid_disease_confirmation_from_agent", True
        ) and _contains_unnegated_claim(text, _DISEASE_CONFIRMATION_PATTERNS):
            raise SafetyViolationError("response contains a disease confirmation claim")
        if self.policy.get(
            "forbid_disease_exclusion_from_agent", True
        ) and _contains_unnegated_claim(text, _DISEASE_EXCLUSION_PATTERNS):
            raise SafetyViolationError("response contains a disease exclusion claim")
        if self.policy.get("forbid_personalized_regimen", True):
            check_regimen = (
                response.response_kind != ResponseKind.GENERAL_ANSWER
                or _MEDICATION_CONTEXT_PATTERN.search(text) is not None
            )
            for pattern in _PERSONALIZED_REGIMEN_PATTERNS:
                if check_regimen and pattern.search(text):
                    raise SafetyViolationError(
                        "response contains a personalized medication instruction"
                    )
        if self.policy.get("forbid_start_stop_switch_medication_instruction", True):
            check_medication_change = (
                response.response_kind != ResponseKind.GENERAL_ANSWER
                or _MEDICATION_CONTEXT_PATTERN.search(text) is not None
            )
            for pattern in _MEDICATION_CHANGE_PATTERNS:
                if check_medication_change and pattern.search(text):
                    raise SafetyViolationError(
                        "response contains a medication start/stop/switch instruction"
                    )
        has_guideline_content = bool(
            response.diagnostic_information
            or response.next_step_information
            or response.treatment_education
        )
        if (
            has_guideline_content
            and self.policy.get("require_citation_for_guideline_claims", True)
            and not response.citations
            and response.urgency != Urgency.EMERGENCY
        ):
            raise SafetyViolationError("guideline content lacks an approved citation")
        citation_keys = [(item.source_id, item.chunk_id) for item in response.citations]
        if len(citation_keys) != len(set(citation_keys)):
            raise SafetyViolationError("response contains duplicate guideline citations")
        if any(not item.support_text.strip() for item in response.citations):
            raise SafetyViolationError("response contains a citation without support text")
        if response.safety_policy_id != self.policy_id:
            raise SafetyViolationError("response safety policy version mismatch")
        if (
            self.policy.get("require_scope_notice", True)
            and response.urgency != Urgency.EMERGENCY
            and response.response_kind != ResponseKind.GENERAL_ANSWER
            and not any(pattern.search(text) for pattern in _NON_DIAGNOSTIC_SCOPE_PATTERNS)
        ):
            raise SafetyViolationError("response lacks an explicit non-diagnostic scope notice")
        if (
            self.policy.get("require_review_state_when_pending", True)
            and response.visual_result == VisualResult.PENDING_HUMAN_REVIEW
            and response.review_status not in {ReviewStatus.NOT_REQUIRED, ReviewStatus.PENDING}
        ):
            raise SafetyViolationError(
                "uncertain visual result has an invalid conversation or review state"
            )
        return response
