from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from .config import PROJECT_ROOT
from .schemas import (
    ActiveScreeningSession,
    AgentResponse,
    Citation,
    ResponseKind,
    ScreeningQuestion,
    ScreeningSummary,
    Urgency,
    utc_now,
)

DEFAULT_QUESTION_BANK_PATH = PROJECT_ROOT / "knowledge" / "active_screening_questions.json"


class ScreeningError(ValueError):
    """Base error for deterministic active-screening operations."""


class ScreeningStateError(ScreeningError):
    """Raised when an action is invalid for the current session state."""


class ScreeningInputError(ScreeningError):
    """Raised when a submitted answer does not match the question contract."""


@dataclass(frozen=True, slots=True)
class ScreeningQuestionBank:
    guideline_rule_version: str
    consent_notice: str
    scope_notice: str
    special_answers: dict[str, str]
    questions: tuple[ScreeningQuestion, ...]
    rules: dict[str, Any]
    summary_citations: dict[str, Citation]

    def get_question(self, question_id: str) -> ScreeningQuestion:
        for question in self.questions:
            if question.question_id == question_id:
                return question
        raise KeyError(f"unknown screening question: {question_id}")


def load_question_bank(
    path: str | Path = DEFAULT_QUESTION_BANK_PATH,
) -> ScreeningQuestionBank:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScreeningInputError("active-screening question bank must be an object")
    allowed_top_level = {
        "schema_version",
        "guideline_rule_version",
        "consent_notice",
        "scope_notice",
        "sources",
        "special_answers",
        "questions",
        "rules",
        "summary_citations",
    }
    unknown_top_level = set(payload) - allowed_top_level
    if unknown_top_level:
        raise ScreeningInputError(
            f"question bank contains unknown fields: {sorted(unknown_top_level)}"
        )
    if payload.get("schema_version") != 1:
        raise ScreeningInputError("unsupported active-screening question-bank schema")

    raw_questions = payload.get("questions", [])
    if not isinstance(raw_questions, list):
        raise ScreeningInputError("question bank questions must be a list")
    try:
        questions = tuple(ScreeningQuestion.model_validate(raw) for raw in raw_questions)
    except ValidationError as exc:
        raise ScreeningInputError("question bank contains an invalid question") from exc
    if not questions:
        raise ScreeningInputError("active-screening question bank is empty")

    question_ids = [question.question_id for question in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ScreeningInputError("active-screening question ids must be unique")

    seen: dict[str, ScreeningQuestion] = {}
    for question in questions:
        unknown_dependencies = set(question.ask_if) - set(seen)
        if unknown_dependencies:
            unknown = ", ".join(sorted(unknown_dependencies))
            raise ScreeningInputError(
                f"question {question.question_id} has forward or unknown dependencies: {unknown}"
            )
        _validate_question_shape(question)
        _validate_question_dependencies(question, seen)
        seen[question.question_id] = question

    raw_special_answers = payload.get("special_answers", {})
    if not isinstance(raw_special_answers, dict):
        raise ScreeningInputError("question bank special answers must be a mapping")
    special_answers = dict(raw_special_answers)
    required_tokens = {"unknown", "skip", "cancel"}
    if set(special_answers) != required_tokens:
        raise ScreeningInputError("question bank must define unknown, skip, and cancel tokens")
    if any(not isinstance(value, str) or not value for value in special_answers.values()):
        raise ScreeningInputError("question bank special-answer tokens must be non-empty strings")
    if len(set(special_answers.values())) != len(special_answers):
        raise ScreeningInputError("question bank special-answer tokens must be distinct")

    raw_rules = payload.get("rules", {})
    if not isinstance(raw_rules, dict):
        raise ScreeningInputError("question bank rules must be a mapping")
    rules = dict(raw_rules)
    try:
        _validate_rule_shape(rules)
        _validate_rule_question_ids(rules, set(question_ids))
        _validate_rule_values(rules, seen)
    except ScreeningInputError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ScreeningInputError("question bank contains invalid rule values") from exc
    try:
        citations = {
            key: Citation.model_validate(value)
            for key, value in payload.get("summary_citations", {}).items()
        }
    except (AttributeError, TypeError, ValidationError) as exc:
        raise ScreeningInputError("question bank contains invalid citations") from exc
    required_citations = {"base", "symptoms", "high_risk", "priority", "referral"}
    if not required_citations.issubset(citations):
        raise ScreeningInputError("question bank is missing required summary citations")

    version = str(payload.get("guideline_rule_version", "")).strip()
    consent_notice = str(payload.get("consent_notice", "")).strip()
    scope_notice = str(payload.get("scope_notice", "")).strip()
    if not version or not consent_notice or not scope_notice:
        raise ScreeningInputError("question bank version and notices are required")

    return ScreeningQuestionBank(
        guideline_rule_version=version,
        consent_notice=consent_notice,
        scope_notice=scope_notice,
        special_answers=special_answers,
        questions=questions,
        rules=rules,
        summary_citations=citations,
    )


def _validate_question_shape(question: ScreeningQuestion) -> None:
    choice_types = {"single_choice", "multi_choice"}
    if question.answer_type in choice_types:
        if not question.choices or len(question.choices) != len(set(question.choices)):
            raise ScreeningInputError(
                f"question {question.question_id} choices must be non-empty and unique"
            )
        if any(not item.strip() or len(item) > 200 for item in question.choices):
            raise ScreeningInputError(f"question {question.question_id} contains an invalid choice")
    elif question.choices:
        raise ScreeningInputError(
            f"question {question.question_id} must not declare unused choices"
        )


def _validate_question_dependencies(
    question: ScreeningQuestion,
    previous: dict[str, ScreeningQuestion],
) -> None:
    special_values = {"__unknown__", "__skipped__"}
    for dependency, raw_expected in question.ask_if.items():
        parent = previous[dependency]
        expected = raw_expected if isinstance(raw_expected, list) else [raw_expected]
        if not expected:
            raise ScreeningInputError(
                f"question {question.question_id} has an empty dependency condition"
            )
        if parent.answer_type == "boolean":
            allowed: set[Any] = {True, False, *special_values}
        elif parent.answer_type in {"single_choice", "multi_choice"}:
            allowed = {*parent.choices, *special_values}
        else:
            # Free-text/integer branching would require an explicit comparison
            # schema rather than implicit Python equality.
            raise ScreeningInputError(
                f"question {question.question_id} depends on unsupported answer type"
            )
        if any(value not in allowed for value in expected):
            raise ScreeningInputError(
                f"question {question.question_id} has an invalid dependency value"
            )


def _validate_rule_shape(rules: dict[str, Any]) -> None:
    required = {
        "emergency_question_id",
        "emergency_none_choice",
        "emergency_choice_codes",
        "age_question_id",
        "child_age_value",
        "adult_age_values",
        "older_age_value",
        "symptom_question_ids",
        "symptom_none_choices",
        "high_risk_boolean_question_ids",
        "high_risk_multi_question_ids",
        "priority_multi_question_ids",
        "high_incidence_question_id",
        "exclusive_choices",
    }
    missing = required - set(rules)
    unknown = set(rules) - required
    if missing or unknown:
        raise ScreeningInputError(
            f"question bank rule fields mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _validate_rule_values(
    rules: dict[str, Any],
    questions: dict[str, ScreeningQuestion],
) -> None:
    emergency = questions[rules["emergency_question_id"]]
    if emergency.answer_type != "multi_choice":
        raise ScreeningInputError("emergency question must be multi-choice")
    if rules["emergency_none_choice"] not in emergency.choices:
        raise ScreeningInputError("emergency none choice is outside the question choices")
    emergency_codes = rules["emergency_choice_codes"]
    if not isinstance(emergency_codes, dict) or any(
        choice not in emergency.choices or not isinstance(code, str) or not code
        for choice, code in emergency_codes.items()
    ):
        raise ScreeningInputError("emergency choice-code mapping is invalid")

    age = questions[rules["age_question_id"]]
    age_values = {
        rules["child_age_value"],
        rules["older_age_value"],
        *rules["adult_age_values"],
    }
    if age.answer_type != "single_choice" or not age_values.issubset(age.choices):
        raise ScreeningInputError("age routing values are outside the age question")

    for question_id in rules["symptom_question_ids"]:
        if questions[question_id].answer_type not in {"boolean", "multi_choice"}:
            raise ScreeningInputError("symptom rules must reference boolean or multi-choice")
    for question_id in rules["high_risk_boolean_question_ids"]:
        if questions[question_id].answer_type != "boolean":
            raise ScreeningInputError("high-risk boolean rule references a non-boolean question")
    if questions[rules["high_incidence_question_id"]].answer_type != "boolean":
        raise ScreeningInputError("high-incidence rule references a non-boolean question")

    none_choice_mappings = (
        rules["symptom_none_choices"],
        rules["high_risk_multi_question_ids"],
        rules["priority_multi_question_ids"],
    )
    for mapping in none_choice_mappings:
        if not isinstance(mapping, dict):
            raise ScreeningInputError("none-choice rule must be a mapping")
        for question_id, none_choice in mapping.items():
            question = questions[question_id]
            if question.answer_type != "multi_choice" or none_choice not in question.choices:
                raise ScreeningInputError("none-choice rule is inconsistent with its question")
    for question_id, exclusive in rules["exclusive_choices"].items():
        question = questions[question_id]
        if not isinstance(exclusive, list) or not set(exclusive).issubset(question.choices):
            raise ScreeningInputError("exclusive-choice rule is inconsistent with its question")


def _validate_rule_question_ids(rules: dict[str, Any], known_ids: set[str]) -> None:
    scalar_keys = {
        "emergency_question_id",
        "age_question_id",
        "high_incidence_question_id",
    }
    list_keys = {"symptom_question_ids", "high_risk_boolean_question_ids"}
    mapping_keys = {
        "symptom_none_choices",
        "high_risk_multi_question_ids",
        "priority_multi_question_ids",
        "exclusive_choices",
    }
    referenced: set[str] = set()
    for key in scalar_keys:
        value = rules.get(key)
        if isinstance(value, str):
            referenced.add(value)
    for key in list_keys:
        referenced.update(str(item) for item in rules.get(key, []))
    for key in mapping_keys:
        referenced.update(str(item) for item in rules.get(key, {}))
    unknown = referenced - known_ids
    if unknown:
        raise ScreeningInputError(f"rules reference unknown questions: {sorted(unknown)}")


class ScreeningEngine:
    """Versioned, deterministic questionnaire engine; it never estimates TB probability."""

    _UNKNOWN_ALIASES = frozenset({"unknown", "不知道", "不清楚", "不确定"})
    _SKIP_ALIASES = frozenset({"skip", "跳过", "不愿回答", "暂不回答"})
    _CANCEL_ALIASES = frozenset({"cancel", "取消", "退出", "停止问询"})
    _TRUE_ALIASES = frozenset({"true", "yes", "y", "1", "是", "有"})
    _FALSE_ALIASES = frozenset({"false", "no", "n", "0", "否", "没有", "无"})

    def __init__(
        self,
        question_bank_path: str | Path = DEFAULT_QUESTION_BANK_PATH,
    ) -> None:
        self.bank = load_question_bank(question_bank_path)

    def start_session(
        self,
        *,
        thread_id: str,
        user_id: str,
        owner_scope: str,
        case_id: str | None = None,
        consent: bool | None = None,
        session_id: str | None = None,
    ) -> ActiveScreeningSession:
        if consent is None:
            status = "consent_pending"
            consent_value = False
        elif consent:
            status = "collecting"
            consent_value = True
        else:
            status = "cancelled"
            consent_value = False

        session = ActiveScreeningSession(
            session_id=session_id or f"screen-{uuid4().hex}",
            thread_id=thread_id,
            user_id=user_id,
            owner_scope=owner_scope,
            case_id=case_id,
            consent=consent_value,
            guideline_rule_version=self.bank.guideline_rule_version,
            status=status,
        )
        return self._advance(session) if consent else session

    def set_consent(
        self,
        session: ActiveScreeningSession,
        *,
        granted: bool,
    ) -> ActiveScreeningSession:
        self._ensure_version(session)
        if session.status != "consent_pending":
            raise ScreeningStateError("consent can only be set while consent is pending")
        updated = session.model_copy(deep=True)
        if not granted:
            return self.cancel(updated)
        updated.consent = True
        updated.status = "collecting"
        updated.updated_at = utc_now()
        return self._advance(updated)

    def cancel(self, session: ActiveScreeningSession) -> ActiveScreeningSession:
        self._ensure_version(session)
        if session.status == "complete":
            raise ScreeningStateError("a completed screening session cannot be cancelled")
        updated = session.model_copy(deep=True)
        updated.consent = False
        updated.status = "cancelled"
        updated.answers = {}
        updated.next_question_id = None
        updated.result = None
        updated.updated_at = utc_now()
        return updated

    def current_question(self, session: ActiveScreeningSession) -> ScreeningQuestion | None:
        self._ensure_version(session)
        if session.status != "collecting" or not session.next_question_id:
            return None
        return self.bank.get_question(session.next_question_id).model_copy(deep=True)

    def submit_answer(
        self,
        session: ActiveScreeningSession,
        answer: Any,
        *,
        question_id: str | None = None,
    ) -> ActiveScreeningSession:
        self._ensure_version(session)
        if session.status == "consent_pending":
            raise ScreeningStateError("explicit consent is required before answering questions")
        if session.status != "collecting" or not session.next_question_id:
            raise ScreeningStateError("screening session is not collecting answers")
        if self._is_cancel(answer):
            return self.cancel(session)

        expected_id = session.next_question_id
        if question_id is not None and question_id != expected_id:
            raise ScreeningStateError(
                f"out-of-order answer: expected {expected_id}, received {question_id}"
            )
        question = self.bank.get_question(expected_id)
        normalized = self._normalize_answer(question, answer)

        updated = session.model_copy(deep=True)
        updated.answers[expected_id] = normalized
        updated.next_question_id = None
        updated.updated_at = utc_now()
        return self._advance(updated)

    def evaluate(self, session: ActiveScreeningSession) -> ScreeningSummary:
        self._ensure_version(session)
        if session.status == "cancelled":
            raise ScreeningStateError("cancelled screening sessions have no result")

        rules = self.bank.rules
        answers = session.answers
        emergency_labels = self._values_from_answers(answers, rules["emergency_question_id"])
        emergency_labels.discard(rules["emergency_none_choice"])
        emergency_codes = [
            rules["emergency_choice_codes"][label]
            for label in sorted(emergency_labels)
            if label in rules["emergency_choice_codes"]
        ]

        symptom_values: set[str] = set()
        for question_id in rules["symptom_question_ids"]:
            value = answers.get(question_id)
            if value is True:
                symptom_values.add(question_id)
            elif isinstance(value, list):
                none_choice = rules["symptom_none_choices"].get(question_id)
                symptom_values.update(item for item in value if item != none_choice)

        high_risk_reasons = [
            question_id
            for question_id in rules["high_risk_boolean_question_ids"]
            if answers.get(question_id) is True
        ]
        for question_id, none_choice in rules["high_risk_multi_question_ids"].items():
            if self._values_from_answers(answers, question_id) - {none_choice}:
                high_risk_reasons.append(question_id)

        priority_reasons: list[str] = []
        age_value = answers.get(rules["age_question_id"])
        if age_value == rules["older_age_value"]:
            priority_reasons.append("older_adult_65_plus")
        for question_id, none_choice in rules["priority_multi_question_ids"].items():
            if self._values_from_answers(answers, question_id) - {none_choice}:
                priority_reasons.append(question_id)
        if answers.get(rules["high_incidence_question_id"]) is True:
            priority_reasons.append("high_incidence_community")

        triggers: list[str] = []
        if symptom_values:
            triggers.append("category:symptoms_reported")
        if high_risk_reasons:
            triggers.append("category:high_risk")
        if priority_reasons:
            triggers.append("category:priority_population")
        triggers.extend(
            f"local_clinical_safety_policy:emergency:{code}" for code in emergency_codes
        )

        information_gaps = self._information_gaps(session)
        # Emergency escalation belongs to local_clinical_safety_policy. It must not borrow
        # citations from the TB active-screening guideline to support the local safety rule.
        citation_keys: list[str] = [] if emergency_codes else ["base"]
        if emergency_codes:
            urgency = Urgency.EMERGENCY
            information_gaps = []
            next_steps = [
                "立即停止本问询，联系当地急救服务或尽快前往急诊；若无法安全到达，请呼叫急救。",
                "这些紧急警示表现可能由多种原因引起，本系统不判断其病因。",
                self.bank.scope_notice,
            ]
        elif symptom_values:
            urgency = Urgency.PROMPT_EVALUATION
            citation_keys.extend(["symptoms", "referral"])
            next_steps = self._prompt_evaluation_steps(age_value, answers)
        elif high_risk_reasons or priority_reasons:
            urgency = Urgency.PRIORITY_SCREENING
            if high_risk_reasons:
                citation_keys.append("high_risk")
            if priority_reasons:
                citation_keys.append("priority")
            citation_keys.append("referral")
            next_steps = self._priority_screening_steps(age_value, answers)
        else:
            urgency = Urgency.ROUTINE_INFORMATION
            next_steps = [
                "本次已回答项目未触发升级分层；这不等于排除肺结核。",
                "如之后出现相关症状、明确接触史或免疫状态变化，请重新筛查或联系医疗机构。",
                self.bank.scope_notice,
            ]
            if information_gaps:
                next_steps.insert(1, "存在未回答项目，不应据此降低筛查优先级。")

        citations = [
            self.bank.summary_citations[key].model_copy(deep=True)
            for key in dict.fromkeys(citation_keys)
        ]
        return ScreeningSummary(
            urgency=urgency,
            triggers=triggers,
            information_gaps=information_gaps,
            next_steps=next_steps,
            citations=citations,
        )

    def build_response(
        self,
        session: ActiveScreeningSession,
        *,
        request_id: str,
        trace_id: str,
    ) -> AgentResponse:
        self._ensure_version(session)
        common = {
            "request_id": request_id,
            "trace_id": trace_id,
            "thread_id": session.thread_id,
            "case_id": session.case_id,
            "limitations": [self.bank.scope_notice],
        }
        if session.status == "consent_pending":
            return AgentResponse(
                **common,
                response_kind=ResponseKind.ACTIVE_SCREENING_QUESTION,
                summary=self.bank.consent_notice,
            )
        if session.status == "collecting":
            question = self.current_question(session)
            return AgentResponse(
                **common,
                response_kind=ResponseKind.ACTIVE_SCREENING_QUESTION,
                summary="请回答下一项；也可以回答不知道、跳过或取消。",
                next_question=question,
            )
        if session.status == "cancelled":
            return AgentResponse(
                **common,
                response_kind=ResponseKind.ACTIVE_SCREENING_SUMMARY,
                summary="主动筛查问询已取消，本次已填写答案未保留。",
            )
        if session.status != "complete" or session.result is None:
            raise ScreeningStateError("screening session has no renderable result")
        # The answer set is authoritative. Recompute rather than trusting a
        # serialized summary that could be stale or locally tampered with.
        result = self.evaluate(session)
        return AgentResponse(
            **common,
            response_kind=ResponseKind.ACTIVE_SCREENING_SUMMARY,
            summary="主动筛查问询已完成；以下是筛查优先级与下一步信息，不是诊断。",
            next_step_information=list(result.next_steps),
            citations=[item.model_copy(deep=True) for item in result.citations],
            urgency=result.urgency,
        )

    def _advance(self, session: ActiveScreeningSession) -> ActiveScreeningSession:
        self._ensure_version(session)
        if session.status != "collecting":
            return session
        updated = session.model_copy(deep=True)
        if self._has_emergency_red_flag(updated):
            updated.result = self.evaluate(updated)
            updated.status = "complete"
            updated.next_question_id = None
            updated.updated_at = utc_now()
            return updated

        for question in self.bank.questions:
            if question.question_id in updated.answers:
                continue
            if self._question_applies(question, updated.answers):
                updated.next_question_id = question.question_id
                updated.updated_at = utc_now()
                return updated

        updated.result = self.evaluate(updated)
        updated.status = "complete"
        updated.next_question_id = None
        updated.updated_at = utc_now()
        return updated

    def _question_applies(
        self,
        question: ScreeningQuestion,
        answers: dict[str, Any],
    ) -> bool:
        for dependency, expected in question.ask_if.items():
            if dependency not in answers:
                return False
            allowed = expected if isinstance(expected, list) else [expected]
            if answers[dependency] not in allowed:
                return False
        return True

    def _normalize_answer(self, question: ScreeningQuestion, answer: Any) -> Any:
        special = self._special_answer(answer)
        if special is not None:
            return special

        if question.answer_type == "boolean":
            if isinstance(answer, bool):
                return answer
            if isinstance(answer, str):
                normalized = answer.strip().casefold()
                if normalized in self._TRUE_ALIASES:
                    return True
                if normalized in self._FALSE_ALIASES:
                    return False
            raise ScreeningInputError(f"{question.question_id} requires a boolean answer")

        if question.answer_type == "integer":
            if isinstance(answer, bool) or not isinstance(answer, int):
                raise ScreeningInputError(f"{question.question_id} requires an integer")
            if abs(answer) > 1_000_000:
                raise ScreeningInputError(
                    f"{question.question_id} integer answer is outside the supported range"
                )
            return answer

        if question.answer_type == "single_choice":
            if not isinstance(answer, str) or answer not in question.choices:
                raise ScreeningInputError(
                    f"{question.question_id} requires one of {question.choices}"
                )
            return answer

        if question.answer_type == "multi_choice":
            submitted = [answer] if isinstance(answer, str) else answer
            if not isinstance(submitted, list | tuple | set) or not submitted:
                raise ScreeningInputError(f"{question.question_id} requires one or more choices")
            values = list(dict.fromkeys(submitted))
            if any(not isinstance(item, str) or item not in question.choices for item in values):
                raise ScreeningInputError(
                    f"{question.question_id} contains a choice outside {question.choices}"
                )
            exclusive = set(self.bank.rules["exclusive_choices"].get(question.question_id, []))
            if len(values) > 1 and exclusive.intersection(values):
                raise ScreeningInputError(
                    f"{question.question_id} combines an exclusive choice with another choice"
                )
            return values

        if question.answer_type == "text":
            if not isinstance(answer, str) or not answer.strip():
                raise ScreeningInputError(f"{question.question_id} requires non-empty text")
            normalized = answer.strip()
            if len(normalized) > 2_000:
                raise ScreeningInputError(
                    f"{question.question_id} text answer exceeds 2000 characters"
                )
            return normalized
        raise ScreeningInputError(f"unsupported answer type: {question.answer_type}")

    def _special_answer(self, answer: Any) -> str | None:
        if not isinstance(answer, str):
            return None
        value = answer.strip().casefold()
        if value in self._UNKNOWN_ALIASES:
            return self.bank.special_answers["unknown"]
        if value in self._SKIP_ALIASES:
            return self.bank.special_answers["skip"]
        return None

    def _is_cancel(self, answer: Any) -> bool:
        return isinstance(answer, str) and answer.strip().casefold() in self._CANCEL_ALIASES

    def _values_from_answers(self, answers: dict[str, Any], question_id: str) -> set[str]:
        value = answers.get(question_id)
        return set(value) if isinstance(value, list) else set()

    def _has_emergency_red_flag(self, session: ActiveScreeningSession) -> bool:
        question_id = self.bank.rules["emergency_question_id"]
        selected = self._values_from_answers(session.answers, question_id)
        return bool(selected - {self.bank.rules["emergency_none_choice"]})

    def _information_gaps(self, session: ActiveScreeningSession) -> list[str]:
        unknown = self.bank.special_answers["unknown"]
        skipped = self.bank.special_answers["skip"]
        gaps: list[str] = []
        for question in self.bank.questions:
            value = session.answers.get(question.question_id)
            if isinstance(value, str) and value in {unknown, skipped}:
                label = "不知道" if value == unknown else "已跳过"
                gaps.append(f"{question.question_id}:{label}")
        return gaps

    def _ensure_version(self, session: ActiveScreeningSession) -> None:
        if session.guideline_rule_version != self.bank.guideline_rule_version:
            raise ScreeningStateError(
                "screening session rule version does not match the loaded question bank"
            )

    def _prompt_evaluation_steps(
        self,
        age_value: Any,
        answers: dict[str, Any],
    ) -> list[str]:
        steps = ["请尽快联系结核病定点医疗机构或其他合格医疗机构完成专业评估。"]
        if age_value == self.bank.rules["child_age_value"]:
            steps.append(
                "2026指南对15岁以下有症状者采用儿童路径，优先考虑病原学检查；"
                "具体标本和检查由专业人员决定。"
            )
        elif age_value in self.bank.rules["adult_age_values"]:
            steps.append(
                "2026指南对15岁及以上有症状者采用成人路径，通常结合胸部X线与"
                "病原学检查；具体项目由专业人员决定。"
            )
        else:
            steps.append("年龄分组缺失，无法选择儿童或成人筛查路径，请由专业人员确认。")
        if answers.get("hiv_status") is True or answers.get("immunosuppressed") is True:
            steps.append("请同步告知负责HIV或免疫抑制情况的临床团队，以确定专门筛查路径。")
        steps.append(self.bank.scope_notice)
        return steps

    def _priority_screening_steps(
        self,
        age_value: Any,
        answers: dict[str, Any],
    ) -> list[str]:
        steps = ["即使当前未报告相关症状，已报告的高风险或重点人群信息仍支持优先安排主动筛查。"]
        if age_value == self.bank.rules["child_age_value"]:
            steps.append(
                "2026指南对15岁以下高风险人群采用儿童路径，强调症状筛查和病原学检查；"
                "请由定点医疗机构确定实施方式。"
            )
        elif age_value in self.bank.rules["adult_age_values"]:
            steps.append(
                "2026指南对15岁及以上高风险人群采用成人路径，通常结合症状、胸部X线和"
                "病原学检查；请由专业人员确定实施方式。"
            )
        else:
            steps.append("年龄分组缺失，请由专业人员确定儿童或成人主动筛查路径。")
        if answers.get("hiv_status") is True or answers.get("immunosuppressed") is True:
            steps.append("请联系负责相关免疫状态的临床团队，避免常规路径遗漏专门评估。")
        if answers.get("high_incidence_community") is True:
            steps.append("社区筛查安排和频次应以当地疾控或卫生部门公布的方案为准。")
        steps.append(self.bank.scope_notice)
        return steps
