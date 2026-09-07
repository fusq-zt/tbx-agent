from __future__ import annotations

import json

import pytest

from tbx_agent.schemas import ResponseKind, Urgency
from tbx_agent.screening import (
    ScreeningEngine,
    ScreeningInputError,
    ScreeningStateError,
    load_question_bank,
)


def _start(engine: ScreeningEngine):
    return engine.start_session(
        thread_id="thread-1",
        user_id="user-1",
        owner_scope="owner-1",
        consent=True,
        session_id="screen-1",
    )


def _finish(engine: ScreeningEngine, session, answers: dict[str, object]):
    while session.status == "collecting":
        question = engine.current_question(session)
        assert question is not None
        answer = answers.get(question.question_id, "跳过")
        session = engine.submit_answer(
            session,
            answer,
            question_id=question.question_id,
        )
    return session


def _no_red_flags() -> list[str]:
    return ["以上均无"]


def test_question_bank_is_versioned_and_has_unique_questions():
    bank = load_question_bank()
    ids = [question.question_id for question in bank.questions]

    assert bank.guideline_rule_version == "tb-active-screening-2026-v1"
    assert len(ids) == len(set(ids))
    assert "child_tb_symptoms" in ids
    assert "adult_tb_symptoms" in ids
    assert "immunosuppression_type" in ids
    safety_questions = [
        question for question in bank.questions if question.source_id == "tbx_local_safety_policy"
    ]
    guideline_questions = [
        question for question in bank.questions if question.source_id != "tbx_local_safety_policy"
    ]
    assert [question.question_id for question in safety_questions] == ["emergency_red_flags"]
    assert guideline_questions
    assert all(
        question.source_id == "china_active_screening_2026" for question in guideline_questions
    )
    assert all(
        citation.source_id == "china_active_screening_2026"
        for citation in bank.summary_citations.values()
    )


def test_explicit_consent_is_required_and_decline_cancels():
    engine = ScreeningEngine()
    pending = engine.start_session(
        thread_id="thread-1",
        user_id="user-1",
        owner_scope="owner-1",
    )

    assert pending.status == "consent_pending"
    assert pending.next_question_id is None
    with pytest.raises(ScreeningStateError, match="explicit consent"):
        engine.submit_answer(pending, True)

    cancelled = engine.set_consent(pending, granted=False)
    assert cancelled.status == "cancelled"
    assert cancelled.consent is False
    assert cancelled.answers == {}


def test_consent_starts_with_emergency_question_and_cancel_clears_answers():
    engine = ScreeningEngine()
    session = _start(engine)
    assert session.next_question_id == "emergency_red_flags"

    session = engine.submit_answer(session, _no_red_flags())
    assert session.answers
    cancelled = engine.submit_answer(session, "取消")

    assert cancelled.status == "cancelled"
    assert cancelled.answers == {}
    assert cancelled.next_question_id is None


def test_unknown_and_skip_are_accepted_and_recorded_as_information_gaps():
    engine = ScreeningEngine()
    session = _finish(
        engine,
        _start(engine),
        {
            "emergency_red_flags": "不知道",
            "age_group": "跳过",
            "age_unknown_tb_symptoms": False,
            "hiv_status": False,
            "close_contact": False,
            "immunosuppressed": False,
            "other_high_risk_factors": ["以上均无"],
            "priority_population_factors": ["以上均无"],
            "high_incidence_community": False,
        },
    )

    assert session.status == "complete"
    assert session.result is not None
    assert "emergency_red_flags:不知道" in session.result.information_gaps
    assert "age_group:已跳过" in session.result.information_gaps
    assert session.result.urgency == Urgency.ROUTINE_INFORMATION
    assert any("不应据此降低" in step for step in session.result.next_steps)


def test_child_and_adult_questions_are_mutually_exclusive():
    engine = ScreeningEngine()
    child = _finish(
        engine,
        _start(engine),
        {
            "emergency_red_flags": _no_red_flags(),
            "age_group": "15岁以下",
            "hiv_status": False,
            "child_tb_symptoms": ["以上均无"],
        },
    )
    adult = _finish(
        engine,
        _start(engine),
        {
            "emergency_red_flags": _no_red_flags(),
            "age_group": "15～64岁",
            "hiv_status": False,
            "adult_tb_symptoms": ["以上均无"],
        },
    )

    assert "child_tb_symptoms" in child.answers
    assert "adult_tb_symptoms" not in child.answers
    assert "adult_tb_symptoms" in adult.answers
    assert "child_tb_symptoms" not in adult.answers


def test_hiv_branch_uses_four_symptom_question():
    engine = ScreeningEngine()
    session = _finish(
        engine,
        _start(engine),
        {
            "emergency_red_flags": _no_red_flags(),
            "age_group": "15岁以下",
            "hiv_status": True,
            "hiv_four_symptoms": ["以上均无"],
        },
    )

    assert "hiv_four_symptoms" in session.answers
    assert "child_tb_symptoms" not in session.answers
    assert session.result is not None
    assert "category:high_risk" in session.result.triggers
    assert session.result.urgency == Urgency.PRIORITY_SCREENING


def test_immunosuppression_detail_branch_and_priority_summary():
    engine = ScreeningEngine()
    session = _finish(
        engine,
        _start(engine),
        {
            "emergency_red_flags": _no_red_flags(),
            "age_group": "15～64岁",
            "hiv_status": False,
            "adult_tb_symptoms": ["以上均无"],
            "close_contact": False,
            "immunosuppressed": True,
            "immunosuppression_type": ["器官移植术后"],
            "other_high_risk_factors": ["以上均无"],
            "priority_population_factors": ["以上均无"],
            "high_incidence_community": False,
        },
    )

    assert session.answers["immunosuppression_type"] == ["器官移植术后"]
    assert session.result is not None
    assert session.result.urgency == Urgency.PRIORITY_SCREENING
    assert "category:high_risk" in session.result.triggers
    assert any("免疫" in step for step in session.result.next_steps)


def test_symptoms_trigger_prompt_evaluation_not_diagnosis():
    engine = ScreeningEngine()
    session = _finish(
        engine,
        _start(engine),
        {
            "emergency_red_flags": _no_red_flags(),
            "age_group": "15～64岁",
            "hiv_status": False,
            "adult_tb_symptoms": ["咳嗽或咳痰持续2周及以上"],
            "close_contact": False,
            "immunosuppressed": False,
            "other_high_risk_factors": ["以上均无"],
            "priority_population_factors": ["以上均无"],
            "high_incidence_community": False,
        },
    )

    assert session.result is not None
    assert session.result.urgency == Urgency.PROMPT_EVALUATION
    assert "category:symptoms_reported" in session.result.triggers
    payload = json.dumps(session.result.model_dump(mode="json"), ensure_ascii=False)
    assert "disease_probability" not in payload
    assert "患病概率为" not in payload
    assert any("不能确诊或排除" in step for step in session.result.next_steps)


def test_child_close_contact_uses_priority_child_path():
    engine = ScreeningEngine()
    session = _finish(
        engine,
        _start(engine),
        {
            "emergency_red_flags": _no_red_flags(),
            "age_group": "15岁以下",
            "hiv_status": False,
            "child_tb_symptoms": ["以上均无"],
            "close_contact": True,
            "immunosuppressed": False,
            "other_high_risk_factors": ["以上均无"],
            "priority_population_factors": ["以上均无"],
            "high_incidence_community": False,
        },
    )

    assert session.result is not None
    assert session.result.urgency == Urgency.PRIORITY_SCREENING
    assert any("15岁以下" in step for step in session.result.next_steps)


def test_older_or_key_population_is_prioritized_without_probability_score():
    engine = ScreeningEngine()
    session = _finish(
        engine,
        _start(engine),
        {
            "emergency_red_flags": _no_red_flags(),
            "age_group": "65岁及以上",
            "hiv_status": False,
            "adult_tb_symptoms": ["以上均无"],
            "close_contact": False,
            "immunosuppressed": False,
            "other_high_risk_factors": ["以上均无"],
            "priority_population_factors": ["糖尿病患者"],
            "high_incidence_community": False,
        },
    )

    assert session.result is not None
    assert session.result.urgency == Urgency.PRIORITY_SCREENING
    assert "category:priority_population" in session.result.triggers
    assert not hasattr(session.result, "disease_probability")


def test_emergency_red_flag_stops_questionnaire_immediately():
    engine = ScreeningEngine()
    session = _start(engine)
    session = engine.submit_answer(session, ["严重呼吸困难"])

    assert session.status == "complete"
    assert session.next_question_id is None
    assert session.result is not None
    assert session.result.urgency == Urgency.EMERGENCY
    assert "local_clinical_safety_policy:emergency:severe_breathlessness" in session.result.triggers
    assert session.result.citations == []
    assert any("急救" in step or "急诊" in step for step in session.result.next_steps)
    assert any("不判断其病因" in step for step in session.result.next_steps)


def test_exclusive_none_choice_cannot_be_combined_with_symptom():
    engine = ScreeningEngine()
    session = _start(engine)
    with pytest.raises(ScreeningInputError, match="exclusive choice"):
        engine.submit_answer(session, ["严重呼吸困难", "以上均无"])


def test_build_response_uses_existing_agent_response_schema():
    engine = ScreeningEngine()
    pending = engine.start_session(
        thread_id="thread-1",
        user_id="user-1",
        owner_scope="owner-1",
    )
    pending_response = engine.build_response(
        pending,
        request_id="request-1",
        trace_id="trace-1",
    )
    assert pending_response.response_kind == ResponseKind.ACTIVE_SCREENING_QUESTION
    assert "是否同意" in pending_response.summary

    emergency = engine.submit_answer(_start(engine), ["大量或持续咯血"])
    summary_response = engine.build_response(
        emergency,
        request_id="request-2",
        trace_id="trace-2",
    )
    assert summary_response.response_kind == ResponseKind.ACTIVE_SCREENING_SUMMARY
    assert summary_response.urgency == Urgency.EMERGENCY
    assert summary_response.citations == []
    assert summary_response.diagnostic_information == []
    assert summary_response.treatment_education == []
