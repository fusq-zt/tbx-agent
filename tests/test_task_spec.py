import json

import pytest

from tbx_agent.agent_state import EvidenceKind
from tbx_agent.schemas import ThreadState
from tbx_agent.task_spec import (
    GoalEvidenceSource,
    GuidelineScenarioTag,
    GuidelineScope,
    GuidelineTaskContext,
    TaskGoal,
    TaskSpecSource,
    continuation_intent_for_answer,
    interpret_task_spec,
    parse_task_spec,
)


class _TaskGenerator:
    backend_id = "test-qwen"
    model = "test-model"

    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls = 0

    def complete_structured(self, **kwargs):
        self.calls += 1
        if not isinstance(self.payload, dict):
            payload = self.payload
        elif kwargs["schema_name"] == "tbx_agent_tool_selection":
            # This fixture intentionally exercises the one-release migration
            # adapter for persisted providers that still emit legacy goals.
            # New planner contract coverage lives in
            # test_task_spec_semantic_router.py and emits {"tools": [...]}.
            payload = {
                "intents": [
                    item["goal"] for item in self.payload.get("selections", [])
                ]
            }
        else:
            raise AssertionError(f"unexpected schema: {kwargs['schema_name']}")
        return json.dumps(payload), {"prompt_tokens": 17, "completion_tokens": 4}


class _FailingTaskGenerator:
    backend_id = "test-qwen"
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, **_kwargs):
        self.calls += 1
        raise RuntimeError("task interpreter unavailable")


def test_compound_query_preserves_all_goals_and_evidence_requirements() -> None:
    spec = parse_task_spec("为什么模型认为是 TB？病灶在哪里？下一步应该做什么检查？")

    assert spec.task_goals == [
        TaskGoal.EXPLAIN_CLASSIFICATION,
        TaskGoal.LOCALIZE,
        TaskGoal.GUIDELINE_DIAGNOSTIC_TESTING,
    ]
    assert spec.required_evidence == [
        EvidenceKind.CLASSIFICATION,
        EvidenceKind.LOCALIZATION,
        EvidenceKind.DIAGNOSTIC,
    ]
    assert spec.guideline_scopes == [GuidelineScope.DIAGNOSTIC_TESTING]
    assert spec.is_compound is True
    assert all(item.evidence_span in spec.current_query for item in spec.goal_evidence)


def test_natural_screening_compound_preserves_classification_localization_and_guidance() -> None:
    spec = parse_task_spec(
        "这张片是体检发现的，患者目前没有明显症状。"
        "你先告诉我模型更倾向于健康、非结核异常还是 TB；"
        "如果异常，请标出主要候选区域，然后告诉我这种筛查异常一般下一步需要做什么。"
    )

    assert spec.task_goals == [
        TaskGoal.SCREEN_CLASSIFICATION,
        TaskGoal.LOCALIZE,
        TaskGoal.GUIDELINE_DIAGNOSTIC_TESTING,
    ]
    assert spec.required_evidence == [
        EvidenceKind.CLASSIFICATION,
        EvidenceKind.LOCALIZATION,
        EvidenceKind.DIAGNOSTIC,
    ]
    assert spec.guideline_scope == GuidelineScope.DIAGNOSTIC_TESTING
    assert spec.subtopic == "diagnostic_pathway"
    assert GuidelineScenarioTag.AFTER_ABNORMAL_CXR in spec.scenario_tags


def test_later_next_step_clause_does_not_erase_explicit_image_reading_clause() -> None:
    spec = parse_task_spec(
        "请先看一下这张胸片；然后告诉我筛查异常后下一步需要做什么检查。"
    )

    assert TaskGoal.SCREEN_CLASSIFICATION in spec.task_goals
    assert TaskGoal.GUIDELINE_DIAGNOSTIC_TESTING in spec.task_goals


def test_pure_classification_never_authorizes_localization() -> None:
    spec = parse_task_spec("这张胸片有没有结核病？")

    assert spec.task_goals == [TaskGoal.SCREEN_CLASSIFICATION]
    assert EvidenceKind.LOCALIZATION not in spec.required_evidence


def test_natural_three_class_question_routes_to_current_cxr_classifier() -> None:
    spec = parse_task_spec("这张胸片是什么分类？")

    assert spec.task_goals == [TaskGoal.SCREEN_CLASSIFICATION]
    assert spec.required_evidence == [EvidenceKind.CLASSIFICATION]


def test_llm_cannot_add_localization_without_spatial_evidence() -> None:
    generator = _TaskGenerator(
        {
            "selections": [
                {"goal": "screen_classification", "evidence_span": "有没有结核"},
                {"goal": "localize", "evidence_span": "这张胸片"},
            ]
        }
    )

    interpreted = interpret_task_spec(
        "这张胸片有没有结核病？",
        generator=generator,
    )

    assert interpreted.source == TaskSpecSource.LLM_WITH_RULE_GUARD
    assert interpreted.authorization_validated is False
    assert interpreted.task_spec.task_goals == [TaskGoal.CLARIFICATION_REQUIRED]
    assert interpreted.task_spec.required_evidence == []


def test_guideline_query_dimensions_are_scope_specific() -> None:
    high_risk = parse_task_spec("哪些人属于 TB 高风险人群？")
    xpert = parse_task_spec("Xpert MTB/RIF、Xpert Ultra 在什么情况下使用？")
    duration = parse_task_spec("标准疗程大概是什么？")
    mask = parse_task_spec("怀疑肺结核时是否需要佩戴口罩？")

    assert high_risk.guideline_scope == GuidelineScope.SCREENING
    assert high_risk.subtopic == "risk_groups"
    assert high_risk.population == ["tb_high_risk_population"]
    assert xpert.guideline_scope == GuidelineScope.DIAGNOSTIC_TESTING
    assert xpert.subtopic == "rapid_molecular_diagnostics"
    assert xpert.product_terms == ["Xpert MTB/RIF", "Xpert Ultra"]
    assert duration.guideline_scope == GuidelineScope.TREATMENT_EDUCATION
    assert duration.subtopic == "standard_regimen_duration"
    assert mask.guideline_scope == GuidelineScope.INFECTION_CONTROL
    assert mask.subtopic == "respiratory_protection"


@pytest.mark.parametrize(
    "query",
    [
        "肺结核必须住院吗？",
        "确诊后能在门诊或社区治疗吗？",
        "耐药肺结核必须住院治疗吗？",
    ],
)
def test_care_setting_questions_route_to_bounded_treatment_education(query: str) -> None:
    spec = parse_task_spec(query)

    assert spec.task_goals == [TaskGoal.GUIDELINE_TREATMENT_EDUCATION]
    assert spec.guideline_scope == GuidelineScope.TREATMENT_EDUCATION
    assert spec.subtopic == "care_setting"


def test_common_knowledge_questions_get_direct_intent_subtopics() -> None:
    precautions = parse_task_spec("怀疑有传染性结核时平时要注意什么？")
    resistance_comparison = parse_task_spec("耐药结核和普通结核治疗一样吗？")

    assert precautions.task_goals == [TaskGoal.GUIDELINE_INFECTION_CONTROL]
    assert precautions.guideline_scope == GuidelineScope.INFECTION_CONTROL
    assert precautions.subtopic == "infection_control_precautions"
    assert resistance_comparison.task_goals == [TaskGoal.GUIDELINE_TREATMENT_EDUCATION]
    assert resistance_comparison.guideline_scope == GuidelineScope.TREATMENT_EDUCATION
    assert resistance_comparison.subtopic == "drug_resistant_treatment_comparison"


def test_sputum_smear_negative_phrasings_are_deterministic_diagnostic_questions() -> None:
    for query in (
        "痰片没查到菌是不是就能排除结核？",
        "痰涂片阴性是否排除肺结核",
        "痰片没查到菌",
    ):
        spec = parse_task_spec(query)

        assert spec.task_goals == [TaskGoal.GUIDELINE_DIAGNOSTIC_TESTING]
        assert spec.required_evidence == [EvidenceKind.DIAGNOSTIC]
        assert spec.guideline_scope == GuidelineScope.DIAGNOSTIC_TESTING
        assert spec.subtopic == "negative_test_interpretation"


def test_sputum_smear_rule_prevents_semantic_classifier_drift() -> None:
    query = "痰片没查到菌是不是就能排除结核？"
    generator = _TaskGenerator(
        {
            "selections": [
                {
                    "goal": "screen_classification",
                    "evidence_span": "是不是就能排除结核",
                }
            ]
        }
    )

    interpreted = interpret_task_spec(query, generator=generator)

    assert interpreted.source == TaskSpecSource.LLM_WITH_RULE_GUARD
    assert interpreted.authorization_validated is False
    assert interpreted.task_spec.task_goals == [TaskGoal.CLARIFICATION_REQUIRED]
    assert interpreted.task_spec.required_evidence == []


def test_explicit_guideline_phrases_beat_ambiguous_cross_scope_keywords() -> None:
    xpert = parse_task_spec("Xpert如何检测利福平耐药？")
    close_contact = parse_task_spec("密切接触者属于高风险人群吗？")

    assert xpert.task_goals == [TaskGoal.GUIDELINE_DIAGNOSTIC_TESTING]
    assert xpert.guideline_scope == GuidelineScope.DIAGNOSTIC_TESTING
    assert xpert.subtopic == "rapid_molecular_diagnostics"
    assert xpert.product_terms == ["Xpert MTB/RIF"]
    assert xpert.goal_evidence[0].evidence_span.casefold() == "xpert"

    assert close_contact.task_goals == [TaskGoal.GUIDELINE_SCREENING]
    assert close_contact.guideline_scope == GuidelineScope.SCREENING
    assert close_contact.subtopic == "risk_groups"
    assert close_contact.population == [
        "close_contacts",
        "tb_high_risk_population",
    ]
    assert close_contact.goal_evidence[0].evidence_span == "高风险人群"


def test_natural_classification_rationale_followup_is_not_case_status() -> None:
    spec = parse_task_spec("为什么这样分类？")

    assert spec.task_goals == [TaskGoal.EXPLAIN_CLASSIFICATION]
    assert spec.required_evidence == [EvidenceKind.CLASSIFICATION]


def test_lung_field_only_does_not_require_localization() -> None:
    spec = parse_task_spec("请显示肺野分割")

    assert spec.task_goals == [TaskGoal.LUNG_FIELDS]
    assert spec.required_evidence == [EvidenceKind.ANATOMY]


def test_candidate_anatomy_requires_localization_then_anatomy() -> None:
    spec = parse_task_spec("候选区域位于左肺上部吗？")

    assert TaskGoal.ANATOMICAL_CONTEXT in spec.task_goals
    assert spec.required_evidence == [EvidenceKind.LOCALIZATION, EvidenceKind.ANATOMY]


def test_no_tool_queries_are_explicit() -> None:
    assert parse_task_spec("你好").no_tool_only is True
    assert parse_task_spec("谢谢").no_tool_only is True
    assert parse_task_spec("这个系统可以做什么？").no_tool_only is True
    assert parse_task_spec("定位模型运行过了吗？").no_tool_only is True
    assert parse_task_spec("1+1 = ？").task_goals == [TaskGoal.GENERAL_CHAT]
    assert parse_task_spec("人每天应该摄入多少糖分").task_goals == [TaskGoal.GENERAL_CHAT]


def test_prior_query_forbids_invented_longitudinal_change() -> None:
    spec = parse_task_spec("和半年前相比恶化了吗？")

    assert spec.task_goals == [TaskGoal.PRIOR_COMPARISON]
    assert spec.required_evidence == [EvidenceKind.PRIOR, EvidenceKind.LONGITUDINAL]
    assert "longitudinal_change_without_prior_and_validated_capability" in spec.forbidden_claims


def test_short_continuation_inherits_only_the_last_audited_tool_intent() -> None:
    diagnostic = parse_task_spec(
        "给出",
        active_intent="guideline:diagnostic_testing",
    )
    unknown = parse_task_spec("给出", active_intent="untrusted_unknown_tool")

    assert diagnostic.task_goals == [TaskGoal.GUIDELINE_DIAGNOSTIC_TESTING]
    assert diagnostic.required_evidence == [EvidenceKind.DIAGNOSTIC]
    assert unknown.task_goals == [TaskGoal.GENERAL_CHAT]


def test_short_why_followup_uses_audited_classification_context_only() -> None:
    rationale = parse_task_spec(
        "为什么？",
        active_intent="classify_current_cxr",
    )
    without_context = parse_task_spec("为什么？")

    assert rationale.task_goals == [TaskGoal.EXPLAIN_CLASSIFICATION]
    assert rationale.goal_evidence[0].evidence_source == "context_memory"
    assert without_context.task_goals == [TaskGoal.GENERAL_CHAT]


def test_only_explicit_case_state_wording_routes_to_case_status() -> None:
    state = parse_task_spec("当前病例状态是什么？")
    execution_state = parse_task_spec("当前病例的分类和定位运行了吗？")
    analysis_summary = parse_task_spec("整理成一个简短的 AI 辅助分析摘要")
    capabilities = parse_task_spec("你会干什么？")
    generic = parse_task_spec("解释一下有限状态机")

    assert state.task_goals == [TaskGoal.CASE_STATUS]
    assert execution_state.task_goals == [TaskGoal.CASE_STATUS]
    assert analysis_summary.task_goals == [TaskGoal.CASE_STATUS]
    assert capabilities.task_goals == [TaskGoal.CAPABILITIES]
    assert generic.task_goals == [TaskGoal.GENERAL_CHAT]


def test_uncovered_medical_phrasings_do_not_bypass_tbx_tools() -> None:
    assert parse_task_spec("这张胸片看起来正常吗？").task_goals == [TaskGoal.SCREEN_CLASSIFICATION]
    self_assessment = parse_task_spec("我是不是得了肺结核？")
    assert self_assessment.task_goals == [TaskGoal.GUIDELINE_DIAGNOSTIC_TESTING]
    assert self_assessment.guideline_scope == GuidelineScope.DIAGNOSTIC_TESTING
    assert self_assessment.subtopic == "diagnostic_pathway"
    assert parse_task_spec("异烟肼该吃多少？").task_goals == [
        TaskGoal.GUIDELINE_TREATMENT_EDUCATION
    ]
    assert parse_task_spec("能不能停利福平？").task_goals == [
        TaskGoal.GUIDELINE_TREATMENT_EDUCATION
    ]


def test_cached_rationale_exposes_a_bounded_continuation_intent() -> None:
    rationale = parse_task_spec("为什么这样分类？")
    semantic_intent = continuation_intent_for_answer(rationale)
    continuation = parse_task_spec("展开", active_intent=semantic_intent)

    assert semantic_intent == "get_exact_case_and_explain"
    assert continuation.task_goals == [TaskGoal.EXPLAIN_CLASSIFICATION]
    assert continuation.required_evidence == [EvidenceKind.CLASSIFICATION]


def test_llm_task_interpreter_handles_natural_request_without_keyword_route() -> None:
    generator = _TaskGenerator(
        {
            "selections": [
                {
                    "goal": "explain_classification",
                    "evidence_span": "为什么会归到这一类",
                }
            ]
        }
    )

    interpreted = interpret_task_spec(
        "这个结果为什么会归到这一类？",
        generator=generator,
    )

    assert interpreted.source == TaskSpecSource.LLM
    assert interpreted.schema_validated is True
    assert interpreted.task_spec.task_goals == [TaskGoal.EXPLAIN_CLASSIFICATION]
    assert interpreted.task_spec.required_evidence == [EvidenceKind.CLASSIFICATION]
    assert interpreted.backend == "test-qwen"
    assert interpreted.prompt_tokens == 17


def test_domain_semantic_interpreter_routes_unlisted_cxr_request() -> None:
    query = "这份CXR可能属于哪种训练类别"
    assert parse_task_spec(query).task_goals == [TaskGoal.GENERAL_CHAT]
    generator = _TaskGenerator(
        {
            "selections": [
                {
                    "goal": "screen_classification",
                    "evidence_span": query,
                }
            ]
        }
    )

    interpreted = interpret_task_spec(query, generator=generator)

    assert generator.calls == 1
    assert interpreted.source == TaskSpecSource.LLM
    assert interpreted.authorization_validated is True
    assert interpreted.task_spec.task_goals == [TaskGoal.SCREEN_CLASSIFICATION]
    assert interpreted.task_spec.required_evidence == [EvidenceKind.CLASSIFICATION]
    assert len(interpreted.task_spec.goal_evidence) == 1
    semantic_evidence = interpreted.task_spec.goal_evidence[0]
    assert semantic_evidence.goal == TaskGoal.SCREEN_CLASSIFICATION
    assert semantic_evidence.evidence_span == query
    assert semantic_evidence.evidence_source == GoalEvidenceSource.MODEL_SEMANTIC


def test_domain_semantic_interpreter_cannot_add_location_without_spatial_intent() -> None:
    query = "请帮我看看这张片子的情况"
    generator = _TaskGenerator(
        {
            "selections": [
                {
                    "goal": "screen_classification",
                    "evidence_span": query,
                },
                {
                    "goal": "localize",
                    "evidence_span": query,
                },
            ]
        }
    )

    interpreted = interpret_task_spec(query, generator=generator)

    assert generator.calls == 1
    assert interpreted.source == TaskSpecSource.LLM_WITH_RULE_GUARD
    assert interpreted.authorization_validated is False
    assert interpreted.task_spec.task_goals == [TaskGoal.CLARIFICATION_REQUIRED]
    assert interpreted.task_spec.required_evidence == []


@pytest.mark.parametrize(
    ("query", "span"),
    (("1+1 = ？", "1+1"), ("人每天应该摄入多少糖分", "摄入多少糖分")),
)
def test_general_questions_use_the_semantic_interpreter_when_available(
    query: str,
    span: str,
) -> None:
    generator = _TaskGenerator(
        {"selections": [{"goal": "general_chat", "evidence_span": span}]}
    )

    interpreted = interpret_task_spec(query, generator=generator)

    assert generator.calls == 1
    assert interpreted.source == TaskSpecSource.LLM
    assert interpreted.task_spec.task_goals == [TaskGoal.GENERAL_CHAT]
    assert interpreted.task_spec.goal_evidence[0].evidence_source == "model_semantic"


def test_domain_semantic_interpreter_failure_returns_existing_rule_result() -> None:
    query = "请帮我看看这张片子的情况"
    fallback = parse_task_spec(query)
    generator = _FailingTaskGenerator()

    interpreted = interpret_task_spec(query, generator=generator)

    assert generator.calls == 1
    assert interpreted.source == TaskSpecSource.RULE_FALLBACK
    assert interpreted.schema_validated is False
    assert interpreted.authorization_validated is False
    assert interpreted.task_spec == fallback


def test_next_test_question_about_abnormal_cxr_does_not_rerun_classification() -> None:
    spec = parse_task_spec("胸片异常后应该做什么检查？")

    assert spec.task_goals == [TaskGoal.GUIDELINE_DIAGNOSTIC_TESTING]
    assert EvidenceKind.CLASSIFICATION not in spec.required_evidence


@pytest.mark.parametrize(
    "query",
    (
        "孕妇怀疑肺结核时检查有什么不同？",
        "孕妇怀疑肺结核时应该做什么检查",
    ),
)
def test_pregnancy_testing_synonyms_keep_special_population_scope(query: str) -> None:
    spec = parse_task_spec(query)

    assert spec.task_goals == [TaskGoal.GUIDELINE_SPECIAL_POPULATION]
    assert spec.guideline_scope == GuidelineScope.SPECIAL_POPULATION
    assert spec.subtopic == "special_population_testing"
    assert spec.population == ["pregnant_people"]
    assert spec.product_terms == []


def test_symptomatic_pregnancy_self_assessment_routes_to_testing_not_chat() -> None:
    spec = parse_task_spec("我今年32岁，怀孕8周，最近咳嗽严重，该怎么判断自己有没有肺结核")

    assert spec.task_goals == [TaskGoal.GUIDELINE_SPECIAL_POPULATION]
    assert spec.required_evidence == [EvidenceKind.DIAGNOSTIC]
    assert spec.guideline_scope == GuidelineScope.SPECIAL_POPULATION
    assert spec.subtopic == "special_population_testing"
    assert spec.population == ["pregnant_people"]
    assert EvidenceKind.CLASSIFICATION not in spec.required_evidence


def test_specific_next_step_followup_inherits_special_population_testing_context() -> None:
    context = GuidelineTaskContext(
        scope=GuidelineScope.SPECIAL_POPULATION,
        subtopic="special_population_testing",
        population=["pregnant_people"],
    )

    interpreted = interpret_task_spec(
        "具体该怎么做",
        active_intent="guideline:special_population",
        active_guideline_context=context,
    )

    assert interpreted.source == TaskSpecSource.CONTEXT_MEMORY
    assert interpreted.task_spec.task_goals == [TaskGoal.GUIDELINE_SPECIAL_POPULATION]
    assert interpreted.task_spec.subtopic == "special_population_testing"
    assert interpreted.task_spec.population == ["pregnant_people"]
    assert interpreted.task_spec.goal_evidence[0].evidence_source == "context_memory"


def test_child_initial_testing_keeps_special_population_scope() -> None:
    spec = parse_task_spec("15岁以下儿童一般优先做什么检查？")

    assert spec.task_goals == [TaskGoal.GUIDELINE_SPECIAL_POPULATION]
    assert spec.guideline_scope == GuidelineScope.SPECIAL_POPULATION
    assert spec.subtopic == "special_population_testing"
    assert spec.population == ["children"]
    assert spec.product_terms == []


def test_child_without_sputum_routes_to_sample_testing() -> None:
    spec = parse_task_spec("儿童咳不出痰时怎么办？")

    assert spec.task_goals == [TaskGoal.GUIDELINE_SPECIAL_POPULATION]
    assert spec.guideline_scope == GuidelineScope.SPECIAL_POPULATION
    assert spec.subtopic == "special_population_testing"
    assert spec.population == ["children"]


def test_child_infection_test_keeps_population_scope_and_named_test_subtopic() -> None:
    spec = parse_task_spec("儿童TST或IGRA阳性能诊断活动性肺结核吗？")

    assert spec.task_goals == [TaskGoal.GUIDELINE_SPECIAL_POPULATION]
    assert spec.guideline_scope == GuidelineScope.SPECIAL_POPULATION
    assert spec.subtopic == "tb_infection_test_interpretation"
    assert spec.population == ["children"]


def test_non_image_words_do_not_authorize_anatomy_or_image_quality_tools() -> None:
    hospitalization = parse_task_spec("什么情况下肺结核需要住院？")
    vision_adverse_effect = parse_task_spec("吃抗结核药后视力变模糊怎么办？")

    assert hospitalization.task_goals == [TaskGoal.GUIDELINE_TREATMENT_EDUCATION]
    assert vision_adverse_effect.task_goals == [TaskGoal.GUIDELINE_TREATMENT_EDUCATION]
    assert hospitalization.subtopic == "care_setting"
    assert vision_adverse_effect.subtopic == "adverse_effects"


def test_rule_guard_blocks_llm_omission_without_rerouting() -> None:
    generator = _TaskGenerator(
        {"selections": [{"goal": "case_status", "evidence_span": "病灶在哪"}]}
    )

    interpreted = interpret_task_spec("病灶在哪？", generator=generator)

    assert interpreted.source == TaskSpecSource.LLM_WITH_RULE_GUARD
    assert interpreted.schema_validated is True
    assert interpreted.authorization_validated is False
    assert interpreted.task_spec.task_goals == [TaskGoal.CLARIFICATION_REQUIRED]
    assert interpreted.task_spec.required_evidence == []


def test_invalid_llm_task_object_falls_back_to_rules() -> None:
    generator = _TaskGenerator(
        {"selections": [{"goal": "run_arbitrary_shell", "evidence_span": "下一步"}]}
    )

    interpreted = interpret_task_spec("下一步做什么检查？", generator=generator)

    assert interpreted.source == TaskSpecSource.RULE_FALLBACK
    assert interpreted.schema_validated is False
    assert interpreted.task_spec.task_goals == [TaskGoal.GUIDELINE_DIAGNOSTIC_TESTING]


def test_short_continuation_uses_last_successful_tool_without_llm_goal_drift() -> None:
    generator = _TaskGenerator(
        {"selections": [{"goal": "explain_classification", "evidence_span": "展开"}]}
    )

    interpreted = interpret_task_spec(
        "展开",
        active_intent="guideline:diagnostic_testing",
        generator=generator,
    )

    assert interpreted.source == TaskSpecSource.LLM_WITH_RULE_GUARD
    assert interpreted.task_spec.task_goals == [TaskGoal.CLARIFICATION_REQUIRED]
    assert interpreted.task_spec.goal_evidence[0].evidence_source == "rule_guard"
    assert generator.calls == 1


def test_guideline_continuation_inherits_all_structured_retrieval_dimensions() -> None:
    cases = (
        (
            "guideline:screening",
            GuidelineTaskContext(
                scope=GuidelineScope.SCREENING,
                subtopic="risk_groups",
                population=["tb_high_risk_population"],
            ),
        ),
        (
            "guideline:diagnostic_testing",
            GuidelineTaskContext(
                scope=GuidelineScope.DIAGNOSTIC_TESTING,
                subtopic="rapid_molecular_diagnostics",
                product_terms=["Xpert MTB/RIF", "Xpert Ultra"],
            ),
        ),
        (
            "guideline:treatment_education",
            GuidelineTaskContext(
                scope=GuidelineScope.TREATMENT_EDUCATION,
                subtopic="standard_regimen_duration",
            ),
        ),
    )

    for active_intent, context in cases:
        continuation = interpret_task_spec(
            "展开",
            active_intent=active_intent,
            active_guideline_context=context,
        )

        assert continuation.source == TaskSpecSource.CONTEXT_MEMORY
        assert continuation.task_spec.guideline_scope == context.scope
        assert continuation.task_spec.subtopic == context.subtopic
        assert continuation.task_spec.population == context.population
        assert continuation.task_spec.product_terms == context.product_terms
        assert continuation.task_spec.goal_evidence[0].evidence_source == "context_memory"


def test_legacy_thread_payload_defaults_structured_guideline_memory_to_none() -> None:
    state = ThreadState.model_validate(
        {
            "thread_id": "legacy-thread",
            "user_id": "legacy-user",
            "owner_scope": "tenant:legacy",
            "active_intent": "guideline:screening",
        }
    )

    assert state.recent_guideline_task is None


@pytest.mark.parametrize(
    ("query", "products"),
    (
        ("如果 Xpert 阴性呢？", ["Xpert MTB/RIF"]),
        ("那培养阴性呢？", []),
        ("胸片正常是不是就没有肺结核？", []),
    ),
)
def test_named_negative_result_fallback_keeps_entity_semantics(
    query: str,
    products: list[str],
) -> None:
    spec = parse_task_spec(query)

    assert spec.task_goals == [TaskGoal.GUIDELINE_DIAGNOSTIC_TESTING]
    assert spec.subtopic == "negative_test_interpretation"
    assert spec.product_terms == products


@pytest.mark.parametrize(
    "query",
    ("结核培养有什么作用？", "那涂片呢？"),
)
def test_single_microbiology_test_role_fallback_is_not_a_negative_result(
    query: str,
) -> None:
    spec = parse_task_spec(query)

    assert spec.task_goals == [TaskGoal.GUIDELINE_DIAGNOSTIC_TESTING]
    assert spec.subtopic == "test_comparison"


def test_child_infection_test_choice_is_a_special_population_testing_task() -> None:
    spec = parse_task_spec("5岁以下儿童做 TST 还是 IGRA？")

    assert spec.task_goals == [TaskGoal.GUIDELINE_SPECIAL_POPULATION]
    assert spec.subtopic == "special_population_testing"
    assert spec.population == ["children"]


def test_explicit_adverse_effect_wording_routes_to_medication_safety() -> None:
    spec = parse_task_spec("如果吃药后视物模糊呢？")

    assert spec.task_goals == [TaskGoal.GUIDELINE_TREATMENT_EDUCATION]
    assert spec.subtopic == "adverse_effects"


def test_generic_testing_followup_inherits_close_contact_task() -> None:
    context = GuidelineTaskContext(
        scope=GuidelineScope.INFECTION_CONTROL,
        subtopic="contact_evaluation",
        population=["close_contacts"],
    )

    spec = parse_task_spec(
        "具体做什么检查？",
        active_intent="guideline:infection_control",
        active_guideline_context=context,
    )

    assert spec.task_goals == [TaskGoal.GUIDELINE_INFECTION_CONTROL]
    assert spec.subtopic == "contact_evaluation"
    assert spec.population == ["close_contacts"]
    assert spec.goal_evidence[0].evidence_source == GoalEvidenceSource.CONTEXT_MEMORY


def test_elliptical_special_population_switch_keeps_testing_operation() -> None:
    context = GuidelineTaskContext(
        scope=GuidelineScope.SPECIAL_POPULATION,
        subtopic="special_population_testing",
        population=["children"],
    )

    spec = parse_task_spec(
        "那 HIV 感染的成年人疑似肺结核呢？",
        active_intent="guideline:special_population",
        active_guideline_context=context,
    )

    assert spec.task_goals == [TaskGoal.GUIDELINE_SPECIAL_POPULATION]
    assert spec.subtopic == "special_population_testing"
    assert spec.population == ["people_living_with_hiv"]
    assert spec.goal_evidence[0].evidence_source == GoalEvidenceSource.CONTEXT_MEMORY


def test_infection_status_and_home_precaution_followups_are_context_bounded() -> None:
    infection_test = parse_task_spec(
        "就是活动性肺结核吗？",
        active_intent="guideline:diagnostic_testing",
        active_guideline_context=GuidelineTaskContext(
            scope=GuidelineScope.DIAGNOSTIC_TESTING,
            subtopic="tb_infection_test_interpretation",
        ),
    )
    home = parse_task_spec(
        "在家里具体还要注意什么？",
        active_intent="guideline:infection_control",
        active_guideline_context=GuidelineTaskContext(
            scope=GuidelineScope.INFECTION_CONTROL,
            subtopic="respiratory_protection",
        ),
    )

    assert infection_test.task_goals == [TaskGoal.GUIDELINE_DIAGNOSTIC_TESTING]
    assert infection_test.subtopic == "infection_vs_disease"
    assert home.task_goals == [TaskGoal.GUIDELINE_INFECTION_CONTROL]
    assert home.subtopic == "infection_control_precautions"
