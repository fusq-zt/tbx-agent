import pytest

from tbx_agent.retrieval.query_understanding import (
    is_guidance_contextual_followup,
    understand_guidance_query,
)
from tbx_agent.task_spec import GuidelineScenarioTag, GuidelineScope


def test_stop_medication_wording_resolves_to_treatment_principles() -> None:
    profile = understand_guidance_query("服药后不舒服，我是否现在就自行停掉所有药？")

    assert profile is not None
    assert profile.scope.value == "treatment_education"
    assert profile.subtopic == "treatment_principles"


def test_negated_mask_entity_does_not_override_requested_cleaning_topic() -> None:
    profile = understand_guidance_query(
        "怀疑传染性肺结核，我不是问口罩，而是家里怎么清洁消毒？"
    )

    assert profile is not None
    assert profile.scope == GuidelineScope.INFECTION_CONTROL
    assert profile.subtopic == "infection_control_precautions"
    assert profile.population == ()


def test_positive_mask_question_still_selects_respiratory_protection() -> None:
    profile = understand_guidance_query("怀疑肺结核时需要戴口罩吗？")

    assert profile is not None
    assert profile.scope == GuidelineScope.INFECTION_CONTROL
    assert profile.subtopic == "respiratory_protection"


@pytest.mark.parametrize(
    "query",
    (
        "共用餐具会传播肺结核吗？",
        "共用碗筷会得结核吗？",
        "和肺结核患者一起吃饭有餐具传播风险吗？",
    ),
)
def test_shared_utensil_questions_select_direct_transmission_subtopic(
    query: str,
) -> None:
    profile = understand_guidance_query(query)

    assert profile is not None
    assert profile.scope == GuidelineScope.INFECTION_CONTROL
    assert profile.subtopic == "shared_utensil_transmission"


def test_negated_culture_entity_does_not_pollute_xpert_scenario() -> None:
    profile = understand_guidance_query("不是问培养，是想问Xpert阴性能否排除肺结核？")

    assert profile is not None
    assert profile.scope == GuidelineScope.DIAGNOSTIC_TESTING
    assert profile.subtopic == "negative_test_interpretation"
    assert profile.product_terms == ("Xpert MTB/RIF",)
    assert profile.scenario_tags == (GuidelineScenarioTag.TEST_NAAT,)


def test_population_only_followup_replaces_inherited_population() -> None:
    profile = understand_guidance_query(
        "那儿童呢？",
        prior_scope=GuidelineScope.SPECIAL_POPULATION,
        prior_subtopic="special_population_testing",
        prior_population=("pregnant_people",),
    )

    assert profile is not None
    assert profile.scope == GuidelineScope.SPECIAL_POPULATION
    assert profile.subtopic == "special_population_testing"
    assert profile.population == ("children",)
    assert profile.resolution_code == "query_context_refinement_v1"


def test_negative_followup_overrides_inherited_test_use_subtopic() -> None:
    profile = understand_guidance_query(
        "如果阴性呢？",
        prior_scope=GuidelineScope.DIAGNOSTIC_TESTING,
        prior_subtopic="rapid_molecular_diagnostics",
        prior_product_terms=("Xpert MTB/RIF",),
        prior_scenario_tags=(GuidelineScenarioTag.TEST_NAAT,),
    )

    assert profile is not None
    assert profile.scope == GuidelineScope.DIAGNOSTIC_TESTING
    assert profile.subtopic == "negative_test_interpretation"
    assert profile.product_terms == ("Xpert MTB/RIF",)
    assert profile.scenario_tags == (GuidelineScenarioTag.TEST_NAAT,)


def test_general_topic_switch_is_not_a_guidance_followup() -> None:
    assert is_guidance_contextual_followup("具体该怎么做？") is True
    assert is_guidance_contextual_followup("1+1 = ？") is False


def test_negative_smear_overrides_cad_result_interpretation() -> None:
    profile = understand_guidance_query(
        "这个患者胸片模型提示 TB，但是痰涂片阴性，是不是基本可以排除了？"
    )

    assert profile is not None
    assert profile.scope == GuidelineScope.DIAGNOSTIC_TESTING
    assert profile.subtopic == "negative_test_interpretation"
    assert GuidelineScenarioTag.TEST_SMEAR in profile.scenario_tags


def test_negative_naat_overrides_cad_result_interpretation() -> None:
    profile = understand_guidance_query("胸片模型提示TB，但Xpert阴性是否能排除？")

    assert profile is not None
    assert profile.scope == GuidelineScope.DIAGNOSTIC_TESTING
    assert profile.subtopic == "negative_test_interpretation"
    assert profile.product_terms == ("Xpert MTB/RIF",)
    assert GuidelineScenarioTag.TEST_NAAT in profile.scenario_tags


def test_screening_abnormal_next_step_resolves_to_diagnostic_pathway() -> None:
    profile = understand_guidance_query(
        "这张片是体检发现的，患者目前没有明显症状。"
        "模型更倾向于非结核异常；这种筛查异常一般下一步需要做什么？"
    )

    assert profile is not None
    assert profile.scope == GuidelineScope.DIAGNOSTIC_TESTING
    assert profile.subtopic == "diagnostic_pathway"
    assert GuidelineScenarioTag.AFTER_ABNORMAL_CXR in profile.scenario_tags


def test_first_line_drug_milligram_questions_resolve_to_medication_dose() -> None:
    for drug in ("利福平", "异烟肼", "吡嗪酰胺", "乙胺丁醇"):
        profile = understand_guidance_query(
            f"患者60 kg，肝肾功能正常，{drug}具体多少毫克？"
        )

        assert profile is not None
        assert profile.scope == GuidelineScope.TREATMENT_EDUCATION
        assert profile.subtopic == "medication_dose"


def test_child_who_has_difficulty_producing_sputum_sets_no_sputum_tag() -> None:
    profile = understand_guidance_query("如果孩子很难咳出痰怎么办？")

    assert profile is not None
    assert profile.scope == GuidelineScope.SPECIAL_POPULATION
    assert profile.population == ("children",)
    assert GuidelineScenarioTag.NO_SPUTUM in profile.scenario_tags
