"""Retrieval-local understanding for the single TB guidance search tool.

The main agent only decides whether a guideline lookup is useful.  This module
then converts the original question into a small, auditable retrieval profile.
It does *not* choose an agent tool or generate medical content.  Its output is
used solely to narrow the reviewed corpus and to enforce population, test and
scenario applicability before evidence can be returned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypeVar

from ..task_spec import GuidelineScenarioTag, GuidelineScope


@dataclass(frozen=True, slots=True)
class GuidanceQueryProfile:
    """Internal constraints resolved from one original user question."""

    scope: GuidelineScope
    subtopic: str
    population: tuple[str, ...] = ()
    product_terms: tuple[str, ...] = ()
    scenario_tags: tuple[GuidelineScenarioTag, ...] = ()
    resolution_code: str = "query_profile_v1"


_POPULATION_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("people_living_with_hiv", ("hiv", "艾滋")),
    ("children", ("儿童", "孩子", "小孩", "婴幼儿", "未成年人", "15岁以下")),
    ("pregnant_people", ("孕妇", "妊娠", "怀孕", "孕期")),
    ("immunosuppressed_people", ("免疫抑制", "免疫缺陷", "器官移植")),
    ("older_adults", ("老年", "65岁", "65 岁")),
    ("people_with_diabetes", ("糖尿病",)),
    (
        "close_contacts",
        (
            "密切接触",
            "接触者",
            "家庭成员",
            "家里有人",
            "家人",
            "同住者",
            "同住",
            "住在一起",
            "共同居住",
            "室友",
        ),
    ),
    ("tb_high_risk_population", ("高风险人群", "高危人群")),
)

_TEST_TERMS: dict[GuidelineScenarioTag, tuple[str, ...]] = {
    GuidelineScenarioTag.TEST_SMEAR: ("痰涂片", "痰片", "涂片"),
    GuidelineScenarioTag.TEST_CULTURE: ("痰培养", "结核培养", "培养"),
    GuidelineScenarioTag.TEST_NAAT: (
        "xpert",
        "ultra",
        "naat",
        "核酸",
        "分子检测",
        "分子诊断",
    ),
    GuidelineScenarioTag.TEST_CXR: ("胸片", "胸部x线", "胸部 x线", "x光"),
}

_SPECIAL_POPULATIONS = frozenset(
    {
        "people_living_with_hiv",
        "children",
        "pregnant_people",
        "immunosuppressed_people",
    }
)

_TREATMENT_CUES = (
    "治疗",
    "用药",
    "服药",
    "药物",
    "抗结核药",
    "疗程",
    "方案",
    "剂量",
    "漏服",
    "漏药",
    "停药",
    "停掉",
    "自行停",
    "换药",
    "不良反应",
    "副作用",
    "住院",
    "门诊",
    "出院",
    "社区照护",
    "利福平",
    "异烟肼",
    "吡嗪酰胺",
    "乙胺丁醇",
    "毫克",
    "mg",
)
_INFECTION_CONTROL_CUES = (
    "传染",
    "传播",
    "共用餐具",
    "餐具",
    "碗筷",
    "共餐",
    "共用水杯",
    "共用杯子",
    "一起吃饭",
    "分享食物",
    "分享饮料",
    "口罩",
    "防护",
    "隔离",
    "通风",
    "咳嗽礼仪",
    "密切接触",
    "接触者",
    "家庭成员",
    "同住",
    "住在一起",
    "返工",
    "返校",
    "上班",
    "上学",
)
_SCREENING_CUES = (
    "主动筛查",
    "优先筛查",
    "筛查对象",
    "高风险人群",
    "高危人群",
    "重点人群",
    "哪些人需要筛查",
    "哪些人建议筛查",
)
_DIAGNOSTIC_CUES = (
    "检查",
    "检测",
    "诊断",
    "确诊",
    "排除",
    "判断",
    "痰",
    "xpert",
    "ultra",
    "naat",
    "培养",
    "涂片",
    "胸片",
    "胸部x线",
    "ct",
    "tst",
    "igra",
    "结核感染",
    "咳嗽",
    "低热",
    "盗汗",
    "是不是得",
    "有没有肺结核",
)
_CAD_CUES = ("cad", "人工智能", "ai筛查", "ai结果", "胸片模型", "模型结果")
_NEGATIVE_RESULT_CUES = (
    "阴性",
    "没查到",
    "未查到",
    "未检出",
    "没有查到",
    "正常",
)
_COMPARISON_CUES = (
    "区别",
    "不同",
    "比较",
    "分别",
    "各自作用",
    "哪个优先",
    "怎么选",
)
_TB_INFECTION_TEST_CUES = ("tst", "igra", "结核菌素", "感染检测", "结核感染试验")
_CONTEXTUAL_FOLLOWUP_CUES = (
    "具体",
    "给出",
    "怎么做",
    "怎么办",
    "然后",
    "接下来",
    "展开",
    "继续",
    "那",
    "呢",
)


def _contains(text: str, *terms: str) -> bool:
    return any(term in text for term in terms)


_T = TypeVar("_T")


def _dedupe(items: list[_T]) -> tuple[_T, ...]:
    return tuple(dict.fromkeys(items))


def _term_is_negated(text: str, term: str) -> bool:
    """Recognize explicit entity rejection without treating result negation as rejection."""

    escaped = re.escape(term)
    patterns = (
        rf"(?:不是(?:想)?问|并非(?:想)?问|不想问|不要问|不讨论)\s*.{{0,5}}{escaped}",
        rf"(?:排除|忽略)\s*{escaped}(?:这个|这一)?(?:检查|项目)?",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _active_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text and not _term_is_negated(text, term) for term in terms)


def _extract_populations(text: str) -> tuple[str, ...]:
    found = [
        canonical
        for canonical, terms in _POPULATION_TERMS
        if _active_term(text, terms)
    ]
    return _dedupe(found)


def _extract_products(text: str) -> tuple[str, ...]:
    products: list[str] = []
    ultra_active = _active_term(text, ("xpert ultra", "ultra"))
    mtb_rif_active = _active_term(
        text,
        ("xpert mtb/rif", "xpert mtb rif", "xpert mtb-rif"),
    )
    generic_xpert_active = _active_term(text, ("xpert",))
    if mtb_rif_active or (generic_xpert_active and not ultra_active):
        products.append("Xpert MTB/RIF")
    if ultra_active:
        products.append("Xpert Ultra")
    return _dedupe(products)


def _extract_test_tags(text: str) -> list[GuidelineScenarioTag]:
    return [tag for tag, terms in _TEST_TERMS.items() if _active_term(text, terms)]


def _diagnostic_subtopic(
    text: str,
    *,
    test_tags: list[GuidelineScenarioTag],
) -> str:
    if _contains(
        text,
        "结核感染和活动性结核病",
        "结核感染和结核病",
        "感染和活动性",
        "感染和发病",
        "潜伏感染和活动性",
    ):
        return "infection_vs_disease"
    if _contains(text, *_TB_INFECTION_TEST_CUES):
        return "tb_infection_test_interpretation"
    if len(test_tags) >= 2 and _contains(text, *_COMPARISON_CUES):
        return "test_comparison"
    if test_tags and _contains(text, *_NEGATIVE_RESULT_CUES):
        return "negative_test_interpretation"
    if _contains(text, "所有", "一律", "都做", "必须") and "ct" in text:
        return "imaging_modality_selection"
    if GuidelineScenarioTag.TEST_NAAT in test_tags:
        return "rapid_molecular_diagnostics"
    if test_tags and _contains(text, "作用", "用途", "用来做什么", "是什么检查"):
        return "test_comparison"
    return "diagnostic_pathway"


def _treatment_subtopic(text: str) -> str:
    if _contains(text, "住院", "门诊", "出院", "社区照护", "社区治疗"):
        return "care_setting"
    if _contains(text, "漏服", "漏药", "忘记吃药", "依从"):
        return "treatment_adherence"
    if _contains(
        text,
        "不良反应",
        "副作用",
        "视力",
        "视物",
        "看不清",
        "色觉",
    ):
        return "adverse_effects"
    if _contains(
        text,
        "剂量",
        "用量",
        "毫克",
        "mg",
        "多少毫克",
        "多少 mg",
        "多少mg",
        "每次多少",
        "每天多少",
    ):
        return "medication_dose"
    resistant = _contains(text, "耐药", "耐多药", "mdr", "rr-tb", "rrtb")
    if resistant and _contains(text, "普通", "药物敏感", "敏感结核") and _contains(
        text, "一样", "相同", "区别", "不同"
    ):
        return "drug_resistant_treatment_comparison"
    if _contains(text, "疗程", "多久", "多长时间"):
        return "standard_regimen_duration"
    return "treatment_principles"


def _infection_control_subtopic(text: str, populations: tuple[str, ...]) -> str:
    # Keep a direct-transmission misconception separate from the broader
    # infection-control topic.  Otherwise a generic airborne-transmission
    # passage can be relevant yet fail to answer whether the named object is
    # itself a transmission route.
    if _active_term(
        text,
        (
            "共用餐具",
            "餐具",
            "碗筷",
            "共餐",
            "共用水杯",
            "共用杯子",
            "一起吃饭",
            "分享食物",
            "分享饮料",
        ),
    ):
        return "shared_utensil_transmission"
    if _contains(text, "返工", "返校", "上班", "上学", "恢复工作", "恢复上学"):
        return "return_to_work_school"
    if _contains(
        text,
        "什么时候不传染",
        "何时不传染",
        "不再传染",
        "传染性消失",
        "没有传染性",
        "没传染性",
    ):
        return "infectiousness_clearance"
    if "close_contacts" in populations and _contains(
        text, "检查", "检测", "评估", "怎么办", "怎么做", "处理"
    ):
        return "contact_evaluation"
    # A named entity may be present only because the user explicitly rejected
    # it (for example, “不是问口罩，而是家里怎么清洁消毒”).  The retrieval
    # profile must follow the requested topic rather than raw substring
    # presence.
    if _active_term(text, ("口罩", "呼吸防护", "佩戴")):
        return "respiratory_protection"
    if _contains(
        text,
        "平时要注意",
        "平时注意",
        "日常要注意",
        "日常注意",
        "注意什么",
        "注意哪些",
        "减少传播",
        "通风",
        "咳嗽礼仪",
        "清洁",
        "消毒",
        "家庭环境",
        "家里怎么",
    ):
        return "infection_control_precautions"
    return "infection_control"


def _scenario_tags(
    text: str,
    *,
    subtopic: str,
    test_tags: list[GuidelineScenarioTag],
) -> tuple[GuidelineScenarioTag, ...]:
    tags = list(test_tags)
    if _contains(text, *_TB_INFECTION_TEST_CUES):
        tags.append(GuidelineScenarioTag.TB_INFECTION_TEST)
    if _contains(
        text,
        "胸片异常",
        "胸部x线异常",
        "筛查阳性",
        "筛查异常",
    ):
        tags.append(GuidelineScenarioTag.AFTER_ABNORMAL_CXR)
    if _contains(
        text,
        "无痰",
        "咳不出痰",
        "没有痰",
        "难以获得痰",
        "难咳痰",
        "难以咳痰",
        "咳痰困难",
        "难咳出痰",
        "很难咳出痰",
    ):
        tags.append(GuidelineScenarioTag.NO_SPUTUM)
    if _contains(text, "耐药", "耐多药", "药敏", "mdr", "rr-tb"):
        tags.append(GuidelineScenarioTag.DRUG_RESISTANCE)
    if _contains(text, "高风险人群", "高危人群"):
        tags.append(GuidelineScenarioTag.RISK_HIGH_RISK_GROUPS)
    if "重点人群" in text:
        tags.append(GuidelineScenarioTag.RISK_KEY_GROUPS)
    if subtopic == "care_setting":
        if _contains(text, "都必须", "必须住院", "是否都", "都要住院", "所有"):
            tags.append(GuidelineScenarioTag.CARE_UNIVERSAL_HOSPITALIZATION)
        if _contains(text, "什么情况", "何时", "哪些情况", "需要住院"):
            tags.append(GuidelineScenarioTag.CARE_INPATIENT_INDICATIONS)
        if _contains(text, "出院", "转门诊", "转到门诊", "社区照护"):
            tags.append(GuidelineScenarioTag.CARE_AMBULATORY_TRANSITION)
    if subtopic == "adverse_effects":
        if _contains(text, "视力", "视物", "看不清", "色觉", "模糊"):
            tags.append(GuidelineScenarioTag.ADVERSE_VISUAL)
        elif _contains(text, "常见", "有哪些", "列举", "不良反应", "副作用"):
            tags.append(GuidelineScenarioTag.ADVERSE_GENERAL_LIST)
    return _dedupe(tags)


def is_guidance_contextual_followup(query: str) -> bool:
    """Return whether an otherwise elliptical query may refine prior retrieval state.

    This predicate only controls the lifetime of already resolved retrieval
    context.  It never selects an agent tool or creates medical evidence.
    """

    text = re.sub(r"\s+", " ", query.strip()).casefold()
    if not text:
        return False
    return bool(
        _extract_populations(text)
        or _contains(text, *_NEGATIVE_RESULT_CUES)
        or (len(text) <= 24 and _contains(text, *_CONTEXTUAL_FOLLOWUP_CUES))
    )


def understand_guidance_query(
    query: str,
    *,
    prior_scope: GuidelineScope | str | None = None,
    prior_subtopic: str | None = None,
    prior_population: tuple[str, ...] | list[str] = (),
    prior_product_terms: tuple[str, ...] | list[str] = (),
    prior_scenario_tags: tuple[GuidelineScenarioTag | str, ...]
    | list[GuidelineScenarioTag | str] = (),
) -> GuidanceQueryProfile | None:
    """Resolve retrieval constraints from the original question.

    ``None`` means the question did not contain enough TB-guidance semantics to
    choose a safe corpus scope.  Callers must report an evidence/query gap
    rather than defaulting to a broad diagnostic search.
    """

    text = re.sub(r"\s+", " ", query.strip()).casefold()
    if not text:
        return None
    populations = _extract_populations(text)
    products = _extract_products(text)
    test_tags = _extract_test_tags(text)
    asks_after_abnormal_screening = _contains(
        text,
        "胸片异常",
        "胸部x线异常",
        "筛查阳性",
        "筛查异常",
    ) and _contains(
        text,
        "下一步",
        "做什么",
        "怎么办",
        "怎么做",
        "进一步检查",
        "进一步检测",
    )

    if _contains(text, *_TREATMENT_CUES):
        scope = GuidelineScope.TREATMENT_EDUCATION
        subtopic = _treatment_subtopic(text)
    elif _contains(text, *_INFECTION_CONTROL_CUES):
        scope = GuidelineScope.INFECTION_CONTROL
        subtopic = _infection_control_subtopic(text, populations)
    elif asks_after_abnormal_screening:
        scope = GuidelineScope.DIAGNOSTIC_TESTING
        subtopic = "diagnostic_pathway"
    elif _contains(text, *_SCREENING_CUES) or (
        "筛查" in text and _contains(text, "哪些人", "谁", "人群", "对象", "优先")
    ):
        scope = GuidelineScope.SCREENING
        subtopic = (
            "risk_groups"
            if _contains(text, "高风险人群", "高危人群", "重点人群")
            else "active_screening_population"
        )
    elif _SPECIAL_POPULATIONS.intersection(populations) and _contains(
        text, *_DIAGNOSTIC_CUES
    ):
        scope = GuidelineScope.SPECIAL_POPULATION
        subtopic = (
            "tb_infection_test_interpretation"
            if _contains(text, *_TB_INFECTION_TEST_CUES)
            and _contains(text, "阳性", "说明", "意味着", "活动性")
            else "special_population_testing"
        )
    # A named CAD/model result is contextual evidence, but an explicit
    # bacteriological test with a negative-result question is a diagnostic
    # interpretation request.  Resolve that entity before the broader CAD cue
    # so “模型提示TB，但痰涂片阴性，能否排除” reaches the applicable evidence.
    elif test_tags and _contains(text, *_NEGATIVE_RESULT_CUES):
        scope = GuidelineScope.DIAGNOSTIC_TESTING
        subtopic = "negative_test_interpretation"
    elif _contains(text, *_CAD_CUES):
        scope = GuidelineScope.CAD_INTERPRETATION
        subtopic = "cad_result_interpretation"
    elif _contains(text, *_DIAGNOSTIC_CUES):
        scope = GuidelineScope.DIAGNOSTIC_TESTING
        subtopic = _diagnostic_subtopic(text, test_tags=test_tags)
    else:
        if prior_scope is None or not prior_subtopic:
            return None
        current_populations = populations
        negative_followup = _contains(text, *_NEGATIVE_RESULT_CUES)
        if not (
            current_populations
            or negative_followup
            or (len(text) <= 24 and _contains(text, *_CONTEXTUAL_FOLLOWUP_CUES))
        ):
            return None
        try:
            inherited_scope = GuidelineScope(str(prior_scope))
            inherited_tags = tuple(
                GuidelineScenarioTag(str(item)) for item in prior_scenario_tags
            )
        except ValueError:
            return None
        if negative_followup:
            inherited_scope = GuidelineScope.DIAGNOSTIC_TESTING
            inherited_subtopic = "negative_test_interpretation"
        else:
            inherited_subtopic = prior_subtopic
        return GuidanceQueryProfile(
            scope=inherited_scope,
            subtopic=inherited_subtopic,
            population=(current_populations or tuple(prior_population)),
            product_terms=tuple(prior_product_terms),
            scenario_tags=_dedupe([*inherited_tags, *test_tags]),
            resolution_code="query_context_refinement_v1",
        )

    return GuidanceQueryProfile(
        scope=scope,
        subtopic=subtopic,
        population=populations,
        product_terms=products,
        scenario_tags=_scenario_tags(text, subtopic=subtopic, test_tags=test_tags),
    )
