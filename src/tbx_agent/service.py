from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Any

from .anatomy_runs import (
    AnatomyRunRecord,
    AnatomyRunStatus,
    RefinementRunStatus,
    build_anatomy_pipeline_generation_key,
)
from .artifacts import ArtifactManager, default_artifact_root, load_manifest
from .capability_answer import TBX_CAPABILITY_ANSWER
from .config import Settings
from .knowledge import GuidelineRetriever, RetrievalAttestation
from .narrator import (
    NARRATOR_POLICY_ID,
    LlamaCppNarrator,
    NarrationError,
    NarrationRejectedError,
    OllamaNarrator,
    OptionalOpenAINarrator,
    validate_grounded_synthesis_summary,
    validate_narration_summary,
)
from .reports import generate_case_report
from .retrieval.query_understanding import understand_guidance_query
from .routing import ROUTER_POLICY_ID
from .safety import (
    SafetyVerifier,
    SafetyViolationError,
    is_medication_change_request,
)
from .schemas import (
    ActorRole,
    AgentResponse,
    CaseRecord,
    ClassificationExecutionStatus,
    ClassifierClass,
    FusionDecision,
    GroundedGuidelineClaim,
    GuidelineAnswerStatus,
    LocalizationEvidence,
    NarrationStatus,
    ResponseKind,
    RetrievedGuidelineEvidence,
    ReviewOrigin,
    ReviewRecord,
    ThreadMemoryEvent,
    Urgency,
    UserPreferences,
    VisionEvidence,
    VisualResult,
)
from .screening import ScreeningEngine
from .state_locks import ScopedLockPool as _ScopedLockPool
from .state_locks import state_lock_key as _state_lock_key
from .storage import AccessDeniedError, SQLiteStore, VersionConflictError
from .task_spec import GuidelineScenarioTag, GuidelineScope
from .tools.builtin import build_builtin_registry
from .tools.contracts import (
    ToolCallStatus,
    ToolInvocation,
    ToolName,
    ToolResult,
    ToolUnavailableError,
)
from .vision import MockRank03Backend, Rank03Backend, VisionBackend, fuse_rank03, validate_image
from .vision.anatomy import (
    SPATIAL_PRESENTATION_POLICY_ID,
    AnatomyBackend,
    AnatomyBackendError,
    AnatomyBackendUnavailable,
    LungFieldLocalizationPolicy,
    LungFieldZone,
    LungSide,
    TorchXRayVisionPSPNetBackend,
    XRVPSPNetConfig,
    build_chat_spatial_summary,
    build_spatial_summary,
    localize_detection_boxes,
)
from .vision.base import VisionBackendError
from .vision.display import DetectionDisplayPolicy, select_display_detections
from .vision.image_validator import ImageValidationError, ValidatedImage
from .vision.refinement import (
    ContourRefinementBackend,
    HFMedSAMBoxRefinementBackend,
    HFMedSAMConfig,
    RefinementBackendError,
    RefinementBackendUnavailable,
)


class ConsentRequiredError(PermissionError):
    pass


class AssessmentStateConflictError(RuntimeError):
    """Existing immutable assessment identity needs an explicit migration/rerun path."""


class AnatomyNotConfiguredError(RuntimeError):
    """The optional anatomy evidence service is disabled for this deployment."""


_BATCH_REVIEW_NAMESPACE = uuid.UUID("552b5d18-471e-4f8a-9e1e-bdd487135f51")
_BATCH_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DETECTION_DISPLAY_POLICY = DetectionDisplayPolicy()
_CLASS_EVIDENCE_LABELS = {
    ClassifierClass.HEALTHY: "健康类",
    ClassifierClass.SICK_NON_TB: "非结核异常类",
    ClassifierClass.TB: "结核类",
}
_LUNG_SIDE_LABELS = {LungSide.LEFT: "左", LungSide.RIGHT: "右"}
_LUNG_ZONE_LABELS = {
    LungFieldZone.UPPER: "上肺野",
    LungFieldZone.MIDDLE: "中肺野",
    LungFieldZone.LOWER: "下肺野",
}
_IMAGE_QUALITY_LABELS = {
    "dicom_chest_body_part_not_verified": "DICOM 信息未能确认检查部位为胸部",
    "dicom_frontal_view_not_verified": "DICOM 信息未能确认是正位 AP/PA 胸片",
    "dicom_burned_in_annotation_not_excluded": "DICOM 未能确认不存在烧录文字或标注",
    "resolution_below_512": "图像短边低于 512 像素",
    "outside_rank03_validated_512x512_domain": "图像尺寸不是 512×512",
    "unusual_aspect_ratio": "图像宽高比异常",
    "very_low_dynamic_range": "图像灰度动态范围过低",
}
_DETECTOR_NOT_REQUESTED_REF = "detector_execution:not_requested"
_DETECTOR_COMPLETED_REF = "detector_execution:on_demand_completed"


_GUIDELINE_SCOPE_POLICY: dict[str, dict[str, set[str]]] = {
    # Claim scopes are the hard authority boundary.  Topic/subtopic signals are
    # deliberately absent here because they only reorder evidence inside scope.
    "screening": {
        "claim_scopes": {
            "active_screening",
            "risk_groups",
            "screening_pathway",
            "cad_boundary",
            "screen_positive_referral",
        },
        "jurisdictions": {"China", "WHO"},
    },
    "diagnostic_testing": {
        "claim_scopes": {
            "initial_diagnostic_testing",
            "diagnostic_test_roles",
            "rapid_drug_resistance_testing",
            "special_population_testing",
            "test_limitations",
            "test_interpretation",
            "culture_interpretation",
            "tb_infection_test_interpretation",
            "infection_status_education",
            "imaging_result_context",
            "imaging_modality_selection",
            "sample_acquisition",
            "low_incidence_context",
            "china_diagnostic_principles",
            "confirmation_basis",
            "confirmation_boundary",
        },
        "jurisdictions": {"China", "WHO", "US"},
    },
    # Drug/regimen education is limited to the reviewed WHO Module 4 guideline
    # excerpts. Care-setting education is separately limited to the reviewed
    # operational-handbook clauses; its MDR-TB recommendation is never widened
    # into a same-grade recommendation for all drug-susceptible TB. The local
    # product policy remains a safety boundary, not medical treatment evidence.
    "treatment_education": {
        "claim_scopes": {
            "treatment_education",
            "treatment_principles",
            "standard_regimen_duration",
            "care_setting_education",
            "hospitalization_indications",
            "treatment_adherence",
            "adverse_effects",
            "medication_safety",
        },
        "jurisdictions": {"China", "WHO", "US"},
    },
    "infection_control": {
        "claim_scopes": {
            "infection_control",
            "respiratory_protection",
            "contact_evaluation",
            "infection_status_education",
            "infectiousness_assessment",
            "return_to_activities",
        },
        "jurisdictions": {"China", "WHO", "US"},
    },
    "special_population": {
        "claim_scopes": {
            "special_population_testing",
            "tb_infection_test_interpretation",
            "infection_status_education",
            "active_screening",
            "risk_groups",
            "screening_pathway",
        },
        "jurisdictions": {"China", "WHO", "US"},
    },
    "cad_interpretation": {
        "claim_scopes": {
            "cad_boundary",
            "imaging_role_education",
            "imaging_nonspecificity",
            "confirmation_boundary",
        },
        "jurisdictions": {"China", "WHO"},
    },
}

_GUIDELINE_SUBTOPIC_TOPICS: dict[str, set[str]] = {
    "risk_groups": {"risk_groups"},
    "active_screening_population": {"active_screening", "risk_groups", "screening_pathway"},
    "rapid_molecular_diagnostics": {"rapid_diagnostics", "naat", "next_tests"},
    "negative_test_interpretation": {"test_limitations", "microbiology", "naat"},
    "test_comparison": {
        "rapid_diagnostics",
        "naat",
        "microbiology",
        "culture",
        "smear",
        "diagnostic_test_roles",
    },
    "tb_infection_test_interpretation": {
        "tb_infection_test",
        "tb_infection_testing",
        "test_interpretation",
        "infection_status",
        "infection_status_education",
    },
    "infection_vs_disease": {
        "infection_status",
        "infection_status_education",
        "tb_infection_test",
        "tb_infection_testing",
    },
    "imaging_modality_selection": {"imaging", "diagnosis", "next_tests"},
    "diagnostic_pathway": {"diagnosis", "next_tests", "microbiology"},
    "treatment_principles": {"treatment_education", "monitoring"},
    "standard_regimen_duration": {"treatment_education", "treatment_regimen"},
    "medication_dose": {"medication_dose"},
    "care_setting": {
        "treatment_education",
        "treatment_support",
        "care_setting",
        "ambulatory_care",
        "inpatient_care",
    },
    "treatment_adherence": {"treatment_adherence", "treatment_safety"},
    "adverse_effects": {"adverse_effects", "treatment_safety"},
    "respiratory_protection": {"infection_control", "respiratory_protection"},
    "infection_control": {"transmission", "infection_status"},
    "shared_utensil_transmission": {
        "transmission",
        "infection_control",
        "food_drink_utensils",
    },
    "infection_control_precautions": {"respiratory_protection"},
    "contact_evaluation": {"contact_evaluation", "infection_control"},
    "infectiousness_clearance": {"infectiousness", "infection_control"},
    "return_to_work_school": {
        "return_to_activities",
        "infectiousness",
        "infection_control",
    },
    "cad_result_interpretation": {
        "cad_boundary",
        "imaging",
        "imaging_role_education",
        "imaging_nonspecificity",
    },
    "special_population_guidance": {
        "special_population",
        "risk_groups",
        "screening_pathway",
    },
    "special_population_testing": {
        "special_population_testing",
        "pregnancy",
        "pediatrics",
        "hiv",
        "screening_pathway",
        "next_tests",
        "rapid_diagnostics",
    },
}

_GUIDELINE_QUERY_EXPANSION: dict[str, str] = {
    "risk_groups": "肺结核 主动筛查 高风险人群 重点人群",
    "active_screening_population": "肺结核 主动筛查 筛查对象 人群 路径",
    "rapid_molecular_diagnostics": "肺结核 快速分子检测 NAAT 核酸检测 病原学",
    "negative_test_interpretation": "肺结核 痰涂片 阴性 未检出 排除 检测局限",
    "test_comparison": "肺结核 Xpert NAAT 痰培养 痰涂片 作用 区别 比较",
    "tb_infection_test_interpretation": "结核感染检测 TST IGRA 阳性 活动性结核病",
    "infection_vs_disease": "结核感染 潜伏结核感染 活动性结核病 区别",
    "imaging_modality_selection": "WHO 疑似肺结核 胸部X线 CT 诊断检查",
    "diagnostic_pathway": "肺结核 诊断 检查 病原学",
    "treatment_principles": "肺结核 治疗原则 治疗教育",
    "standard_regimen_duration": "肺结核 标准治疗方案 疗程",
    "medication_dose": "肺结核 抗结核药 剂量 用量",
    "care_setting": "肺结核 照护模式 住院 门诊 社区 去中心化 医学安全",
    "treatment_adherence": "抗结核治疗 漏服 漏药 加倍 联系治疗机构",
    "adverse_effects": "抗结核药 不良反应 副作用 视力模糊 处理",
    "respiratory_protection": "肺结核 感染控制 呼吸防护 佩戴口罩",
    "infection_control": "肺结核 是否传染 空气传播 活动性结核 潜伏感染",
    "shared_utensil_transmission": (
        "肺结核 共用餐具 碗筷 共餐 食物 饮料 不经餐具传播 空气传播"
    ),
    "infection_control_precautions": "疑似传染性肺结核 日常 通风 咳嗽礼仪 口罩",
    "contact_evaluation": "肺结核 密切接触者 同住 家庭成员 评估 检查",
    "infectiousness_clearance": "肺结核 传染性 何时降低 痰检查 治疗反应",
    "return_to_work_school": "肺结核 返工 返校 上班 上学 传染性 医疗评估",
    "cad_result_interpretation": "肺结核 胸部X线 计算机辅助检测 影像解释",
    "special_population_guidance": "肺结核 特殊人群 检测 筛查",
    "special_population_testing": "肺结核 特殊人群 检查 检测 病原学",
}

_GUIDELINE_MAX_CHAT_CLAIMS: dict[str, int] = {
    "risk_groups": 2,
    "active_screening_population": 2,
    "rapid_molecular_diagnostics": 2,
    "negative_test_interpretation": 2,
    "test_comparison": 3,
    "tb_infection_test_interpretation": 2,
    "infection_vs_disease": 2,
    "imaging_modality_selection": 2,
    "diagnostic_pathway": 2,
    "treatment_principles": 3,
    "standard_regimen_duration": 2,
    "medication_dose": 2,
    "care_setting": 3,
    "treatment_adherence": 2,
    "adverse_effects": 3,
    "shared_utensil_transmission": 1,
    "contact_evaluation": 3,
    "infectiousness_clearance": 2,
    "return_to_work_school": 2,
    "special_population_testing": 3,
}


# Subtopic-specific claim gates prevent a same-scope but different test or
# safety clause from being substituted merely because it ranks lexically well.
_GUIDELINE_SUBTOPIC_CLAIM_SCOPES: dict[str, set[str]] = {
    "test_comparison": {"diagnostic_test_roles"},
    "tb_infection_test_interpretation": {
        "tb_infection_test_interpretation",
        "infection_status_education",
    },
    "infection_vs_disease": {
        "infection_status_education",
        "tb_infection_test_interpretation",
    },
    "imaging_modality_selection": {"imaging_modality_selection"},
    "negative_test_interpretation": {
        "test_limitations",
        "test_interpretation",
        "culture_interpretation",
        "imaging_result_context",
    },
    "special_population_testing": {"special_population_testing"},
    "treatment_adherence": {"treatment_adherence", "medication_safety"},
    "adverse_effects": {"adverse_effects", "medication_safety"},
    # A dose answer requires a reviewed dose-specific claim.  General
    # treatment-principle or regimen-duration text must never substitute for
    # it merely because both live in the treatment scope.
    "medication_dose": {"medication_dose"},
    "contact_evaluation": {"contact_evaluation", "infection_status_education"},
    "infection_control": {"infection_status_education"},
    "shared_utensil_transmission": {"infection_control"},
    "infection_control_precautions": {"respiratory_protection"},
    "respiratory_protection": {"respiratory_protection"},
    "infectiousness_clearance": {"infectiousness_assessment"},
    "return_to_work_school": {"return_to_activities", "infectiousness_assessment"},
}


_GUIDELINE_SUBTOPIC_SOURCE_IDS: dict[str, set[str]] = {
    "risk_groups": {"china_active_screening_2026"},
    "active_screening_population": {"china_active_screening_2026"},
    "diagnostic_pathway": {
        "who_tb_diagnosis_module3_2025",
        "china_ws288_2017",
    },
    "test_comparison": {"cdc_tb_clinical_lab_2025"},
    "tb_infection_test_interpretation": {"cdc_tb_clinical_lab_2025"},
    "infection_vs_disease": {
        "cdc_tb_clinical_lab_2025",
        "cdc_tb_exposure_2024",
    },
    "treatment_adherence": {"cdc_tb_treatment_public_2025"},
    "adverse_effects": {"cdc_tb_adverse_events_2025"},
    "contact_evaluation": {"cdc_tb_exposure_2024"},
    "infection_control": {"cdc_tb_exposure_2024"},
    "shared_utensil_transmission": {"cdc_tb_exposure_2024"},
    "infectiousness_clearance": {"cdc_tb_prevention_2025"},
    "return_to_work_school": {"cdc_tb_prevention_2025"},
    "care_setting": {"who_tb_treatment_handbook_module4_2025"},
}


def _guideline_required_source_ids(
    *,
    subtopic: str,
    population: list[str],
    scenario_tags: set[GuidelineScenarioTag],
) -> set[str]:
    """Select reviewed evidence packets for entity-sensitive questions.

    Claim scope remains the authority boundary. This narrower source gate stops
    a crowded multi-topic page from displacing the exact test, population or
    safety packet before deterministic evidence selection can run.
    """

    if subtopic == "negative_test_interpretation":
        if GuidelineScenarioTag.TEST_SMEAR in scenario_tags:
            return {"ats_cdc_idsa_tb_diagnosis_2017"}
        return {"cdc_tb_clinical_lab_2025"}
    if subtopic == "special_population_testing":
        if "pregnant_people" in population:
            return {"cdc_tb_pregnancy_2025"}
        if "people_living_with_hiv" in population:
            return {"who_tb_diagnosis_module3_2025"}
        if "children" in population:
            if GuidelineScenarioTag.TB_INFECTION_TEST in scenario_tags:
                return {"cdc_tb_clinical_lab_2025"}
            return {
                "china_active_screening_2026",
                "who_tb_diagnosis_module3_2025",
            }
    return set(_GUIDELINE_SUBTOPIC_SOURCE_IDS.get(subtopic, ()))


_DRUG_RESISTANT_TREATMENT_TERMS = (
    "耐药",
    "耐多药",
    "广泛耐药",
    "mdr-tb",
    "mdrtb",
    "mdr/rr",
    "rr-tb",
    "rrtb",
    "dr-tb",
    "drtb",
    "xdr-tb",
    "xdrtb",
    "pre-xdr",
    "drug-resistant",
    "rifampicin-resistant",
    "multidrug-resistant",
)
_PERSONALIZED_TREATMENT_CONTEXT_TERMS = (
    "给我",
    "为我",
    "按我的",
    "根据我的",
    "我体重",
    "本人",
    "体重",
    "肝功能",
    "肾功能",
    "怀孕",
    "妊娠",
    "糖尿病",
    "hiv",
)
_PERSONALIZED_TREATMENT_DETAIL_TERMS = (
    "剂量",
    "用量",
    "每天",
    "每次",
    "疗程",
    "方案",
    "开药",
    "处方",
)


def _is_drug_resistant_treatment_question(value: str) -> bool:
    """Keep DS-TB education from crossing into a resistant-TB question."""

    normalized = value.casefold().replace("—", "-").replace("–", "-")
    return any(term in normalized for term in _DRUG_RESISTANT_TREATMENT_TERMS)


def _is_personalized_treatment_request(value: str) -> bool:
    """Detect personal regimen selection without blocking general education."""

    normalized = value.casefold()
    return any(term in normalized for term in _PERSONALIZED_TREATMENT_CONTEXT_TERMS) and any(
        term in normalized for term in _PERSONALIZED_TREATMENT_DETAIL_TERMS
    )


_LOCATOR_PAGE_RE = re.compile(r"(?:(?:PDF|期刊)?\s*页|第\s*)(\d{1,4})")


def _locator_page(locator: str) -> str | None:
    match = _LOCATOR_PAGE_RE.search(locator)
    return match.group(1) if match else None


def _normalized_evidence_term(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()


def _guideline_hit_priority(
    subtopic: str,
    hit: Any,
    *,
    population: list[str],
    scenario_tags: set[GuidelineScenarioTag],
) -> int:
    """Prefer direct answer clauses over merely same-topic background text."""

    text = str(getattr(hit, "text", ""))
    priority = 0
    if subtopic == "risk_groups":
        if "高风险人群包括" in text:
            priority = 300
        elif "重点人群包括" in text:
            priority = 200
        if "people_with_diabetes" in population and "糖尿病" in text:
            priority += 500
        if "older_adults" in population and any(term in text for term in ("65岁", "老年")):
            priority += 500
    elif subtopic == "active_screening_population":
        if "主动筛查优先对象包括" in text:
            priority = 300
        elif "重点人群包括" in text:
            priority = 200
        elif "高风险人群包括" in text:
            priority = 100
        if "older_adults" in population:
            if "高发病率地区" in text and "65岁" in text and "胸部X线" in text:
                priority += 1_200
            elif "65岁" in text:
                priority += 500
    elif subtopic == "rapid_molecular_diagnostics":
        if "作为初始诊断检测" in text:
            priority = 300
        elif "利福平耐药的初始检测" in text:
            priority = 250
    elif subtopic == "negative_test_interpretation":
        requested_modality = next(
            (
                name
                for tag, name in (
                    (GuidelineScenarioTag.TEST_CULTURE, "culture"),
                    (GuidelineScenarioTag.TEST_NAAT, "naat"),
                    (GuidelineScenarioTag.TEST_SMEAR, "smear"),
                    (GuidelineScenarioTag.TEST_CXR, "cxr"),
                )
                if tag in scenario_tags
            ),
            "",
        )
        modality_terms = {
            "culture": ("培养",),
            "naat": ("xpert", "naat", "核酸"),
            "cxr": ("胸片", "胸部x线", "x线"),
            "smear": ("涂片",),
        }
        lowered_text = text.casefold()
        if requested_modality and any(
            term in lowered_text for term in modality_terms[requested_modality]
        ):
            priority += 1000
        if requested_modality == "smear" and "痰抗酸杆菌涂片阴性不能排除肺结核" in text:
            priority += 600
        if ("培养阴性" in text or "xpert阴性" in lowered_text) and "不能" in text:
            priority += 500
        elif "单次naat阴性不能用于排除肺结核" in lowered_text:
            priority += 450
        elif "痰抗酸杆菌涂片阴性不能排除肺结核" in text:
            priority += 400
        elif "胸片正常" in text and "不能" in text:
            priority += 500
    elif subtopic == "test_comparison":
        lowered_text = text.casefold()
        modality_terms = {
            GuidelineScenarioTag.TEST_NAAT: ("xpert", "naat", "核酸"),
            GuidelineScenarioTag.TEST_CULTURE: ("培养",),
            GuidelineScenarioTag.TEST_SMEAR: ("涂片",),
        }
        for tag, terms in modality_terms.items():
            if tag in scenario_tags and any(term in lowered_text for term in terms):
                priority += 300
        if any(term in lowered_text for term in ("作用", "用于", "检出", "金标准")):
            priority += 100
    elif subtopic == "diagnostic_pathway":
        lowered_text = text.casefold()
        image_abnormal_question = GuidelineScenarioTag.AFTER_ABNORMAL_CXR in scenario_tags
        if (
            image_abnormal_question
            and "naat" in lowered_text
            and any(term in lowered_text for term in ("初始诊断", "初始检测"))
        ):
            priority += 800
        elif "综合分析" in text:
            priority += 600
        elif "作为初始诊断检测" in text:
            priority += 500
    elif subtopic in {"tb_infection_test_interpretation", "infection_vs_disease"}:
        lowered_text = text.casefold()
        if any(term in lowered_text for term in ("tst", "igra", "感染检测")):
            priority += 400
        if "活动性" in lowered_text and any(
            term in lowered_text for term in ("不能", "不等于", "不同")
        ):
            priority += 500
        if "children" in population and "儿童" in lowered_text:
            priority += 1_000
        if "pregnant_people" in population and any(
            term in lowered_text for term in ("孕妇", "妊娠", "怀孕")
        ):
            priority += 1_000
    elif subtopic == "special_population_testing":
        if "15岁以下" in text and "可疑症状" in text and "痰标本NAAT" in text:
            priority = 500
        elif "儿童" in text and "呼吸道和粪便标本" in text and "初始诊断" in text:
            priority = 400
        elif "15岁以下" in text and "无痰" in text and "粪便NAAT" in text:
            priority = 300
        if (
            GuidelineScenarioTag.NO_SPUTUM in scenario_tags
            and "粪便naat" in text.casefold()
            and any(term in text for term in ("无痰", "难以获得"))
        ):
            priority += 1000
        if "pregnant_people" in population and any(
            term in text for term in ("孕妇", "妊娠", "怀孕")
        ):
            priority += 1000
            if "医学评估" in text and "胸部X线" in text and "痰" in text:
                priority += 600
        if "people_living_with_hiv" in population:
            if "hiv感染的成人" in text.casefold():
                priority += 1_500
            elif "hiv" in text.casefold():
                priority += 300
    elif subtopic == "care_setting":
        if "适用于多数结核病患者" in text and "安全监测" in text:
            priority = 500
        elif "可能需要住院" in text and "情况稳定" in text:
            priority = 400
        elif "缩短住院时间" in text and "尽早衔接门诊" in text:
            priority = 300
        if (
            GuidelineScenarioTag.CARE_UNIVERSAL_HOSPITALIZATION in scenario_tags
            and "适用于多数结核病患者" in text
        ):
            priority += 1_500
        if (
            GuidelineScenarioTag.CARE_INPATIENT_INDICATIONS in scenario_tags
            and "可能需要住院" in text
        ):
            priority += 1_200
        if (
            GuidelineScenarioTag.CARE_AMBULATORY_TRANSITION in scenario_tags
            and any(term in text for term in ("尽早衔接门诊", "缩短住院时间"))
        ):
            priority += 1_200
    elif subtopic == "treatment_adherence":
        if "漏服" in text and any(term in text for term in ("不要加倍", "下一次", "联系")):
            priority = 500
    elif subtopic == "adverse_effects":
        if GuidelineScenarioTag.ADVERSE_VISUAL in scenario_tags and any(
            term in text for term in ("视力", "视觉", "模糊")
        ):
            priority = 700
        elif any(term in text for term in ("不良反应", "副作用")):
            priority = 400
        if GuidelineScenarioTag.ADVERSE_GENERAL_LIST in scenario_tags and all(
            term in text for term in ("胃肠道", "肝损伤", "皮疹")
        ):
            priority += 800
    elif subtopic == "contact_evaluation":
        if any(term in text for term in ("接触者", "密切接触", "同住")):
            priority = 500
    elif subtopic == "infectiousness_clearance":
        if any(term in text for term in ("传染性", "不再传播", "痰")):
            priority = 500
    elif subtopic == "return_to_work_school":
        if any(term in text for term in ("返工", "返校", "工作", "上学", "上班")):
            priority = 500
    return priority


_SPECIAL_POPULATION_EVIDENCE_TERMS: dict[str, tuple[str, ...]] = {
    "pregnant_people": ("孕妇", "妊娠", "怀孕"),
    "children": ("儿童", "15岁以下", "婴幼儿", "未成年人"),
    "people_living_with_hiv": ("HIV", "艾滋"),
    "immunosuppressed_people": ("免疫抑制", "免疫缺陷"),
}

# Population-specific query expansion is deliberately limited to populations
# with reviewed, directly conditional clauses in the current snapshot.  It is
# a ranking hint only and never widens the claim-scope filter.
_SPECIAL_POPULATION_QUERY_EXPANSION: dict[str, str] = {
    "pregnant_people": ("孕妇 妊娠 怀孕 肺结核可疑症状 胸部X线不适合 优先NAAT 病原学 检查"),
    "children": (
        "15岁以下 儿童 肺结核可疑症状 筛查阳性 初始诊断 病原学 痰标本NAAT 呼吸道 粪便 无痰"
    ),
}


def _hit_matches_special_population(hit: Any, population: list[str]) -> bool:
    """Require direct textual coverage for requested special populations."""

    requested_terms = {
        term.casefold()
        for canonical in population
        for term in _SPECIAL_POPULATION_EVIDENCE_TERMS.get(canonical, ())
    }
    if not requested_terms:
        return False
    # Only the extractive claim text is user-visible as a claim. Metadata or a
    # reviewer note mentioning the population cannot make a generic claim
    # population-specific.
    evidence_material = str(getattr(hit, "text", "")).casefold()
    return any(term in evidence_material for term in requested_terms)


def _hit_answers_special_population_testing(hit: Any, population: list[str]) -> bool:
    """Accept direct testing clauses and narrow, population-conditional pathways.

    A directly reviewed ``special_population_testing`` clause may answer this
    question. For a symptomatic child, the reviewed active-screening pathway
    may also answer the exact next-test condition it states. Pregnancy clauses
    from screening are not promoted into a general diagnostic rule.
    """

    allowed_scopes = set(getattr(hit, "allowed_claim_scope", ()))
    text = str(getattr(hit, "text", "")).casefold()
    contains_test = any(
        term in text
        for term in (
            "检查",
            "检测",
            "naat",
            "xpert",
            "胸部x线",
            "胸片",
            "病原学",
        )
    )
    if "special_population_testing" in allowed_scopes:
        return contains_test
    screening_pathway = bool({"active_screening", "screening_pathway"}.intersection(allowed_scopes))
    if "children" in population and screening_pathway:
        contains_child = "儿童" in text or "15岁以下" in text
        contains_explicit_condition = any(
            term in text
            for term in (
                "可疑症状",
                "筛查阳性",
                "初筛任一项目阳性",
                "无痰",
                "难以获得合格痰标本",
            )
        )
        return contains_test and contains_child and contains_explicit_condition
    return False


def _select_guideline_hits(
    hits: list[Any],
    *,
    subtopic: str,
    population: list[str],
    scenario_tags: set[GuidelineScenarioTag],
) -> list[Any]:
    candidates = list(hits)

    def chunk_id(hit: Any) -> str:
        return str(getattr(getattr(hit, "citation", None), "chunk_id", ""))

    if subtopic == "risk_groups":
        # A named-person question must not be answered by a different official
        # population list merely because it shares the broad risk-group topic.
        if "people_with_diabetes" in population:
            direct = [hit for hit in candidates if "糖尿病" in str(hit.text)]
            candidates = direct
        elif {
            GuidelineScenarioTag.RISK_HIGH_RISK_GROUPS,
            GuidelineScenarioTag.RISK_KEY_GROUPS,
        }.issubset(scenario_tags):
            pass
        elif GuidelineScenarioTag.RISK_HIGH_RISK_GROUPS in scenario_tags:
            direct = [
                hit
                for hit in candidates
                if str(hit.text).startswith("主动筛查高风险人群包括")
            ]
            candidates = direct
        elif "older_adults" in population:
            direct = [
                hit
                for hit in candidates
                if any(term in str(hit.text) for term in ("65岁", "老年"))
            ]
            candidates = direct
    if subtopic in {"special_population_guidance", "special_population_testing"} and population:
        # Do not answer a pregnancy/child/HIV question with a generic adult or
        # unrelated high-risk-group clause. If no directly matching reviewed
        # chunk exists, the caller emits an explicit evidence gap.
        candidates = [
            hit
            for hit in candidates
            if _hit_matches_special_population(hit, population)
            and (
                subtopic != "special_population_testing"
                or _hit_answers_special_population_testing(hit, population)
            )
        ]
    if subtopic == "negative_test_interpretation":
        if GuidelineScenarioTag.TEST_CULTURE in scenario_tags:
            direct = [hit for hit in candidates if "培养阴性" in str(hit.text)]
        elif GuidelineScenarioTag.TEST_NAAT in scenario_tags:
            direct = [
                hit
                for hit in candidates
                if "阴性" in str(hit.text)
                and any(term in str(hit.text).casefold() for term in ("xpert", "naat", "核酸"))
            ]
        elif GuidelineScenarioTag.TEST_SMEAR in scenario_tags:
            direct = [
                hit
                for hit in candidates
                if "涂片阴性" in str(hit.text) and "不能排除" in str(hit.text)
            ]
        elif GuidelineScenarioTag.TEST_CXR in scenario_tags:
            direct = [
                hit
                for hit in candidates
                if any(term in str(hit.text).casefold() for term in ("胸片正常", "胸部x线"))
                and any(term in str(hit.text) for term in ("排除", "不能"))
            ]
        else:
            direct = []
        # A named test is an applicability boundary, not a ranking hint. If
        # its direct clause is absent, the caller must emit an evidence gap
        # instead of borrowing another test's limitation.
        if scenario_tags.intersection(
            {
                GuidelineScenarioTag.TEST_CULTURE,
                GuidelineScenarioTag.TEST_NAAT,
                GuidelineScenarioTag.TEST_CXR,
                GuidelineScenarioTag.TEST_SMEAR,
            }
        ):
            candidates = direct
    if subtopic == "test_comparison":
        requested_modalities = {
            modality
            for tag, modality in (
                (GuidelineScenarioTag.TEST_NAAT, "xpert"),
                (GuidelineScenarioTag.TEST_CULTURE, "culture"),
                (GuidelineScenarioTag.TEST_SMEAR, "smear"),
            )
            if tag in scenario_tags
        }
        modality_by_chunk = {
            "cdc25_xpert_role": "xpert",
            "cdc25_culture_role": "culture",
            "cdc25_smear_role": "smear",
        }
        direct = [
            hit
            for hit in candidates
            if modality_by_chunk.get(chunk_id(hit)) in requested_modalities
        ]
        if requested_modalities:
            candidates = direct
    if subtopic == "diagnostic_pathway" and not population:
        asks_after_abnormal_cxr = GuidelineScenarioTag.AFTER_ABNORMAL_CXR in scenario_tags
        population_neutral = [
            hit
            for hit in candidates
            if not {"hiv", "pediatrics"}.intersection(set(hit.topics))
            and not any(
                marker in str(hit.text).casefold()
                for marker in ("hiv感染的成人", "hiv阴性或hiv状态未知且")
            )
        ]
        if GuidelineScenarioTag.DRUG_RESISTANCE not in scenario_tags:
            population_neutral = [
                hit
                for hit in population_neutral
                if "drug_resistance" not in set(hit.topics)
            ]
        if asks_after_abnormal_cxr:
            direct = [
                hit
                for hit in population_neutral
                if chunk_id(hit) == "who25_initial_lc_anaat"
            ]
            population_neutral = direct
        if population_neutral:
            candidates = population_neutral
    if subtopic == "special_population_testing":
        if "pregnant_people" in population:
            direct = [
                hit
                for hit in candidates
                if chunk_id(hit) == "cdc25_pregnancy_tb_evaluation"
            ]
            candidates = direct
        elif "people_living_with_hiv" in population:
            direct = [
                hit
                for hit in candidates
                if "hiv感染的成人" in str(hit.text).casefold()
            ]
            candidates = direct
        elif (
            "children" in population
            and GuidelineScenarioTag.NO_SPUTUM in scenario_tags
        ):
            direct = [
                hit
                for hit in candidates
                if chunk_id(hit) == "as26_symptomatic_under15_no_sputum"
            ]
            candidates = direct
    if subtopic == "tb_infection_test_interpretation":
        allowed_chunks = {"cdc25_infection_test_positive"}
        if "children" in population:
            allowed_chunks.add("cdc25_child_tb_infection_test_selection")
        direct = [hit for hit in candidates if chunk_id(hit) in allowed_chunks]
        candidates = direct
    if subtopic == "infection_vs_disease":
        direct = [
            hit
            for hit in candidates
            if chunk_id(hit)
            in {"cdc25_infection_test_positive", "cdc24_only_active_spreads"}
        ]
        candidates = direct
    if subtopic == "shared_utensil_transmission":
        # This question needs an explicit object-level clause.  A generic
        # airborne-transmission hit is related but does not answer whether
        # sharing utensils is itself a route of spread.
        candidates = [
            hit
            for hit in candidates
            if chunk_id(hit) == "cdc24_shared_utensils_not_transmission"
        ]
    elif subtopic == "infection_control":
        # A passage correcting an object-specific misconception must not become
        # the lead answer to a general question about transmission.
        candidates = [hit for hit in candidates
                      if chunk_id(hit) != "cdc24_shared_utensils_not_transmission"]
    if subtopic == "care_setting":
        asks_transition = GuidelineScenarioTag.CARE_AMBULATORY_TRANSITION in scenario_tags
        asks_universal = (
            GuidelineScenarioTag.CARE_UNIVERSAL_HOSPITALIZATION in scenario_tags
        )
        asks_indications = GuidelineScenarioTag.CARE_INPATIENT_INDICATIONS in scenario_tags
        if asks_transition and (asks_universal or asks_indications):
            allowed_chunks = {
                "who25_care_setting_ambulatory_majority",
                "who25_care_setting_inpatient_indications",
                "who25_care_setting_early_ambulatory_transition",
            }
        elif asks_transition:
            allowed_chunks = {"who25_care_setting_early_ambulatory_transition"}
        elif asks_indications:
            allowed_chunks = {"who25_care_setting_inpatient_indications"}
        else:
            allowed_chunks = {
                "who25_care_setting_ambulatory_majority",
                "who25_care_setting_inpatient_indications",
            }
        direct = [hit for hit in candidates if chunk_id(hit) in allowed_chunks]
        candidates = direct
    if subtopic == "infectiousness_clearance":
        direct = [
            hit
            for hit in candidates
            if chunk_id(hit) == "cdc25_infectiousness_followup"
        ]
        candidates = direct
    if subtopic == "return_to_work_school":
        direct = [
            hit
            for hit in candidates
            if chunk_id(hit) == "cdc25_return_to_activities"
        ]
        candidates = direct
    if subtopic == "adverse_effects":
        if GuidelineScenarioTag.ADVERSE_VISUAL in scenario_tags:
            direct = [
                hit
                for hit in candidates
                if chunk_id(hit) == "cdc25_blurred_vision_serious"
            ]
        else:
            direct = [
                hit
                for hit in candidates
                if chunk_id(hit) == "cdc25_common_tb_drug_adverse_events"
            ]
        if scenario_tags.intersection(
            {
                GuidelineScenarioTag.ADVERSE_VISUAL,
                GuidelineScenarioTag.ADVERSE_GENERAL_LIST,
            }
        ) or direct:
            candidates = direct
    if subtopic == "rapid_molecular_diagnostics" and not population:
        # A generic product question should not be expanded into unrelated HIV,
        # paediatric, isoniazid or fluoroquinolone recommendations merely because
        # they share the broad `rapid_diagnostics` topic.
        direct_general = [
            hit
            for hit in candidates
            if _guideline_hit_priority(
                subtopic,
                hit,
                population=population,
                scenario_tags=scenario_tags,
            )
            > 0
            and not {"hiv", "pediatrics"}.intersection(set(hit.topics))
        ]
        if GuidelineScenarioTag.DRUG_RESISTANCE not in scenario_tags:
            direct_general = [
                hit
                for hit in direct_general
                if "利福平耐药的初始检测" not in str(hit.text)
            ]
        candidates = direct_general
    candidates.sort(
        key=lambda hit: _guideline_hit_priority(
            subtopic,
            hit,
            population=population,
            scenario_tags=scenario_tags,
        ),
        reverse=True,
    )
    return candidates[: _GUIDELINE_MAX_CHAT_CLAIMS.get(subtopic, 4)]


def _screening_disposition(decision: FusionDecision | None) -> str:
    if decision is None:
        return "insufficient_evidence"
    return {
        VisualResult.MODEL_FLAGGED: "screen_positive",
        VisualResult.MODEL_NOT_FLAGGED: "screen_negative",
        VisualResult.NON_TB_ABNORMAL: "non_tb_abnormal",
        VisualResult.PENDING_HUMAN_REVIEW: "review_required",
        VisualResult.INDETERMINATE: "insufficient_evidence",
        VisualResult.TECHNICAL_FAILURE: "technical_failure",
    }[decision.visual_result]


_NARRATABLE_RESPONSE_KINDS = frozenset(
    {
        ResponseKind.VISUAL_SCREENING_RESULT,
        ResponseKind.LOCALIZATION_RESULT,
        ResponseKind.CASE_EXPLANATION,
        ResponseKind.DIAGNOSTIC_INFORMATION,
        ResponseKind.NEXT_TEST_INFORMATION,
        ResponseKind.TREATMENT_EDUCATION,
        ResponseKind.SAFE_ABSTENTION,
    }
)
_NARRATION_MUTABLE_FIELDS = frozenset(
    {
        "summary",
        "claims",
        # Grounded narrator validation derives these three presentation lists
        # only by filtering the already-authoritative selected claims.  They are
        # mutable so the API cannot re-expand a deliberately concise answer.
        "diagnostic_information",
        "next_step_information",
        "treatment_education",
        "narrator_backend",
        "narrator_model",
        "narrator_model_digest",
        "narrator_policy_id",
        "narration_status",
        "narrator_generation_invoked",
        "narrator_prompt_tokens",
        "narrator_completion_tokens",
    }
)


def _ids() -> tuple[str, str]:
    return str(uuid.uuid4()), str(uuid.uuid4())


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _narrator_metadata(narrator: Any, status: NarrationStatus) -> dict[str, Any]:
    return {
        "narrator_backend": narrator.backend_id,
        "narrator_model": narrator.model,
        "narrator_model_digest": getattr(narrator, "model_digest", None),
        "narrator_policy_id": getattr(narrator, "policy_id", NARRATOR_POLICY_ID),
        "narration_status": status,
        "narrator_generation_invoked": False,
        "narrator_prompt_tokens": None,
        "narrator_completion_tokens": None,
    }


def _narration_invariants_hold(original: AgentResponse, candidate: AgentResponse) -> bool:
    original_fields = original.model_dump(exclude=_NARRATION_MUTABLE_FIELDS)
    candidate_fields = candidate.model_dump(exclude=_NARRATION_MUTABLE_FIELDS)
    if original_fields != candidate_fields:
        return False

    information_fields = (
        "diagnostic_information",
        "next_step_information",
        "treatment_education",
    )
    if original.answer_status is None:
        return all(
            getattr(candidate, field) == getattr(original, field) for field in information_fields
        )

    original_claims = {(claim.text, tuple(claim.chunk_ids)) for claim in original.claims}
    if any(
        (claim.text, tuple(claim.chunk_ids)) not in original_claims for claim in candidate.claims
    ):
        return False
    selected_claim_texts = {claim.text.strip() for claim in candidate.claims}
    return all(
        getattr(candidate, field)
        == [item for item in getattr(original, field) if item.strip() in selected_claim_texts]
        for field in information_fields
    )


class TBXAgentService:
    def __init__(
        self,
        settings: Settings,
        *,
        store: SQLiteStore | None = None,
        vision_backend: VisionBackend | None = None,
        anatomy_backend: AnatomyBackend | None = None,
        refinement_backend: ContourRefinementBackend | None = None,
        tool_timeout_seconds: float = 8.0,
        max_tool_steps: int | None = None,
    ):
        self.settings = settings
        self._close_lock = RLock()
        self._closed = False
        self._owns_store = store is None
        self._tool_timeout_seconds = tool_timeout_seconds
        self._state_locks = _ScopedLockPool()
        self._general_chat_memory_lock = RLock()
        self._general_chat_memory: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        settings.ensure_runtime_dirs()
        self.store = store or SQLiteStore(settings.db_path)
        interrupted_anatomy_runs = self.store.reconcile_interrupted_anatomy_runs()
        for interrupted in interrupted_anatomy_runs:
            self._audit(
                request_id=str(uuid.uuid4()),
                actor_id="tbx-agent-startup",
                actor_role=ActorRole.ADMIN,
                action="anatomy_worker_interrupted",
                owner_scope=interrupted.owner_scope,
                case_id=interrupted.case_id,
                details={
                    "run_id": interrupted.run_id,
                    "error_code": interrupted.error_code,
                    "record_version": interrupted.record_version,
                },
            )
        self.policy = settings.fusion_policy()
        self.runtime_config = settings.rank03_config()
        self.safety = SafetyVerifier(settings.safety_policy())
        self.retriever = GuidelineRetriever(
            settings.knowledge_dir,
            retrieval_config_path=settings.retrieval_config_path,
        )
        self.screening = ScreeningEngine(settings.knowledge_dir / "active_screening_questions.json")
        if vision_backend is not None:
            self.vision = vision_backend
        elif settings.vision_backend == "rank03":
            self.vision = Rank03Backend(self.policy, self.runtime_config)
        elif settings.vision_backend == "mock":
            self.vision = MockRank03Backend(self.policy, self.runtime_config)
        else:
            raise ValueError(f"unsupported vision backend: {settings.vision_backend}")
        if anatomy_backend is not None:
            self.anatomy = anatomy_backend
        elif settings.anatomy_backend == "none":
            self.anatomy = None
        elif settings.anatomy_backend == "xrv_pspnet":
            model_manifest = load_manifest(settings.config_dir / "model_sources.yaml")
            pspnet_spec = model_manifest.artifacts["xrv_chestxdet_pspnet"]
            artifact_manager = ArtifactManager(
                model_manifest,
                cache_dir=default_artifact_root(),
            )
            pspnet_path = artifact_manager.path_for(pspnet_spec)
            self.anatomy = TorchXRayVisionPSPNetBackend(
                config=XRVPSPNetConfig(
                    cache_dir=pspnet_path.parent,
                    weight_filename=pspnet_path.name,
                    expected_weight_file_sha256=pspnet_spec.sha256,
                )
            )
        else:
            raise ValueError(f"unsupported anatomy backend: {settings.anatomy_backend}")
        if refinement_backend is not None:
            self.refinement = refinement_backend
        elif settings.contour_refinement_backend == "none":
            self.refinement = None
        elif settings.contour_refinement_backend == "medsam_hf":
            model_manifest = load_manifest(settings.config_dir / "model_sources.yaml")
            artifact_manager = ArtifactManager(
                model_manifest,
                cache_dir=default_artifact_root(),
            )
            weight_spec = model_manifest.artifacts["medsam_vit_base_weights"]
            config_spec = model_manifest.artifacts["medsam_vit_base_config"]
            processor_spec = model_manifest.artifacts["medsam_vit_base_preprocessor"]
            medsam_paths = [
                artifact_manager.path_for(spec)
                for spec in (weight_spec, config_spec, processor_spec)
            ]
            if len({path.parent for path in medsam_paths}) != 1:
                raise ValueError("MedSAM artifacts must share one immutable model directory")
            self.refinement = HFMedSAMBoxRefinementBackend(
                config=HFMedSAMConfig(
                    artifact_dir=medsam_paths[0].parent,
                    expected_weight_sha256=weight_spec.sha256,
                    expected_config_sha256=config_spec.sha256,
                    expected_preprocessor_sha256=processor_spec.sha256,
                    model_revision=weight_spec.revision,
                    device=settings.contour_refinement_device,
                )
            )
        else:
            raise ValueError(
                f"unsupported contour refinement backend: {settings.contour_refinement_backend}"
            )
        if self.refinement is not None and self.anatomy is None:
            raise ValueError("contour refinement requires a configured anatomy backend")
        if settings.anatomy_max_workers < 1 or settings.anatomy_max_workers > 4:
            raise ValueError("anatomy_max_workers must be in [1, 4]")
        self._anatomy_executor = (
            ThreadPoolExecutor(
                max_workers=settings.anatomy_max_workers,
                thread_name_prefix="tbx-anatomy",
            )
            if self.anatomy is not None
            else None
        )
        if settings.narrator_backend == "none":
            self.narrator = None
        elif settings.narrator_backend == "openai":
            self.narrator = OptionalOpenAINarrator(settings.openai_model)
        elif settings.narrator_backend == "ollama":
            self.narrator = OllamaNarrator(
                model=settings.ollama_model,
                expected_digest=settings.ollama_model_digest,
                base_url=settings.ollama_base_url,
                timeout_seconds=settings.ollama_timeout_seconds,
                max_response_bytes=settings.ollama_max_response_bytes,
                allow_remote=settings.ollama_allow_remote,
            )
        elif settings.narrator_backend == "llama_cpp":
            self.narrator = LlamaCppNarrator(
                model_alias=settings.llama_cpp_model_alias,
                model_path=settings.llama_cpp_model_path,
                expected_model_sha256=settings.llama_cpp_model_sha256,
                expected_server_build=settings.llama_cpp_server_build,
                base_url=settings.llama_cpp_base_url,
                timeout_seconds=settings.llama_cpp_timeout_seconds,
                max_response_bytes=settings.llama_cpp_max_response_bytes,
                allow_remote=settings.llama_cpp_allow_remote,
                api_key=settings.resolved_llama_cpp_api_key(),
            )
        else:
            raise ValueError(f"unsupported narrator backend: {settings.narrator_backend}")
        self.tool_registry = build_builtin_registry(
            {
                ToolName.EMERGENCY_TRIAGE: self._tool_emergency_triage,
                ToolName.GET_EXACT_CASE_AND_EXPLAIN: self._tool_exact_case,
                ToolName.CLASSIFY_CURRENT_CXR: self._tool_classify_current_cxr,
                ToolName.LOCALIZE_CURRENT_CXR: self._tool_localize_current_cxr,
                ToolName.INSPECT_ANATOMICAL_CONTEXT: self._tool_inspect_anatomical_context,
                ToolName.INSPECT_IMAGE_QUALITY: self._tool_inspect_image_quality,
                ToolName.COMPARE_WITH_PRIOR_CXR: self._tool_compare_with_prior_cxr,
                ToolName.SEARCH_TB_GUIDANCE: self._tool_search_tb_guidance,
                ToolName.DESCRIBE_AGENT_CAPABILITIES: self._tool_capabilities,
            },
            timeout_seconds=tool_timeout_seconds,
            max_steps=(settings.max_tool_calls if max_tool_steps is None else max_tool_steps),
            health_checks={
                ToolName.INSPECT_ANATOMICAL_CONTEXT: self._anatomy_tool_ready,
            },
        )

    def close(self) -> None:
        """Drain accepted work before releasing its dependencies.

        The host must stop accepting requests first. Python model threads cannot
        be killed safely, so even timed-out tools finish before SQLite closes.
        An injected store remains the caller's responsibility.
        """
        with self._close_lock:
            if self._closed:
                return
            self.tool_registry.close(wait=True)
            if self._anatomy_executor is not None:
                # Running tools may enqueue anatomy work while being drained.
                self._anatomy_executor.shutdown(wait=True, cancel_futures=False)
            try:
                close_retriever = getattr(self.retriever, "close", None)
                if callable(close_retriever):
                    close_retriever()
            finally:
                with self._general_chat_memory_lock:
                    self._general_chat_memory.clear()
                if self._owns_store:
                    self.store.close()
                self._closed = True

    def _anatomy_tool_ready(self) -> bool:
        """Fail closed unless the configured anatomy model and weights load."""

        if self.anatomy is None:
            return False
        try:
            probe = self.anatomy.probe_runtime(load=True)
        except Exception:  # noqa: BLE001 - tool availability must fail closed
            return False
        return bool(probe.loaded and probe.available == "yes")

    def _verify_real_vision_evidence(
        self,
        evidence: VisionEvidence,
        *,
        case_id: str,
        image_sha256: str,
        image_source_format: str,
        input_transform_id: str,
        image_quality_status: str,
        image_quality_codes: tuple[str, ...],
    ) -> None:
        """Reject synthetic or drifted evidence before fusion and persistence."""

        if not self.settings.require_real_inference:
            return
        classifier = self.runtime_config["classifier"]
        detector = self.runtime_config["detector"]
        bundle_id = str(self.runtime_config["model_bundle_id"])
        expected = {
            "case_id": case_id,
            "image_sha256": image_sha256,
            "image_source_format": image_source_format,
            "input_transform_id": input_transform_id,
            "classifier_model_id": f"{bundle_id}:convnext_tiny",
            "classifier_checkpoint_sha256": classifier["checkpoint_sha256"],
            "detector_model_id": f"{bundle_id}:dfine_l",
            "detector_checkpoint_sha256": detector["checkpoint_sha256"],
            "preprocessing_version": "rank03-frozen-official-submission-v1",
            "threshold_config_version": str(self.policy["policy_id"]),
        }
        observed = {key: getattr(evidence, key) for key in expected}
        quality_provenance_matches = (
            evidence.image_quality_status == image_quality_status
            and evidence.image_quality_codes == list(image_quality_codes)
        )
        synthetic_marker = evidence.run_id.casefold().startswith("mock-") or any(
            "mock" in value.casefold() or "synthetic" in value.casefold()
            for value in evidence.artifact_refs
        )
        backend_attested = getattr(
            self.vision, "runtime_contract", None
        ) == "rank03-frozen-runtime-v1" and not bool(getattr(self.vision, "synthetic", True))
        if (
            observed != expected
            or not quality_provenance_matches
            or synthetic_marker
            or not backend_attested
        ):
            raise VisionBackendError(
                "real-inference evidence failed the frozen rank03 attestation contract"
            )

    def _discard_retained_artifact(self, artifact_ref: str) -> None:
        if artifact_ref == "not_retained":
            return
        try:
            artifact_path = Path(artifact_ref).resolve(strict=False)
            artifact_root = self.settings.case_artifact_root.resolve(strict=False)
            if not artifact_path.is_relative_to(artifact_root):
                return
            artifact_path.unlink(missing_ok=True)
            # persist_original creates exactly artifact_root/<case_id>/<file>.
            # Remove only that now-empty case directory; never recurse.
            if artifact_path.parent.parent == artifact_root:
                artifact_path.parent.rmdir()
        except OSError:
            # Cleanup must never hide the original inference failure. Retention
            # reconciliation is handled by deployment-level artifact monitoring.
            pass

    def _require_case_access(
        self,
        *,
        case_id: str,
        owner_scope: str,
        user_id: str,
    ) -> CaseRecord:
        """Load a case only when both tenant scope and subject binding match.

        Legacy records without a subject binding fail closed. A migration must bind
        them before production use; tenant membership alone is not case access.
        """

        case = self.store.get_case(case_id, owner_scope)
        if case.user_id is None or case.user_id != user_id:
            raise AccessDeniedError("case subject binding does not match the caller")
        return case

    def _validated_case_image(
        self,
        case: CaseRecord,
        payload: bytes | None,
    ) -> ValidatedImage:
        """Load the exact case image without trusting a persisted arbitrary path."""

        if payload is None:
            if case.image_artifact_ref == "not_retained":
                raise ValueError(
                    "this deployment does not retain uploads; re-upload the same image"
                )
            artifact = Path(case.image_artifact_ref).resolve(strict=False)
            root = self.settings.case_artifact_root.resolve(strict=False)
            if not artifact.is_relative_to(root) or not artifact.is_file():
                raise ValueError("the retained case image is unavailable")
            payload = artifact.read_bytes()
        image = validate_image(payload, max_bytes=self.settings.max_upload_bytes)
        if (
            image.sha256 != case.image_sha256
            or image.width != case.image_width
            or image.height != case.image_height
        ):
            raise ValueError("the supplied image does not match the immutable case image")
        return image

    @staticmethod
    def _case_detections(case: CaseRecord) -> list[Any]:
        localization = case.localization_evidence
        if localization.status in {"completed", "completed_no_detection"}:
            return list(localization.detections)
        # Compatibility for records created by the former eager detector path.
        if case.vision_evidence is not None:
            return list(case.vision_evidence.detections)
        return []

    @staticmethod
    def _case_detector_identity(case: CaseRecord) -> tuple[str | None, str | None]:
        localization = case.localization_evidence
        if localization.status in {"completed", "completed_no_detection"}:
            return localization.run_id, localization.detector_checkpoint_sha256
        if case.vision_evidence is not None and case.vision_evidence.detections:
            return (
                case.vision_evidence.run_id,
                case.vision_evidence.detector_checkpoint_sha256,
            )
        return None, None

    def request_anatomy_run(
        self,
        *,
        case_id: str,
        owner_scope: str,
        user_id: str,
        payload: bytes | None = None,
    ) -> AnatomyRunRecord:
        """Queue routing-neutral lung-field evidence for an existing case."""

        request_id, _ = _ids()
        if self.anatomy is None or self._anatomy_executor is None:
            raise AnatomyNotConfiguredError("anatomy segmentation is not configured")
        case = self._require_case_access(
            case_id=case_id,
            owner_scope=owner_scope,
            user_id=user_id,
        )
        image = self._validated_case_image(case, payload)
        if image.width * image.height > 16_000_000:
            raise ValueError("anatomy output is limited to 16 million source pixels")
        try:
            anatomy_generation_key = self.anatomy.generation_key_for(image.sha256)
        except AnatomyBackendError:
            self._audit(
                request_id=request_id,
                actor_id=user_id,
                action="anatomy_request_rejected",
                owner_scope=owner_scope,
                case_id=case_id,
                details={"reason_code": "backend_unavailable"},
            )
            raise
        localization_policy = LungFieldLocalizationPolicy()
        boxes = [item.bbox_xyxy for item in self._case_detections(case)]
        detector_run_id, detector_checkpoint_sha256 = self._case_detector_identity(case)
        refinement_generation_key: str | None = None
        refinement_status = RefinementRunStatus.DISABLED
        refinement_error_code: str | None = None
        if self.refinement is not None:
            refinement_status = RefinementRunStatus.PENDING
            try:
                refinement_generation_key = self.refinement.generation_key_for(
                    image_sha256=image.sha256,
                    boxes=boxes,
                    anatomy_generation_key=anatomy_generation_key,
                )
            except RefinementBackendError:
                # Generation identity must not require model loading.  A custom
                # backend that violates that contract degrades only its branch;
                # the fallback key remains deterministic and retryable.
                refinement_generation_key = _canonical_sha256(
                    {
                        "schema": "tbx-refinement-key-failure-v1",
                        "backend_id": self.refinement.backend_id,
                        "image_sha256": image.sha256,
                        "anatomy_generation_key": anatomy_generation_key,
                        "boxes": boxes,
                    }
                )
                refinement_status = RefinementRunStatus.TECHNICAL_FAILURE
                refinement_error_code = "generation_identity_unavailable"
        generation_key = build_anatomy_pipeline_generation_key(
            anatomy_generation_key=anatomy_generation_key,
            detector_run_id=detector_run_id,
            detector_checkpoint_sha256=detector_checkpoint_sha256,
            detector_boxes=boxes,
            localization_policy_id=localization_policy.policy_id,
            localization_minimum_box_overlap_fraction=(
                localization_policy.minimum_box_overlap_fraction
            ),
            presentation_policy_id=SPATIAL_PRESENTATION_POLICY_ID,
            refinement_generation_key=refinement_generation_key,
        )
        existing = self.store.find_anatomy_generation(
            case_id=case_id,
            owner_scope=owner_scope,
            user_id=user_id,
            generation_key=generation_key,
        )
        if existing is not None:
            self._audit(
                request_id=request_id,
                actor_id=user_id,
                action="anatomy_reused",
                owner_scope=owner_scope,
                case_id=case_id,
                details={
                    "run_id": existing.run_id,
                    "status": existing.status.value,
                    "generation_key": generation_key,
                },
            )
            return existing.model_copy(update={"reused_existing_run": True})

        run = AnatomyRunRecord(
            run_id=str(uuid.uuid4()),
            case_id=case_id,
            owner_scope=owner_scope,
            user_id=user_id,
            image_sha256=image.sha256,
            generation_key=generation_key,
            anatomy_generation_key=anatomy_generation_key,
            backend_id=self.anatomy.backend_id,
            localization_policy_id=localization_policy.policy_id,
            localization_minimum_box_overlap_fraction=(
                localization_policy.minimum_box_overlap_fraction
            ),
            presentation_policy_id=SPATIAL_PRESENTATION_POLICY_ID,
            refinement_backend_id=(
                self.refinement.backend_id if self.refinement is not None else None
            ),
            refinement_generation_key=refinement_generation_key,
            refinement_status=refinement_status,
            refinement_error_code=refinement_error_code,
        )
        with self._state_locks.hold(_state_lock_key(case_id, generation_key)):
            # Recheck after acquiring the in-process generation lock.
            raced = self.store.find_anatomy_generation(
                case_id=case_id,
                owner_scope=owner_scope,
                user_id=user_id,
                generation_key=generation_key,
            )
            if raced is not None:
                return raced.model_copy(update={"reused_existing_run": True})
            self.store.create_anatomy_run(run)
            self._anatomy_executor.submit(self._execute_anatomy_run, run, image, request_id)
        self._audit(
            request_id=request_id,
            actor_id=user_id,
            action="anatomy_requested",
            owner_scope=owner_scope,
            case_id=case_id,
            details={
                "run_id": run.run_id,
                "backend_id": run.backend_id,
                "generation_key": generation_key,
            },
        )
        return run

    def _execute_anatomy_run(
        self,
        run: AnatomyRunRecord,
        image: ValidatedImage,
        request_id: str,
    ) -> None:
        """Worker entry point; failures never mutate the case or rank03 evidence."""

        assert self.anatomy is not None
        running = run.model_copy(update={"status": AnatomyRunStatus.RUNNING})
        try:
            running = self.store.update_anatomy_run(
                running,
                expected_version=run.record_version,
            )
            raw_evidence = self.anatomy.infer(case_id=run.case_id, image=image)
            if (
                raw_evidence.case_id != run.case_id
                or raw_evidence.image_sha256 != run.image_sha256
                or raw_evidence.generation_key != (run.anatomy_generation_key or run.generation_key)
                or raw_evidence.backend_id != run.backend_id
                or raw_evidence.image_width != image.width
                or raw_evidence.image_height != image.height
                or raw_evidence.routing_effect != "none"
                or raw_evidence.clinical_validation
            ):
                raise AnatomyBackendError("anatomy evidence failed its immutable identity contract")
            # The API owns run identity; backend-local UUIDs are never exposed as
            # a second competing job identifier.
            evidence = raw_evidence.model_copy(update={"run_id": run.run_id})
            case = self._require_case_access(
                case_id=run.case_id,
                owner_scope=run.owner_scope,
                user_id=run.user_id,
            )
            boxes = [item.bbox_xyxy for item in self._case_detections(case)]
            detector_run_id, detector_checkpoint_sha256 = self._case_detector_identity(case)
            observed_pipeline_key = build_anatomy_pipeline_generation_key(
                anatomy_generation_key=(running.anatomy_generation_key or running.generation_key),
                detector_run_id=detector_run_id,
                detector_checkpoint_sha256=detector_checkpoint_sha256,
                detector_boxes=boxes,
                localization_policy_id=running.localization_policy_id,
                localization_minimum_box_overlap_fraction=(
                    running.localization_minimum_box_overlap_fraction
                ),
                presentation_policy_id=running.presentation_policy_id,
                refinement_generation_key=running.refinement_generation_key,
            )
            if observed_pipeline_key != running.generation_key:
                raise AnatomyBackendError(
                    "anatomy pipeline inputs changed after the run was queued"
                )
            localization_policy = LungFieldLocalizationPolicy(
                policy_id=running.localization_policy_id,
                minimum_box_overlap_fraction=(running.localization_minimum_box_overlap_fraction),
            )
            locations = localize_detection_boxes(
                boxes,
                anatomy=evidence,
                policy=localization_policy,
            )
            spatial_summary = build_spatial_summary(
                locations,
                anatomy_qc_status=evidence.qc.status,
                policy_id=running.presentation_policy_id,
            )
            refinement_evidence = None
            refinement_status = running.refinement_status
            refinement_error_code = running.refinement_error_code
            if refinement_status == RefinementRunStatus.PENDING:
                if self.refinement is None:
                    refinement_status = RefinementRunStatus.TECHNICAL_FAILURE
                    refinement_error_code = "backend_not_configured"
                else:
                    try:
                        candidate = self.refinement.refine(
                            case_id=run.case_id,
                            image=image,
                            boxes=boxes,
                            anatomy=evidence,
                        )
                        if (
                            candidate.case_id != run.case_id
                            or candidate.image_sha256 != run.image_sha256
                            or candidate.image_width != image.width
                            or candidate.image_height != image.height
                            or candidate.backend_id != running.refinement_backend_id
                            or candidate.generation_key != running.refinement_generation_key
                            or candidate.anatomy_generation_key
                            != (running.anatomy_generation_key or running.generation_key)
                            or candidate.routing_effect != "none"
                            or candidate.clinical_validation
                            or len(candidate.items) != len(boxes)
                        ):
                            raise RefinementBackendError(
                                "contour refinement failed its immutable identity contract"
                            )
                    except RefinementBackendUnavailable:
                        refinement_status = RefinementRunStatus.TECHNICAL_FAILURE
                        refinement_error_code = "backend_unavailable"
                    except RefinementBackendError:
                        refinement_status = RefinementRunStatus.TECHNICAL_FAILURE
                        refinement_error_code = "inference_contract_failed"
                    except Exception:  # noqa: BLE001 - optional branch records stable code
                        refinement_status = RefinementRunStatus.TECHNICAL_FAILURE
                        refinement_error_code = "unexpected_runtime_failure"
                    else:
                        refinement_evidence = candidate
                        refinement_status = RefinementRunStatus.COMPLETED
                        refinement_error_code = None
            completed_status = (
                AnatomyRunStatus.COMPLETED_WITH_REFINEMENT_FAILURE
                if refinement_status == RefinementRunStatus.TECHNICAL_FAILURE
                else AnatomyRunStatus.COMPLETED
            )
            completed = running.model_copy(
                update={
                    "status": completed_status,
                    "evidence": evidence,
                    "detector_locations": locations,
                    "spatial_summary": spatial_summary,
                    "refinement_status": refinement_status,
                    "refinement_evidence": refinement_evidence,
                    "refinement_error_code": refinement_error_code,
                }
            )
            self.store.update_anatomy_run(
                completed,
                expected_version=running.record_version,
            )
        except AnatomyBackendUnavailable:
            self._fail_anatomy_run(running, "backend_unavailable")
        except AnatomyBackendError:
            self._fail_anatomy_run(running, "inference_contract_failed")
        except (KeyError, ValueError, AccessDeniedError, VersionConflictError):
            self._fail_anatomy_run(running, "state_or_identity_conflict")
        except Exception:  # noqa: BLE001 - worker boundary records only a stable code
            self._fail_anatomy_run(running, "unexpected_runtime_failure")
        else:
            self._audit(
                request_id=request_id,
                actor_id=run.user_id,
                action="anatomy_completed",
                owner_scope=run.owner_scope,
                case_id=run.case_id,
                details={
                    "run_id": run.run_id,
                    "generation_key": run.generation_key,
                    "runtime_ms": evidence.runtime_ms,
                    "qc_status": evidence.qc.status.value,
                    "structure_count": len(evidence.masks),
                    "refinement_status": refinement_status.value,
                    "refined_prompt_count": (
                        len(refinement_evidence.items) if refinement_evidence is not None else 0
                    ),
                },
            )

    def _fail_anatomy_run(self, running: AnatomyRunRecord, error_code: str) -> None:
        refinement_update = (
            {
                "refinement_status": RefinementRunStatus.TECHNICAL_FAILURE,
                "refinement_error_code": "anatomy_dependency_failed",
            }
            if running.refinement_status == RefinementRunStatus.PENDING
            else {}
        )
        failed = running.model_copy(
            update={
                "status": AnatomyRunStatus.TECHNICAL_FAILURE,
                "error_code": error_code,
                **refinement_update,
            }
        )
        try:
            self.store.update_anatomy_run(
                failed,
                expected_version=running.record_version,
            )
        except (KeyError, AccessDeniedError, VersionConflictError):
            return
        self._audit(
            request_id=str(uuid.uuid4()),
            actor_id=running.user_id,
            action="anatomy_failed",
            owner_scope=running.owner_scope,
            case_id=running.case_id,
            details={"run_id": running.run_id, "error_code": error_code},
        )

    def get_anatomy_run(
        self,
        *,
        run_id: str,
        case_id: str,
        owner_scope: str,
        user_id: str,
    ) -> AnatomyRunRecord:
        self._require_case_access(
            case_id=case_id,
            owner_scope=owner_scope,
            user_id=user_id,
        )
        run = self.store.get_anatomy_run(
            run_id,
            owner_scope,
            subject_user_id=user_id,
        )
        if run.case_id != case_id:
            raise AccessDeniedError("anatomy run is outside the requested case")
        return run

    @staticmethod
    def _bind_thread_case(thread: Any, case_id: str | None) -> None:
        if case_id is None:
            return
        if thread.current_case_id not in {None, case_id}:
            raise AccessDeniedError(
                "thread is already bound to a different case; start a new thread"
            )
        thread.current_case_id = case_id

    @staticmethod
    def _memory_event(
        *,
        role: str,
        content: str,
        request_id: str,
        kind: str,
    ) -> ThreadMemoryEvent:
        """Persist a turn reference without retaining free-text health content."""

        return ThreadMemoryEvent(
            role=role,
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            request_id=request_id,
            kind=kind,
        )

    @staticmethod
    def _general_chat_memory_key(
        *,
        owner_scope: str,
        user_id: str,
        thread_id: str,
    ) -> str:
        return hashlib.sha256(
            f"general-chat\0{owner_scope}\0{user_id}\0{thread_id}".encode()
        ).hexdigest()

    def _general_chat_history(
        self,
        *,
        owner_scope: str,
        user_id: str,
        thread_id: str,
    ) -> list[dict[str, str]]:
        """Return a bounded, process-local window; no raw text is persisted."""

        key = self._general_chat_memory_key(
            owner_scope=owner_scope,
            user_id=user_id,
            thread_id=thread_id,
        )
        with self._general_chat_memory_lock:
            messages = self._general_chat_memory.get(key, [])
            if key in self._general_chat_memory:
                self._general_chat_memory.move_to_end(key)
            return [dict(item) for item in messages]

    def _remember_general_chat(
        self,
        *,
        owner_scope: str,
        user_id: str,
        thread_id: str,
        query: str,
        answer: str,
    ) -> None:
        """Remember three Q/A pairs for this process and identity-bound thread."""

        key = self._general_chat_memory_key(
            owner_scope=owner_scope,
            user_id=user_id,
            thread_id=thread_id,
        )
        pair = (
            {"role": "user", "content": query.strip()[:2_000]},
            {"role": "assistant", "content": answer.strip()[:2_000]},
        )
        if not pair[0]["content"] or not pair[1]["content"]:
            return
        with self._general_chat_memory_lock:
            messages = [*self._general_chat_memory.get(key, []), *pair][-6:]
            self._general_chat_memory[key] = messages
            self._general_chat_memory.move_to_end(key)
            while len(self._general_chat_memory) > 256:
                self._general_chat_memory.popitem(last=False)

    def _audit(
        self,
        *,
        request_id: str,
        actor_id: str,
        action: str,
        owner_scope: str,
        case_id: str | None = None,
        details: dict[str, Any] | None = None,
        actor_role: ActorRole = ActorRole.USER,
    ) -> None:
        self.store.audit(
            request_id=request_id,
            actor_id=actor_id,
            actor_role=actor_role.value,
            action=action,
            owner_scope=owner_scope,
            case_id=case_id,
            details=details,
        )

    def _artifact_intact(self, case: CaseRecord) -> bool:
        if case.image_artifact_ref == "not_retained":
            return not self.settings.retain_uploaded_image
        path = Path(case.image_artifact_ref)
        if not path.is_file():
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == case.image_sha256

    def _can_reuse(self, case: CaseRecord, *, image: ValidatedImage) -> bool:
        """Return whether an upload can reuse the existing immutable case.

        Upload reuse deliberately does not depend on classification state.  A
        case with ``not_requested`` classification is a complete upload result,
        not a failed assessment.
        """

        if not self._artifact_intact(case):
            return False
        if image.sha256 != case.image_sha256:
            return False
        return not (
            case.image_source_format != image.source_format
            or case.input_transform_id != image.input_transform_id
            or case.image_quality_status != image.quality_status
            or case.image_quality_codes != list(image.quality_warnings)
        )

    def _classification_generation_key(self, case: CaseRecord) -> str:
        classifier = self.runtime_config["classifier"]
        return _canonical_sha256(
            {
                "schema": "tbx-classification-generation-v1",
                "case_id": case.case_id,
                "image_sha256": case.image_sha256,
                "vision_backend": self.settings.vision_backend,
                "backend_id": getattr(self.vision, "backend_id", "unknown"),
                "model_bundle_id": self.runtime_config["model_bundle_id"],
                "checkpoint_sha256": classifier["checkpoint_sha256"],
                "config_sha256": classifier["config_sha256"],
                "input_size": classifier["input_size"],
                "probability_order": classifier["probability_order"],
                "amp": classifier["amp"],
                "policy_id": self.policy["policy_id"],
                "input_transform_id": case.input_transform_id,
            }
        )

    def _classification_is_reusable(self, case: CaseRecord) -> bool:
        evidence = case.vision_evidence
        fusion = case.fusion_decision
        if (
            case.classification_status != ClassificationExecutionStatus.COMPLETED
            or evidence is None
            or fusion is None
            or case.classification_generation_key != self._classification_generation_key(case)
        ):
            return False
        if (
            evidence.case_id != case.case_id
            or evidence.image_sha256 != case.image_sha256
            or evidence.threshold_config_version != self.policy["policy_id"]
            or fusion.policy_id != self.policy["policy_id"]
        ):
            return False
        return fuse_rank03(evidence, self.policy) == fusion

    def _case_upload_response(
        self,
        case: CaseRecord,
        *,
        request_id: str,
        trace_id: str,
        reused: bool,
    ) -> AgentResponse:
        return self.safety.verify(
            AgentResponse(
                request_id=request_id,
                trace_id=trace_id,
                thread_id="assessment",
                case_id=case.case_id,
                response_kind=ResponseKind.CASE_EXPLANATION,
                summary="胸片已载入，尚未运行分类。",
                limitations=["本系统不用于确诊或排除肺结核。"],
                reused_existing_assessment=reused,
                safety_policy_id=self.safety.policy_id,
            )
        )

    def _case_response(
        self,
        case: CaseRecord,
        *,
        request_id: str,
        trace_id: str,
        reused: bool,
        include_guidance: bool = True,
    ) -> AgentResponse:
        if case.classification_status == ClassificationExecutionStatus.NOT_REQUESTED:
            return self._case_upload_response(
                case,
                request_id=request_id,
                trace_id=trace_id,
                reused=reused,
            )
        decision = case.fusion_decision
        result = decision.visual_result if decision else VisualResult.TECHNICAL_FAILURE
        predicted_class = decision.predicted_class if decision is not None else None
        decision_rule = (
            decision.classifier_decision_rule
            if decision is not None
            else str(self.policy.get("classifier_rule", "unknown"))
        )
        if result == VisualResult.MODEL_FLAGGED:
            if decision_rule == "p_tb_gte_threshold":
                summary = "本次模型触发筛查标记，建议进一步评估。"
            elif decision_rule == "native_three_class_argmax":
                summary = "模型识别为结核类，建议进一步检查。"
            else:
                summary = "本次模型触发筛查标记，建议进一步评估。"
        elif result == VisualResult.MODEL_NOT_FLAGGED:
            if decision_rule == "p_tb_gte_threshold":
                summary = "本次模型未触发筛查标记。"
            elif decision_rule == "native_three_class_argmax":
                summary = "模型更倾向于健康类。"
            else:
                summary = "本次模型未触发筛查标记。"
        elif result == VisualResult.NON_TB_ABNORMAL:
            summary = "模型识别为非结核异常。"
        elif result == VisualResult.PENDING_HUMAN_REVIEW:
            if decision is not None and "image_quality_warning" in decision.review_reasons:
                if (
                    decision_rule == "native_three_class_argmax"
                    and predicted_class is not None
                    and predicted_class.value == "sick_non_tb"
                ):
                    summary = "本次模型识别为非结核异常，但图像质量不足，建议重新上传。"
                elif (
                    decision_rule == "native_three_class_argmax"
                    and "classifier_exact_argmax_tie" in decision.review_reasons
                ):
                    summary = "分类器最高分并列且图像质量存在问题，建议重新上传。"
                else:
                    summary = "存在图像质量或输入域警告，模型原始分流仅作记录。"
            elif (
                decision_rule == "native_three_class_argmax"
                and predicted_class is not None
                and predicted_class.value == "sick_non_tb"
            ):
                summary = "本次模型识别为非结核异常。"
            elif (
                decision_rule == "native_three_class_argmax"
                and decision is not None
                and "classifier_exact_argmax_tie" in decision.review_reasons
            ):
                summary = "分类器最高分并列，本轮无法给出单一类别。"
            elif (
                decision is not None
                and "fusion_policy_evidence_mismatch" in decision.review_reasons
            ):
                summary = "模型证据与冻结策略不匹配，不能用于分流。"
            else:
                summary = "现有模型证据不足以形成直接分流。"
        elif result == VisualResult.TECHNICAL_FAILURE:
            summary = "本次未获得完整可靠模型证据，请检查图像或模型服务后重试。"
        else:
            summary = "现有证据不足，需人工复核或补充检查。"
        prefix = "【演示后端】" if self.settings.vision_backend == "mock" else ""
        visual_guidance_topics = {
            "diagnosis",
            "next_tests",
            "confirmation_boundary",
            "imaging_nonspecificity",
            "imaging_role_education",
        }
        visual_guidance_claim_scopes = {
            "initial_diagnostic_testing",
            "test_limitations",
            "china_diagnostic_principles",
            "confirmation_basis",
            "confirmation_boundary",
            "imaging_nonspecificity",
            "imaging_role_education",
        }
        visual_guidance_jurisdictions = {"China", "WHO"}
        hits = []
        if include_guidance:
            raw_hits = self.retriever.retrieve(
                "胸片或AI筛查异常后如何进一步检查，影像能否确诊肺结核",
                topics=visual_guidance_topics,
                jurisdictions=visual_guidance_jurisdictions,
                required_claim_scopes=visual_guidance_claim_scopes,
                top_k=3,
            )
            attestation = self.retriever.attest_hits(
                raw_hits,
                required_claim_scopes=visual_guidance_claim_scopes,
                jurisdictions=visual_guidance_jurisdictions,
            )
            hits = list(attestation.hits)
        next_step_information = (
            [
                "结合症状、暴露史和临床评估决定是否进行呼吸道标本快速分子检测、培养及药敏等检查。",
                "若有可疑影像、持续症状或高风险因素，尽快前往结核病定点医疗机构评估。",
            ]
            if include_guidance and hits
            else []
        )
        retrieval_limitations = (
            ["指南检索证据未通过完整性或范围校验，本次未生成进一步检查信息。"]
            if include_guidance and not hits
            else []
        )
        response = AgentResponse(
            request_id=request_id,
            trace_id=trace_id,
            thread_id="assessment",
            case_id=case.case_id,
            response_kind=ResponseKind.VISUAL_SCREENING_RESULT,
            summary=prefix + summary,
            visual_result=result,
            predicted_class=predicted_class,
            next_step_information=next_step_information,
            limitations=[
                "这是辅助筛查结果，不用于确诊或排除肺结核。",
                *retrieval_limitations,
            ],
            citations=[hit.citation for hit in hits],
            review_status=case.review_status,
            reused_existing_assessment=reused,
            safety_policy_id=self.safety.policy_id,
        )
        return self.safety.verify(response)

    def assess_cxr(
        self,
        payload: bytes,
        *,
        user_id: str,
        owner_scope: str,
        consent_to_process: bool,
        attested_chest_radiograph: bool,
    ) -> tuple[CaseRecord, AgentResponse]:
        request_id, trace_id = _ids()
        if not consent_to_process:
            raise ConsentRequiredError("需要明确同意处理上传图像。")
        if not attested_chest_radiograph:
            raise ConsentRequiredError("请确认上传内容是已去标识化的胸部X线图像。")
        image = validate_image(payload, max_bytes=self.settings.max_upload_bytes)
        lock_key = _state_lock_key(
            "assessment",
            owner_scope,
            user_id,
            image.sha256,
        )
        with self._state_locks.hold(lock_key):
            return self._assess_validated_cxr(
                payload,
                image=image,
                user_id=user_id,
                owner_scope=owner_scope,
                request_id=request_id,
                trace_id=trace_id,
            )

    @staticmethod
    def _validated_batch_identifier(value: str, *, field_name: str) -> str:
        normalized = value.strip()
        if not _BATCH_IDENTIFIER_PATTERN.fullmatch(normalized):
            raise ValueError(
                f"{field_name} must be 1-128 ASCII letters, digits, dot, underscore, colon or dash"
            )
        return normalized

    @staticmethod
    def _batch_review_reasons(case: CaseRecord) -> list[str]:
        """Return deterministic reasons, or an empty list for a healthy direct route."""

        decision = case.fusion_decision
        if (
            decision is not None
            and decision.visual_result == VisualResult.MODEL_NOT_FLAGGED
            and decision.predicted_class == ClassifierClass.HEALTHY
        ):
            return []

        reasons = list(decision.review_reasons if decision is not None else [])
        result = decision.visual_result if decision is not None else VisualResult.TECHNICAL_FAILURE
        reason_for_result = {
            VisualResult.MODEL_FLAGGED: "batch_model_flagged",
            VisualResult.NON_TB_ABNORMAL: "batch_non_tb_abnormal",
            VisualResult.PENDING_HUMAN_REVIEW: "batch_uncertain_result",
            VisualResult.INDETERMINATE: "batch_indeterminate_result",
            VisualResult.TECHNICAL_FAILURE: "batch_technical_failure",
            VisualResult.MODEL_NOT_FLAGGED: "batch_non_healthy_result",
        }[result]
        reasons.append(reason_for_result)
        return list(dict.fromkeys(reasons))

    def assess_cxr_for_batch(
        self,
        payload: bytes,
        *,
        batch_id: str,
        batch_item_id: str,
        user_id: str,
        owner_scope: str,
        consent_to_process: bool,
        attested_chest_radiograph: bool,
    ) -> tuple[CaseRecord, AgentResponse, ReviewRecord | None]:
        """Run/reuse inference, then independently enroll eligible batch items.

        A normal assessment never creates a central review task. This wrapper is
        the sole product path that may enroll one, and its deterministic review
        identity makes retries safe even when the inference case was created by
        an earlier interactive request.
        """

        batch_id = self._validated_batch_identifier(batch_id, field_name="batch_id")
        batch_item_id = self._validated_batch_identifier(batch_item_id, field_name="batch_item_id")
        case, response = self.assess_cxr(
            payload,
            user_id=user_id,
            owner_scope=owner_scope,
            consent_to_process=consent_to_process,
            attested_chest_radiograph=attested_chest_radiograph,
        )
        # Batch screening is the explicit eager-classification path.  It still
        # runs ConvNeXt only; localization and anatomy remain review-time tools.
        case, response = self.classify_cxr_case(
            case_id=case.case_id,
            owner_scope=owner_scope,
            user_id=user_id,
            payload=payload,
        )
        reasons = self._batch_review_reasons(case)
        if not reasons:
            self._audit(
                request_id=response.request_id,
                actor_id=user_id,
                action="batch_assessment_not_enrolled",
                owner_scope=owner_scope,
                case_id=case.case_id,
                details={"batch_id": batch_id, "batch_item_id": batch_item_id},
            )
            return case, response, None

        review_id = str(
            uuid.uuid5(
                _BATCH_REVIEW_NAMESPACE,
                f"{owner_scope}\0{batch_id}\0{case.case_id}",
            )
        )
        proposed = ReviewRecord(
            review_id=review_id,
            case_id=case.case_id,
            owner_scope=owner_scope,
            trigger_reasons=reasons,
            origin=ReviewOrigin.BATCH_SCREENING,
            batch_id=batch_id,
            batch_item_id=batch_item_id,
        )
        lock_key = _state_lock_key("batch-review", owner_scope, batch_id, case.case_id)
        with self._state_locks.hold(lock_key):
            case, review = self.store.ensure_batch_review(proposed)

        response.review_status = review.status
        response = self.safety.verify(response)
        self._audit(
            request_id=response.request_id,
            actor_id=user_id,
            action="batch_review_enrolled",
            owner_scope=owner_scope,
            case_id=case.case_id,
            details={
                "batch_id": batch_id,
                "batch_item_id": batch_item_id,
                "review_id": review.review_id,
                "review_status": review.status.value,
            },
        )
        return case, response, review

    def _assess_validated_cxr(
        self,
        payload: bytes,
        *,
        image: ValidatedImage,
        user_id: str,
        owner_scope: str,
        request_id: str,
        trace_id: str,
    ) -> tuple[CaseRecord, AgentResponse]:
        """Serialize one exact subject/image generation within this process."""

        existing = self.store.find_case_by_hash(owner_scope, user_id, image.sha256)
        if existing is not None and (existing.user_id is None or existing.user_id != user_id):
            self._audit(
                request_id=request_id,
                actor_id=user_id,
                action="assessment_subject_binding_rejected",
                owner_scope=owner_scope,
                case_id=existing.case_id,
                details={"reason_code": "case_subject_binding_mismatch"},
            )
            raise AccessDeniedError("existing assessment is bound to another subject")
        if existing is not None and self._can_reuse(existing, image=image):
            self._audit(
                request_id=request_id,
                actor_id=user_id,
                action="assessment_reused",
                owner_scope=owner_scope,
                case_id=existing.case_id,
                details={"policy_id": self.policy["policy_id"]},
            )
            return existing, self._case_upload_response(
                existing, request_id=request_id, trace_id=trace_id, reused=True
            )
        if existing is not None:
            self._audit(
                request_id=request_id,
                actor_id=user_id,
                action="assessment_rerun_requires_generation_migration",
                owner_scope=owner_scope,
                case_id=existing.case_id,
                details={
                    "policy_id": self.policy["policy_id"],
                    "reason_code": "existing_assessment_not_reusable",
                },
            )
            raise AssessmentStateConflictError(
                "existing assessment cannot be reused under the current runtime; "
                "an explicit assessment-generation migration is required"
            )

        case_id = str(uuid.uuid4())
        artifact_ref = (
            str(image.persist_original(payload, self.settings.case_artifact_root, case_id))
            if self.settings.retain_uploaded_image
            else "not_retained"
        )
        case = CaseRecord(
            case_id=case_id,
            owner_scope=owner_scope,
            user_id=user_id,
            image_artifact_ref=artifact_ref,
            image_sha256=image.sha256,
            image_width=image.width,
            image_height=image.height,
            image_source_format=image.source_format,
            input_transform_id=image.input_transform_id,
            image_quality_status=image.quality_status,
            image_quality_codes=list(image.quality_warnings),
            consent_scope="cxr_auxiliary_screening",
        )
        try:
            self.store.save_case(case)
        except VersionConflictError as exc:
            return self._resolve_concurrent_assessment(
                image=image,
                artifact_ref=artifact_ref,
                owner_scope=owner_scope,
                user_id=user_id,
                request_id=request_id,
                trace_id=trace_id,
                conflict=exc,
            )
        self._audit(
            request_id=request_id,
            actor_id=user_id,
            action="cxr_case_created",
            owner_scope=owner_scope,
            case_id=case_id,
            details={
                "classification_status": case.classification_status.value,
                "localization_status": case.localization_evidence.status,
                "quality_status": case.image_quality_status,
            },
        )
        return case, self._case_upload_response(
            case, request_id=request_id, trace_id=trace_id, reused=False
        )

    def _resolve_concurrent_assessment(
        self,
        *,
        image: ValidatedImage,
        artifact_ref: str,
        owner_scope: str,
        user_id: str,
        request_id: str,
        trace_id: str,
        conflict: VersionConflictError,
    ) -> tuple[CaseRecord, AgentResponse]:
        """Resolve a cross-process unique-key race without leaking SQLite errors."""

        self._discard_retained_artifact(artifact_ref)
        winner = self.store.find_case_by_hash(owner_scope, user_id, image.sha256)
        if winner is not None and self._can_reuse(winner, image=image):
            self._audit(
                request_id=request_id,
                actor_id=user_id,
                action="assessment_concurrent_reused",
                owner_scope=owner_scope,
                case_id=winner.case_id,
                details={"policy_id": self.policy["policy_id"]},
            )
            return winner, self._case_upload_response(
                winner,
                request_id=request_id,
                trace_id=trace_id,
                reused=True,
            )
        self._audit(
            request_id=request_id,
            actor_id=user_id,
            action="assessment_concurrent_conflict",
            owner_scope=owner_scope,
            details={"reason_code": "case_generation_conflict"},
        )
        raise AssessmentStateConflictError(
            "a concurrent assessment generation won the immutable image identity; "
            "retry after reading the existing case"
        ) from conflict

    def _tool_emergency_triage(self, invocation: ToolInvocation) -> AgentResponse:
        return AgentResponse(
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            thread_id=invocation.thread_id,
            case_id=invocation.case_id,
            response_kind=ResponseKind.SAFE_ABSTENTION,
            summary=(
                "你描述的信息可能涉及急症。立即联系当地急救服务（中国大陆可拨打120）"
                "或前往最近急诊；不要等待本系统继续筛查。"
            ),
            limitations=["本系统不能评估出血量、生命体征或替代急诊分诊。"],
            urgency=Urgency.EMERGENCY,
            safety_policy_id=self.safety.policy_id,
        )

    @staticmethod
    def _image_region_label(
        bbox_xyxy: tuple[float, float, float, float],
        *,
        image_width: int,
        image_height: int,
    ) -> str:
        x1, y1, x2, y2 = bbox_xyxy
        centre_x = (x1 + x2) / 2.0
        centre_y = (y1 + y2) / 2.0
        horizontal = "左侧" if centre_x < image_width / 2.0 else "右侧"
        if centre_y < image_height / 3.0:
            vertical = "上部"
        elif centre_y < image_height * 2.0 / 3.0:
            vertical = "中部"
        else:
            vertical = "下部"
        return f"图像{horizontal}{vertical}"

    @staticmethod
    def _anatomy_location_label(location: Any) -> str | None:
        if location is None or getattr(location, "status", None) != "localized":
            return None
        labels = []
        for assignment in getattr(location, "assignments", []):
            side = _LUNG_SIDE_LABELS.get(assignment.lung)
            zone = _LUNG_ZONE_LABELS.get(assignment.primary_zone)
            if side and zone:
                labels.append(f"{side}{zone}")
        return "、".join(dict.fromkeys(labels)) or None

    def _case_rationale_response(
        self,
        case: CaseRecord,
        response: AgentResponse,
    ) -> AgentResponse:
        evidence = case.vision_evidence
        decision = case.fusion_decision
        if evidence is None or decision is None:
            return response

        predicted = decision.predicted_class
        if predicted is None:
            summary = "胸片分类模型本轮没有形成单一类别。"
        else:
            label = _CLASS_EVIDENCE_LABELS[predicted]
            summary = f"因为胸片分类模型在三个训练类别中将{label}判为最高类别。"
        notes: list[str] = []
        if decision.visual_result == VisualResult.PENDING_HUMAN_REVIEW:
            notes.append(
                "当前最终状态为结果不确定，原因：" + "、".join(decision.review_reasons) + "。"
            )
        return response.model_copy(
            update={
                "summary": summary,
                "visual_evidence_notes": notes,
                "diagnostic_information": [],
                "next_step_information": [],
                "citations": [],
            }
        )

    def _case_localization_response(
        self,
        case: CaseRecord,
        response: AgentResponse,
        anatomy_run: AnatomyRunRecord | None,
    ) -> AgentResponse:
        localization = case.localization_evidence
        if localization.status not in {
            "completed",
            "completed_no_detection",
        }:
            return response
        localization_only = response.visual_result is None
        localization_fields = (
            {
                "response_kind": ResponseKind.LOCALIZATION_RESULT,
                "visual_result": None,
                "predicted_class": None,
                "review_status": None,
            }
            if localization_only
            else {}
        )
        selected = select_display_detections(
            [item.model_dump(mode="json") for item in localization.detections],
            policy=_DETECTION_DISPLAY_POLICY,
            image_width=case.image_width,
        )
        if not selected:
            summary = "定位检测器没有发现达到显示门槛的候选区域。"
            return response.model_copy(
                update={
                    **localization_fields,
                    "summary": summary,
                    "visual_evidence_notes": [],
                    "diagnostic_information": [],
                    "next_step_information": [],
                    "citations": [],
                }
            )

        notes: list[str] = []
        for display_index, candidate in enumerate(selected, start=1):
            anatomy_location = None
            if anatomy_run is not None and candidate.raw_index < len(
                anatomy_run.detector_locations
            ):
                anatomy_location = self._anatomy_location_label(
                    anatomy_run.detector_locations[candidate.raw_index]
                )
            location_label = anatomy_location or self._image_region_label(
                candidate.bbox_xyxy,
                image_width=case.image_width,
                image_height=case.image_height,
            )
            notes.append(f"候选区域 {display_index}：{location_label}。")
        return response.model_copy(
            update={
                **localization_fields,
                "summary": f"检测到 {len(selected)} 个候选区域，已标在右侧胸片上。",
                "visual_evidence_notes": notes,
                "diagnostic_information": [],
                "next_step_information": [],
                "citations": [],
            }
        )

    def _commit_classification_failure(
        self,
        case: CaseRecord,
        *,
        status: ClassificationExecutionStatus,
        generation_key: str,
        error_code: str,
    ) -> CaseRecord:
        candidate = case.model_copy(
            deep=True,
            update={
                "classification_status": status,
                "classification_generation_key": generation_key,
                "classification_attempt_count": max(0, case.classification_attempt_count) + 1,
                "classification_error_code": error_code,
                "vision_evidence": None,
                "fusion_decision": None,
            },
        )
        candidate.screening_disposition = "technical_failure"
        candidate.conflict_flags = list(
            dict.fromkeys([*case.conflict_flags, "classification_tool_failure"])
        )
        gap = (
            "classification_unavailable"
            if status == ClassificationExecutionStatus.UNAVAILABLE
            else "classification_failed"
        )
        candidate.evidence_gaps = list(dict.fromkeys([*case.evidence_gaps, gap]))
        # Interactive failures are returned in the chat.  Only the explicit
        # batch-screening wrapper may enroll a case in the review workbench.
        candidate.human_review_required = False
        candidate.human_review_reason_codes = [
            reason
            for reason in case.human_review_reason_codes
            if reason != "classification_tool_failed"
        ]
        committed, _ = self.store.commit_classification_state(
            candidate,
            expected_version=case.record_version,
        )
        return committed

    def _ensure_case_classification(
        self,
        *,
        case_id: str,
        owner_scope: str,
        user_id: str,
        request_id: str,
        payload: bytes | None = None,
        raise_on_failure: bool,
    ) -> tuple[CaseRecord, bool]:
        lock_key = _state_lock_key("classification", owner_scope, user_id, case_id)
        with self._state_locks.hold(lock_key):
            case = self._require_case_access(
                case_id=case_id,
                owner_scope=owner_scope,
                user_id=user_id,
            )
            if self._classification_is_reusable(case):
                self._audit(
                    request_id=request_id,
                    actor_id=user_id,
                    action="case_classification_reused",
                    owner_scope=owner_scope,
                    case_id=case.case_id,
                    details={"classification_generation_key": (case.classification_generation_key)},
                )
                return case, True

            generation_key = self._classification_generation_key(case)
            try:
                image = self._validated_case_image(case, payload)
            except (ImageValidationError, ValueError) as exc:
                case = self._commit_classification_failure(
                    case,
                    status=ClassificationExecutionStatus.UNAVAILABLE,
                    generation_key=generation_key,
                    error_code="classification_image_artifact_unavailable",
                )
                self._audit(
                    request_id=request_id,
                    actor_id=user_id,
                    action="case_classification_unavailable",
                    owner_scope=owner_scope,
                    case_id=case.case_id,
                    details={"error_type": type(exc).__name__},
                )
                if raise_on_failure and not self._classification_is_reusable(case):
                    raise ToolUnavailableError(
                        "the immutable case image is unavailable for classification"
                    ) from exc
                return case, False

            try:
                evidence = self.vision.infer(case_id=case.case_id, image=image)
                self._verify_real_vision_evidence(
                    evidence,
                    case_id=case.case_id,
                    image_sha256=case.image_sha256,
                    image_source_format=image.source_format,
                    input_transform_id=image.input_transform_id,
                    image_quality_status=image.quality_status,
                    image_quality_codes=image.quality_warnings,
                )
                fusion = fuse_rank03(evidence, self.policy)
            except VisionBackendError as exc:
                case = self._commit_classification_failure(
                    case,
                    status=ClassificationExecutionStatus.FAILED,
                    generation_key=generation_key,
                    error_code="classification_backend_failed",
                )
                self._audit(
                    request_id=request_id,
                    actor_id=user_id,
                    action="case_classification_failed",
                    owner_scope=owner_scope,
                    case_id=case.case_id,
                    details={"error_type": type(exc).__name__},
                )
                if raise_on_failure and not self._classification_is_reusable(case):
                    raise
                return case, False

            candidate = case.model_copy(
                deep=True,
                update={
                    "classification_status": ClassificationExecutionStatus.COMPLETED,
                    "classification_generation_key": generation_key,
                    "classification_attempt_count": max(0, case.classification_attempt_count) + 1,
                    "classification_error_code": None,
                    "vision_evidence": evidence,
                    "fusion_decision": fusion,
                },
            )
            candidate.screening_disposition = _screening_disposition(fusion)
            candidate.conflict_flags = [
                flag for flag in case.conflict_flags if flag != "classification_tool_failure"
            ]
            candidate.evidence_gaps = [
                gap
                for gap in case.evidence_gaps
                if gap not in {"classification_failed", "classification_unavailable"}
            ]
            candidate.uncertainty_flags = [
                flag for flag in case.uncertainty_flags if flag != "classification_exact_argmax_tie"
            ]
            if evidence.classifier_argmax_tied:
                candidate.uncertainty_flags.append("classification_exact_argmax_tie")
            candidate.human_review_reason_codes = [
                reason
                for reason in case.human_review_reason_codes
                if reason != "classification_tool_failed"
            ]
            candidate.human_review_required = bool(candidate.human_review_reason_codes)
            if candidate.localization_evidence.status != "not_requested":
                self._apply_localization_system_state(candidate)

            try:
                committed, reused = self.store.commit_classification_state(
                    candidate,
                    expected_version=case.record_version,
                )
            except VersionConflictError:
                winner = self._require_case_access(
                    case_id=case_id,
                    owner_scope=owner_scope,
                    user_id=user_id,
                )
                if self._classification_is_reusable(winner):
                    return winner, True
                raise
            self._audit(
                request_id=request_id,
                actor_id=user_id,
                action="case_classification_completed",
                owner_scope=owner_scope,
                case_id=committed.case_id,
                details={
                    "classification_generation_key": generation_key,
                    "model_version": evidence.classifier_model_id,
                    "weight_version": evidence.classifier_weight_version,
                    "predicted_class": (
                        evidence.predicted_class.value
                        if evidence.predicted_class is not None
                        else None
                    ),
                    "cached": reused,
                },
            )
            return committed, reused

    def classify_cxr_case(
        self,
        *,
        case_id: str,
        owner_scope: str,
        user_id: str,
        payload: bytes | None = None,
    ) -> tuple[CaseRecord, AgentResponse]:
        """Run/reuse ConvNeXt only; used by explicit batch screening."""

        request_id, trace_id = _ids()
        case, reused = self._ensure_case_classification(
            case_id=case_id,
            owner_scope=owner_scope,
            user_id=user_id,
            request_id=request_id,
            payload=payload,
            raise_on_failure=self.settings.require_real_inference,
        )
        return case, self._case_response(
            case,
            request_id=request_id,
            trace_id=trace_id,
            reused=reused,
            include_guidance=False,
        )

    def _tool_classify_current_cxr(self, invocation: ToolInvocation) -> AgentResponse:
        if invocation.case_id is None:
            raise KeyError("case_id is required for classification")
        case, reused = self._ensure_case_classification(
            case_id=invocation.case_id,
            owner_scope=invocation.owner_scope,
            user_id=invocation.user_id,
            request_id=invocation.request_id,
            raise_on_failure=True,
        )
        response = self._case_response(
            case,
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            reused=reused,
            include_guidance=False,
        )
        # ``analyze_current_cxr`` intentionally combines first analysis and
        # explanation into one high-level capability.  Presentation intent is
        # owned by this handler; it never changes the classifier result or
        # calls another tool.
        normalized_question = invocation.message.casefold()
        if any(
            cue in normalized_question
            for cue in ("为什么", "为何", "依据", "原因", "怎么判", "如何判")
        ):
            response = self._case_rationale_response(case, response)
        response.thread_id = invocation.thread_id
        return self.safety.verify(response)

    def _tool_exact_case(self, invocation: ToolInvocation) -> AgentResponse:
        if invocation.case_id is None:
            raise KeyError("case_id is required for exact-case explanation")
        case = self._require_case_access(
            case_id=invocation.case_id,
            owner_scope=invocation.owner_scope,
            user_id=invocation.user_id,
        )
        response = self._case_response(
            case,
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            reused=True,
            include_guidance=False,
        )
        # The tool contract already says that this is an explanation request.
        # Do not run a second keyword router over the raw message here: after a
        # TaskSpec has authorized this tool, rules may validate or reject the
        # call but must not choose a different answer shape.
        response = self._case_rationale_response(case, response)
        response.thread_id = invocation.thread_id
        return self.safety.verify(response)

    def _localization_generation_key(self, case: CaseRecord) -> tuple[str, str]:
        detector = self.runtime_config["detector"]
        preprocessing_version = (
            f"dfine-localization-v1:{detector['input_size']}:"
            f"{detector['bbox_format']}:{detector['precision']}"
        )
        return (
            _canonical_sha256(
                {
                    "schema": "tbx-localization-generation-v1",
                    "image_sha256": case.image_sha256,
                    "model_bundle_id": self.runtime_config["model_bundle_id"],
                    "checkpoint_sha256": detector["checkpoint_sha256"],
                    "resolved_config_sha256": detector["resolved_config_sha256"],
                    "input_size": detector["input_size"],
                    "native_label": detector["native_label"],
                    "export_floor": detector["export_floor"],
                    "bbox_format": detector["bbox_format"],
                    "precision": detector["precision"],
                }
            ),
            preprocessing_version,
        )

    @staticmethod
    def _apply_localization_system_state(case: CaseRecord) -> None:
        localization = case.localization_evidence
        flags = [
            flag
            for flag in case.conflict_flags
            if flag
            not in {
                "classifier_positive_detector_negative",
                "classifier_negative_detector_positive",
                "tool_failure_conflict",
            }
        ]
        gaps = [
            gap
            for gap in case.evidence_gaps
            if gap not in {"localization_missing", "localization_failed"}
        ]
        reasons = [
            reason
            for reason in case.human_review_reason_codes
            if reason
            not in {
                "cross_model_conflict",
                "localization_tool_failed",
            }
        ]
        if localization.status == "failed":
            gaps.append("localization_failed")
        case.conflict_flags = list(dict.fromkeys(flags))
        case.evidence_gaps = list(dict.fromkeys(gaps))
        case.human_review_reason_codes = list(dict.fromkeys(reasons))
        case.human_review_required = bool(case.human_review_reason_codes)

    def _legacy_localization_evidence(
        self,
        case: CaseRecord,
        *,
        generation_key: str,
        preprocessing_version: str,
    ) -> LocalizationEvidence | None:
        evidence = case.vision_evidence
        if evidence is None or _DETECTOR_NOT_REQUESTED_REF in evidence.artifact_refs:
            return None
        status = "completed" if evidence.detections else "completed_no_detection"
        return LocalizationEvidence(
            status=status,
            run_id=evidence.run_id,
            generation_key=generation_key,
            case_id=case.case_id,
            image_sha256=case.image_sha256,
            detector_model_id=evidence.detector_model_id,
            detector_checkpoint_sha256=evidence.detector_checkpoint_sha256,
            preprocessing_version=preprocessing_version,
            detections=list(evidence.detections),
            runtime_ms=evidence.runtime_ms,
            attempt_count=1,
        )

    def _ensure_case_localization(
        self,
        *,
        invocation: ToolInvocation,
    ) -> CaseRecord:
        if invocation.case_id is None:
            raise KeyError("case_id is required for localization")
        lock_key = _state_lock_key(
            "localization",
            invocation.owner_scope,
            invocation.user_id,
            invocation.case_id,
        )
        with self._state_locks.hold(lock_key):
            case = self._require_case_access(
                case_id=invocation.case_id,
                owner_scope=invocation.owner_scope,
                user_id=invocation.user_id,
            )
            evidence = case.vision_evidence
            detector = self.runtime_config["detector"]
            detector_model_id = (
                evidence.detector_model_id
                if evidence is not None
                else f"{self.runtime_config['model_bundle_id']}:dfine_l"
            )
            detector_checkpoint_sha256 = (
                evidence.detector_checkpoint_sha256
                if evidence is not None
                else detector["checkpoint_sha256"]
            )
            generation_key, preprocessing_version = self._localization_generation_key(case)
            existing = case.localization_evidence
            if (
                existing.status in {"completed", "completed_no_detection"}
                and existing.generation_key == generation_key
            ):
                return case

            legacy = self._legacy_localization_evidence(
                case,
                generation_key=generation_key,
                preprocessing_version=preprocessing_version,
            )
            if legacy is not None:
                candidate = case.model_copy(deep=True)
                candidate.localization_evidence = legacy
                self._apply_localization_system_state(candidate)
                committed, _ = self.store.commit_localization_state(
                    candidate,
                    expected_version=case.record_version,
                )
                return committed

            localize = getattr(self.vision, "localize", None)
            if not callable(localize):
                raise VisionBackendError("the configured backend has no localization worker")
            image = self._validated_case_image(case, None)
            started = time.perf_counter()
            run_id = str(uuid.uuid4())
            attempt_count = max(0, existing.attempt_count) + 1
            try:
                detections = localize(case_id=case.case_id, image=image)
                detector_runtime_ms = max(
                    0,
                    int((time.perf_counter() - started) * 1000),
                )
                localization = LocalizationEvidence(
                    status="completed" if detections else "completed_no_detection",
                    run_id=run_id,
                    generation_key=generation_key,
                    case_id=case.case_id,
                    image_sha256=case.image_sha256,
                    detector_model_id=detector_model_id,
                    detector_checkpoint_sha256=detector_checkpoint_sha256,
                    preprocessing_version=preprocessing_version,
                    detections=detections,
                    runtime_ms=detector_runtime_ms,
                    attempt_count=attempt_count,
                )
            except Exception as exc:
                detector_runtime_ms = max(
                    0,
                    int((time.perf_counter() - started) * 1000),
                )
                localization = LocalizationEvidence(
                    status="failed",
                    run_id=run_id,
                    generation_key=generation_key,
                    case_id=case.case_id,
                    image_sha256=case.image_sha256,
                    detector_model_id=detector_model_id,
                    detector_checkpoint_sha256=detector_checkpoint_sha256,
                    preprocessing_version=preprocessing_version,
                    runtime_ms=detector_runtime_ms,
                    attempt_count=attempt_count,
                    error_code=f"localization_{type(exc).__name__.lower()}",
                )
                failed = case.model_copy(deep=True)
                failed.localization_evidence = localization
                self._apply_localization_system_state(failed)
                self.store.commit_localization_state(
                    failed,
                    expected_version=case.record_version,
                )
                raise

            candidate = case.model_copy(deep=True)
            candidate.localization_evidence = localization
            self._apply_localization_system_state(candidate)
            try:
                case, reused = self.store.commit_localization_state(
                    candidate,
                    expected_version=case.record_version,
                )
            except VersionConflictError:
                winner = self._require_case_access(
                    case_id=invocation.case_id,
                    owner_scope=invocation.owner_scope,
                    user_id=invocation.user_id,
                )
                if (
                    winner.localization_evidence.status in {"completed", "completed_no_detection"}
                    and winner.localization_evidence.generation_key == generation_key
                ):
                    return winner
                raise
            self._audit(
                request_id=invocation.request_id,
                actor_id=invocation.user_id,
                action="case_localization_completed",
                owner_scope=invocation.owner_scope,
                case_id=case.case_id,
                details={
                    "candidate_count": len(localization.detections),
                    "detector_runtime_ms": detector_runtime_ms,
                    "classifier_route_unchanged": True,
                    "localization_generation_key": generation_key,
                    "reused": reused,
                },
            )
            return case

    def _tool_localize_current_cxr(self, invocation: ToolInvocation) -> AgentResponse:
        case = self._ensure_case_localization(invocation=invocation)
        if case.vision_evidence is None:
            response = AgentResponse(
                request_id=invocation.request_id,
                trace_id=invocation.trace_id,
                thread_id=invocation.thread_id,
                case_id=case.case_id,
                response_kind=ResponseKind.CASE_EXPLANATION,
                summary="候选区域定位已完成。",
                limitations=["候选区域定位不用于确诊或排除肺结核。"],
                safety_policy_id=self.safety.policy_id,
            )
        else:
            response = self._case_response(
                case,
                request_id=invocation.request_id,
                trace_id=invocation.trace_id,
                reused=True,
                include_guidance=False,
            )
        anatomy_run = self.store.find_latest_completed_anatomy_run(
            case_id=case.case_id,
            owner_scope=invocation.owner_scope,
            user_id=invocation.user_id,
        )
        response = self._case_localization_response(case, response, anatomy_run)
        response.thread_id = invocation.thread_id
        return self.safety.verify(response)

    def _tool_inspect_anatomical_context(
        self,
        invocation: ToolInvocation,
    ) -> AgentResponse:
        """Run the optional lung-field worker only when the controller asks for it."""

        if invocation.case_id is None:
            raise KeyError("case_id is required for anatomical context")
        run = self.request_anatomy_run(
            case_id=invocation.case_id,
            owner_scope=invocation.owner_scope,
            user_id=invocation.user_id,
        )
        deadline = time.monotonic() + max(0.25, self._tool_timeout_seconds - 0.75)
        while run.status in {AnatomyRunStatus.PENDING, AnatomyRunStatus.RUNNING}:
            if time.monotonic() >= deadline:
                raise ToolUnavailableError("anatomy worker did not finish within this turn")
            time.sleep(0.02)
            run = self.get_anatomy_run(
                run_id=run.run_id,
                case_id=invocation.case_id,
                owner_scope=invocation.owner_scope,
                user_id=invocation.user_id,
            )
        if run.status == AnatomyRunStatus.TECHNICAL_FAILURE:
            raise ToolUnavailableError(run.error_code or "anatomy worker failed")

        return self._anatomy_run_response(
            run,
            case=self._require_case_access(
                case_id=run.case_id,
                owner_scope=run.owner_scope,
                user_id=run.user_id,
            ),
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            thread_id=invocation.thread_id,
        )

    def _anatomy_run_response(
        self,
        run: AnatomyRunRecord,
        *,
        case: CaseRecord,
        request_id: str,
        trace_id: str,
        thread_id: str,
    ) -> AgentResponse:
        """Project one completed anatomy record onto the concise chat surface.

        The controller also uses this projection when the required anatomy
        evidence was completed in an earlier turn.  Reusing the persisted
        record here must not create a tool receipt or a new anatomy run; the
        complete per-box overlap metrics remain available in ``run`` for audit
        and case-detail rendering.
        """

        if run.status not in {
            AnatomyRunStatus.COMPLETED,
            AnatomyRunStatus.COMPLETED_WITH_REFINEMENT_FAILURE,
        }:
            raise ValueError("anatomy chat projection requires a completed run")

        if run.spatial_summary is not None:
            selected = select_display_detections(
                [item.model_dump(mode="json") for item in case.localization_evidence.detections],
                policy=_DETECTION_DISPLAY_POLICY,
                image_width=case.image_width,
            )
            visible_locations = [
                run.detector_locations[candidate.raw_index]
                for candidate in selected
                if candidate.raw_index < len(run.detector_locations)
            ]
            summary = build_chat_spatial_summary(
                visible_locations,
                anatomy_qc_status=run.spatial_summary.anatomy_qc_status,
            )
        else:
            # Completed records normally always contain the structured spatial
            # summary. Keep a concise compatibility response for legacy data.
            summary = "肺野结构分析已完成，可在病例详情查看结果。"
        return self.safety.verify(
            AgentResponse(
                request_id=request_id,
                trace_id=trace_id,
                thread_id=thread_id,
                case_id=run.case_id,
                response_kind=ResponseKind.CASE_EXPLANATION,
                summary=summary,
                limitations=["肺野结构结果用于辅助查看，不用于确诊或排除肺结核。"],
                safety_policy_id=self.safety.policy_id,
            )
        )

    def _case_quality_response(
        self,
        case: CaseRecord,
        *,
        request_id: str,
        trace_id: str,
        thread_id: str,
    ) -> AgentResponse:
        """Project stored upload QC into a quality-only response.

        This is shared by the explicit quality tool and the controller's cached
        evidence path so a quality question cannot fall back to a classification
        response or lose the re-upload guidance merely because QC already ran at
        upload time.
        """

        if case.image_quality_codes:
            findings = [
                _IMAGE_QUALITY_LABELS.get(code, f"未识别的质控代码：{code}")
                for code in case.image_quality_codes
            ]
            summary = (
                "基础输入可用性检查发现："
                + "；".join(dict.fromkeys(findings))
                + "。请换用更清晰的原始胸片重新上传。"
            )
        else:
            summary = (
                "基础输入可用性检查：已覆盖文件解码、尺寸与宽高比、基础灰度动态范围，"
                "未发现问题。未覆盖摆位、吸气、曝光和轻度运动模糊，无法据此评价这些项目。"
            )
        return self.safety.verify(
            AgentResponse(
                request_id=request_id,
                trace_id=trace_id,
                thread_id=thread_id,
                case_id=case.case_id,
                response_kind=ResponseKind.CASE_EXPLANATION,
                summary=summary,
                limitations=["本系统不用于确诊或排除肺结核。"],
                safety_policy_id=self.safety.policy_id,
            )
        )

    def _tool_inspect_image_quality(self, invocation: ToolInvocation) -> AgentResponse:
        if invocation.case_id is None:
            raise KeyError("case_id is required for image quality inspection")
        case = self._require_case_access(
            case_id=invocation.case_id,
            owner_scope=invocation.owner_scope,
            user_id=invocation.user_id,
        )
        return self._case_quality_response(
            case,
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            thread_id=invocation.thread_id,
        )

    def _tool_compare_with_prior_cxr(self, invocation: ToolInvocation) -> AgentResponse:
        if invocation.case_id is None:
            raise KeyError("case_id is required for longitudinal comparison")
        case = self._require_case_access(
            case_id=invocation.case_id,
            owner_scope=invocation.owner_scope,
            user_id=invocation.user_id,
        )
        return self.safety.verify(
            AgentResponse(
                request_id=invocation.request_id,
                trace_id=invocation.trace_id,
                thread_id=invocation.thread_id,
                case_id=case.case_id,
                response_kind=ResponseKind.SAFE_ABSTENTION,
                summary=(
                    "当前版本未接入可用于纵向比较的既往胸片，也没有执行前后片比较，"
                    "因此无法判断与半年前相比是否恶化。"
                ),
                limitations=["本系统不用于确诊或排除肺结核。"],
                safety_policy_id=self.safety.policy_id,
            )
        )

    def _retrieval_abstention(
        self,
        invocation: ToolInvocation,
        *,
        information_type: str,
    ) -> AgentResponse:
        return AgentResponse(
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            thread_id=invocation.thread_id,
            case_id=invocation.case_id,
            response_kind=ResponseKind.SAFE_ABSTENTION,
            summary=(
                f"当前受审核知识快照未检索到达到相关性与适用范围门槛的{information_type}证据，"
                "因此本次不生成医学建议。"
            ),
            limitations=[
                "本系统不用于确诊或排除肺结核。",
                "本系统不使用低相关性检索结果补全医学内容。",
                "如问题涉及当前症状或个体治疗，请由结核病定点医疗机构评估。",
            ],
            urgency=Urgency.PROMPT_EVALUATION,
            safety_policy_id=self.safety.policy_id,
        )

    def _retrieval_content_quarantine(
        self,
        invocation: ToolInvocation,
        *,
        response_kind: ResponseKind = ResponseKind.NEXT_TEST_INFORMATION,
    ) -> AgentResponse:
        """Return a code-owned boundary notice without echoing rejected retrieval data."""

        return AgentResponse(
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            thread_id=invocation.thread_id,
            case_id=invocation.case_id,
            response_kind=response_kind,
            summary=(
                "检索结果未通过知识快照完整性或适用范围校验，已被隔离；"
                "其中的文本不会被执行，也不会作为医学证据引用。"
            ),
            limitations=[
                "本系统不用于确诊或排除肺结核。",
                "本次未使用被隔离的检索文本生成诊断、检查或治疗内容。",
                "请改用已审核且可追溯的来源重新提问。",
            ],
            urgency=Urgency.PROMPT_EVALUATION,
            safety_policy_id=self.safety.policy_id,
        )

    def _guideline_gap_response(
        self,
        invocation: ToolInvocation,
        *,
        scope: str,
        subtopic: str,
        gap: str,
    ) -> AgentResponse:
        response_kind = (
            ResponseKind.TREATMENT_EDUCATION
            if scope == "treatment_education"
            else ResponseKind.SAFE_ABSTENTION
        )
        limitations = [
            "本系统不用于确诊或排除肺结核。",
            "本系统不会使用其他指南范围的文献替代缺失证据。",
        ]
        if scope == "treatment_education":
            limitations.append("不根据本次知识缺口生成个体化药物、剂量或疗程建议。")
        return AgentResponse(
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            thread_id=invocation.thread_id,
            case_id=invocation.case_id,
            response_kind=response_kind,
            summary=gap,
            limitations=limitations,
            source_query=invocation.message,
            guideline_scope=scope,
            guideline_subtopic=subtopic,
            answer_status=GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE,
            evidence_gap=gap,
            urgency=Urgency.ROUTINE_INFORMATION,
            safety_policy_id=self.safety.policy_id,
        )

    @staticmethod
    def _guideline_evidence_from_hit(hit: Any) -> RetrievedGuidelineEvidence:
        citation = hit.citation
        return RetrievedGuidelineEvidence(
            chunk_id=citation.chunk_id,
            text=hit.text,
            source_id=citation.source_id,
            source=citation.title,
            organization=citation.organization,
            publication_year=citation.publication_year,
            section=citation.section,
            locator=citation.locator,
            page=_locator_page(citation.locator),
            url=citation.url,
            score=hit.score,
            lexical_score=hit.lexical_score,
            semantic_score=hit.semantic_score,
            metadata={
                "topics": list(hit.topics),
                "jurisdiction": hit.jurisdiction,
                "allowed_claim_scope": list(hit.allowed_claim_scope),
                "treatment_details_allowed": hit.treatment_details_allowed,
                "claim_type": hit.claim_type,
                "recommendation_strength": hit.recommendation_strength,
                "certainty": hit.certainty,
                "support_text": citation.support_text,
            },
        )

    def _tool_search_tb_guidance(self, invocation: ToolInvocation) -> AgentResponse:
        """Resolve the original question inside the retrieval tool, then search.

        The agent-facing contract is intentionally just ``message``.  The
        resolved dimensions are internal retrieval constraints, never medical
        claims.  They are copied onto the invocation so the immutable tool
        receipt records exactly which population/entity/scenario gates were
        applied.  Explicit dimensions remain a compatibility fallback for old
        direct adapters only when the raw question cannot be resolved.
        """

        thread = self.store.get_or_create_thread(
            invocation.thread_id,
            invocation.user_id,
            invocation.owner_scope,
        )
        memory = thread.recent_guideline_task
        profile = understand_guidance_query(
            invocation.message,
            prior_scope=(memory.scope if memory is not None else None),
            prior_subtopic=(memory.subtopic if memory is not None else None),
            prior_population=(memory.population if memory is not None else ()),
            prior_product_terms=(memory.product_terms if memory is not None else ()),
            prior_scenario_tags=(memory.scenario_tags if memory is not None else ()),
        )
        if profile is None:
            if invocation.guideline_scope is not None and invocation.subtopic:
                return self._tool_retrieve_guideline(invocation)
            # Elliptical follow-ups such as “具体该怎么做？” carry no safe
            # retrieval dimensions by themselves.  Reuse only the structured
            # dimensions written after this same tenant/user/thread completed
            # a successful knowledge search.  The raw question is always
            # parsed first, so an explicit topic or population change wins.
            if memory is not None:
                invocation.guideline_scope = GuidelineScope(memory.scope)
                invocation.subtopic = memory.subtopic
                invocation.population = list(memory.population)
                invocation.product_terms = list(memory.product_terms)
                invocation.scenario_tags = [
                    GuidelineScenarioTag(item) for item in memory.scenario_tags
                ]
                return self._tool_retrieve_guideline(invocation)
            return self._guideline_gap_response(
                invocation,
                scope="unspecified",
                subtopic="unspecified",
                gap=(
                    "当前问题不足以确定可安全检索的结核病指南范围；"
                    "请补充想了解的是筛查、检查、治疗、传染防护还是特殊人群。"
                ),
            )

        # This object is private to one registry execution. Updating it makes
        # the post-execution receipt expose the resolved retrieval contract;
        # the original query remains the only model/user supplied argument.
        invocation.guideline_scope = profile.scope
        invocation.subtopic = profile.subtopic
        invocation.population = list(profile.population)
        invocation.product_terms = list(profile.product_terms)
        invocation.scenario_tags = list(profile.scenario_tags)
        return self._tool_retrieve_guideline(invocation)

    def _tool_retrieve_guideline(self, invocation: ToolInvocation) -> AgentResponse:
        """Return extractive, scope-bound guideline evidence for one question."""

        scope = str(getattr(invocation, "guideline_scope", None) or "").strip()
        subtopic = str(getattr(invocation, "subtopic", None) or "").strip()
        population = [
            str(item).strip() for item in getattr(invocation, "population", []) if str(item).strip()
        ]
        product_terms = [
            str(item).strip()
            for item in getattr(invocation, "product_terms", [])
            if str(item).strip()
        ]
        scenario_tags = {
            GuidelineScenarioTag(str(item))
            for item in getattr(invocation, "scenario_tags", [])
        }
        policy = _GUIDELINE_SCOPE_POLICY.get(scope)
        if policy is None:
            return self._guideline_gap_response(
                invocation,
                scope=scope or "unspecified",
                subtopic=subtopic or "unspecified",
                gap="当前问题没有经过验证的指南范围，未执行跨范围检索。",
            )
        required_claim_scopes = set(policy["claim_scopes"])
        if subtopic in _GUIDELINE_SUBTOPIC_CLAIM_SCOPES:
            required_claim_scopes = set(_GUIDELINE_SUBTOPIC_CLAIM_SCOPES[subtopic])
        if subtopic == "special_population_testing" and "children" in population:
            required_claim_scopes.update({"active_screening", "screening_pathway"})
        if scope == "treatment_education":
            if subtopic == "care_setting":
                required_claim_scopes = {
                    "care_setting_education",
                    "hospitalization_indications",
                }
            elif subtopic not in _GUIDELINE_SUBTOPIC_CLAIM_SCOPES:
                required_claim_scopes = {
                    "treatment_education",
                    (
                        "standard_regimen_duration"
                        if subtopic == "standard_regimen_duration"
                        else "treatment_principles"
                    ),
                }
        if scope == "treatment_education" and _is_personalized_treatment_request(
            invocation.message
        ):
            return self._guideline_gap_response(
                invocation,
                scope=scope,
                subtopic=subtopic or "treatment_principles",
                gap=(
                    "这是个体治疗方案或剂量请求；当前系统只提供一般治疗教育，"
                    "不根据体重、合并症或个人资料生成处方、剂量或个人疗程。"
                    "请由结核病定点医疗机构结合药物敏感性和完整病史制定方案。"
                ),
            )
        if (
            scope == "treatment_education"
            and subtopic != "care_setting"
            and _is_drug_resistant_treatment_question(invocation.message)
        ):
            return self._guideline_gap_response(
                invocation,
                scope=scope,
                subtopic=subtopic or "treatment_principles",
                gap=(
                    "当前受审核知识快照只包含WHO药物敏感性肺结核的一般治疗教育，"
                    "不包含可用于回答耐药肺结核治疗的审核来源；不能把药物敏感性方案"
                    "套用于耐药情形。请由结核病定点医疗机构根据完整耐药谱制定方案。"
                ),
            )
        if scope == "treatment_education" and is_medication_change_request(invocation.message):
            return self._guideline_gap_response(
                invocation,
                scope=scope,
                subtopic=subtopic or "treatment_principles",
                gap=(
                    "这是个体用药调整问题：不要自行停药、换药或调整剂量；"
                    "请联系原治疗机构评估。本轮不生成个体化用药建议。"
                ),
            )

        preferred_topics = _GUIDELINE_SUBTOPIC_TOPICS.get(subtopic, set())
        required_source_ids = _guideline_required_source_ids(
            subtopic=subtopic,
            population=population,
            scenario_tags=scenario_tags,
        )
        expansion = _GUIDELINE_QUERY_EXPANSION.get(subtopic, "肺结核 指南")
        retrieval_query = "\n".join(
            item
            for item in (
                invocation.message,
                expansion,
                " ".join(population),
                " ".join(
                    _SPECIAL_POPULATION_QUERY_EXPANSION[item]
                    for item in population
                    if item in _SPECIAL_POPULATION_QUERY_EXPANSION
                ),
                " ".join(product_terms),
            )
            if item
        )
        retrieve_scoped = getattr(self.retriever, "retrieve_scoped", None)
        if callable(retrieve_scoped):
            raw_hits = retrieve_scoped(
                retrieval_query,
                required_claim_scopes=required_claim_scopes,
                preferred_topics=preferred_topics,
                ranking_terms=[
                    subtopic,
                    *population,
                    *product_terms,
                    *(item.value for item in sorted(scenario_tags, key=lambda item: item.value)),
                ],
                jurisdictions=policy["jurisdictions"],
                required_source_ids=required_source_ids,
                minimum_lexical_score=0.0,
                minimum_semantic_score=0.0,
                top_k=8,
            )
        else:
            # Compatibility for narrow test doubles and older adapters. Scope
            # remains a hard retrieval filter; non-empty results are never used
            # unless the immutable attestation boundary is also available.
            raw_hits = self.retriever.retrieve(
                retrieval_query,
                required_claim_scopes=required_claim_scopes,
                jurisdictions=policy["jurisdictions"],
                required_source_ids=required_source_ids,
                minimum_lexical_score=0.0,
                minimum_semantic_score=0.0,
                top_k=8,
            )
        attest_hits = getattr(self.retriever, "attest_hits", None)
        if callable(attest_hits):
            attestation = attest_hits(
                raw_hits,
                required_claim_scopes=required_claim_scopes,
                jurisdictions=policy["jurisdictions"],
            )
        elif raw_hits:
            return self._guideline_gap_response(
                invocation,
                scope=scope,
                subtopic=subtopic or "unspecified",
                gap="检索适配器没有提供知识快照完整性校验，结果未被采用。",
            )
        else:
            attestation = RetrievalAttestation(
                hits=(),
                rejected_count=0,
                rejection_reasons=(),
            )
        if not attestation.hits and attestation.rejected_count:
            return self._guideline_gap_response(
                invocation,
                scope=scope,
                subtopic=subtopic or "unspecified",
                gap="检索结果未通过知识快照完整性或指南范围校验。",
            )
        hits = list(attestation.hits)
        if not hits:
            if subtopic == "standard_regimen_duration":
                gap = "当前受审核知识库没有标准治疗方案与疗程的可引用条款。"
            elif subtopic == "medication_dose":
                gap = (
                    "当前受审核知识库没有可直接回答具体抗结核药剂量的条款；"
                    "未使用一般治疗原则代替剂量证据。"
                )
            elif subtopic == "treatment_principles":
                gap = "当前受审核知识库没有肺结核一般治疗原则的可引用医学条款。"
            elif subtopic == "care_setting":
                gap = "当前受审核知识库没有住院、门诊或社区照护模式的可引用条款。"
            elif subtopic == "respiratory_protection":
                gap = "当前受审核知识库没有呼吸防护或佩戴口罩的可引用条款。"
            elif subtopic == "imaging_modality_selection":
                gap = (
                    "当前受审核WHO证据不支持所有疑似肺结核者都必须做CT；"
                    "本知识快照也没有可用于制定个人CT检查指征的直接条款。"
                )
            else:
                gap = f"当前受审核知识库没有足够证据回答该{subtopic or scope}问题。"
            return self._guideline_gap_response(
                invocation,
                scope=scope,
                subtopic=subtopic or "unspecified",
                gap=gap,
            )

        topic_hits = [hit for hit in hits if preferred_topics.intersection(set(hit.topics))]
        selection_pool = topic_hits if preferred_topics else hits
        if preferred_topics and not selection_pool:
            return self._guideline_gap_response(
                invocation,
                scope=scope,
                subtopic=subtopic or "unspecified",
                gap="检索结果没有命中当前子主题的直接条款，未使用同范围其他内容代答。",
            )
        selected_hits = _select_guideline_hits(
            selection_pool,
            subtopic=subtopic,
            population=population,
            scenario_tags=scenario_tags,
        )
        if (
            subtopic in {"special_population_guidance", "special_population_testing"}
            and population
            and not selected_hits
        ):
            gap = (
                "当前受审核知识库没有直接覆盖该特殊人群疑似肺结核诊断检查路径的"
                "可引用条款；主动筛查场景条款未外推为诊断规则。"
                if subtopic == "special_population_testing"
                else (
                    "当前受审核知识库没有直接覆盖该特殊人群检查路径的可引用条款，"
                    "未使用通用或其他人群条款代替。"
                )
            )
            return self._guideline_gap_response(
                invocation,
                scope=scope,
                subtopic=subtopic,
                gap=gap,
            )
        if not selected_hits:
            return self._guideline_gap_response(
                invocation,
                scope=scope,
                subtopic=subtopic or "unspecified",
                gap=(
                    "检索结果没有覆盖当前实体、人群或适用场景，"
                    "未使用同主题的其他条款代答。"
                ),
            )
        answer_status = (
            GuidelineAnswerStatus.ANSWERED
            if topic_hits or not preferred_topics
            else GuidelineAnswerStatus.PARTIAL
        )
        evidence_gap: str | None = None

        if product_terms:
            evidence_material = " ".join(
                [
                    *(hit.text for hit in selected_hits),
                    *(hit.citation.support_text for hit in selected_hits),
                    *(hit.citation.section for hit in selected_hits),
                ]
            )
            normalized_material = _normalized_evidence_term(evidence_material)
            missing_products = [
                term
                for term in product_terms
                if _normalized_evidence_term(term) not in normalized_material
            ]
            if missing_products:
                answer_status = GuidelineAnswerStatus.PARTIAL
                evidence_gap = (
                    "现有证据只覆盖通用快速分子检测/NAAT信息，未覆盖产品级适用条件："
                    + "、".join(missing_products)
                    + "。"
                )

        # Certain subtopics require direct, subtopic-specific authority.  A
        # same-scope fallback is informative for ordinary questions but cannot
        # answer a missing regimen or infection-control clause.
        if subtopic == "standard_regimen_duration" and not any(
            hit.treatment_details_allowed and "疗程" in hit.text for hit in topic_hits
        ):
            return self._guideline_gap_response(
                invocation,
                scope=scope,
                subtopic=subtopic,
                gap="当前受审核知识库没有标准治疗方案与疗程的可引用条款。",
            )
        if subtopic == "respiratory_protection" and not topic_hits:
            return self._guideline_gap_response(
                invocation,
                scope=scope,
                subtopic=subtopic,
                gap="当前受审核知识库没有呼吸防护或佩戴口罩的可引用条款。",
            )
        if subtopic == "care_setting" and not topic_hits:
            return self._guideline_gap_response(
                invocation,
                scope=scope,
                subtopic=subtopic,
                gap="当前受审核知识库没有住院、门诊或社区照护模式的可引用条款。",
            )

        evidence = [self._guideline_evidence_from_hit(hit) for hit in selected_hits]
        claims = [
            GroundedGuidelineClaim(
                text=hit.text,
                chunk_ids=[hit.citation.chunk_id],
            )
            for hit in selected_hits
        ]
        claim_texts = [claim.text for claim in claims]
        if evidence_gap is None and answer_status == GuidelineAnswerStatus.PARTIAL:
            evidence_gap = "检索到同一指南范围内的证据，但没有命中该子主题的直接条款。"
        summary = claim_texts[0]
        if evidence_gap:
            summary = f"{summary}\n\n证据范围：{evidence_gap}"
        response_kind = {
            "treatment_education": ResponseKind.TREATMENT_EDUCATION,
            "diagnostic_testing": ResponseKind.NEXT_TEST_INFORMATION,
            "cad_interpretation": ResponseKind.NEXT_TEST_INFORMATION,
        }.get(scope, ResponseKind.DIAGNOSTIC_INFORMATION)
        response_lists: dict[str, Any]
        if scope == "treatment_education":
            response_lists = {"treatment_education": claim_texts}
        elif scope in {"diagnostic_testing", "cad_interpretation"}:
            response_lists = {"next_step_information": claim_texts}
        else:
            response_lists = {"diagnostic_information": claim_texts}
        return AgentResponse(
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            thread_id=invocation.thread_id,
            case_id=invocation.case_id,
            response_kind=response_kind,
            summary=summary,
            citations=[hit.citation for hit in selected_hits],
            limitations=[
                "本系统不用于确诊或排除肺结核。",
                "回答仅使用本轮列出的受审核知识块。",
            ],
            source_query=invocation.message,
            guideline_scope=scope,
            guideline_subtopic=subtopic or None,
            answer_status=answer_status,
            retrieved_evidence=evidence,
            claims=claims,
            evidence_gap=evidence_gap,
            urgency=Urgency.ROUTINE_INFORMATION,
            safety_policy_id=self.safety.policy_id,
            **response_lists,
        )

    # Deprecated adapters remain only for direct-call compatibility while the
    # runtime uses the single retrieve_guideline contract.
    def _tool_treatment_education(self, invocation: ToolInvocation) -> AgentResponse:
        return self._tool_retrieve_guideline(
            invocation.model_copy(
                update={
                    "guideline_scope": "treatment_education",
                    "subtopic": getattr(invocation, "subtopic", None) or "treatment_principles",
                }
            )
        )

    def _tool_diagnostic_guidance(self, invocation: ToolInvocation) -> AgentResponse:
        return self._tool_retrieve_guideline(
            invocation.model_copy(
                update={
                    "guideline_scope": "diagnostic_testing",
                    "subtopic": getattr(invocation, "subtopic", None) or "diagnostic_pathway",
                }
            )
        )

    def _tool_capabilities(self, invocation: ToolInvocation) -> AgentResponse:
        return AgentResponse(
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            thread_id=invocation.thread_id,
            case_id=invocation.case_id,
            response_kind=ResponseKind.SAFE_ABSTENTION,
            summary=TBX_CAPABILITY_ANSWER,
            limitations=["不用于确诊或排除肺结核，也不提供个体化处方。"],
            safety_policy_id=self.safety.policy_id,
        )

    def _tool_failure_fallback(
        self,
        invocation: ToolInvocation,
        status: ToolCallStatus,
        error_code: str,
    ) -> AgentResponse:
        if invocation.tool_name == ToolName.EMERGENCY_TRIAGE.value:
            return self._tool_emergency_triage(invocation)
        if invocation.tool_name == ToolName.CLASSIFY_CURRENT_CXR.value:
            return AgentResponse(
                request_id=invocation.request_id,
                trace_id=invocation.trace_id,
                thread_id=invocation.thread_id,
                case_id=invocation.case_id,
                response_kind=ResponseKind.SAFE_ABSTENTION,
                summary="分类模型本次未能完成运行。",
                limitations=[
                    "本系统不用于确诊或排除肺结核。",
                    f"工具状态：{status.value}；代码：{error_code}。",
                ],
                safety_policy_id=self.safety.policy_id,
            )
        if invocation.tool_name == ToolName.LOCALIZE_CURRENT_CXR.value:
            return AgentResponse(
                request_id=invocation.request_id,
                trace_id=invocation.trace_id,
                thread_id=invocation.thread_id,
                case_id=invocation.case_id,
                response_kind=ResponseKind.SAFE_ABSTENTION,
                summary="候选区域定位暂时没有完成。三分类结果仍可查看，你可以稍后重试定位。",
                limitations=[
                    "本系统不用于确诊或排除肺结核。",
                    f"工具降级代码：{error_code}；状态：{status.value}。",
                ],
                safety_policy_id=self.safety.policy_id,
            )
        return AgentResponse(
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            thread_id=invocation.thread_id,
            case_id=invocation.case_id,
            response_kind=ResponseKind.SAFE_ABSTENTION,
            summary="本次所需工具未能在受控执行边界内完成，因此没有生成医学建议。",
            limitations=[
                "本系统不用于确诊或排除肺结核。",
                "请稍后重试；若存在持续症状或担忧，请联系结核病定点医疗机构。",
                f"工具降级代码：{error_code}；状态：{status.value}。",
            ],
            safety_policy_id=self.safety.policy_id,
        )

    def _apply_narrator(
        self,
        response: AgentResponse,
        *,
        narrator_override: Any | None = None,
        source_query: str | None = None,
    ) -> AgentResponse:
        active_narrator = narrator_override if narrator_override is not None else self.narrator
        if active_narrator is None:
            return response
        narrator_required = self.settings.require_llm_inference or narrator_override is not None
        authoritative = response.model_copy(
            update=(
                {"source_query": source_query}
                if source_query is not None and response.source_query is None
                else {}
            ),
            deep=True,
        )
        if authoritative.urgency == Urgency.EMERGENCY:
            return authoritative.model_copy(
                update=_narrator_metadata(active_narrator, NarrationStatus.SKIPPED_EMERGENCY)
            )
        if authoritative.response_kind not in _NARRATABLE_RESPONSE_KINDS:
            return authoritative.model_copy(
                update=_narrator_metadata(active_narrator, NarrationStatus.SKIPPED_RESPONSE_KIND)
            )
        try:
            candidate = active_narrator.narrate(authoritative.model_copy(deep=True))
            if not _narration_invariants_hold(authoritative, candidate):
                raise NarrationRejectedError("narrator changed a protected response field")
            if narrator_required:
                metadata_attested = (
                    candidate.narration_status == NarrationStatus.APPLIED
                    and candidate.narrator_backend == active_narrator.backend_id
                    and candidate.narrator_model == active_narrator.model
                    and candidate.narrator_policy_id == NARRATOR_POLICY_ID
                    and candidate.narrator_generation_invoked is True
                    and candidate.narrator_prompt_tokens is not None
                    and candidate.narrator_prompt_tokens > 0
                    and candidate.narrator_completion_tokens is not None
                    and candidate.narrator_completion_tokens > 0
                )
                if active_narrator.backend_id == "llama_cpp":
                    metadata_attested = bool(
                        metadata_attested
                        and candidate.narrator_model == self.settings.llama_cpp_model_alias
                        and candidate.narrator_model_digest == self.settings.llama_cpp_model_sha256
                    )
                if not metadata_attested:
                    raise NarrationRejectedError(
                        "selected LLM response lacks generation attestation"
                    )
            if authoritative.answer_status is not None:
                validate_grounded_synthesis_summary(
                    authoritative,
                    candidate.summary,
                    candidate.claims,
                )
            else:
                validate_narration_summary(authoritative, candidate.summary)
            return self.safety.verify(candidate)
        except (NarrationRejectedError, SafetyViolationError) as exc:
            if narrator_required:
                raise NarrationError(
                    "selected LLM output was rejected by the evidence or safety contract"
                ) from exc
            return authoritative.model_copy(
                update=_narrator_metadata(
                    active_narrator,
                    NarrationStatus.REJECTED_BY_SAFETY,
                )
            )
        except Exception as exc:
            if narrator_required:
                detail = (
                    "selected LLM inference failed"
                    if narrator_override is not None
                    else "required local LLM inference failed"
                )
                raise NarrationError(detail) from exc
            return authoritative.model_copy(
                update=_narrator_metadata(active_narrator, NarrationStatus.FALLBACK_ERROR)
            )

    @staticmethod
    def _merge_unique_strings(*groups: list[str], limit: int = 32) -> list[str]:
        return list(dict.fromkeys(value for group in groups for value in group if value))[:limit]

    def respond_with_controller(
        self,
        *,
        message: str,
        thread_id: str,
        user_id: str,
        owner_scope: str,
        case_id: str | None = None,
        generator: Any | None = None,
        narrator_override: Any | None = None,
    ):
        """Execute the compiled LangGraph Plan + ReAct workflow.

        The import is local to keep the domain controller independent from the
        service construction graph and to preserve the direct-tool API for
        explicit non-agent integrations.
        """

        from .agent_runtime import run_agent_turn

        return run_agent_turn(
            self,
            message=message,
            thread_id=thread_id,
            user_id=user_id,
            owner_scope=owner_scope,
            case_id=case_id,
            generator=generator,
            narrator_override=narrator_override,
        )

    def respond_with_tool(
        self,
        *,
        selected_tool: ToolName | str,
        message: str,
        thread_id: str,
        user_id: str,
        owner_scope: str,
        case_id: str | None = None,
        step_index: int = 0,
        narrator_override: Any | None = None,
    ) -> ToolResult:
        """Execute a code-selected tool and commit one audited turn.

        API requests cannot supply ``selected_tool``. The registry only runs its
        fixed allowlist and the narrator cannot alter this invocation contract.
        """

        request_id, trace_id = _ids()
        requested_tool = (
            selected_tool.value if isinstance(selected_tool, ToolName) else selected_tool
        )
        try:
            tool_name = ToolName(requested_tool).value
        except ValueError as exc:
            raise ValueError("selected_tool is not in the fixed tool allowlist") from exc
        lock_key = _state_lock_key("thread", owner_scope, user_id, thread_id)
        with self._state_locks.hold(lock_key):
            try:
                if case_id is not None:
                    self._require_case_access(
                        case_id=case_id,
                        owner_scope=owner_scope,
                        user_id=user_id,
                    )
                thread = self.store.get_or_create_thread(thread_id, user_id, owner_scope)
                self._bind_thread_case(thread, case_id)
            except (KeyError, PermissionError) as exc:
                self._audit(
                    request_id=request_id,
                    actor_id=user_id,
                    action="tool_context_rejected",
                    owner_scope=owner_scope,
                    case_id=case_id,
                    details={
                        "error_type": type(exc).__name__,
                        "input_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
                        "routing_policy_id": ROUTER_POLICY_ID,
                        "tool_name": tool_name,
                    },
                )
                raise

            invocation = ToolInvocation(
                tool_name=tool_name,
                message=message,
                thread_id=thread_id,
                user_id=user_id,
                owner_scope=owner_scope,
                request_id=request_id,
                trace_id=trace_id,
                routing_policy_id=ROUTER_POLICY_ID,
                case_id=case_id,
                step_index=step_index,
                max_steps=self.tool_registry.max_steps,
                safety_policy_id=self.safety.policy_id,
            )

            try:
                result = self.tool_registry.execute(
                    invocation,
                    fallback_factory=self._tool_failure_fallback,
                )
            except (KeyError, PermissionError) as exc:
                self._audit(
                    request_id=request_id,
                    actor_id=user_id,
                    action="tool_domain_rejected",
                    owner_scope=owner_scope,
                    case_id=case_id,
                    details={
                        "error_type": type(exc).__name__,
                        "input_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
                        "routing_policy_id": ROUTER_POLICY_ID,
                        "tool_name": tool_name,
                    },
                )
                raise

            response = self.safety.verify(result.response)
            # Degraded output is already the final safe fallback. Letting an LLM
            # elaborate it would defeat fail-closed execution.
            if not result.receipt.fallback_used:
                response = self._apply_narrator(
                    response,
                    narrator_override=narrator_override,
                    source_query=message,
                )
            response_sha256 = _canonical_sha256(response.model_dump(mode="json"))
            receipt = result.receipt.model_copy(update={"response_sha256": response_sha256})
            result = result.model_copy(update={"response": response, "receipt": receipt})

            thread.recent_messages.extend(
                [
                    self._memory_event(
                        role="user",
                        content=message,
                        request_id=request_id,
                        kind="input_digest",
                    ),
                    self._memory_event(
                        role="assistant",
                        content=response.summary,
                        request_id=request_id,
                        kind=response.response_kind.value,
                    ),
                ]
            )
            action = result.audit_action
            thread.tool_call_counts[action] = thread.tool_call_counts.get(action, 0) + 1
            if tool_name in {
                ToolName.GET_EXACT_CASE_AND_EXPLAIN.value,
                ToolName.LOCALIZE_CURRENT_CXR.value,
                ToolName.INSPECT_IMAGE_QUALITY.value,
                ToolName.SEARCH_TB_GUIDANCE.value,
            }:
                thread.active_intent = tool_name
            self.store.save_thread(thread)
            self._audit(
                request_id=request_id,
                actor_id=user_id,
                action=action,
                owner_scope=owner_scope,
                case_id=case_id,
                details={
                    "response_kind": response.response_kind.value,
                    "narrator_backend": response.narrator_backend,
                    "narrator_model": response.narrator_model,
                    "narrator_model_digest": response.narrator_model_digest,
                    "narrator_policy_id": response.narrator_policy_id,
                    "narration_status": response.narration_status.value,
                    "route_override_applied": False,
                    "tool_receipt": receipt.model_dump(mode="json"),
                },
            )
            return result

    def respond(
        self,
        *,
        message: str,
        thread_id: str,
        user_id: str,
        owner_scope: str,
        case_id: str | None = None,
    ) -> AgentResponse:
        return self.respond_with_controller(
            message=message,
            thread_id=thread_id,
            user_id=user_id,
            owner_scope=owner_scope,
            case_id=case_id,
        ).response

    def complete_review(
        self,
        *,
        review_id: str,
        owner_scope: str,
        reviewer_id: str,
        expected_version: int,
        decision: str,
        note: str | None,
        subject_user_id: str | None = None,
        allow_cross_subject: bool = False,
    ) -> ReviewRecord:
        from datetime import UTC, datetime

        request_id, _ = _ids()
        lock_key = _state_lock_key("review", owner_scope, review_id)
        with self._state_locks.hold(lock_key):
            review = self.store.get_review(review_id, owner_scope)
            case = self.store.get_case(review.case_id, owner_scope)
            effective_subject = subject_user_id or reviewer_id
            if not allow_cross_subject and (
                case.user_id is None or case.user_id != effective_subject
            ):
                raise AccessDeniedError(
                    "review subject binding does not match the authenticated user"
                )
            review.reviewer_decision = decision  # validated by pydantic assignment
            review.reviewer_note = note
            review.reviewed_by = reviewer_id
            review.reviewed_at = datetime.now(UTC)
            review, case = self.store.complete_review_with_case_update(
                review, expected_version=expected_version
            )
            self._audit(
                request_id=request_id,
                actor_id=reviewer_id,
                actor_role=ActorRole.REVIEWER,
                action="review_completed",
                owner_scope=owner_scope,
                case_id=case.case_id,
                details={
                    "review_id": review_id,
                    "decision": decision,
                    "version": review.version,
                    "cross_subject_authorized": allow_cross_subject,
                },
            )
            return review

    def update_preferences(self, preferences: UserPreferences) -> UserPreferences:
        return self.store.save_preferences(preferences)

    def build_active_screening_response(
        self,
        session: Any,
        *,
        request_id: str,
        trace_id: str,
        user_id: str,
        owner_scope: str,
    ) -> AgentResponse:
        """Compose questionnaire output with the exact bound CXR evidence, when present."""

        response = self.screening.build_response(
            session,
            request_id=request_id,
            trace_id=trace_id,
        )
        if session.case_id is None:
            return self.safety.verify(response)

        case = self._require_case_access(
            case_id=session.case_id,
            owner_scope=owner_scope,
            user_id=user_id,
        )
        decision = case.fusion_decision
        if decision is None or case.vision_evidence is None:
            return self.safety.verify(response)

        visual_note = {
            VisualResult.MODEL_FLAGGED: "当前胸片模型分流为结核样本训练类别。",
            VisualResult.MODEL_NOT_FLAGGED: "当前胸片模型分流为健康样本训练类别。",
            VisualResult.NON_TB_ABNORMAL: "当前胸片模型分流为非结核病变样本训练类别。",
            VisualResult.PENDING_HUMAN_REVIEW: "当前胸片模型分流结果不确定。",
            VisualResult.INDETERMINATE: "当前胸片模型证据不足以形成稳定分流。",
            VisualResult.TECHNICAL_FAILURE: "当前胸片模型处理未形成可用结果。",
        }[decision.visual_result]
        response.visual_result = decision.visual_result
        response.predicted_class = decision.predicted_class
        response.review_status = case.review_status
        response.visual_evidence_notes = [visual_note]
        response.limitations.append("问询分层与胸片模型分流是并列证据；模型训练类别不是临床诊断。")

        if response.response_kind == ResponseKind.ACTIVE_SCREENING_QUESTION:
            response.summary = f"已载入当前胸片模型结果。{response.summary}"
        elif session.status == "complete":
            response.summary = "主动筛查问询已完成，并已纳入当前胸片模型分流。"
            if (
                response.urgency != Urgency.EMERGENCY
                and decision.visual_result != VisualResult.MODEL_NOT_FLAGGED
            ):
                response.urgency = Urgency.PROMPT_EVALUATION
                response.next_step_information.insert(
                    0,
                    "当前胸片模型结果需要进一步评估，请携带原始影像和本次问询信息就诊。",
                )
        return self.safety.verify(response)

    def start_active_screening(
        self,
        *,
        thread_id: str,
        user_id: str,
        owner_scope: str,
        consent: bool,
        case_id: str | None = None,
    ) -> tuple[Any, AgentResponse]:
        request_id, trace_id = _ids()
        lock_key = _state_lock_key("thread", owner_scope, user_id, thread_id)
        with self._state_locks.hold(lock_key):
            if case_id is not None:
                self._require_case_access(
                    case_id=case_id,
                    owner_scope=owner_scope,
                    user_id=user_id,
                )
            thread = self.store.get_or_create_thread(thread_id, user_id, owner_scope)
            self._bind_thread_case(thread, case_id)
            if thread.active_screening_session_id is not None:
                active = self.store.get_screening_session(
                    thread.active_screening_session_id, owner_scope
                )
                if active.user_id != user_id or active.thread_id != thread_id:
                    raise AccessDeniedError("active screening identity mismatch")
                if active.status == "collecting":
                    if not consent:
                        if active.case_id is not None:
                            self._require_case_access(
                                case_id=active.case_id,
                                owner_scope=owner_scope,
                                user_id=user_id,
                            )
                        active = self.screening.cancel(active)
                        self.store.save_screening_session(active)
                        thread.active_screening_session_id = None
                        self.store.save_thread(thread)
                        response = self.build_active_screening_response(
                            active,
                            request_id=request_id,
                            trace_id=trace_id,
                            user_id=user_id,
                            owner_scope=owner_scope,
                        )
                        self._audit(
                            request_id=request_id,
                            actor_id=user_id,
                            action="active_screening_cancelled_on_consent_withdrawal",
                            owner_scope=owner_scope,
                            case_id=active.case_id,
                            details={
                                "session_id": active.session_id,
                                "rule_version": active.guideline_rule_version,
                            },
                        )
                        return active, response
                    if active.case_id != case_id:
                        raise AccessDeniedError(
                            "active screening is bound to a different case context"
                        )
                    response = self.build_active_screening_response(
                        active,
                        request_id=request_id,
                        trace_id=trace_id,
                        user_id=user_id,
                        owner_scope=owner_scope,
                    )
                    self._audit(
                        request_id=request_id,
                        actor_id=user_id,
                        action="active_screening_resumed",
                        owner_scope=owner_scope,
                        case_id=case_id,
                        details={
                            "session_id": active.session_id,
                            "rule_version": active.guideline_rule_version,
                        },
                    )
                    return active, response
            session = self.screening.start_session(
                thread_id=thread_id,
                user_id=user_id,
                owner_scope=owner_scope,
                case_id=case_id,
                consent=consent,
            )
            self.store.save_screening_session(session)
            thread.active_screening_session_id = (
                session.session_id if session.status == "collecting" else None
            )
            self.store.save_thread(thread)
            response = self.build_active_screening_response(
                session,
                request_id=request_id,
                trace_id=trace_id,
                user_id=user_id,
                owner_scope=owner_scope,
            )
            self._audit(
                request_id=request_id,
                actor_id=user_id,
                action=("active_screening_started" if consent else "active_screening_declined"),
                owner_scope=owner_scope,
                case_id=case_id,
                details={
                    "session_id": session.session_id,
                    "rule_version": session.guideline_rule_version,
                },
            )
            return session, response

    def answer_active_screening(
        self,
        *,
        session_id: str,
        user_id: str,
        owner_scope: str,
        question_id: str,
        answer: Any,
    ) -> tuple[Any, AgentResponse]:
        request_id, trace_id = _ids()
        initial = self.store.get_screening_session(session_id, owner_scope)
        if initial.user_id != user_id:
            raise AccessDeniedError("screening session user identity mismatch")
        # A thread is the aggregate consistency boundary for its active session.
        # Every screening mutation takes this same lock so consent withdrawal,
        # answers, and cancellation cannot resurrect stale session state.
        thread_lock = _state_lock_key("thread", owner_scope, user_id, initial.thread_id)
        with self._state_locks.hold(thread_lock):
            session = self.store.get_screening_session(session_id, owner_scope)
            if session.user_id != user_id or session.thread_id != initial.thread_id:
                raise AccessDeniedError("screening session identity changed")
            if session.case_id is not None:
                self._require_case_access(
                    case_id=session.case_id,
                    owner_scope=owner_scope,
                    user_id=user_id,
                )
            thread = self.store.get_or_create_thread(
                session.thread_id, session.user_id, session.owner_scope
            )
            self._bind_thread_case(thread, session.case_id)
            if thread.active_screening_session_id != session.session_id:
                raise AccessDeniedError("screening session is not the thread's active session")
            session = self.screening.submit_answer(session, answer, question_id=question_id)
            self.store.save_screening_session(session)
            thread.active_screening_session_id = (
                session.session_id if session.status == "collecting" else None
            )
            self.store.save_thread(thread)
            response = self.build_active_screening_response(
                session,
                request_id=request_id,
                trace_id=trace_id,
                user_id=user_id,
                owner_scope=owner_scope,
            )
            self._audit(
                request_id=request_id,
                actor_id=user_id,
                action=(
                    "active_screening_completed"
                    if session.status == "complete"
                    else "active_screening_answered"
                ),
                owner_scope=owner_scope,
                case_id=session.case_id,
                details={
                    "session_id": session_id,
                    "question_id": question_id,
                    "status": session.status,
                },
            )
            return session, response

    def cancel_active_screening(
        self,
        *,
        session_id: str,
        user_id: str,
        owner_scope: str,
    ) -> tuple[Any, AgentResponse]:
        request_id, trace_id = _ids()
        initial = self.store.get_screening_session(session_id, owner_scope)
        if initial.user_id != user_id:
            raise AccessDeniedError("screening session user identity mismatch")
        thread_lock = _state_lock_key("thread", owner_scope, user_id, initial.thread_id)
        with self._state_locks.hold(thread_lock):
            session = self.store.get_screening_session(session_id, owner_scope)
            if session.user_id != user_id or session.thread_id != initial.thread_id:
                raise AccessDeniedError("screening session identity changed")
            if session.case_id is not None:
                self._require_case_access(
                    case_id=session.case_id,
                    owner_scope=owner_scope,
                    user_id=user_id,
                )
            thread = self.store.get_or_create_thread(
                session.thread_id, session.user_id, session.owner_scope
            )
            self._bind_thread_case(thread, session.case_id)
            if thread.active_screening_session_id != session.session_id:
                raise AccessDeniedError("screening session is not the thread's active session")
            session = self.screening.cancel(session)
            self.store.save_screening_session(session)
            thread.active_screening_session_id = None
            self.store.save_thread(thread)
            response = self.build_active_screening_response(
                session,
                request_id=request_id,
                trace_id=trace_id,
                user_id=user_id,
                owner_scope=owner_scope,
            )
            self._audit(
                request_id=request_id,
                actor_id=user_id,
                action="active_screening_cancelled_and_answers_cleared",
                owner_scope=owner_scope,
                case_id=session.case_id,
                details={"session_id": session_id},
            )
            return session, response

    def create_report(
        self,
        *,
        case_id: str,
        owner_scope: str,
        actor_id: str,
        subject_user_id: str | None = None,
        allow_cross_subject: bool = False,
    ) -> dict[str, str]:
        request_id, _ = _ids()
        lock_key = _state_lock_key("report", owner_scope, case_id)
        with self._state_locks.hold(lock_key):
            effective_subject = subject_user_id or actor_id
            if allow_cross_subject:
                case = self.store.get_case(case_id, owner_scope)
            else:
                case = self._require_case_access(
                    case_id=case_id,
                    owner_scope=owner_scope,
                    user_id=effective_subject,
                )
            review = self.store.get_review(case.review_id, owner_scope) if case.review_id else None
            report_topics = {
                "diagnosis",
                "next_tests",
                "confirmation_boundary",
                "imaging_nonspecificity",
                "imaging_role_education",
            }
            report_claim_scopes = {
                "initial_diagnostic_testing",
                "test_limitations",
                "china_diagnostic_principles",
                "confirmation_basis",
                "confirmation_boundary",
                "imaging_nonspecificity",
                "imaging_role_education",
            }
            report_jurisdictions = {"China", "WHO"}
            raw_hits = self.retriever.retrieve(
                "影像筛查异常后进一步诊断检查",
                topics=report_topics,
                jurisdictions=report_jurisdictions,
                required_claim_scopes=report_claim_scopes,
                top_k=4,
            )
            attestation = self.retriever.attest_hits(
                raw_hits,
                required_claim_scopes=report_claim_scopes,
                jurisdictions=report_jurisdictions,
            )
            hits = list(attestation.hits)
            anatomy_run = (
                self.store.find_latest_completed_anatomy_run(
                    case_id=case.case_id,
                    owner_scope=case.owner_scope,
                    user_id=case.user_id,
                )
                if case.user_id is not None
                else None
            )
            artifact = generate_case_report(
                case=case,
                review=review,
                citations=[hit.citation for hit in hits],
                artifact_root=self.settings.case_artifact_root,
                knowledge_snapshot_id=self.retriever.snapshot_id,
                knowledge_manifest_sha256=self.retriever.manifest_sha256,
                knowledge_chunks_sha256=self.retriever.chunks_sha256,
                anatomy_run=anatomy_run,
            )
            self._audit(
                request_id=request_id,
                actor_id=actor_id,
                action="report_created",
                owner_scope=owner_scope,
                case_id=case_id,
                details={
                    "report_id": artifact.report_id,
                    "markdown_sha256": artifact.markdown_sha256,
                    "json_sha256": artifact.json_sha256,
                    "cross_subject_authorized": allow_cross_subject,
                    "retrieval_rejected_count": attestation.rejected_count,
                },
            )
            return {
                "report_id": artifact.report_id,
                "markdown_path": str(artifact.markdown_path),
                "json_path": str(artifact.json_path),
                "generated_at": artifact.generated_at.isoformat(),
                "markdown_sha256": artifact.markdown_sha256,
                "json_sha256": artifact.json_sha256,
            }
