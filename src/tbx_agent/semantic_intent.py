"""Compact semantic tasks: the model chooses meaning, runtime supplies plan labels."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SemanticTask(StrEnum):
    CLASSIFY = "classify_image"
    BOXES = "show_detection_boxes"
    ANATOMY = "locate_within_lungs"
    RATIONALE = "explain_classification"
    STATUS = "summarize_completed_work"
    CAPABILITIES = "explain_app_capabilities"
    QUALITY = "explain_image_quality"
    COMPARISON = "compare_with_prior_image"
    LOBE_LIMIT = "explain_lung_lobe_limits"
    SCREENING_LIMIT = "explain_screening_limits"
    KNOWLEDGE = "search_tb_knowledge"
    CHAT = "general_chat"


class IntentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: SemanticTask
    when: Literal["always", "classification_abnormal"]


class TurnIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tasks: list[IntentItem] = Field(min_length=1, max_length=4)


INTENT_SCHEMA = TurnIntent.model_json_schema()
INTENT_PROMPT = """You identify the user's CURRENT requested tasks for a chest X-ray assistant.
Return only JSON tasks. Do not answer, generate a verbose plan, or select tasks merely
because evidence is cached. History resolves pronouns, never overrides the latest question.

Task meanings:
classify_image: overall image classification/screening (是否有结核可疑).
show_detection_boxes: draw/show/reuse suspicious bounding boxes, or coarse IMAGE positions.
locate_within_lungs: locate a finding relative to left/right LUNG or upper/middle/lower LUNG
FIELDS; lung segmentation. This requires lung masks even when boxes already exist.
"图像右侧中部" is an image coordinate, NOT a lung-field location. Image right is not right lung.
explain_classification: explain WHY/how a prior classifier result was reached.
summarize_completed_work: ask which analyses have run or summarize existing results, no new work.
explain_app_capabilities: ask what the application can do.
explain_image_quality: explain upload clarity/quality.
compare_with_prior_image: compare with an earlier image, even if the prior image is missing.
search_tb_knowledge: TB transmission, prevention, treatment or next diagnostic tests need retrieval.
general_chat: social chat, arithmetic or unsupported requests.

Select ALL tasks explicitly requested this turn, at most four. Classification does not imply
boxes. Boxes do not imply lung anatomy. But a follow-up asking where in the lungs requires
locate_within_lungs, not repeating cached boxes. Reading status is not running those analyses.
Select explain_classification for a short "为什么" following a classification result.
Use when=always unless the user explicitly makes a task conditional on abnormal classification;
then use classification_abnormal for that task and include classify_image(always).
Do NOT evaluate IF branches now. Always include the requested conditional tasks in JSON,
even if the cached classification is healthy. Runtime will decide whether they run.
Cached results or missing image do not change the requested task: runtime handles prerequisites.
No tool arguments, identifiers, private reasoning or extra fields.

Examples of output structure (choose by the current meaning, not by word matching):
用户：现在处理到哪一步了，哪些分析已经完成？
{"tasks":[{"task":"summarize_completed_work","when":"always"}]}
用户：候选框在图像的哪一边？
{"tasks":[{"task":"show_detection_boxes","when":"always"}]}
用户：候选区域属于哪个肺区？
{"tasks":[{"task":"locate_within_lungs","when":"always"}]}
用户：家人该怎么预防传染？
{"tasks":[{"task":"search_tb_knowledge","when":"always"}]}
用户：先筛查，只有异常时再标记。
{"tasks":[{"task":"classify_image","when":"always"},
{"task":"show_detection_boxes","when":"classification_abnormal"}]}
用户：为什么给出这个分类？
{"tasks":[{"task":"explain_classification","when":"always"}]}
用户：不管分类是什么，都把可疑区域标出来。
{"tasks":[{"task":"show_detection_boxes","when":"always"}]}
用户：你会做些什么？
{"tasks":[{"task":"explain_app_capabilities","when":"always"}]}
用户：谢谢。
{"tasks":[{"task":"general_chat","when":"always"}]}
用户：分类、圈出位置，再查下一步一般要做什么检查。
{"tasks":[{"task":"classify_image","when":"always"},{"task":"show_detection_boxes","when":"always"},{"task":"search_tb_knowledge","when":"always"}]}
"""


def intent_plan_payload(intent: TurnIntent, query: str) -> dict[str, Any]:
    """Translate semantic choices, without inspecting query words, into a public plan."""
    contracts = {
        SemanticTask.CLASSIFY: ("胸片分类", "classification", "general"),
        SemanticTask.BOXES: ("显示候选区域", "localization", "general"),
        SemanticTask.ANATOMY: ("分析候选区与肺野的空间关系", "lung_anatomy", "general"),
        SemanticTask.RATIONALE: ("解释分类依据", "classification", "classification_rationale"),
        SemanticTask.STATUS: ("汇总已完成分析", "none", "case_status"),
        SemanticTask.CAPABILITIES: ("介绍系统能力", "none", "capabilities"),
        SemanticTask.QUALITY: ("解释上传质量", "none", "image_quality"),
        SemanticTask.COMPARISON: ("比较既往影像", "none", "prior_comparison"),
        SemanticTask.LOBE_LIMIT: ("说明肺野与肺叶的区别", "none", "lung_lobe_limit"),
        SemanticTask.SCREENING_LIMIT: ("说明筛查结果的适用范围", "none", "screening_limit"),
        SemanticTask.KNOWLEDGE: ("检索结核知识", "tb_knowledge", "general"),
        SemanticTask.CHAT: ("直接回答", "none", "general"),
    }
    steps = []
    focus = "general"
    for item in intent.tasks:
        objective, evidence, candidate_focus = contracts[item.task]
        if candidate_focus != "general":
            focus = candidate_focus
        steps.append({"objective": objective, "evidence_need": evidence,
                      "condition": item.when})
    if {SemanticTask.STATUS, SemanticTask.CAPABILITIES} <= {t.task for t in intent.tasks}:
        focus = "capabilities_and_status"
    return {"goal": query[:160], "answer_focus": focus, "steps": steps}
