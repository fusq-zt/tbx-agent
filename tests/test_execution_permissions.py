from __future__ import annotations

import pytest

from tbx_agent.execution_permissions import prohibited_tools
from tbx_agent.llm.tool_calling import HighLevelToolName as Tool


@pytest.mark.parametrize("query", [
    "肺野分割启动过没有？现在不要运行。",
    "先不运行，只告诉我进度。",
    "请不要执行任何新的工具。",
    "现在禁止启动。",
    "先别跑，先看已有结果。",
    "不用重新分析。",
    "不需要继续调用工具。",
    "暂不执行。",
])
def test_bare_explicit_execution_prohibition_blocks_every_new_tool(query):
    assert prohibited_tools(query) == set(Tool)


@pytest.mark.parametrize(("query", "expected"), [
    ("不要分类", {Tool.CLASSIFY_CXR}),
    ("不用重新分类，标框", {Tool.CLASSIFY_CXR}),
    ("请别再执行胸片分类", {Tool.CLASSIFY_CXR}),
    ("分类不要重新运行", {Tool.CLASSIFY_CXR}),
    ("只显示检测框，不需要分类，也不要肺野分割", {
        Tool.CLASSIFY_CXR, Tool.ANALYZE_LUNG_ANATOMY,
    }),
    ("先不定位，告诉我分类结果", {Tool.LOCALIZE_CXR}),
    ("不用检测，直接看分类结果", {Tool.LOCALIZE_CXR}),
    ("别画框", {Tool.LOCALIZE_CXR}),
    ("不要运行定位和分割", {Tool.LOCALIZE_CXR, Tool.ANALYZE_LUNG_ANATOMY}),
    ("分类和定位都别运行", {Tool.CLASSIFY_CXR, Tool.LOCALIZE_CXR}),
    ("不要肺野分割", {Tool.ANALYZE_LUNG_ANATOMY}),
    ("分割别启动", {Tool.ANALYZE_LUNG_ANATOMY}),
    ("不用分析肺野", {Tool.ANALYZE_LUNG_ANATOMY}),
    ("禁止知识检索", {Tool.SEARCH_TB_KNOWLEDGE}),
    ("不要搜索指南，告诉我分类结果", {Tool.SEARCH_TB_KNOWLEDGE}),
    ("先不检索，也不要分类", {Tool.SEARCH_TB_KNOWLEDGE, Tool.CLASSIFY_CXR}),
])
def test_explicit_tool_objects_narrow_the_denial(query, expected):
    assert prohibited_tools(query) == expected


@pytest.mark.parametrize("query", [
    "先分类，正常就不要再定位",
    "如果正常就不定位，如果异常再标框",
    "如果正常，就不要运行定位",
    "若分类正常，则不要检测",
    "只有异常才运行定位",
    "如果结果正常就不用分割",
    "为什么不运行分类？",
    "为什么不用分类模型？",
    "为什么不要启动分割？",
    "是否不要运行分类？",
    "肺野分割是不是不用运行？",
    "需不需要运行检测？",
    "要不要分类？",
    "没有检测到候选框",
    "分类没运行，分割还没启动",
    "不用担心，把可疑位置标出来",
    "不要紧，先分类",
    "分别运行分类和定位",
    "特别分析肺野的位置",
    "告诉我识别结果",
    "不要解释分类结果，直接画框",
    "不要分类概率，只说最终类别",
    "不要定位结果，只显示胸片",
    "执行分类、检测、分割、知识检索",
    "只显示检测框",
    "",
])
def test_questions_conditions_results_and_positive_requests_are_not_prohibitions(query):
    assert prohibited_tools(query) == set()


def test_unconditional_denial_remains_separate_from_conditional_branch():
    assert prohibited_tools("如果正常就不要定位，但不要分类") == {Tool.CLASSIFY_CXR}


def test_positive_request_never_cancels_an_explicit_denial():
    # This function only removes permissions; resolving contradictory requests
    # is a separate model/user interaction, not implicit execution permission.
    assert prohibited_tools("不要分类。请分类。") == {Tool.CLASSIFY_CXR}
