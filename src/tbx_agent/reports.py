from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .anatomy_runs import AnatomyRunRecord
from .product_identity import VISION_MODEL_DISPLAY_NAME
from .schemas import CaseRecord, Citation, ReviewRecord


@dataclass(frozen=True, slots=True)
class ReportArtifact:
    report_id: str
    markdown_path: Path
    json_path: Path
    generated_at: datetime
    markdown_sha256: str
    json_sha256: str


_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]+")
_MARKDOWN_META = re.compile(r"([\\`*_{}\[\]()<>#+.!|~-])")
_HIDDEN_CLASSIFIER_SCORE_FIELDS = frozenset(
    {
        "class_probability_order",
        "class_probabilities",
        "top1_score",
        "top2_score",
        "top1_top2_margin",
        "classifier_threshold",
    }
)


def _markdown_literal(value: object, *, max_length: int = 2_000) -> str:
    """Render untrusted record text as one inert Markdown line."""

    text = _CONTROL_CHARACTERS.sub(" ", str(value)).strip()
    text = " ".join(text.split())[:max_length]
    return _MARKDOWN_META.sub(r"\\\1", text)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _report_safe_case_payload(value: object) -> object:
    """Remove classifier scores from the user-downloadable report projection."""

    if isinstance(value, dict):
        return {
            key: _report_safe_case_payload(item)
            for key, item in value.items()
            if key not in _HIDDEN_CLASSIFIER_SCORE_FIELDS
        }
    if isinstance(value, list):
        return [_report_safe_case_payload(item) for item in value]
    return value


def _visual_label(case: CaseRecord) -> str:
    if case.fusion_decision is None:
        return "技术失败/尚无可用模型证据"
    rule = case.fusion_decision.classifier_decision_rule
    if rule == "p_tb_gte_threshold":
        routes = {
            "model_flagged": "固定筛查规则已触发，需进一步评估",
            "model_not_flagged": "固定筛查规则未触发",
        }
    elif rule == "native_three_class_argmax":
        routes = {
            "model_flagged": "模型识别为结核类，建议进一步检查",
            "model_not_flagged": "模型没有识别出结核",
            "non_tb_abnormal": "模型识别为非结核异常",
        }
    else:
        routes = {
            "model_flagged": "模型辅助筛查标记：历史复现策略满足标记条件",
            "model_not_flagged": "模型辅助筛查未标记：历史复现策略未满足标记条件",
        }
    return {
        **routes,
        "pending_human_review": "训练类分流、并列或图像质量状态需要人工复核",
        "technical_failure": "技术失败：需要重新上传或检查模型服务",
        "indeterminate": "无法判定：需要人工复核或补充检查",
    }[case.fusion_decision.visual_result.value]


def _anatomy_report_summary(run: AnatomyRunRecord | None) -> dict[str, object] | None:
    """Project a completed anatomy run to report-safe, routing-neutral provenance."""

    if run is None:
        return None
    evidence = run.evidence
    spatial = run.spatial_summary
    refinement = run.refinement_evidence
    refinement_summary: dict[str, object] = {
        "status": run.refinement_status.value,
        "backend_id": run.refinement_backend_id,
        "error_code": run.refinement_error_code,
        "model_identity": None,
        "prompt_count": 0,
        "refined_count": 0,
        "capacity_abstained_count": 0,
    }
    if refinement is not None:
        refinement_summary.update(
            {
                "model_identity": {
                    "model_id": refinement.model_id,
                    "model_revision": refinement.model_revision,
                    "model_weight_sha256": refinement.model_weight_sha256,
                    "preprocessing_id": refinement.preprocessing_id,
                    "policy_id": refinement.policy_id,
                },
                "prompt_count": len(refinement.items),
                "refined_count": sum(item.status.value == "refined" for item in refinement.items),
                "capacity_abstained_count": refinement.capacity_abstained_count,
            }
        )
    return {
        "run_id": run.run_id,
        "status": run.status.value,
        "routing_effect": run.routing_effect,
        "clinical_validation": run.clinical_validation,
        "anatomy_model_identity": (
            {
                "backend_id": evidence.backend_id,
                "model_id": evidence.model_id,
                "model_weight_sha256": evidence.model_weight_sha256,
                "preprocessing_id": evidence.preprocessing_id,
                "policy_id": evidence.policy_id,
            }
            if evidence is not None
            else None
        ),
        "anatomy_qc": (
            {"status": evidence.qc.status.value, "codes": list(evidence.qc.codes)}
            if evidence is not None
            else None
        ),
        "spatial_summary": spatial.model_dump(mode="json") if spatial is not None else None,
        "refinement": refinement_summary,
    }


def _anatomy_markdown_lines(summary: dict[str, object] | None) -> list[str]:
    if summary is None:
        return ["- 未生成可用的肺野分割/空间关系证据。"]
    anatomy_qc = summary.get("anatomy_qc") or {}
    spatial = summary.get("spatial_summary") or {}
    refinement = summary.get("refinement") or {}
    model_identity = summary.get("anatomy_model_identity") or {}
    refinement_model = refinement.get("model_identity") or {}
    statements = spatial.get("statements") or []
    lines = [
        f"- 解剖 run 状态：{_markdown_literal(summary['status'])}",
        f"- 肺野 QC：{_markdown_literal(anatomy_qc.get('status', 'unavailable'))}",
        f"- 肺野模型：{_markdown_literal(model_identity.get('model_id', 'unavailable'))}",
        f"- 候选框数：{spatial.get('candidate_count', 0)}；已定位："
        f"{spatial.get('localized_count', 0)}；肺野外/未定位："
        f"{spatial.get('outside_lungs_count', 0)}",
        f"- 轮廓细化状态：{_markdown_literal(refinement.get('status', 'disabled'))}",
        f"- 轮廓模型：{_markdown_literal(refinement_model.get('model_id', 'not_used'))}",
        f"- 对 {VISION_MODEL_DISPLAY_NAME} 分类影响：无；临床验证：false。",
    ]
    if refinement.get("error_code"):
        lines.append(f"- 轮廓细化错误代码：{_markdown_literal(refinement['error_code'])}")
    if statements:
        lines.extend(["", "代码生成的二维空间陈述："])
        lines.extend(f"- {_markdown_literal(statement)}" for statement in statements)
    return lines


def generate_case_report(
    *,
    case: CaseRecord,
    review: ReviewRecord | None,
    citations: list[Citation],
    artifact_root: Path,
    knowledge_snapshot_id: str,
    knowledge_manifest_sha256: str,
    knowledge_chunks_sha256: str,
    anatomy_run: AnatomyRunRecord | None = None,
) -> ReportArtifact:
    generated_at = datetime.now(UTC)
    citations = list(
        {(citation.source_id, citation.chunk_id): citation for citation in citations}.values()
    )
    if (
        not case.case_id
        or case.case_id in {".", ".."}
        or "/" in case.case_id
        or "\\" in case.case_id
        or _CONTROL_CHARACTERS.search(case.case_id)
    ):
        raise ValueError("case id is unsafe for report artifact paths")
    report_id = (
        f"tbx-report-{case.case_id}-{generated_at.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
    )
    artifact_root = artifact_root.resolve()
    report_root = (artifact_root / case.case_id / "reports").resolve()
    if not report_root.is_relative_to(artifact_root):
        raise ValueError("report artifact path escaped its configured root")
    report_root.mkdir(parents=True, exist_ok=True)
    markdown_path = report_root / f"{report_id}.md"
    json_path = report_root / f"{report_id}.json"

    visual = _visual_label(case)
    anatomy_summary = _anatomy_report_summary(anatomy_run)
    anatomy_lines = _anatomy_markdown_lines(anatomy_summary)
    branch_lines: list[str] = []
    if case.vision_evidence is not None:
        evidence = case.vision_evidence
        predicted_class = (
            evidence.predicted_class.value if evidence.predicted_class is not None else "unresolved"
        )
        argmax_role = (
            "描述性、非路由、非诊断"
            if evidence.classifier_decision_rule == "p_tb_gte_threshold"
            else "冻结策略分流证据、非诊断"
        )
        branch_lines = [
            f"- 原生 argmax 训练类（{argmax_role}）：{predicted_class}",
            f"- 分类决策规则：{evidence.classifier_decision_rule}",
            f"- 分类最高分并列：{evidence.classifier_argmax_tied}",
            f"- 检测器角色：{evidence.detector_decision_role}",
            f"- 候选区域数：{len(evidence.detections)}",
            f"- 图像传输/域检查：{evidence.image_quality_status}",
            "- 图像质量代码：" + (", ".join(evidence.image_quality_codes) or "none"),
            f"- 输入格式：{evidence.image_source_format}",
            f"- 输入转码契约：{evidence.input_transform_id}",
            f"- 模型运行标识：{evidence.run_id}",
        ]
    citation_lines = [
        f"- {_markdown_literal(item.title)}（{_markdown_literal(item.organization)}, "
        f"{item.publication_year}），{_markdown_literal(item.locator)}："
        f"{_markdown_literal(item.url, max_length=1_000)}"
        for item in citations
    ]
    review_lines = [f"- 人工复核状态：{case.review_status.value}"]
    if review is not None:
        review_lines.extend(
            [
                f"- 复核记录：{_markdown_literal(review.review_id)}（version {review.version}）",
                f"- 复核结论：{_markdown_literal(review.reviewer_decision or 'pending')}",
                f"- 复核人员：{_markdown_literal(review.reviewed_by or 'pending')}",
            ]
        )
        if review.reviewer_note:
            review_lines.append(f"- 复核备注：{_markdown_literal(review.reviewer_note)}")
    markdown = "\n".join(
        [
            "# TBX-Agent 辅助筛查报告",
            "",
            f"- 报告编号：{report_id}",
            f"- 病例编号：{case.case_id}",
            f"- 生成时间（UTC）：{generated_at.isoformat()}",
            f"- 人工复核状态：{case.review_status.value}",
            "",
            "## 辅助筛查结果",
            "",
            visual,
            "",
            *branch_lines,
            "",
            "## 人工复核",
            "",
            *review_lines,
            "",
            "## 可选解剖空间证据",
            "",
            *anatomy_lines,
            "",
            "## 下一步信息",
            "",
            "该结果不是肺结核诊断，也不能排除肺结核。请结合症状、暴露史、胸部影像、"
            "病原学/分子检测及临床判断；存在可疑信号或高风险信息时，应尽快前往结核病"
            "定点医疗机构评估。",
            "healthy、sick_non_tb 和 tb 均为模型训练类，不是胸片正常、具体疾病或肺结核诊断。"
            "检测候选框只作定位证据，不改变当前分类器策略的分流。",
            "",
            "## 证据来源",
            "",
            *(citation_lines or ["- 本报告未附加指南性主张。"]),
            "",
            "## 审计与限制",
            "",
            f"- 图像 SHA256：{case.image_sha256}",
            "- 筛查策略："
            + (case.fusion_decision.policy_id if case.fusion_decision else "unavailable"),
            f"- 知识快照：{knowledge_snapshot_id}",
            f"- 知识清单 SHA256：{knowledge_manifest_sha256}",
            f"- 知识分块 SHA256：{knowledge_chunks_sha256}",
            f"- {VISION_MODEL_DISPLAY_NAME} 仅在 TBX11K 数据集范围内验证，尚无临床验证。",
            "- 本系统不提供个体化处方，不应替代医生、病原学检查或紧急医疗服务。",
            "",
        ]
    )
    payload = {
        "schema_version": "tbx.report.v4",
        "report_id": report_id,
        "generated_at": generated_at.isoformat(),
        "case": _report_safe_case_payload(case.model_dump(mode="json")),
        "review": review.model_dump(mode="json") if review is not None else None,
        "visual_label": visual,
        "anatomy_evidence_summary": anatomy_summary,
        "citations": [item.model_dump(mode="json") for item in citations],
        "knowledge_snapshot": {
            "snapshot_id": knowledge_snapshot_id,
            "manifest_sha256": knowledge_manifest_sha256,
            "chunks_sha256": knowledge_chunks_sha256,
        },
        "clinical_validation": False,
        "disclaimer": "辅助筛查，不用于确诊或排除肺结核，不提供个体化处方。",
    }
    markdown_bytes = markdown.encode("utf-8")
    json_bytes = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    # Both payloads are fully materialized before either destination is committed.
    # Unique report IDs prevent one request from overwriting another.
    _atomic_write(json_path, json_bytes)
    _atomic_write(markdown_path, markdown_bytes)
    return ReportArtifact(
        report_id=report_id,
        markdown_path=markdown_path,
        json_path=json_path,
        generated_at=generated_at,
        markdown_sha256=_sha256(markdown_bytes),
        json_sha256=_sha256(json_bytes),
    )
