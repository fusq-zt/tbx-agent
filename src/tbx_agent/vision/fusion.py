from __future__ import annotations

import math

from ..schemas import ClassifierClass, FusionDecision, VisionEvidence, VisualResult


def fuse_rank03(evidence: VisionEvidence, policy: dict) -> FusionDecision:
    """Apply the declared classifier route while retaining legacy compatibility.

    For active threshold and native-argmax policies detector output is advisory and
    cannot change the classifier route. Legacy evidence keeps its historical
    two-branch disagreement behavior.
    """

    reasons: list[str] = []
    policy_rule = policy.get("classifier_rule")
    policy_detector_role = policy.get("detector_role")
    if policy_rule is None:
        # Old persisted cases were historically fused with only a policy_id.
        policy_mismatch = False
    elif policy_rule == "native_three_class_argmax":
        policy_mismatch = (
            evidence.classifier_decision_rule != "native_three_class_argmax"
            or evidence.detector_decision_role != "advisory_localization_only"
            or policy_detector_role != "advisory_localization_only"
        )
    elif policy_rule == "p_tb_gte_threshold":
        if policy_detector_role == "advisory_localization_only":
            try:
                raw_threshold = policy["classifier_threshold"]
                if isinstance(raw_threshold, bool):
                    raise TypeError
                policy_threshold = float(raw_threshold)
            except (KeyError, TypeError, ValueError):
                policy_threshold = None
            policy_mismatch = (
                evidence.classifier_decision_rule != "p_tb_gte_threshold"
                or evidence.detector_decision_role != "advisory_localization_only"
                or policy_threshold is None
                or not math.isfinite(policy_threshold)
                or evidence.classifier_threshold != policy_threshold
                or "detector_rule" in policy
                or "detector_threshold" in policy
            )
        else:
            detector_rule = policy.get("detector_rule")
            legacy_policy = policy_detector_role in {None, "legacy_vote"} and detector_rule in {
                None,
                "max_category_agnostic_score_gte_threshold",
            }
            policy_mismatch = not legacy_policy or (
                evidence.classifier_decision_rule != "legacy_p_tb_threshold"
                or evidence.detector_decision_role != "legacy_vote"
            )
    else:
        policy_mismatch = True

    if policy_mismatch:
        result = VisualResult.TECHNICAL_FAILURE
        reasons.append("fusion_policy_evidence_mismatch")
    elif evidence.image_quality_status == "technical_failure":
        result = VisualResult.TECHNICAL_FAILURE
        reasons.append("technical_failure")
    elif evidence.classifier_decision_rule == "native_three_class_argmax":
        # ``predicted_class`` is the immutable output of the declared
        # probability-order argmax.  D-FINE and score-margin heuristics do not
        # override it; even an exact numeric tie retains the backend's stable
        # argmax result and is recorded separately as uncertainty metadata.
        predicted_class = evidence.predicted_class
        if predicted_class is None and evidence.classifier_argmax_tied:
            maximum = max(evidence.class_probabilities.values())
            predicted_class = next(
                ClassifierClass(name)
                for name in evidence.class_probability_order
                if evidence.class_probabilities[name] == maximum
            )
        if predicted_class == ClassifierClass.TB:
            result = VisualResult.MODEL_FLAGGED
        elif predicted_class == ClassifierClass.SICK_NON_TB:
            result = VisualResult.NON_TB_ABNORMAL
        elif predicted_class == ClassifierClass.HEALTHY:
            result = VisualResult.MODEL_NOT_FLAGGED
        else:  # Protected by the evidence schema; retained as a fail-closed guard.
            result = VisualResult.TECHNICAL_FAILURE
            reasons.append("classifier_argmax_contract_failure")
    elif evidence.classifier_decision_rule == "p_tb_gte_threshold":
        # D-FINE detections remain localization evidence only and never vote here.
        result = (
            VisualResult.MODEL_FLAGGED
            if evidence.classifier_flagged
            else VisualResult.MODEL_NOT_FLAGGED
        )
    else:
        classifier_flagged = evidence.classifier_flagged
        detector_flagged = bool(evidence.detector_flagged)
        if classifier_flagged != detector_flagged:
            result = VisualResult.PENDING_HUMAN_REVIEW
            reasons.append("rank03_branch_disagreement")
        elif classifier_flagged:
            result = VisualResult.MODEL_FLAGGED
        else:
            result = VisualResult.MODEL_NOT_FLAGGED

    if evidence.image_quality_status == "warning":
        retain_argmax_with_advisory = (
            policy.get("quality_warning") == "retain_argmax_with_advisory"
            and policy_rule == "native_three_class_argmax"
            and policy_detector_role == "advisory_localization_only"
            and evidence.classifier_decision_rule == "native_three_class_argmax"
            and evidence.detector_decision_role == "advisory_localization_only"
        )
        if not retain_argmax_with_advisory:
            # Historical policies intentionally route warnings to review.  The
            # active native-argmax policy instead retains warning details on the
            # VisionEvidence without turning them into review reasons.
            reasons.append("image_quality_warning")
            if result != VisualResult.TECHNICAL_FAILURE:
                result = VisualResult.PENDING_HUMAN_REVIEW

    decision_predicted_class = evidence.predicted_class
    if (
        evidence.classifier_decision_rule == "native_three_class_argmax"
        and decision_predicted_class is None
        and evidence.classifier_argmax_tied
    ):
        maximum = max(evidence.class_probabilities.values())
        decision_predicted_class = next(
            ClassifierClass(name)
            for name in evidence.class_probability_order
            if evidence.class_probabilities[name] == maximum
        )

    return FusionDecision(
        policy_id=str(policy["policy_id"]),
        visual_result=result,
        review_required=result == VisualResult.PENDING_HUMAN_REVIEW,
        review_reasons=reasons,
        classifier_decision_rule=evidence.classifier_decision_rule,
        predicted_class=decision_predicted_class,
        classifier_flagged=evidence.classifier_flagged,
        detector_decision_role=evidence.detector_decision_role,
        detector_flagged=evidence.detector_flagged,
        max_detector_score=max((item.score for item in evidence.detections), default=None),
        clinical_validation=False,
    )
