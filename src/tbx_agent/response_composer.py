"""Deterministic composition of already validated tool responses."""

from __future__ import annotations

from .schemas import AgentResponse, ResponseKind


def _unique(*groups: list[str], limit: int = 32) -> list[str]:
    return list(dict.fromkeys(item for group in groups for item in group if item))[:limit]


def merge_agent_responses(responses: list[AgentResponse]) -> AgentResponse:
    """Merge any number of executed-tool responses without inventing evidence.

    Every sentence and citation in the result originates in a response that has
    already passed its tool contract.  The function performs no retrieval and
    no medical inference.
    """

    if not responses:
        raise ValueError("at least one response is required")
    if len(responses) == 1:
        return responses[0]

    visual = next(
        (
            item
            for item in responses
            if item.response_kind
            in {ResponseKind.VISUAL_SCREENING_RESULT, ResponseKind.LOCALIZATION_RESULT}
        ),
        None,
    )
    base = visual or responses[-1]
    summaries = list(dict.fromkeys(item.summary.strip() for item in responses if item.summary))
    citations = []
    citation_keys: set[tuple[str, str]] = set()
    for response in responses:
        for citation in response.citations:
            key = (citation.source_id, citation.chunk_id)
            if key not in citation_keys:
                citation_keys.add(key)
                citations.append(citation)
    guideline = next(
        (item for item in reversed(responses) if item.answer_status is not None),
        None,
    )
    # One TaskSpec owns one guideline scope. Preserve that response as one
    # atomic grounded bundle; combining claims from different status/scope
    # objects would invalidate their evidence boundary.
    retrieved_evidence = list(guideline.retrieved_evidence) if guideline else []
    claims = list(guideline.claims) if guideline else []

    return base.model_copy(
        update={
            "summary": "\n\n".join(summaries)[:2_000],
            "visual_evidence_notes": _unique(*(item.visual_evidence_notes for item in responses)),
            "diagnostic_information": _unique(*(item.diagnostic_information for item in responses)),
            "next_step_information": _unique(*(item.next_step_information for item in responses)),
            "treatment_education": _unique(*(item.treatment_education for item in responses)),
            "limitations": _unique(*(item.limitations for item in responses)),
            "citations": citations[:16],
            "source_query": guideline.source_query if guideline else None,
            "guideline_scope": guideline.guideline_scope if guideline else None,
            "guideline_subtopic": guideline.guideline_subtopic if guideline else None,
            "answer_status": guideline.answer_status if guideline else None,
            "retrieved_evidence": retrieved_evidence[:8],
            "claims": claims[:8],
            "evidence_gap": guideline.evidence_gap if guideline else None,
            "urgency": next(
                (item.urgency for item in reversed(responses) if item.urgency is not None),
                base.urgency,
            ),
        }
    )
