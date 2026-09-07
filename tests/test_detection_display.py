from tbx_agent.vision.display import DetectionDisplayPolicy, select_display_detections


def _candidate(box, score):
    return {"bbox_xyxy": box, "score": score, "label": "tb_lesion_candidate"}


def test_display_selection_suppresses_weak_overlapping_dense_queries():
    detections = [
        _candidate((10, 10, 100, 100), 0.91),
        _candidate((12, 12, 102, 102), 0.89),
        _candidate((200, 30, 260, 90), 0.72),
        _candidate((300, 30, 360, 90), 0.61),
        _candidate((400, 30, 460, 90), 0.59),
        _candidate((20, 200, 80, 260), 0.10),
    ]

    selected = select_display_detections(detections)

    assert [item.raw_index for item in selected] == [0, 2]
    assert len(selected) == 2


def test_display_selection_keeps_weak_scores_auditable_but_not_visible():
    detections = [_candidate((10, 10, 100, 100), 0.09)]
    original = [dict(detections[0])]

    selected = select_display_detections(detections)

    assert selected == []
    assert detections == original


def test_frozen_validation_display_threshold_hides_weak_second_box():
    detections = [
        _candidate((10, 10, 100, 100), 0.879588),
        _candidate((210, 10, 300, 100), 0.165297),
    ]

    selected = select_display_detections(detections, image_width=400)

    assert [item.raw_index for item in selected] == [0]
    assert selected[0].score == 0.879588


def test_display_policy_is_presentation_only_and_configurable():
    detections = [
        _candidate((0, 0, 10, 10), 0.8),
        _candidate((20, 0, 30, 10), 0.7),
    ]
    policy = DetectionDisplayPolicy(
        minimum_score=0.0,
        relative_to_best=0.0,
        nms_iou_threshold=1.0,
        max_boxes=1,
    )

    selected = select_display_detections(detections, policy=policy)

    assert [item.raw_index for item in selected] == [0]
    assert detections[1]["score"] == 0.7


def test_display_selection_keeps_at_most_one_candidate_per_image_half():
    detections = [
        _candidate((10, 10, 90, 90), 0.91),
        _candidate((12, 12, 92, 92), 0.89),
        _candidate((120, 20, 180, 80), 0.88),
        _candidate((220, 10, 290, 90), 0.87),
        _candidate((222, 12, 292, 92), 0.86),
        _candidate((320, 20, 380, 80), 0.85),
    ]

    selected = select_display_detections(detections, image_width=400)

    assert [item.raw_index for item in selected] == [0, 3]
    assert len(selected) == 2


def test_display_selection_orders_half_winners_stably():
    detections = [
        _candidate((220, 10, 280, 70), 0.8),
        _candidate((20, 10, 80, 70), 0.8),
        _candidate((30, 100, 90, 160), 0.7),
        _candidate((230, 100, 290, 160), 0.7),
    ]

    selected = select_display_detections(detections, image_width=400)

    assert [item.raw_index for item in selected] == [0, 1]


def test_display_selection_does_not_mutate_per_half_evidence():
    detections = [
        _candidate((10, 10, 100, 100), 0.9),
        _candidate((15, 15, 105, 105), 0.8),
        _candidate((210, 10, 300, 100), 0.7),
    ]
    original = [dict(candidate) for candidate in detections]

    selected = select_display_detections(detections, image_width=400)

    assert [item.raw_index for item in selected] == [0, 2]
    assert detections == original


def test_invalid_image_width_preserves_global_compatibility_strategy():
    detections = [
        _candidate((10, 10, 80, 80), 0.9),
        _candidate((110, 10, 180, 80), 0.8),
        _candidate((210, 10, 280, 80), 0.7),
        _candidate((310, 10, 380, 80), 0.6),
    ]

    for image_width in (None, 0, -1, float("nan"), float("inf"), True):
        selected = select_display_detections(detections, image_width=image_width)
        assert [item.raw_index for item in selected] == [0, 1]
