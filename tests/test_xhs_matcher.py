from xhs_travel_graph.fit_evaluator import FitEvaluator
from xhs_travel_graph.profile_parser import parse_traveler_profile


def test_low_age_child_water_activity_without_safety_evidence_is_blocked():
    profile = parse_traveler_profile("带5岁小孩，轻松游")
    payload = {
        "play_mode_id": "pm1",
        "name": "海边冲浪",
        "risks": [{"risk_type": "water_safety", "severity": "unknown", "evidence": "海边冲浪"}],
    }

    assessment = FitEvaluator().evaluate_route_variant(profile, payload)

    assert assessment.hard_fail is True
    assert assessment.decision == "unknown"
    assert "water_safety_detail" in assessment.missing_evidence
