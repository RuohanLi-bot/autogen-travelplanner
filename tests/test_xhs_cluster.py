from xhs_travel_graph.cluster import route_similarity


def test_route_similarity_uses_explainable_features():
    a = {
        "places": ["天子山", "袁家界"],
        "style_tags": ["family"],
        "segments": [{"from_place": "天子山", "to_place": "袁家界", "transport_mode": "cable_car"}],
        "requirements": [{"requirement_type": "mobility"}],
        "risks": [{"risk_type": "fatigue", "severity": "low"}],
        "mitigations": [{"mitigation_type": "transport_substitution"}],
    }
    b = {
        "places": ["天子山", "袁家界"],
        "style_tags": ["family"],
        "segments": [{"from_place": "天子山", "to_place": "袁家界", "transport_mode": "cable_car"}],
        "requirements": [{"requirement_type": "mobility"}],
        "risks": [{"risk_type": "fatigue", "severity": "low"}],
        "mitigations": [{"mitigation_type": "transport_substitution"}],
    }

    weight, reasons = route_similarity(a, b)

    assert weight >= 5
    assert "shared_place" in reasons
    assert "shared_route_bigram" in reasons
