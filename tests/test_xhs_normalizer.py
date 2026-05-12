from xhs_travel_graph.models import RouteVariantFact
from xhs_travel_graph.normalizer import normalize_route_variant


def test_normalizer_extracts_stairs_and_cost_constraints():
    fact = RouteVariantFact(
        route_variant_id="rv1",
        post_id="p1",
        name="天门山",
        evidence_span="穿山扶梯需买票32元或爬999台阶",
    )

    normalized = normalize_route_variant(fact)

    assert any(req.demand == "climb_stairs" and req.magnitude == 999 for req in normalized.requirements)
    assert any(constraint.metric == "extra_cost_cny" and constraint.value_num == 32 for constraint in normalized.constraints)
