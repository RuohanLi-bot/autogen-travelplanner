from xhs_travel_graph.graph_repository import RecordingQueryRunner
from xhs_travel_graph.graph_writer import XHSTravelGraphWriter
from xhs_travel_graph.models import RouteVariantFact, XHSPostEvidence
from xhs_travel_graph.normalizer import normalize_route_variant


def test_graph_writer_uses_schema_bound_nodes_and_alternatives():
    post = XHSPostEvidence(
        post_id="p1",
        source_file="/tmp/xhs.json",
        result_index=0,
        result_count=1,
        body="穿山扶梯需买票32元或爬999台阶",
        raw_result="穿山扶梯需买票32元或爬999台阶",
    )
    fact = normalize_route_variant(
        RouteVariantFact(
            route_variant_id="rv1",
            post_id="p1",
            name="天门山",
            evidence_span="穿山扶梯需买票32元或爬999台阶",
        )
    )
    runner = RecordingQueryRunner()

    XHSTravelGraphWriter(runner).write_many([post], {"p1": [fact]})

    cypher = "\n".join(call["cypher"] for call in runner.calls)
    assert "MERGE (post:Post" in cypher
    assert "MERGE (rv:RouteVariant" in cypher
    assert "MERGE (alt:RouteAlternative" in cypher
    assert "MERGE (rv)-[:HAS_CONSTRAINT]->(c)" in cypher
