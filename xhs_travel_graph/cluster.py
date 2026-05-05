from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .graph_repository import QueryRunner
from .normalizer import stable_id

try:
    import networkx as nx
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    nx = None


class XHSPlayModeClusterer:
    def __init__(self, query_runner: QueryRunner):
        self.query_runner = query_runner

    def cluster_and_write(self, run_id: str = "xhs", destination: str = "") -> Dict[str, Any]:
        rows = self.fetch_route_variants(run_id=run_id, destination=destination)
        communities = self.detect_communities(rows)
        summaries = [self._summarize_community(rows, community, run_id, destination) for community in communities]
        for summary in summaries:
            self._write_play_mode(summary)
        return {
            "route_variants": len(rows),
            "play_modes": len(summaries),
            "community_sizes": [len(summary["route_variant_ids"]) for summary in summaries],
        }

    def fetch_route_variants(self, run_id: str = "xhs", destination: str = "") -> List[Dict[str, Any]]:
        return self.query_runner.query(
            """
            MATCH (rv:RouteVariant {run_id: $run_id})
            WHERE $destination = "" OR rv.destination = $destination
            OPTIONAL MATCH (rv)-[:HAS_SEGMENT]->(seg:RouteSegment)
            OPTIONAL MATCH (rv)-[:HAS_CONSTRAINT]->(c:Constraint)
            OPTIONAL MATCH (rv)-[:REQUIRES]->(req:Requirement)
            OPTIONAL MATCH (rv)-[:HAS_RISK]->(risk:Risk)
            OPTIONAL MATCH (rv)-[:HAS_MITIGATION]->(mit:Mitigation)
            RETURN rv.id AS id,
                   rv.name AS name,
                   rv.destination AS destination,
                   rv.places AS places,
                   rv.style_tags AS style_tags,
                   rv.physical_load_rank AS physical_load_rank,
                   rv.duration_min AS duration_min,
                   rv.duration_max_min AS duration_max_min,
                   rv.cost_min_cny AS cost_min_cny,
                   rv.cost_max_cny AS cost_max_cny,
                   collect(DISTINCT {
                       order: seg.order,
                       from_place: seg.from_place,
                       to_place: seg.to_place,
                       transport_mode: seg.transport_mode,
                       physical_load_rank: seg.physical_load_rank
                   }) AS segments,
                   collect(DISTINCT {
                       metric: c.metric,
                       value_num: c.value_num,
                       value_text: c.value_text,
                       unit: c.unit
                   }) AS constraints,
                   collect(DISTINCT {
                       requirement_type: req.requirement_type,
                       demand: req.demand,
                       magnitude: req.magnitude,
                       unit: req.unit
                   }) AS requirements,
                   collect(DISTINCT {
                       risk_type: risk.risk_type,
                       severity: risk.severity
                   }) AS risks,
                   collect(DISTINCT {
                       mitigation_type: mit.mitigation_type,
                       method: mit.method,
                       status: mit.status
                   }) AS mitigations
            """,
            {"run_id": run_id, "destination": destination},
        )

    def detect_communities(self, rows: List[Dict[str, Any]]) -> List[List[str]]:
        if not rows:
            return []
        if nx is None or len(rows) == 1:
            return [[row["id"]] for row in rows if row.get("id")]
        graph = build_route_similarity_graph(rows)
        if graph.number_of_edges() == 0:
            return [[node] for node in graph.nodes]
        try:
            communities = nx.algorithms.community.louvain_communities(graph, weight="weight", seed=42)
        except Exception:
            communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
        return [sorted(list(community)) for community in communities if community]

    def _summarize_community(
        self,
        all_rows: List[Dict[str, Any]],
        route_ids: Sequence[str],
        run_id: str,
        destination: str,
    ) -> Dict[str, Any]:
        by_id = {row.get("id"): row for row in all_rows}
        rows = [by_id[route_id] for route_id in route_ids if route_id in by_id]
        representative_places = _top_values((place for row in rows for place in _list(row.get("places"))), 4)
        transports = _top_values((_segment_transport(seg) for row in rows for seg in _dict_list(row.get("segments"))), 3)
        style_tags = _top_values((tag for row in rows for tag in _list(row.get("style_tags"))), 4)
        load_rank = _max_value(row.get("physical_load_rank") for row in rows)
        duration_min = _min_value(row.get("duration_min") for row in rows)
        duration_max = _max_value(row.get("duration_max_min") for row in rows)
        cost_min = _min_value(row.get("cost_min_cny") for row in rows)
        cost_max = _max_value(row.get("cost_max_cny") for row in rows)
        destination_candidates = _top_values((row.get("destination") for row in rows), 1)
        resolved_destination = destination or (destination_candidates[0] if destination_candidates else "")
        name = _name_play_mode(resolved_destination, representative_places, transports, style_tags, load_rank)
        return {
            "play_mode_id": stable_id(run_id, resolved_destination, ",".join(sorted(route_ids))),
            "run_id": run_id,
            "name": name,
            "destination": resolved_destination,
            "route_variant_ids": list(route_ids),
            "representative_places": representative_places,
            "dominant_transport_modes": transports,
            "style_tags": style_tags,
            "physical_load_rank": load_rank,
            "duration_min": duration_min,
            "duration_max_min": duration_max,
            "cost_min_cny": cost_min,
            "cost_max_cny": cost_max,
            "evidence_count": len(rows),
        }

    def _write_play_mode(self, summary: Dict[str, Any]) -> None:
        self.query_runner.query(
            """
            MERGE (pm:PlayMode {id: $play_mode_id})
            SET pm.run_id = $run_id,
                pm.name = $name,
                pm.destination = $destination,
                pm.representative_places = $representative_places,
                pm.dominant_transport_modes = $dominant_transport_modes,
                pm.style_tags = $style_tags,
                pm.physical_load_rank = $physical_load_rank,
                pm.duration_min = $duration_min,
                pm.duration_max_min = $duration_max_min,
                pm.cost_min_cny = $cost_min_cny,
                pm.cost_max_cny = $cost_max_cny,
                pm.evidence_count = $evidence_count
            """,
            summary,
        )
        for route_variant_id in summary["route_variant_ids"]:
            self.query_runner.query(
                """
                MATCH (pm:PlayMode {id: $play_mode_id})
                MATCH (rv:RouteVariant {id: $route_variant_id})
                MERGE (pm)-[:CONTAINS]->(rv)
                """,
                {"play_mode_id": summary["play_mode_id"], "route_variant_id": route_variant_id},
            )


def build_route_similarity_graph(route_rows: List[Dict[str, Any]]):
    if nx is None:
        raise RuntimeError("networkx is required for route similarity graph construction")
    graph = nx.Graph()
    for row in route_rows:
        if row.get("id"):
            graph.add_node(row["id"], **row)
    for a, b in combinations(route_rows, 2):
        if not a.get("id") or not b.get("id"):
            continue
        weight, reasons = route_similarity(a, b)
        if weight > 0:
            graph.add_edge(a["id"], b["id"], weight=weight, reasons=reasons)
    return graph


def route_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[int, List[str]]:
    reasons: List[str] = []
    if _overlap(_list(a.get("places")), _list(b.get("places"))):
        reasons.append("shared_place")
    if _overlap(_route_bigrams(a), _route_bigrams(b)):
        reasons.append("shared_route_bigram")
    if _overlap(_transport_modes(a), _transport_modes(b)):
        reasons.append("shared_transport_mode")
    if _overlap(_risk_requirement_buckets(a), _risk_requirement_buckets(b)):
        reasons.append("same_risk_or_requirement_bucket")
    if _overlap(_requirement_types(a), _requirement_types(b)):
        reasons.append("shared_requirement_type")
    if _overlap(_mitigation_types(a), _mitigation_types(b)):
        reasons.append("shared_mitigation_type")
    if _overlap(_list(a.get("style_tags")), _list(b.get("style_tags"))):
        reasons.append("shared_style_tag")
    return len(reasons), reasons


def _route_bigrams(row: Dict[str, Any]) -> List[str]:
    places = _list(row.get("places"))
    bigrams = [f"{places[i]}->{places[i + 1]}" for i in range(len(places) - 1)]
    for segment in _dict_list(row.get("segments")):
        start = segment.get("from_place")
        end = segment.get("to_place")
        if start and end:
            bigrams.append(f"{start}->{end}")
    return bigrams


def _transport_modes(row: Dict[str, Any]) -> List[str]:
    return [_segment_transport(segment) for segment in _dict_list(row.get("segments")) if _segment_transport(segment)]


def _requirement_types(row: Dict[str, Any]) -> List[str]:
    return [
        item.get("requirement_type")
        for item in _dict_list(row.get("requirements"))
        if item.get("requirement_type")
    ]


def _mitigation_types(row: Dict[str, Any]) -> List[str]:
    return [
        item.get("mitigation_type")
        for item in _dict_list(row.get("mitigations"))
        if item.get("mitigation_type")
    ]


def _risk_requirement_buckets(row: Dict[str, Any]) -> List[str]:
    buckets = []
    rank = row.get("physical_load_rank")
    if rank is not None:
        buckets.append(f"physical_load:{int(rank)}")
    buckets.extend(
        f"risk:{item.get('risk_type')}:{item.get('severity')}"
        for item in _dict_list(row.get("risks"))
        if item.get("risk_type")
    )
    buckets.extend(f"req:{item}" for item in _requirement_types(row))
    return buckets


def _name_play_mode(destination: str, places: List[str], transports: List[str], style_tags: List[str], load_rank: Any) -> str:
    place_part = "/".join(places[:2]) if places else "综合路线"
    transport_part = "/".join(transports[:2]) if transports else "常规交通"
    style = "玩法"
    if "intensive" in style_tags or (load_rank is not None and load_rank >= 4):
        style = "高强度打卡线"
    elif "family" in style_tags or "relaxed" in style_tags or (load_rank is not None and load_rank <= 2):
        style = "亲子轻松线"
    elif "budget" in style_tags:
        style = "省钱线"
    return f"{destination or '目的地'}-{place_part}-{transport_part}-{style}"


def _segment_transport(segment: Dict[str, Any]) -> str:
    mode = segment.get("transport_mode") if isinstance(segment, dict) else ""
    return mode if mode and mode != "unknown" else ""


def _overlap(a: Iterable[str], b: Iterable[str]) -> bool:
    left = {item for item in a if item}
    right = {item for item in b if item}
    return bool(left & right)


def _list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict) and any(v is not None for v in item.values())]


def _top_values(values: Iterable[Any], limit: int) -> List[str]:
    counter = Counter(str(value) for value in values if value)
    return [value for value, _ in counter.most_common(limit)]


def _max_value(values: Iterable[Any]):
    parsed = [value for value in values if value is not None]
    return max(parsed) if parsed else None


def _min_value(values: Iterable[Any]):
    parsed = [value for value in values if value is not None]
    return min(parsed) if parsed else None
