from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .graph_repository import QueryRunner
from .models import (
    ConstraintFact,
    FitAssessment,
    RequirementFact,
    RiskFact,
    RouteSegmentFact,
    RouteVariantFact,
    XHSPostEvidence,
)
from .normalizer import (
    constraint_id,
    evidence_id,
    requirement_id,
    risk_id,
    stable_id,
)

logger = logging.getLogger(__name__)


class XHSTravelGraphWriter:
    def __init__(self, query_runner: QueryRunner):
        self.query_runner = query_runner

    def ensure_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT route_variant_id IF NOT EXISTS FOR (rv:RouteVariant) REQUIRE rv.id IS UNIQUE",
            "CREATE CONSTRAINT route_segment_id IF NOT EXISTS FOR (seg:RouteSegment) REQUIRE seg.id IS UNIQUE",
            "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (ev:Evidence) REQUIRE ev.id IS UNIQUE",
            "CREATE CONSTRAINT play_mode_id IF NOT EXISTS FOR (pm:PlayMode) REQUIRE pm.id IS UNIQUE",
            "CREATE CONSTRAINT fit_assessment_id IF NOT EXISTS FOR (fa:FitAssessment) REQUIRE fa.id IS UNIQUE",
            "CREATE INDEX place_lookup IF NOT EXISTS FOR (p:Place) ON (p.name, p.run_id)",
            "CREATE INDEX constraint_lookup IF NOT EXISTS FOR (c:Constraint) ON (c.metric, c.value_num, c.unit, c.run_id)",
        ]
        for statement in statements:
            try:
                self.query_runner.query(statement)
            except Exception as exc:
                logger.warning("Neo4j schema statement failed: %s; cypher=%s", exc, statement)

    def write_many(
        self,
        posts: Iterable[XHSPostEvidence],
        facts_by_post: Dict[str, List[RouteVariantFact]],
    ) -> None:
        for post in posts:
            self.write_post(post)
            for fact in facts_by_post.get(post.post_id, []):
                self.write_route_variant(post, fact)

    def write_post(self, post: XHSPostEvidence) -> None:
        self.query_runner.query(
            """
            MERGE (post:Post {id: $post_id})
            SET post.run_id = $run_id,
                post.source_file = $source_file,
                post.result_index = $result_index,
                post.result_count = $result_count,
                post.task = $task,
                post.query = $query,
                post.title = $title,
                post.author = $author,
                post.body = $body,
                post.raw_result = $raw_result,
                post.parse_quality = $parse_quality
            """,
            {
                "post_id": post.post_id,
                "run_id": post.run_id,
                "source_file": post.source_file,
                "result_index": post.result_index,
                "result_count": post.result_count,
                "task": post.task,
                "query": post.query,
                "title": post.title,
                "author": post.author,
                "body": post.body,
                "raw_result": post.raw_result,
                "parse_quality": post.parse_quality,
            },
        )

    def write_route_variant(self, post: XHSPostEvidence, fact: RouteVariantFact) -> None:
        summary = _route_summary(fact)
        self.query_runner.query(
            """
            MERGE (rv:RouteVariant {id: $route_variant_id})
            SET rv.run_id = $run_id,
                rv.post_id = $post_id,
                rv.name = $name,
                rv.destination = $destination,
                rv.places = $places,
                rv.style_tags = $style_tags,
                rv.physical_load_rank = $physical_load_rank,
                rv.duration_min = $duration_min,
                rv.duration_max_min = $duration_max_min,
                rv.cost_min_cny = $cost_min_cny,
                rv.cost_max_cny = $cost_max_cny,
                rv.evidence_span = $evidence_span
            WITH rv
            MATCH (post:Post {id: $post_id})
            MERGE (post)-[:DESCRIBES]->(rv)
            """,
            {
                "route_variant_id": fact.route_variant_id,
                "run_id": fact.run_id,
                "post_id": post.post_id,
                "name": fact.name,
                "destination": fact.destination,
                "places": fact.places,
                "style_tags": fact.style_tags,
                "physical_load_rank": summary["physical_load_rank"],
                "duration_min": summary["duration_min"],
                "duration_max_min": summary["duration_max_min"],
                "cost_min_cny": summary["cost_min_cny"],
                "cost_max_cny": summary["cost_max_cny"],
                "evidence_span": fact.evidence_span,
            },
        )
        for place in fact.places:
            self._write_place_link(fact, place)
        for segment in fact.segments:
            self._write_segment(post, fact, segment)
        for constraint in fact.constraints:
            self._write_constraint(post, fact, constraint)
        for requirement in fact.requirements:
            self._write_requirement(post, fact, requirement)
        for risk in fact.risks:
            self._write_risk(post, fact, risk)

    def write_fit_assessment(self, parent_label: str, parent_id: str, assessment: FitAssessment) -> None:
        if parent_label not in {"RouteVariant", "PlayMode"}:
            raise ValueError("parent_label must be RouteVariant or PlayMode")
        self.query_runner.query(
            f"""
            MERGE (fa:FitAssessment {{id: $assessment_id}})
            SET fa.profile_hash = $profile_hash,
                fa.route_variant_id = $route_variant_id,
                fa.decision = $decision,
                fa.hard_fail = $hard_fail,
                fa.reasons = $reasons,
                fa.required_actions = $required_actions,
                fa.missing_evidence = $missing_evidence,
                fa.evidence_used = $evidence_used
            WITH fa
            MATCH (parent:{parent_label} {{id: $parent_id}})
            MERGE (parent)-[:ASSESSED_AS]->(fa)
            """,
            {
                "assessment_id": assessment.assessment_id,
                "profile_hash": assessment.profile_hash,
                "route_variant_id": assessment.route_variant_id,
                "decision": assessment.decision,
                "hard_fail": assessment.hard_fail,
                "reasons": assessment.reasons,
                "required_actions": assessment.required_actions,
                "missing_evidence": assessment.missing_evidence,
                "evidence_used": assessment.evidence_used,
                "parent_id": parent_id,
            },
        )

    def _write_place_link(self, fact: RouteVariantFact, place: str) -> None:
        self.query_runner.query(
            """
            MERGE (place:Place {name: $place_name, run_id: $run_id})
            WITH place
            MATCH (rv:RouteVariant {id: $route_variant_id})
            MERGE (rv)-[:IN_PLACE]->(place)
            """,
            {
                "place_name": place,
                "run_id": fact.run_id,
                "route_variant_id": fact.route_variant_id,
            },
        )

    def _write_segment(self, post: XHSPostEvidence, fact: RouteVariantFact, segment: RouteSegmentFact) -> None:
        segment_id = stable_id(fact.route_variant_id, "segment", segment.order, segment.evidence_span)
        self.query_runner.query(
            """
            MERGE (seg:RouteSegment {id: $segment_id})
            SET seg.run_id = $run_id,
                seg.order = $order,
                seg.from_place = $from_place,
                seg.to_place = $to_place,
                seg.place_names = $place_names,
                seg.transport_mode = $transport_mode,
                seg.duration_min = $duration_min,
                seg.duration_max_min = $duration_max_min,
                seg.stairs = $stairs,
                seg.extra_cost_cny = $extra_cost_cny,
                seg.physical_load_rank = $physical_load_rank,
                seg.evidence_span = $evidence_span
            WITH seg
            MATCH (rv:RouteVariant {id: $route_variant_id})
            MERGE (rv)-[:HAS_SEGMENT]->(seg)
            """,
            {
                "segment_id": segment_id,
                "run_id": fact.run_id,
                "route_variant_id": fact.route_variant_id,
                "order": segment.order,
                "from_place": segment.from_place,
                "to_place": segment.to_place,
                "place_names": segment.place_names,
                "transport_mode": segment.transport_mode,
                "duration_min": segment.duration_min,
                "duration_max_min": segment.duration_max_min,
                "stairs": segment.stairs,
                "extra_cost_cny": segment.extra_cost_cny,
                "physical_load_rank": segment.physical_load_rank,
                "evidence_span": segment.evidence_span,
            },
        )
        if segment.evidence_span:
            self._write_evidence(post, fact.run_id, segment.evidence_span)

    def _write_constraint(self, post: XHSPostEvidence, fact: RouteVariantFact, item: ConstraintFact) -> str:
        ev_id = self._write_evidence(post, fact.run_id, item.evidence_span or fact.evidence_span)
        cid = constraint_id(fact.route_variant_id, item)
        self.query_runner.query(
            """
            MERGE (c:Constraint {id: $constraint_id})
            SET c.run_id = $run_id,
                c.metric = $metric,
                c.value_num = $value_num,
                c.value_text = $value_text,
                c.unit = $unit,
                c.bound = $bound,
                c.polarity = $polarity
            WITH c
            MATCH (rv:RouteVariant {id: $route_variant_id})
            MERGE (rv)-[:HAS_CONSTRAINT]->(c)
            WITH c
            MATCH (ev:Evidence {id: $evidence_id})
            MERGE (c)-[:SUPPORTED_BY]->(ev)
            """,
            {
                "constraint_id": cid,
                "run_id": fact.run_id,
                "route_variant_id": fact.route_variant_id,
                "metric": item.metric,
                "value_num": item.value_num,
                "value_text": item.value_text,
                "unit": item.unit,
                "bound": item.bound,
                "polarity": item.polarity,
                "evidence_id": ev_id,
            },
        )
        return cid

    def _write_requirement(self, post: XHSPostEvidence, fact: RouteVariantFact, item: RequirementFact) -> str:
        ev_id = self._write_evidence(post, fact.run_id, item.evidence_span or fact.evidence_span)
        rid = requirement_id(fact.route_variant_id, item)
        self.query_runner.query(
            """
            MERGE (req:Requirement {id: $requirement_id})
            SET req.run_id = $run_id,
                req.requirement_type = $requirement_type,
                req.demand = $demand,
                req.magnitude = $magnitude,
                req.unit = $unit
            WITH req
            MATCH (rv:RouteVariant {id: $route_variant_id})
            MERGE (rv)-[:REQUIRES]->(req)
            WITH req
            MATCH (ev:Evidence {id: $evidence_id})
            MERGE (req)-[:SUPPORTED_BY]->(ev)
            """,
            {
                "requirement_id": rid,
                "run_id": fact.run_id,
                "route_variant_id": fact.route_variant_id,
                "requirement_type": item.requirement_type,
                "demand": item.demand,
                "magnitude": item.magnitude,
                "unit": item.unit,
                "evidence_id": ev_id,
            },
        )
        return rid

    def _write_risk(self, post: XHSPostEvidence, fact: RouteVariantFact, item: RiskFact) -> str:
        ev_id = self._write_evidence(post, fact.run_id, item.evidence_span or fact.evidence_span)
        rid = risk_id(fact.route_variant_id, item)
        self.query_runner.query(
            """
            MERGE (risk:Risk {id: $risk_id})
            SET risk.run_id = $run_id,
                risk.risk_type = $risk_type,
                risk.severity = $severity,
                risk.reason = $reason
            WITH risk
            MATCH (rv:RouteVariant {id: $route_variant_id})
            MERGE (rv)-[:HAS_RISK]->(risk)
            WITH risk
            MATCH (ev:Evidence {id: $evidence_id})
            MERGE (risk)-[:SUPPORTED_BY]->(ev)
            """,
            {
                "risk_id": rid,
                "run_id": fact.run_id,
                "route_variant_id": fact.route_variant_id,
                "risk_type": item.risk_type,
                "severity": item.severity,
                "reason": item.reason,
                "evidence_id": ev_id,
            },
        )
        return rid

    def _write_evidence(self, post: XHSPostEvidence, run_id: str, text: str) -> str:
        ev_text = text or post.body[:1000]
        ev_id = evidence_id(post.post_id, ev_text)
        self.query_runner.query(
            """
            MERGE (ev:Evidence {id: $evidence_id})
            SET ev.run_id = $run_id,
                ev.post_id = $post_id,
                ev.text = $text,
                ev.source_file = $source_file,
                ev.result_index = $result_index
            """,
            {
                "evidence_id": ev_id,
                "run_id": run_id,
                "post_id": post.post_id,
                "text": ev_text,
                "source_file": post.source_file,
                "result_index": post.result_index,
            },
        )
        return ev_id


def _route_summary(fact: RouteVariantFact) -> Dict[str, Any]:
    durations_min = [s.duration_min for s in fact.segments if s.duration_min is not None]
    durations_max = [
        s.duration_max_min if s.duration_max_min is not None else s.duration_min
        for s in fact.segments
        if s.duration_max_min is not None or s.duration_min is not None
    ]
    costs = _collect_costs(fact)
    load_ranks = [s.physical_load_rank for s in fact.segments if s.physical_load_rank is not None]
    load_ranks.extend(
        int(c.value_num)
        for c in fact.constraints
        if c.metric == "physical_load_rank" and c.value_num is not None
    )
    return {
        "duration_min": min(durations_min) if durations_min else None,
        "duration_max_min": max(durations_max) if durations_max else None,
        "cost_min_cny": min(costs) if costs else None,
        "cost_max_cny": max(costs) if costs else None,
        "physical_load_rank": max(load_ranks) if load_ranks else None,
    }


def _collect_costs(fact: RouteVariantFact) -> List[float]:
    costs: List[float] = []
    costs.extend(s.extra_cost_cny for s in fact.segments if s.extra_cost_cny is not None)
    costs.extend(c.value_num for c in fact.constraints if c.metric == "extra_cost_cny" and c.value_num is not None)
    return [float(cost) for cost in costs if cost is not None]
