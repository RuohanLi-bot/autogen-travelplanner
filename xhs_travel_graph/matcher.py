from __future__ import annotations

from typing import Any, Dict, List, Optional

from .fit_evaluator import FitEvaluator
from .graph_repository import QueryRunner
from .graph_writer import XHSTravelGraphWriter
from .models import MatchResult, TravelerProfile


DECISION_RANK = {"pass": 0, "conditional": 1, "unknown": 2, "fail": 3}


def query_matching_play_modes(
    *,
    query_runner: QueryRunner,
    run_id: str,
    destination: str,
    profile: TravelerProfile,
    llm_client: Optional[Any] = None,
    write_assessments: bool = True,
    include_blocked: bool = False,
    limit: int = 10,
) -> List[MatchResult]:
    rows = query_runner.query(
        """
        MATCH (pm:PlayMode {run_id: $run_id})
        WHERE $destination = "" OR pm.destination = $destination
        OPTIONAL MATCH (pm)-[:CONTAINS]->(rv:RouteVariant)
        OPTIONAL MATCH (rv)-[:HAS_CONSTRAINT]->(c:Constraint)
        OPTIONAL MATCH (rv)-[:REQUIRES]->(req:Requirement)-[:SUPPORTED_BY]->(req_ev:Evidence)
        OPTIONAL MATCH (rv)-[:HAS_RISK]->(risk:Risk)-[:SUPPORTED_BY]->(risk_ev:Evidence)
        OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(ev:Evidence)
        RETURN pm.id AS play_mode_id,
               pm.name AS name,
               pm.destination AS destination,
               pm.representative_places AS representative_places,
               pm.dominant_transport_modes AS dominant_transport_modes,
               pm.style_tags AS style_tags,
               pm.physical_load_rank AS physical_load_rank,
               pm.duration_min AS duration_min,
               pm.duration_max_min AS duration_max_min,
               pm.cost_min_cny AS cost_min_cny,
               pm.cost_max_cny AS cost_max_cny,
               pm.evidence_count AS evidence_count,
               collect(DISTINCT rv.id) AS route_variant_ids,
               collect(DISTINCT {
                   metric: c.metric,
                   value_num: c.value_num,
                   value_text: c.value_text,
                   unit: c.unit,
                   evidence: ev.text
               }) AS constraints,
               collect(DISTINCT {
                   requirement_type: req.requirement_type,
                   demand: req.demand,
                   magnitude: req.magnitude,
                   unit: req.unit,
                   evidence: req_ev.text
               }) AS requirements,
               collect(DISTINCT {
                   risk_type: risk.risk_type,
                   severity: risk.severity,
                   evidence: risk_ev.text
               }) AS risks
        """,
        {"run_id": run_id, "destination": destination},
    )
    return match_play_modes(
        rows=rows,
        profile=profile,
        query_runner=query_runner if write_assessments else None,
        llm_client=llm_client,
        include_blocked=include_blocked,
        limit=limit,
    )


def match_play_modes(
    *,
    rows: List[Dict[str, Any]],
    profile: TravelerProfile,
    query_runner: Optional[QueryRunner] = None,
    llm_client: Optional[Any] = None,
    include_blocked: bool = False,
    limit: int = 10,
) -> List[MatchResult]:
    evaluator = FitEvaluator(llm_client)
    writer = XHSTravelGraphWriter(query_runner) if query_runner is not None else None
    results: List[MatchResult] = []
    for row in rows:
        normalized = _normalize_row(row)
        assessment = evaluator.evaluate_route_variant(profile, normalized)
        blocked = assessment.hard_fail and assessment.decision == "unknown"
        result = MatchResult(
            play_mode_id=str(normalized.get("play_mode_id") or ""),
            name=str(normalized.get("name") or ""),
            assessment=assessment,
            route_variant_ids=[str(item) for item in normalized.get("route_variant_ids", []) if item],
            evidence_count=int(normalized.get("evidence_count") or 0),
            decision_rank=DECISION_RANK[assessment.decision],
            missing_required_evidence_count=len(assessment.missing_evidence),
            unresolved_risk_count=_unresolved_risk_count(normalized),
            required_action_count=len(assessment.required_actions),
            cost_max_cny=_as_float(normalized.get("cost_max_cny")),
            duration_max_min=_as_int(normalized.get("duration_max_min")),
            blocked_by_safety_floor=blocked,
            raw=normalized,
        )
        if writer is not None:
            writer.write_fit_assessment("PlayMode", result.play_mode_id, assessment)
        if include_blocked or (assessment.decision != "fail" and not blocked):
            results.append(result)
    results.sort(key=_sort_key)
    return _dedupe_match_results(results)[:limit]


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for field in ("constraints", "requirements", "risks"):
        out[field] = _clean_dicts(row.get(field))
    out["style_tags"] = row.get("style_tags") or []
    out["route_variant_ids"] = [item for item in row.get("route_variant_ids", []) if item]
    return out


def _clean_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict) and any(v is not None for v in item.values())]


def _unresolved_risk_count(row: Dict[str, Any]) -> int:
    return sum(
        1
        for item in _clean_dicts(row.get("risks"))
        if item.get("severity") in {"high", "unknown"}
    )


def _sort_key(result: MatchResult):
    return (
        result.decision_rank,
        int(result.assessment.hard_fail),
        result.missing_required_evidence_count,
        result.unresolved_risk_count,
        result.required_action_count,
        result.cost_max_cny if result.cost_max_cny is not None else 999999,
        result.duration_max_min if result.duration_max_min is not None else 999999,
        -result.evidence_count,
    )


def _dedupe_match_results(results: List[MatchResult]) -> List[MatchResult]:
    seen = set()
    deduped: List[MatchResult] = []
    for result in results:
        key = _match_semantic_key(result)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _match_semantic_key(result: MatchResult):
    raw = result.raw or {}
    places = tuple(str(item).strip() for item in (raw.get("representative_places") or [])[:2] if str(item).strip())
    modes = tuple(str(item).strip().lower() for item in (raw.get("dominant_transport_modes") or [])[:2] if str(item).strip())
    style_tags = tuple(str(item).strip().lower() for item in (raw.get("style_tags") or [])[:3] if str(item).strip())
    if places or modes or style_tags:
        return (
            str(raw.get("destination") or "").strip(),
            places,
            modes,
            style_tags,
            str(result.name or "").strip(),
        )
    return (str(result.name or "").strip(),)


def _as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    parsed = _as_float(value)
    return int(parsed) if parsed is not None else None
