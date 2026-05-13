from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from poi_research.llm_client import OpenAILLMClient
from xhs_travel_graph.graph_repository import QueryRunner
from xhs_travel_graph.models import MatchResult, TravelerProfile

from .models import PlanningBudget, PlayModeCostVector, PlayModeFit
from .profile_specs import metrics_for_projection_request
from .transport_canonicalizer import normalize_transport_mode


def build_play_mode_fits(
    *,
    query_runner: QueryRunner,
    matches: Sequence[MatchResult],
    traveler_profile: TravelerProfile,
    llm_client: Optional[Any] = None,
) -> List[PlayModeFit]:
    client = llm_client or OpenAILLMClient()
    fits: List[PlayModeFit] = []
    for match in matches:
        if match.assessment.decision == "fail" or match.blocked_by_safety_floor:
            continue
        selected = _select_projected_route_variant(
            query_runner=query_runner,
            route_variant_ids=match.route_variant_ids,
            play_mode_row=match.raw or {},
            traveler_profile=traveler_profile,
            llm_client=client,
        )
        representative_route_variant_id = str(selected.get("route_variant_id") or "")
        representative_template = selected.get("template") or {}
        projection = selected.get("projection") or {}
        cost_vector = _build_cost_vector(match.raw or {}, representative_template)
        fits.append(
            PlayModeFit(
                play_mode_id=match.play_mode_id,
                name=match.name,
                destination=str((match.raw or {}).get("destination") or ""),
                representative_places=[str(v) for v in (match.raw or {}).get("representative_places") or [] if str(v).strip()],
                dominant_transport_modes=[str(v) for v in (match.raw or {}).get("dominant_transport_modes") or [] if str(v).strip()],
                style_tags=[str(v) for v in (match.raw or {}).get("style_tags") or [] if str(v).strip()],
                route_variant_ids=list(match.route_variant_ids),
                representative_route_variant_id=representative_route_variant_id,
                representative_route_template=representative_template,
                support_confidence=_support_confidence(match, representative_template),
                evidence_count=int(match.evidence_count or 0),
                selected_scenario="default",
                cost_vector=cost_vector,
                constraint_projection=projection,
                raw={
                    "match": match.model_dump(),
                    "play_mode": match.raw,
                },
            )
        )
    return fits


def summarize_play_mode_fits(fits: Sequence[PlayModeFit]) -> str:
    if not fits:
        return "无可拟合的玩法簇"
    return "；".join(
        f"{fit.name}[场景={fit.selected_scenario}, 置信度={fit.support_confidence:.2f}, 点数={len(fit.representative_places)}]"
        for fit in list(fits)[:5]
    )


def format_play_mode_fit_details(fits: Sequence[PlayModeFit]) -> str:
    if not fits:
        return "无玩法拟合明细"
    lines = ["[玩法拟合明细]"]
    for idx, fit in enumerate(fits, 1):
        vector = fit.cost_vector
        lines.extend(
            [
                f"{idx}. {fit.name}",
                f"   scenario={fit.selected_scenario}, support_confidence={fit.support_confidence:.2f}, evidence_count={fit.evidence_count}",
                f"   places={','.join(fit.representative_places) or 'n/a'}",
                f"   systems={','.join(vector.scenic_systems) or 'n/a'}, modules={','.join(vector.modules) or 'n/a'}",
                f"   cost_vector=walk={vector.walk_distance_km if vector.walk_distance_km is not None else 'n/a'}km,"
                f" continuous={vector.continuous_walk_min if vector.continuous_walk_min is not None else 'n/a'}min,"
                f" stairs={vector.stairs_steps if vector.stairs_steps is not None else 'n/a'}steps,"
                f" active={vector.active_hours if vector.active_hours is not None else 'n/a'}h,"
                f" queue={vector.queue_time_min if vector.queue_time_min is not None else 'n/a'}min",
                f"   projection={json_safe_dict(fit.constraint_projection)}",
                f"   transport_modes={','.join(fit.dominant_transport_modes) or 'n/a'}",
            ]
        )
    return "\n".join(lines)


def _select_projected_route_variant(
    *,
    query_runner: QueryRunner,
    route_variant_ids: Sequence[str],
    play_mode_row: Dict[str, Any],
    traveler_profile: TravelerProfile,
    llm_client: Optional[Any],
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for route_variant_id in route_variant_ids:
        if not route_variant_id:
            continue
        template = _fetch_route_variant_template(query_runner, route_variant_id)
        projection = _build_constraint_projection(
            play_mode_row=play_mode_row,
            route_template=template,
            traveler_profile=traveler_profile,
            llm_client=llm_client,
        )
        candidates.append(
            {
                "route_variant_id": route_variant_id,
                "template": template,
                "projection": projection,
                "score": _projection_completeness_score(projection),
            }
        )
    if candidates:
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[0]
    fallback_projection = _build_constraint_projection(
        play_mode_row=play_mode_row,
        route_template={},
        traveler_profile=traveler_profile,
        llm_client=llm_client,
    )
    return {
        "route_variant_id": next((item for item in route_variant_ids if item), ""),
        "template": {},
        "projection": fallback_projection,
        "score": _projection_completeness_score(fallback_projection),
    }


def _fetch_route_variant_template(query_runner: QueryRunner, route_variant_id: str) -> Dict[str, Any]:
    rows = query_runner.query(
        """
        MATCH (rv:RouteVariant {id: $route_variant_id})
        OPTIONAL MATCH (rv)-[:HAS_SEGMENT]->(seg:RouteSegment)
        OPTIONAL MATCH (rv)-[:HAS_CONSTRAINT]->(c:Constraint)
        OPTIONAL MATCH (rv)-[:REQUIRES]->(req:Requirement)-[:SUPPORTED_BY]->(req_ev:Evidence)
        OPTIONAL MATCH (rv)-[:HAS_RISK]->(risk:Risk)-[:SUPPORTED_BY]->(risk_ev:Evidence)
        OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(ev:Evidence)
        RETURN rv.id AS route_variant_id,
               rv.name AS name,
               rv.destination AS destination,
               rv.places AS places,
               rv.style_tags AS style_tags,
               rv.physical_load_rank AS physical_load_rank,
               rv.duration_min AS duration_min,
               rv.duration_max_min AS duration_max_min,
               rv.cost_min_cny AS cost_min_cny,
               rv.cost_max_cny AS cost_max_cny,
               rv.evidence_span AS evidence_span,
               collect(DISTINCT {
                   order: seg.order,
                   from_place: seg.from_place,
                   to_place: seg.to_place,
                   place_names: seg.place_names,
                   transport_mode: seg.transport_mode,
                   duration_min: seg.duration_min,
                   duration_max_min: seg.duration_max_min,
                   stairs: seg.stairs,
                   extra_cost_cny: seg.extra_cost_cny,
                   physical_load_rank: seg.physical_load_rank,
                   evidence_span: seg.evidence_span
               }) AS segments,
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
               }) AS risks,
        """,
        {"route_variant_id": route_variant_id},
    )
    return rows[0] if rows else {}


def _build_cost_vector(play_mode_row: Dict[str, Any], route_template: Dict[str, Any]) -> PlayModeCostVector:
    evidence = _collect_evidence(play_mode_row, route_template)
    places = [str(v).strip() for v in play_mode_row.get("representative_places") or [] if str(v).strip()]
    if not places:
        places = [str(v).strip() for v in route_template.get("places") or [] if str(v).strip()]
    walk_distance = _parse_walk_km("；".join(evidence))
    stairs_steps = _max_numbers(
        [
            _parse_stairs("；".join(evidence)),
            _as_int(play_mode_row.get("stairs_steps")),
            _as_int(route_template.get("stairs_steps")),
            *[_as_int(seg.get("stairs")) for seg in _clean_dicts(route_template.get("segments"))],
            *[_as_int(item.get("value_num")) for item in _clean_dicts(route_template.get("constraints")) if str(item.get("metric") or "") == "stairs"],
        ]
    )


def _build_constraint_projection(
    *,
    play_mode_row: Dict[str, Any],
    route_template: Dict[str, Any],
    traveler_profile: TravelerProfile,
    llm_client: Optional[Any],
) -> Dict[str, Any]:
    heuristic = _heuristic_projection(play_mode_row, route_template)
    requested_metrics = metrics_for_projection_request(traveler_profile)
    if not requested_metrics:
        return heuristic
    if llm_client is not None and getattr(llm_client, "available", lambda: False)():
        system_prompt = (
            "你是旅行路线量化投影器。"
            "你会收到一条 RouteVariant/PlayMode 的结构化事实和证据，以及当前 TravelerProfile 需要衡量的 metrics。"
            "请只针对请求的 metric_key 输出该路线的实际取值 actual_value。"
            "输出 JSON object，格式为 {\"projection\":[{\"metric_key\":\"\",\"actual_value\":null}]}。"
            "要求：1. 只能输出请求里的 metric_key。2. 无法从事实推断时填 null。"
            "3. 数值型 metric 输出 number；集合型 metric 输出 string 或 array[string]。"
            "4. 不要输出解释文字。"
        )
        payload = {
            "traveler_metrics": requested_metrics,
            "play_mode": play_mode_row,
            "route_variant": route_template,
            "heuristic_projection": heuristic,
        }
        result = llm_client.generate_json(
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            temperature=0.0,
            default={"projection": []},
        )
        llm_projection = _sanitize_projection(result.get("projection") if isinstance(result, dict) else None)
        if llm_projection:
            merged = dict(heuristic)
            merged.update({key: value for key, value in llm_projection.items() if value not in (None, [], "")})
            return merged
    return heuristic


def _heuristic_projection(play_mode_row: Dict[str, Any], route_template: Dict[str, Any]) -> Dict[str, Any]:
    vector = _build_cost_vector(play_mode_row, route_template)
    evidence = _collect_evidence(play_mode_row, route_template)
    text_blob = "；".join(evidence)
    tags = _dedupe_strings(
        [str(v).strip().lower() for v in play_mode_row.get("style_tags") or [] if str(v).strip()]
        + _keyword_tags(text_blob)
    )
    activities = _infer_activity_values(play_mode_row, route_template, text_blob)
    projection: Dict[str, Any] = {
        "daily_budget_cny": vector.cost_max_cny,
        "walk_distance_km": vector.walk_distance_km,
        "continuous_walk_min": vector.continuous_walk_min,
        "stairs_steps": vector.stairs_steps,
        "active_duration_h": vector.active_hours,
        "queue_time_min": vector.queue_time_min,
        "preferred_activities": activities,
        "forbidden_activities": [],
        "preferred_tags": tags,
        "forbidden_tags": [],
        "pace": _infer_pace(tags, vector),
        "budget_level": _infer_budget_level(vector.cost_max_cny),
    }
    return {key: value for key, value in projection.items() if value not in (None, [], "")}


def _sanitize_projection(value: Any) -> Dict[str, Any]:
    if not isinstance(value, list):
        return {}
    out: Dict[str, Any] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        metric_key = str(item.get("metric_key") or "").strip()
        if not metric_key:
            continue
        actual_value = item.get("actual_value")
        if isinstance(actual_value, list):
            actual_value = _dedupe_strings(actual_value)
        elif isinstance(actual_value, str):
            actual_value = actual_value.strip()
        out[metric_key] = actual_value
    return out


def _projection_completeness_score(projection: Dict[str, Any]) -> int:
    return sum(1 for value in projection.values() if value not in (None, [], ""))
    active_minutes = _max_numbers(
        [
            _as_int(play_mode_row.get("duration_max_min")),
            _as_int(route_template.get("duration_max_min")),
            _as_int(route_template.get("duration_min")),
            *[_as_int(seg.get("duration_max_min") or seg.get("duration_min")) for seg in _clean_dicts(route_template.get("segments"))],
            _extract_duration_min("；".join(evidence)),
        ]
    )
    queue_time = _parse_queue_time("；".join(evidence))
    physical_load_rank = _max_numbers(
        [
            _as_int(play_mode_row.get("physical_load_rank")),
            _as_int(route_template.get("physical_load_rank")),
            *[_as_int(seg.get("physical_load_rank")) for seg in _clean_dicts(route_template.get("segments"))],
        ]
    )
    return PlayModeCostVector(
        walk_distance_km=walk_distance,
        continuous_walk_min=active_minutes,
        stairs_steps=stairs_steps,
        active_hours=round(active_minutes / 60.0, 2) if active_minutes is not None else None,
        queue_time_min=queue_time,
        transfer_complexity=max(0, len(places) - 1),
        cost_max_cny=_max_floats(
            [
                _as_float(play_mode_row.get("cost_max_cny")),
                _as_float(route_template.get("cost_max_cny")),
                *[_as_float(seg.get("extra_cost_cny")) for seg in _clean_dicts(route_template.get("segments"))],
            ]
        ),
        physical_load_rank=physical_load_rank,
        modules=_infer_modules(play_mode_row, route_template),
        scenic_systems=_infer_scenic_systems(places),
        tags=_dedupe_strings(
            [str(v).strip().lower() for v in play_mode_row.get("style_tags") or [] if str(v).strip()]
            + _keyword_tags("；".join(evidence))
        ),
    )


def _support_confidence(match: MatchResult, route_template: Dict[str, Any]) -> float:
    base = 0.35
    if match.assessment.decision == "pass":
        base += 0.25
    elif match.assessment.decision == "conditional":
        base += 0.10
    elif match.assessment.decision == "unknown":
        base -= 0.08
    evidence_count = int(match.evidence_count or 0)
    base += min(0.20, evidence_count * 0.03)
    if route_template.get("segments"):
        base += 0.08
    return max(0.05, min(0.95, round(base, 2)))


def _collect_evidence(play_mode_row: Dict[str, Any], route_template: Dict[str, Any]) -> List[str]:
    evidence: List[str] = []
    for item in _clean_dicts(play_mode_row.get("constraints")) + _clean_dicts(play_mode_row.get("requirements")) + _clean_dicts(play_mode_row.get("risks")):
        text = item.get("evidence")
        if text:
            evidence.append(str(text))
    if route_template.get("evidence_span"):
        evidence.append(str(route_template["evidence_span"]))
    for field in ("segments", "constraints", "requirements", "risks"):
        for item in _clean_dicts(route_template.get(field)):
            text = item.get("evidence") or item.get("evidence_span")
            if text:
                evidence.append(str(text))
    return _dedupe_strings(evidence)


def _infer_modules(play_mode_row: Dict[str, Any], route_template: Dict[str, Any]) -> List[str]:
    places = [str(v).strip() for v in play_mode_row.get("representative_places") or [] if str(v).strip()]
    if not places:
        places = [str(v).strip() for v in route_template.get("places") or [] if str(v).strip()]
    if len(places) >= 2:
        return ["/".join(places[:2])]
    if places:
        return [places[0]]
    name = str(play_mode_row.get("name") or route_template.get("name") or "")
    parts = [part.strip() for part in re.split(r"[-/]", name) if part.strip()]
    return parts[:1] or ["综合路线"]


def _keyword_tags(text: str) -> List[str]:
    mapping = {
        "family_friendly": ("亲子", "带娃", "孩子", "儿童", "宝宝"),
        "accessibility": ("适合老人", "少走路", "无障碍", "接驳", "省力"),
        "low_crowd": ("人少", "不排队", "避开人流", "安静"),
        "budget_friendly": ("省钱", "便宜", "人均", "性价比"),
        "relaxed": ("轻松", "不累", "休闲"),
    }
    out = []
    for tag, tokens in mapping.items():
        if any(token in text for token in tokens):
            out.append(tag)
    return out


def _infer_activity_values(play_mode_row: Dict[str, Any], route_template: Dict[str, Any], text: str) -> List[str]:
    corpus = " ".join(
        [
            str(play_mode_row.get("name") or ""),
            str(route_template.get("name") or ""),
            " ".join(str(v) for v in play_mode_row.get("representative_places") or []),
            " ".join(str(v) for v in route_template.get("places") or []),
            text,
        ]
    )
    mapping = {
        "mountain_sightseeing": ("山", "索道", "观景", "天门山", "森林公园", "袁家界", "天子山"),
        "theme_park": ("乐园", "主题公园", "迪士尼", "环球"),
        "museum_culture": ("博物馆", "展馆", "文化", "古城"),
        "waterfront_leisure": ("海边", "沙滩", "江边", "湖边", "海岛"),
        "general_sightseeing": tuple(),
    }
    out = []
    for key, tokens in mapping.items():
        if key == "general_sightseeing":
            continue
        if any(token in corpus for token in tokens):
            out.append(key)
    return out or ["general_sightseeing"]


def _infer_pace(tags: List[str], vector: PlayModeCostVector) -> Optional[str]:
    if "relaxed" in tags or "accessibility" in tags:
        return "relaxed"
    if (vector.physical_load_rank or 0) >= 4 or (vector.stairs_steps or 0) >= 800:
        return "intensive"
    return None


def _infer_budget_level(cost_max_cny: Optional[float]) -> Optional[str]:
    if cost_max_cny is None:
        return None
    if cost_max_cny <= 300:
        return "low"
    if cost_max_cny >= 700:
        return "high"
    return "medium"


SCENIC_SYSTEM_KEYWORDS = {
    "tianmenshan": ("天门山",),
    "forest_park": ("国家森林公园", "森林公园", "袁家界", "天子山", "十里画廊", "金鞭溪", "黄石寨", "杨家界"),
    "grand_canyon": ("大峡谷", "玻璃桥"),
    "yellow_dragon_cave": ("黄龙洞",),
    "qixing_mountain": ("七星山",),
    "qilou72": ("七十二奇楼", "72奇楼"),
}


def _infer_scenic_systems(places: Sequence[str]) -> List[str]:
    systems = []
    for place in places:
        system = _scenic_system_for_place(place)
        if system:
            systems.append(system)
    return _dedupe_strings(systems)


def _scenic_system_for_place(place: str) -> str:
    text = str(place or "").strip()
    if not text:
        return ""
    for system, keywords in SCENIC_SYSTEM_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return system
    return ""


def _parse_walk_km(text: str) -> Optional[float]:
    km_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:公里|km)", text or "", re.I)
    if km_match:
        return float(km_match.group(1))
    step_match = re.search(r"(\d{4,6})\s*(?:步)", text or "")
    if step_match:
        return round(float(step_match.group(1)) / 1500.0, 2)
    return None


def _parse_stairs(text: str) -> Optional[int]:
    match = re.search(r"(\d{2,5})\s*(?:级)?\s*(?:台阶|阶)", text or "")
    return int(match.group(1)) if match else None


def _extract_duration_min(text: str) -> Optional[int]:
    hour = re.search(r"(\d+(?:\.\d+)?)\s*(?:小时|h)", text or "", re.I)
    if hour:
        return int(float(hour.group(1)) * 60)
    minute = re.search(r"(\d{1,3})\s*(?:分钟|min)", text or "", re.I)
    return int(minute.group(1)) if minute else None


def _parse_queue_time(text: str) -> Optional[int]:
    if "排队" not in text:
        return None
    return _extract_duration_min(text)


def _clean_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict) and any(v is not None for v in item.values())]


def _dedupe_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


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


def _max_numbers(values: Sequence[Optional[int]]) -> Optional[int]:
    parsed = [int(v) for v in values if v is not None]
    return max(parsed) if parsed else None


def _max_floats(values: Sequence[Optional[float]]) -> Optional[float]:
    parsed = [float(v) for v in values if v is not None]
    return max(parsed) if parsed else None


def json_safe_dict(value: Dict[str, Any]) -> str:
    if not value:
        return "{}"
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)
