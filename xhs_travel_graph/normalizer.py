from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Optional, Tuple

from .models import (
    ConstraintFact,
    RequirementFact,
    RiskFact,
    RouteSegmentFact,
    RouteVariantFact,
)


TRANSPORT_SYNONYMS = {
    "快线索道": "cable_car",
    "索道": "cable_car",
    "缆车": "cable_car",
    "百龙天梯": "elevator",
    "百龙电梯": "elevator",
    "穿山扶梯": "escalator",
    "扶梯": "escalator",
    "景区环保车": "shuttle_bus",
    "环保车": "shuttle_bus",
    "高铁": "high_speed_rail",
    "包车": "chartered_car",
    "专车": "private_car",
    "步行": "walking",
    "爬台阶": "walking_stairs",
}

RISK_HINTS = {
    "不累": ("fatigue", "low"),
    "轻松": ("fatigue", "low"),
    "地势平缓": ("mobility", "low"),
    "省力": ("fatigue", "low"),
    "999级台阶": ("fatigue", "high"),
    "暴走": ("fatigue", "high"),
    "特种兵": ("fatigue", "high"),
    "玻璃栈道": ("height_exposure", "medium"),
    "冲浪": ("water_safety", "unknown"),
    "漂流": ("water_safety", "unknown"),
}


def normalize_place_name(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").strip().strip("。；;，,"))


def normalize_transport_mode(text: str) -> str:
    raw = text or ""
    for key, value in TRANSPORT_SYNONYMS.items():
        if key in raw:
            return value
    return "unknown" if not raw.strip() else raw.strip().lower().replace(" ", "_")


def parse_duration_to_minutes(text: str) -> Tuple[Optional[int], Optional[int]]:
    raw = text or ""
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-~到至]\s*(\d+(?:\.\d+)?)\s*(h|小时|分钟|min)?", raw, re.I)
    if range_match:
        start = float(range_match.group(1))
        end = float(range_match.group(2))
        unit = range_match.group(3) or ""
        factor = 60 if unit.lower() in {"h", "小时"} or "小时" in raw else 1
        return int(start * factor), int(end * factor)
    single_match = re.search(r"(\d+(?:\.\d+)?)\s*(h|小时|分钟|min)", raw, re.I)
    if single_match:
        value = float(single_match.group(1))
        unit = single_match.group(2).lower()
        factor = 60 if unit in {"h", "小时"} else 1
        minutes = int(value * factor)
        return minutes, minutes
    return None, None


def parse_money_cny(text: str) -> Optional[float]:
    raw = text or ""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:元|块|rmb|cny)", raw, re.I)
    if not match:
        return None
    return float(match.group(1))


def parse_stairs_count(text: str) -> Optional[int]:
    raw = text or ""
    match = re.search(r"(\d{2,5})\s*(?:级)?\s*(?:台阶|阶)", raw)
    if not match:
        return None
    return int(match.group(1))


def infer_physical_load_rank(text: str) -> Optional[int]:
    raw = text or ""
    if any(token in raw for token in ("不累", "轻松", "省力", "地势平缓")):
        return 1
    if any(token in raw for token in ("亲子", "带娃", "适合小孩", "适合老人")):
        return 2
    if any(token in raw for token in ("徒步", "爬山", "台阶")):
        return 3
    if any(token in raw for token in ("999级台阶", "暴走", "特种兵", "一天打卡")):
        return 4
    return None


def normalize_route_variant(fact: RouteVariantFact) -> RouteVariantFact:
    fact.name = fact.name.strip() or "未命名路线"
    fact.destination = normalize_place_name(fact.destination)
    fact.places = _dedupe_keep_order(normalize_place_name(place) for place in fact.places if place)

    for segment in fact.segments:
        _normalize_segment(segment)
        for place in segment.place_names:
            if place and place not in fact.places:
                fact.places.append(place)

    for constraint in list(fact.constraints):
        _normalize_constraint(constraint)

    _add_facts_from_text(fact, fact.evidence_span)
    for segment in fact.segments:
        _add_facts_from_text(fact, segment.evidence_span)

    fact.style_tags = _dedupe_keep_order(tag.strip().lower().replace(" ", "_") for tag in fact.style_tags if tag)
    return fact


def validate_route_variant(fact: RouteVariantFact) -> Optional[RouteVariantFact]:
    if not fact.evidence_span.strip():
        return None
    if not fact.route_variant_id:
        fact.route_variant_id = stable_id(fact.post_id, fact.name, fact.evidence_span[:120])
    fact.places = [place for place in fact.places if place]
    return fact


def stable_id(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def constraint_id(route_variant_id: str, fact: ConstraintFact) -> str:
    return stable_id(route_variant_id, "constraint", fact.metric, fact.value_text, fact.value_num, fact.evidence_span)


def requirement_id(route_variant_id: str, fact: RequirementFact) -> str:
    return stable_id(route_variant_id, "requirement", fact.requirement_type, fact.demand, fact.magnitude, fact.evidence_span)


def risk_id(route_variant_id: str, fact: RiskFact) -> str:
    return stable_id(route_variant_id, "risk", fact.risk_type, fact.severity, fact.evidence_span)


def evidence_id(post_id: str, text: str) -> str:
    return stable_id(post_id, "evidence", text)


def _normalize_segment(segment: RouteSegmentFact) -> None:
    segment.from_place = normalize_place_name(segment.from_place)
    segment.to_place = normalize_place_name(segment.to_place)
    segment.place_names = _dedupe_keep_order(normalize_place_name(place) for place in segment.place_names if place)
    segment.transport_mode = normalize_transport_mode(segment.transport_mode)
    if segment.duration_min is None and segment.evidence_span:
        segment.duration_min, segment.duration_max_min = parse_duration_to_minutes(segment.evidence_span)
    if segment.stairs is None and segment.evidence_span:
        segment.stairs = parse_stairs_count(segment.evidence_span)
    if segment.extra_cost_cny is None and segment.evidence_span:
        segment.extra_cost_cny = parse_money_cny(segment.evidence_span)
    if segment.physical_load_rank is None and segment.evidence_span:
        segment.physical_load_rank = infer_physical_load_rank(segment.evidence_span)


def _normalize_constraint(fact: ConstraintFact) -> None:
    metric = (fact.metric or "").strip().lower()
    metric_map = {
        "duration": "duration_min",
        "duration_max": "duration_max_min",
        "extra_cost": "extra_cost_cny",
        "cost": "extra_cost_cny",
        "physical_load": "physical_load_rank",
    }
    fact.metric = metric_map.get(metric, metric)
    if fact.value_num is None:
        if fact.metric == "extra_cost_cny":
            fact.value_num = parse_money_cny(fact.value_text or fact.evidence_span)
            fact.unit = fact.unit or "CNY"
        elif fact.metric == "stairs":
            stairs = parse_stairs_count(fact.value_text or fact.evidence_span)
            fact.value_num = float(stairs) if stairs is not None else None
            fact.unit = fact.unit or "steps"


def _add_facts_from_text(fact: RouteVariantFact, text: str) -> None:
    if not text:
        return
    stairs = parse_stairs_count(text)
    if stairs is not None:
        _append_unique_constraint(
            fact.constraints,
            ConstraintFact(
                metric="stairs",
                value_num=float(stairs),
                value_text=str(stairs),
                unit="steps",
                bound="exact",
                polarity="negative",
                evidence_span=text,
            ),
        )
        _append_unique_requirement(
            fact.requirements,
            RequirementFact(
                requirement_type="mobility",
                demand="climb_stairs",
                magnitude=float(stairs),
                unit="steps",
                evidence_span=text,
            ),
        )
        _append_unique_risk(
            fact.risks,
            RiskFact(risk_type="fatigue", severity="high" if stairs >= 500 else "medium", evidence_span=text),
        )
    money = parse_money_cny(text)
    if money is not None:
        _append_unique_constraint(
            fact.constraints,
            ConstraintFact(
                metric="extra_cost_cny",
                value_num=money,
                value_text=str(money),
                unit="CNY",
                bound="exact",
                polarity="negative",
                evidence_span=text,
            ),
        )
    for token, (risk_type, severity) in RISK_HINTS.items():
        if token in text:
            _append_unique_risk(fact.risks, RiskFact(risk_type=risk_type, severity=severity, evidence_span=text))
    load_rank = infer_physical_load_rank(text)
    if load_rank is not None:
        _append_unique_constraint(
            fact.constraints,
            ConstraintFact(
                metric="physical_load_rank",
                value_num=float(load_rank),
                value_text=str(load_rank),
                unit="rank_1_4",
                bound="exact",
                polarity="negative" if load_rank >= 3 else "positive",
                evidence_span=text,
            ),
        )


def _append_unique_constraint(items: List[ConstraintFact], item: ConstraintFact) -> None:
    key = (item.metric, item.value_num, item.value_text, item.unit, item.evidence_span)
    if all((x.metric, x.value_num, x.value_text, x.unit, x.evidence_span) != key for x in items):
        items.append(item)


def _append_unique_requirement(items: List[RequirementFact], item: RequirementFact) -> None:
    key = (item.requirement_type, item.demand, item.magnitude, item.unit, item.evidence_span)
    if all((x.requirement_type, x.demand, x.magnitude, x.unit, x.evidence_span) != key for x in items):
        items.append(item)


def _append_unique_risk(items: List[RiskFact], item: RiskFact) -> None:
    key = (item.risk_type, item.severity, item.evidence_span)
    if all((x.risk_type, x.severity, x.evidence_span) != key for x in items):
        items.append(item)


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
