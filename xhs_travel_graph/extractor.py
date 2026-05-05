from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from .models import (
    ConstraintFact,
    MitigationFact,
    RequirementFact,
    RiskFact,
    RouteAlternativeFact,
    RouteSegmentFact,
    RouteVariantFact,
    XHSPostEvidence,
)
from .normalizer import normalize_route_variant, stable_id, validate_route_variant

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是旅行网页信息结构化抽取器。
只抽取原文有证据支持的旅行事实，不要补造数值，不要直接判断是否推荐。
输出必须是 JSON object，格式为：
{
  "route_variants": [
    {
      "name": "路线或活动名称",
      "destination": "目的地，可为空",
      "places": ["地点"],
      "style_tags": ["relaxed", "family", "budget", "intensive"],
      "segments": [
        {
          "order": 1,
          "from_place": "",
          "to_place": "",
          "place_names": ["地点"],
          "transport_mode": "walking/cable_car/elevator/escalator/shuttle_bus/unknown",
          "duration_min": null,
          "duration_max_min": null,
          "stairs": null,
          "extra_cost_cny": null,
          "physical_load_rank": null,
          "evidence_span": "原文片段"
        }
      ],
      "alternatives": [
        {
          "option_name": "选项名，例如 escalator 或 stairs",
          "constraints": [],
          "requirements": [],
          "risks": [],
          "mitigations": [],
          "evidence_span": "原文片段"
        }
      ],
      "constraints": [
        {"metric": "stairs/duration_min/duration_max_min/extra_cost_cny/physical_load_rank/transport_mode/style", "value_num": null, "value_text": "", "unit": "", "bound": "exact/min/max/range/unknown", "polarity": "positive/negative/neutral", "evidence_span": "原文片段"}
      ],
      "requirements": [
        {"requirement_type": "mobility/water_activity/time/budget/traffic", "demand": "climb_stairs/balance_and_swimming_or_supervision/etc", "magnitude": null, "unit": "", "evidence_span": "原文片段"}
      ],
      "risks": [
        {"risk_type": "fatigue/water_safety/height_exposure/traffic_safety/crowd", "severity": "low/medium/high/unknown", "reason": "", "evidence_span": "原文片段"}
      ],
      "mitigations": [
        {"mitigation_type": "transport_substitution/coach/safety_equipment/shallow_water/lifeguard/official_service", "method": "", "extra_cost_cny": null, "status": "available/unavailable/unknown", "evidence_span": "原文片段"}
      ],
      "evidence_span": "覆盖该路线的原文片段"
    }
  ]
}
如果原文只有单个活动，也创建一个 route_variant；如果证据不足，对应字段留空或 unknown。"""


class XHSTravelFactExtractor:
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def extract(self, post: XHSPostEvidence) -> List[RouteVariantFact]:
        payload: Dict[str, Any] = {}
        if self.llm_client is not None and getattr(self.llm_client, "available", lambda: False)():
            payload = self.llm_client.generate_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=self._build_user_prompt(post),
                temperature=0.0,
                default={},
            )

        variants = self._parse_payload(post, payload)
        if not variants:
            variants = self._fallback_extract(post)
        return variants

    def _build_user_prompt(self, post: XHSPostEvidence) -> str:
        return (
            f"任务/搜索词：{post.query or post.task}\n"
            f"标题：{post.title}\n"
            f"作者：{post.author}\n"
            f"正文：\n{post.body[:6000]}"
        )

    def _parse_payload(self, post: XHSPostEvidence, payload: Dict[str, Any]) -> List[RouteVariantFact]:
        raw_variants = payload.get("route_variants") if isinstance(payload, dict) else []
        if not isinstance(raw_variants, list):
            return []
        out = []
        for raw in raw_variants:
            if not isinstance(raw, dict):
                continue
            variant = self._route_variant_from_dict(post, raw)
            if variant is not None:
                out.append(variant)
        return out

    def _route_variant_from_dict(self, post: XHSPostEvidence, raw: Dict[str, Any]) -> Optional[RouteVariantFact]:
        evidence_span = _clean_text(raw.get("evidence_span")) or post.body[:1000]
        name = _clean_text(raw.get("name")) or post.title or post.query or "小红书路线"
        try:
            variant = RouteVariantFact(
                route_variant_id=_clean_text(raw.get("route_variant_id"))
                or stable_id(post.post_id, name, evidence_span[:160]),
                post_id=post.post_id,
                run_id=post.run_id,
                name=name,
                destination=_clean_text(raw.get("destination")),
                places=_string_list(raw.get("places")),
                segments=[self._segment_from_dict(i, item) for i, item in enumerate(_dicts(raw.get("segments")), 1)],
                alternatives=[self._alternative_from_dict(item) for item in _dicts(raw.get("alternatives"))],
                constraints=[self._constraint_from_dict(item) for item in _dicts(raw.get("constraints"))],
                requirements=[self._requirement_from_dict(item) for item in _dicts(raw.get("requirements"))],
                risks=[self._risk_from_dict(item) for item in _dicts(raw.get("risks"))],
                mitigations=[self._mitigation_from_dict(item) for item in _dicts(raw.get("mitigations"))],
                style_tags=_string_list(raw.get("style_tags")),
                evidence_span=evidence_span,
            )
        except Exception as exc:
            logger.warning("Drop invalid route variant from post %s: %s", post.post_id, exc)
            return None
        return validate_route_variant(normalize_route_variant(variant))

    def _segment_from_dict(self, order: int, raw: Dict[str, Any]) -> RouteSegmentFact:
        return RouteSegmentFact(
            order=_as_int(raw.get("order")) or order,
            from_place=_clean_text(raw.get("from_place")),
            to_place=_clean_text(raw.get("to_place")),
            place_names=_string_list(raw.get("place_names")),
            transport_mode=_clean_text(raw.get("transport_mode")) or "unknown",
            duration_min=_as_int(raw.get("duration_min")),
            duration_max_min=_as_int(raw.get("duration_max_min")),
            stairs=_as_int(raw.get("stairs")),
            extra_cost_cny=_as_float(raw.get("extra_cost_cny")),
            physical_load_rank=_as_int(raw.get("physical_load_rank")),
            evidence_span=_clean_text(raw.get("evidence_span")),
        )

    def _alternative_from_dict(self, raw: Dict[str, Any]) -> RouteAlternativeFact:
        return RouteAlternativeFact(
            option_name=_clean_text(raw.get("option_name")) or "unknown",
            constraints=[self._constraint_from_dict(item) for item in _dicts(raw.get("constraints"))],
            requirements=[self._requirement_from_dict(item) for item in _dicts(raw.get("requirements"))],
            risks=[self._risk_from_dict(item) for item in _dicts(raw.get("risks"))],
            mitigations=[self._mitigation_from_dict(item) for item in _dicts(raw.get("mitigations"))],
            evidence_span=_clean_text(raw.get("evidence_span")),
        )

    def _constraint_from_dict(self, raw: Dict[str, Any]) -> ConstraintFact:
        return ConstraintFact(
            metric=_clean_text(raw.get("metric")) or "unknown",
            value_num=_as_float(raw.get("value_num")),
            value_text=_clean_text(raw.get("value_text")),
            unit=_clean_text(raw.get("unit")),
            bound=_enum(raw.get("bound"), {"exact", "min", "max", "range", "unknown"}, "unknown"),
            polarity=_enum(raw.get("polarity"), {"positive", "negative", "neutral"}, "neutral"),
            evidence_span=_clean_text(raw.get("evidence_span")),
        )

    def _requirement_from_dict(self, raw: Dict[str, Any]) -> RequirementFact:
        return RequirementFact(
            requirement_type=_clean_text(raw.get("requirement_type")) or "unknown",
            demand=_clean_text(raw.get("demand")) or "unknown",
            magnitude=_as_float(raw.get("magnitude")),
            unit=_clean_text(raw.get("unit")),
            evidence_span=_clean_text(raw.get("evidence_span")),
        )

    def _risk_from_dict(self, raw: Dict[str, Any]) -> RiskFact:
        return RiskFact(
            risk_type=_clean_text(raw.get("risk_type")) or "unknown",
            severity=_enum(raw.get("severity"), {"low", "medium", "high", "unknown"}, "unknown"),
            reason=_clean_text(raw.get("reason")),
            evidence_span=_clean_text(raw.get("evidence_span")),
        )

    def _mitigation_from_dict(self, raw: Dict[str, Any]) -> MitigationFact:
        return MitigationFact(
            mitigation_type=_clean_text(raw.get("mitigation_type")) or "unknown",
            method=_clean_text(raw.get("method")),
            extra_cost_cny=_as_float(raw.get("extra_cost_cny")),
            status=_enum(raw.get("status"), {"available", "unavailable", "unknown"}, "unknown"),
            evidence_span=_clean_text(raw.get("evidence_span")),
        )

    def _fallback_extract(self, post: XHSPostEvidence) -> List[RouteVariantFact]:
        places = _guess_places_from_route_text(post.body)
        segments = []
        if places:
            for idx, place in enumerate(places):
                segments.append(
                    RouteSegmentFact(
                        order=idx + 1,
                        place_names=[place],
                        transport_mode="unknown",
                        evidence_span=post.body[:1000],
                    )
                )
        variant = RouteVariantFact(
            route_variant_id=stable_id(post.post_id, post.title or post.query, post.body[:160]),
            post_id=post.post_id,
            run_id=post.run_id,
            name=post.title or post.query or "小红书路线",
            destination=_guess_destination(post),
            places=places,
            segments=segments,
            evidence_span=post.body[:1000],
        )
        normalized = validate_route_variant(normalize_route_variant(variant))
        return [normalized] if normalized else []


def _dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [_clean_text(item) for item in value if _clean_text(item)]


def _as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _as_int(value: Any) -> Optional[int]:
    parsed = _as_float(value)
    return int(parsed) if parsed is not None else None


def _enum(value: Any, allowed: Iterable[str], default: str) -> str:
    raw = _clean_text(value)
    return raw if raw in allowed else default


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _guess_destination(post: XHSPostEvidence) -> str:
    text = f"{post.query} {post.title} {post.body[:200]}"
    for token in ("张家界", "天门山", "武陵源", "长沙", "三亚", "青岛", "厦门"):
        if token in text:
            return token
    return ""


def _guess_places_from_route_text(text: str) -> List[str]:
    for line in text.splitlines():
        if "->" in line or "→" in line or "—" in line:
            line = re.sub(r"^.*?(?:路线|行程安排|推荐路线)[:：]\s*", "", line)
            line = line.lstrip("-•· 　")
            parts = re.split(r"\s*(?:->|→|—|--|➜|到)\s*", line)
            places = [_clean_text(part).strip("，,。；;") for part in parts]
            return [place for place in places if 1 < len(place) <= 24][:12]
    return []
