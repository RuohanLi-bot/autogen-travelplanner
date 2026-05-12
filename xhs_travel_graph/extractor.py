from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from .models import (
    ConstraintFact,
    RequirementFact,
    RiskFact,
    RouteSegmentFact,
    RouteVariantFact,
    XHSPostEvidence,
)
from .normalizer import normalize_route_variant, stable_id, validate_route_variant

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是旅行网页信息结构化抽取器，只抽取原文有证据支持的旅行事实。
输出必须是 JSON object，格式为：
{
  "route_variants": [
    {
      "name": "路线或活动名称",
      "destination": "目的地，可为空",
      "places": ["路线中出现的地点名称，按原文抽取"],
      "style_tags": ["从原文总结出的风格描述，使用短文本，自由生成，例如：轻松慢游、亲子友好、强体力消耗、等等"],
      "segments": [
        {
          "order": 1,
          "from_place": "",
          "to_place": "",
          "place_names": ["该段涉及的地点名称"],
          "transport_mode": "标准化交通方式：walking/cycling/driving/bus/subway/taxi/boat/cable_car/escalator/stairs/mixed/unknown",
          "duration_min": null,
          "duration_max_min": null,
          "stairs": null,
          "extra_cost_cny": null,
          "physical_load_rank": null,
          "evidence_span": "支持该段信息的原文片段"
        }
      ],
      "constraints": [
        {"metric": "约束指标名称，使用简短自然语言，常见指标仅作参考：stairs/duration_min/duration_max_min/extra_cost_cny/physical_load_rank/transport_mode/style", "value_num": null, "value_text": "", "unit": "", "bound": "exact/min/max/range/unknown", "polarity": "positive/negative/neutral", "evidence_span": "原文片段"}
      ],
      "requirements": [
        {"requirement_type": "对参与者的要求类型，常见指标仅作参考：mobility/water_activity/time/budget/traffic", "demand": "具体要求描述，尽量贴近原文", "magnitude": null, "unit": "", "evidence_span": "原文片段"}
      ],
      "risks": [
        {"risk_type": "风险类型，常见类型仅作参考：fatigue/water_safety/height_exposure/traffic_safety/crowd", "severity": "low/medium/high/unknown", "reason": "风险原因，尽量贴近原文", "evidence_span": "原文片段"}
      ],
      "evidence_span": "覆盖该路线的原文片段"
    }
  ]
}
抽取规则：
1. 只抽取原文明确支持的信息，不要根据常识或经验补充缺失内容。
2. 对于语义空间较大的描述字段，允许模型根据原文自由总结自然语言描述，不要强行映射到少量固定标签。
3. 对于语义空间较小、便于标准化的字段，必须使用受控值或固定量表。
4. style_tags 允许自由生成短描述，重点概括路线风格、节奏和整体体验，不要限定为少数预设词。
5. transport_mode 必须使用列举值；无法判断时填 unknown。
6. physical_load_rank 为整数 1-5，1 表示体力负担最轻，5 表示最重；无法判断时填 null。
7. 如果原文没有明确证据，对应字段填空字符串、空数组或 null，不要猜测。
8. 如果原文只有单个活动，也创建一个 route_variant。
9. evidence_span 必须尽量截取能够直接支持该对象内容的原文片段，优先保留最关键证据。
10. 不要重复填充语义相近字段；同一信息优先放到最贴切的字段中。"""


class XHSTravelFactExtractor:
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def extract(self, post: XHSPostEvidence) -> List[RouteVariantFact]:
        payload = self.llm_client.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(post),
            temperature=0.0,
            default={},
        )
        return self._parse_payload(post, payload)

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
                constraints=[self._constraint_from_dict(item) for item in _dicts(raw.get("constraints"))],
                requirements=[self._requirement_from_dict(item) for item in _dicts(raw.get("requirements"))],
                risks=[self._risk_from_dict(item) for item in _dicts(raw.get("risks"))],
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
