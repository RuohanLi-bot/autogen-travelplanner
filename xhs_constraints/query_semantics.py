from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from poi_research.llm_client import OpenAILLMClient
from xhs_travel_graph.models import TravelerProfile


SPEC_DIMENSIONS = {"budget", "strength", "activity", "preference"}
SPEC_OPERATORS = {"<=", ">=", "==", "in", "not_in"}

METRIC_GUIDANCE: Dict[str, Dict[str, Any]] = {
    "daily_budget_cny": {"dimension": "budget", "sample": {"op": "<=", "value": 500, "description": "daily budget in CNY"}},
    "budget_level": {"dimension": "budget", "sample": {"op": "==", "value": "low", "description": "budget level"}},
    "walk_distance_km": {"dimension": "strength", "sample": {"op": "<=", "value": 3.0, "description": "max walking distance per day"}},
    "continuous_walk_min": {"dimension": "strength", "sample": {"op": "<=", "value": 90, "description": "max continuous walking time"}},
    "stairs_steps": {"dimension": "strength", "sample": {"op": "<=", "value": 500, "description": "max stairs steps per day"}},
    "active_duration_h": {"dimension": "strength", "sample": {"op": "<=", "value": 6, "description": "max active hours per day"}},
    "queue_time_min": {"dimension": "strength", "sample": {"op": "<=", "value": 30, "description": "max queue time tolerance"}},
    "preferred_activities": {"dimension": "activity", "sample": {"op": "in", "value": ["mountain_sightseeing"], "description": "preferred activity types"}},
    "forbidden_activities": {"dimension": "activity", "sample": {"op": "not_in", "value": ["intensive_hiking"], "description": "forbidden activity types"}},
    "preferred_tags": {"dimension": "preference", "sample": {"op": "in", "value": ["relaxed", "low_walking"], "description": "preferred travel tags"}},
    "forbidden_tags": {"dimension": "preference", "sample": {"op": "not_in", "value": ["intensive"], "description": "forbidden travel tags"}},
    "pace": {"dimension": "preference", "sample": {"op": "==", "value": "relaxed", "description": "preferred travel pace"}},
}


def generate_figure_mapping_questions(
    *,
    traveler_profile: TravelerProfile,
    llm_client: Optional[object] = None,
) -> List[Dict[str, Any]]:
    client = llm_client or OpenAILLMClient()
    fallback_specs = _fallback_constraint_specs(traveler_profile)
    fallback_payload = {"specs": fallback_specs}
    if getattr(client, "available", lambda: False)():
        system_prompt = (
            "你是旅行用户画像约束规划器。"
            "你要根据 TravelerProfile 中的 figure、destination、user_query，"
            "从 budget、strength、activity、preference 四个维度出发，落成后续需要验证的可计算约束或可打分偏好。"
            "每个 spec 必须是一个 JSON object，字段严格限制为："
            '{"metric_key":"","dimension":"","op":"","value":null,"description":"","hard":false}。'
            "要求："
            "1. dimension 只能从 budget、strength、activity、preference 中选择。"
            "2. op 只能从 <=、>=、==、in、not_in 中选择。"
            "3. value 可以是 number、string、array[string] 或 null。"
            "4. description 必须是简短、可搜索、可指导后续去小红书求证的短语。"
            "5. hard 表示这是硬约束还是软偏好。"
            "6. 最多输出 10 个 spec，优先保留最重要的。"
            "7. strength 优先落到系统已支持的可计算指标，如 walk_distance_km、continuous_walk_min、stairs_steps、active_duration_h、queue_time_min。"
            "8. budget 推荐使用 daily_budget_cny 或 budget_level。"
            "9. activity/preference 推荐使用 preferred_activities、forbidden_activities、preferred_tags、forbidden_tags、pace。"
            "10. 不要输出任何额外字段，不要输出解释文字。"
        )
        payload = {
            "traveler_profile": traveler_profile.model_dump(),
            "metric_guidance": METRIC_GUIDANCE,
            "fallback_specs": fallback_specs,
        }
        result = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            temperature=0.0,
            default=fallback_payload,
        )
        llm_specs = _sanitize_constraint_specs(result.get("specs") if isinstance(result, dict) else None)
        if llm_specs:
            return llm_specs
    return fallback_specs


def ground_constraint_spec_from_raw_result(
    *,
    spec: Dict[str, Any],
    raw_result: str,
    llm_client: Optional[object] = None,
) -> Dict[str, Any]:
    text = str(raw_result or "").strip()
    if not text:
        return dict(spec)
    client = llm_client or OpenAILLMClient()
    fallback = {"value": spec.get("value")}
    if getattr(client, "available", lambda: False)():
        system_prompt = (
            "你是旅行约束求证结果结构化器。"
            "你会收到一个 TravelerProfile 约束 spec，以及 AutoGLM 在小红书中搜索后的自然语言总结。"
            "你的任务是只提取这个 spec 最终应采用的 grounded value。"
            "输出必须是 JSON object，字段严格限制为 {\"value\": ...}。"
            "如果原文无法支持明确值，则尽量保留原 spec 中的 value；如果仍然没有合适值，就输出 null。"
            "对于 activity/preference 的集合型约束，value 可以是字符串列表。"
            "不要输出任何额外字段。"
        )
        payload = {
            "spec": spec,
            "raw_result": text,
        }
        result = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            temperature=0.0,
            default=fallback,
        )
        if isinstance(result, dict) and "value" in result:
            grounded = dict(spec)
            grounded["value"] = _sanitize_spec_value(grounded, result.get("value"))
            return grounded
    grounded = dict(spec)
    grounded["value"] = spec.get("value")
    return grounded


def apply_constraint_specs_to_profile(
    *,
    traveler_profile: TravelerProfile,
    specs: List[Dict[str, Any]],
    source: str = "grounded",
) -> TravelerProfile:
    grouped = {
        "budget": [],
        "strength": [],
        "activity": [],
        "preference": [],
    }
    for item in specs:
        if not isinstance(item, dict):
            continue
        dimension = str(item.get("dimension") or "").strip().lower()
        if dimension in grouped:
            grouped[dimension].append(item)
    return traveler_profile.model_copy(
        update={
            "budget": grouped["budget"],
            "strength": grouped["strength"],
            "activity": grouped["activity"],
            "preference": grouped["preference"],
            "source": source,
        }
    )


def _sanitize_constraint_specs(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sanitized: List[Dict[str, Any]] = []
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        metric_key = str(item.get("metric_key") or "").strip()
        dimension = str(item.get("dimension") or "").strip().lower()
        op = str(item.get("op") or "").strip()
        description = str(item.get("description") or "").strip()
        if not metric_key or not description:
            continue
        if dimension not in SPEC_DIMENSIONS or op not in SPEC_OPERATORS:
            continue
        dedupe_key = (dimension, metric_key, description)
        if dedupe_key in seen:
            continue
        hard = bool(item.get("hard", False))
        sanitized.append(
            {
                "metric_key": metric_key,
                "dimension": dimension,
                "op": op,
                "value": _sanitize_spec_value(item, item.get("value")),
                "description": description,
                "hard": hard,
            }
        )
        seen.add(dedupe_key)
    return sanitized[:10]


def _sanitize_spec_value(spec: Dict[str, Any], value: Any) -> Any:
    op = str(spec.get("op") or "").strip()
    if op in {"in", "not_in"}:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if value in (None, ""):
            return []
        return [str(value).strip()]
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return int(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            return str(value).strip()


def _fallback_constraint_specs(profile: TravelerProfile) -> List[Dict[str, Any]]:
    figure_text = "、".join(profile.figure) or "当前人群"
    specs: List[Dict[str, Any]] = [
        {
            "metric_key": "walk_distance_km",
            "dimension": "strength",
            "op": "<=",
            "value": None,
            "description": f"{figure_text}一天典型步行距离上限",
            "hard": False,
        },
        {
            "metric_key": "stairs_steps",
            "dimension": "strength",
            "op": "<=",
            "value": None,
            "description": f"{figure_text}一天可接受的台阶/爬升上限",
            "hard": True,
        },
        {
            "metric_key": "active_duration_h",
            "dimension": "strength",
            "op": "<=",
            "value": None,
            "description": f"{figure_text}一天总游玩时长上限",
            "hard": False,
        },
    ]
    if "预算" in profile.user_query or "省钱" in profile.user_query or "经济" in profile.user_query:
        specs.append(
            {
                "metric_key": "budget_level",
                "dimension": "budget",
                "op": "==",
                "value": "low",
                "description": f"{figure_text}的预算敏感度",
                "hard": False,
            }
        )
    if "轻松" in profile.user_query or "少走路" in profile.user_query or "休闲" in profile.user_query:
        specs.append(
            {
                "metric_key": "preferred_tags",
                "dimension": "preference",
                "op": "in",
                "value": ["relaxed", "low_walking"],
                "description": f"{figure_text}偏好的游玩节奏和强度",
                "hard": False,
            }
        )
    return specs[:10]
