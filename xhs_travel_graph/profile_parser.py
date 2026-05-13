from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from poi_research.llm_client import OpenAILLMClient

from .models import TravelerProfile
from .normalizer import stable_id


def parse_traveler_profile(user_text: str, destination: str = "") -> TravelerProfile:
    text = (user_text or "").strip()
    client = OpenAILLMClient()
    payload = _fallback_profile_payload(text)
    if getattr(client, "available", lambda: False)():
        system_prompt = (
            "你是旅行用户画像结构化抽取器。"
            "请仅基于 user_text 抽取 TravelerProfile 所需字段，不要补造原文没有的信息。"
            "输出必须是 JSON object，字段严格限制为："
            '{"figure":[],"budget":{},"strength":{},"activity":[],"preference":[]}。'
            "约束："
            "1. figure 只能输出规范化字符串数组，格式示例：adult、adult:35、child:5、child:12、senior:70:normal、senior:unknown:limited。"
            "2. budget 仅在 query 有明显预算信号时输出，例如 {\"level\":\"low\",\"notes\":[\"预算有限\"]}；否则输出空对象。"
            "3. strength 不能凭空猜测数值；如果 query 没有明确数值或明确强弱描述，输出空对象。"
            "4. activity 只能输出少量稳定活动类型标签，例如 mountain_sightseeing、theme_park、museum_culture、waterfront_leisure、general_sightseeing。"
            "5. preference 只能输出少量偏好标签，例如 relaxed、intensive、avoid_intensive、low_walking。"
            "6. 如果 query 里提到家庭/一家人/带老人孩子，但成人数量不确定，可以保留 adult。"
            "7. 不要输出解释文本，不要输出额外字段。"
        )
        result = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=text,
            temperature=0.0,
            default=payload,
        )
        if isinstance(result, dict):
            payload = result
    return TravelerProfile(
        profile_id=stable_id("traveler_profile_seed", text or "empty"),
        destination=destination,
        user_query=text,
        figure=canonicalize_figures(payload.get("figure") or []),
        budget=_seed_budget_specs(payload.get("budget")),
        strength=_seed_strength_specs(payload.get("strength")),
        activity=_seed_activity_specs(payload.get("activity"), fallback=["general_sightseeing"]),
        preference=_seed_preference_specs(payload.get("preference")),
        source="query_seed",
    )


def canonicalize_figures(figures: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in figures:
        text = str(value or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return sorted(out)


def profile_has_role(profile: TravelerProfile, role: str) -> bool:
    target = str(role or "").strip().lower()
    if not target:
        return False
    return any(_figure_role(item) == target for item in profile.figure)


def profile_max_age(profile: TravelerProfile, role: str) -> Optional[int]:
    ages = [age for age in (_figure_age(item) for item in profile.figure) if age is not None and _figure_role(item) == role]
    return max(ages) if ages else None


def profile_min_age(profile: TravelerProfile, role: str) -> Optional[int]:
    ages = [age for age in (_figure_age(item) for item in profile.figure) if age is not None and _figure_role(item) == role]
    return min(ages) if ages else None


def profile_is_mobility_limited(profile: TravelerProfile) -> bool:
    return any(_figure_mobility(item) == "limited" for item in profile.figure)


def profile_budget_level(profile: TravelerProfile) -> str:
    for item in profile.budget or []:
        if not isinstance(item, dict):
            continue
        metric_key = str(item.get("metric_key") or "").strip()
        value = item.get("value")
        if metric_key == "budget_level" and isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"low", "medium", "high"}:
                return normalized
        if metric_key == "daily_budget_cny" and isinstance(value, (int, float)):
            budget = float(value)
            if budget <= 300:
                return "low"
            if budget >= 700:
                return "high"
            return "medium"
    return "unknown"


def profile_has_preference(profile: TravelerProfile, key: str) -> bool:
    target = str(key or "").strip().lower()
    if not target:
        return False
    for item in profile.preference or []:
        if not isinstance(item, dict):
            continue
        metric_key = str(item.get("metric_key") or "").strip()
        value = item.get("value")
        if metric_key == "pace" and isinstance(value, str) and value.strip().lower() == target:
            return True
        if metric_key in {"preferred_tags", "forbidden_tags"}:
            values = value if isinstance(value, list) else [value]
            normalized_values = {str(entry).strip().lower() for entry in values if str(entry).strip()}
            if target in normalized_values:
                return True
    return False


def profile_activity_key(profile: TravelerProfile) -> str:
    activities = profile_activity_values(profile)
    return activities[0] if activities else "general_sightseeing"


def profile_activity_values(profile: TravelerProfile) -> List[str]:
    values: List[str] = []
    seen = set()
    for item in profile.activity or []:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        entries = value if isinstance(value, list) else [value]
        for entry in entries:
            text = str(entry or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            values.append(text)
    return values


def profile_preference_values(profile: TravelerProfile) -> List[str]:
    values: List[str] = []
    seen = set()
    for item in profile.preference or []:
        if not isinstance(item, dict):
            continue
        raw = item.get("value")
        entries = raw if isinstance(raw, list) else [raw]
        for entry in entries:
            text = str(entry or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            values.append(text)
    return values

def _fallback_profile_payload(text: str) -> Dict[str, object]:
    figure: List[str] = []
    if "老人" in text or "长辈" in text or "父母" in text or "腿脚一般" in text or "腿脚不便" in text:
        mobility = "limited" if ("腿脚一般" in text or "腿脚不便" in text or "少走路" in text) else "normal"
        figure.append(f"senior:unknown:{mobility}")
    if "孩子" in text or "小孩" in text or "儿童" in text or "带娃" in text or "亲子" in text:
        figure.append("child:unknown")
    if "一家" in text or "家庭" in text or figure:
        figure.append("adult")
    budget: Dict[str, object] = {}
    if "预算有限" in text or "省钱" in text or "经济" in text or "穷游" in text:
        budget = {"level": "low", "notes": ["预算有限"]}
    activity = ["general_sightseeing"]
    if any(token in text for token in ("山", "徒步", "爬山", "台阶", "索道", "观景")):
        activity = ["mountain_sightseeing"]
    preference: List[str] = []
    if any(token in text for token in ("轻松", "休闲", "不累", "慢节奏", "少走路")):
        preference.extend(["relaxed", "low_walking"])
    return {"figure": figure, "budget": budget, "strength": {}, "activity": activity, "preference": preference}


def _sanitize_budget(value: object) -> Dict[str, object]:
    if not isinstance(value, dict):
        return {}
    level = str(value.get("level") or "").strip().lower()
    notes = [str(item).strip() for item in value.get("notes") or [] if str(item).strip()]
    out: Dict[str, object] = {}
    if level in {"low", "medium", "high"}:
        out["level"] = level
    if notes:
        out["notes"] = notes
    return out


def _sanitize_strength(value: object) -> Dict[str, Dict[str, object]]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for key, item in value.items():
        metric = str(key or "").strip()
        if not metric or not isinstance(item, dict):
            continue
        payload: Dict[str, object] = {}
        if item.get("soft") is not None:
            payload["soft"] = item.get("soft")
        if item.get("hard") is not None:
            payload["hard"] = item.get("hard")
        unit = str(item.get("unit") or "").strip()
        if unit:
            payload["unit"] = unit
        if item.get("confidence") is not None:
            payload["confidence"] = item.get("confidence")
        source = str(item.get("source") or "").strip()
        if source:
            payload["source"] = source
        if payload:
            out[metric] = payload
    return out


def _sanitize_simple_list(value: object, *, fallback: Optional[List[str]] = None) -> List[str]:
    if not isinstance(value, list):
        return list(fallback or [])
    normalized = canonicalize_figures(value)
    return normalized or list(fallback or [])


def _seed_budget_specs(value: object) -> List[Dict[str, Any]]:
    sanitized = _sanitize_budget(value)
    level = str(sanitized.get("level") or "").strip().lower()
    notes = [str(item).strip() for item in sanitized.get("notes") or [] if str(item).strip()]
    if level not in {"low", "medium", "high"} and not notes:
        return []
    description = notes[0] if notes else "budget level inferred from query"
    return [
        {
            "metric_key": "budget_level",
            "dimension": "budget",
            "op": "==",
            "value": level or "unknown",
            "description": description,
            "hard": False,
        }
    ]


def _seed_strength_specs(value: object) -> List[Dict[str, Any]]:
    sanitized = _sanitize_strength(value)
    specs: List[Dict[str, Any]] = []
    for metric_key, payload in sanitized.items():
        numeric_value = payload.get("hard", payload.get("soft"))
        if numeric_value in (None, ""):
            continue
        specs.append(
            {
                "metric_key": metric_key,
                "dimension": "strength",
                "op": "<=",
                "value": numeric_value,
                "description": f"{metric_key} inferred from query",
                "hard": False,
            }
        )
    return specs


def _seed_activity_specs(value: object, *, fallback: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    activities = _sanitize_simple_list(value, fallback=fallback)
    if not activities:
        return []
    return [
        {
            "metric_key": "preferred_activities",
            "dimension": "activity",
            "op": "in",
            "value": activities,
            "description": "preferred activities inferred from query",
            "hard": False,
        }
    ]


def _seed_preference_specs(value: object) -> List[Dict[str, Any]]:
    preferences = _sanitize_simple_list(value)
    if not preferences:
        return []
    return [
        {
            "metric_key": "preferred_tags",
            "dimension": "preference",
            "op": "in",
            "value": preferences,
            "description": "preferences inferred from query",
            "hard": False,
        }
    ]


def _figure_role(value: str) -> str:
    return str(value or "").split(":", 1)[0].strip().lower()


def _figure_age(value: str) -> Optional[int]:
    parts = str(value or "").split(":")
    if len(parts) < 2 or parts[1] in {"", "unknown"}:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _figure_mobility(value: str) -> str:
    parts = str(value or "").split(":")
    if len(parts) >= 3 and parts[2].strip():
        return parts[2].strip().lower()
    return "unknown"
