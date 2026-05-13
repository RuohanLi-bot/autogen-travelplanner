from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from xhs_travel_graph.models import TravelerProfile


DAY_NUMERIC_METRICS = {
    "daily_budget_cny",
    "walk_distance_km",
    "continuous_walk_min",
    "stairs_steps",
    "active_duration_h",
    "queue_time_min",
    "elevation_gain_m",
}
TRIP_NUMERIC_METRICS = {"total_trip_budget_cny"}
PRESENCE_ACTIVITY_METRICS = {"preferred_activities", "forbidden_activities"}
PRESENCE_TAG_METRICS = {"preferred_tags", "forbidden_tags"}
PRESENCE_SCALAR_METRICS = {"pace", "budget_level"}


def iter_profile_specs(traveler_profile: TravelerProfile) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for field in ("budget", "strength", "activity", "preference"):
        values = getattr(traveler_profile, field, []) or []
        for item in values:
            if isinstance(item, dict):
                specs.append(item)
    return specs


def profile_spec_metric_keys(traveler_profile: TravelerProfile) -> List[str]:
    seen = set()
    keys: List[str] = []
    for spec in iter_profile_specs(traveler_profile):
        metric_key = str(spec.get("metric_key") or "").strip()
        if not metric_key or metric_key in seen:
            continue
        seen.add(metric_key)
        keys.append(metric_key)
    return keys


def spec_metric_key(spec: Dict[str, Any]) -> str:
    return str(spec.get("metric_key") or "").strip()


def spec_scope(spec: Dict[str, Any]) -> str:
    metric_key = spec_metric_key(spec)
    if metric_key in DAY_NUMERIC_METRICS:
        return "day"
    if metric_key in TRIP_NUMERIC_METRICS:
        return "trip"
    return "presence"


def spec_is_numeric(spec: Dict[str, Any]) -> bool:
    metric_key = spec_metric_key(spec)
    return metric_key in DAY_NUMERIC_METRICS or metric_key in TRIP_NUMERIC_METRICS


def normalize_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = [value]
    seen = set()
    out: List[str] = []
    for item in raw:
        text = str(item or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def projection_value(projection: Dict[str, Any], metric_key: str) -> Any:
    if not isinstance(projection, dict):
        return None
    return projection.get(metric_key)


def actual_missing(value: Any) -> bool:
    return value is None or value == [] or value == ""


def compare_spec(spec: Dict[str, Any], actual_value: Any) -> tuple[bool, str]:
    op = str(spec.get("op") or "").strip()
    expected = spec.get("value")
    metric_key = spec_metric_key(spec)
    if actual_missing(actual_value):
        return False, f"{metric_key} 缺少可比较值。"
    if spec_is_numeric(spec):
        if not isinstance(actual_value, (int, float)) or not isinstance(expected, (int, float)):
            return False, f"{metric_key} 不是可比较数值。"
        actual = float(actual_value)
        target = float(expected)
        if op == "<=":
            return actual <= target, f"{metric_key}={actual:g} 应 <= {target:g}"
        if op == ">=":
            return actual >= target, f"{metric_key}={actual:g} 应 >= {target:g}"
        if op == "==":
            return actual == target, f"{metric_key}={actual:g} 应 == {target:g}"
        return False, f"{metric_key} 不支持数值操作符 {op}"
    expected_values = normalize_string_list(expected)
    actual_values = normalize_string_list(actual_value)
    if op == "in":
        hit = any(item in actual_values for item in expected_values)
        return hit, f"{metric_key} 需要命中 {expected_values}"
    if op == "not_in":
        hit = any(item in actual_values for item in expected_values)
        return not hit, f"{metric_key} 不能命中 {expected_values}"
    if op == "==":
        if isinstance(actual_value, list):
            return False, f"{metric_key} 预期单值比较。"
        actual_text = str(actual_value or "").strip().lower()
        expected_text = str(expected or "").strip().lower()
        return actual_text == expected_text, f"{metric_key}={actual_text} 应 == {expected_text}"
    return False, f"{metric_key} 不支持集合操作符 {op}"


def update_trip_numeric_totals(
    totals: Dict[str, float],
    projection: Dict[str, Any],
) -> Dict[str, float]:
    updated = dict(totals)
    for metric_key in DAY_NUMERIC_METRICS | TRIP_NUMERIC_METRICS:
        value = projection_value(projection, metric_key)
        if isinstance(value, (int, float)):
            updated[metric_key] = updated.get(metric_key, 0.0) + float(value)
    return updated


def update_presence_values(
    current: Dict[str, Set[str]],
    projection: Dict[str, Any],
) -> Dict[str, Set[str]]:
    updated = {key: set(values) for key, values in current.items()}
    for metric_key in PRESENCE_ACTIVITY_METRICS:
        values = normalize_string_list(projection_value(projection, metric_key))
        if values:
            updated.setdefault("activities", set()).update(values)
    for metric_key in PRESENCE_TAG_METRICS:
        values = normalize_string_list(projection_value(projection, metric_key))
        if values:
            updated.setdefault("tags", set()).update(values)
    for metric_key in PRESENCE_SCALAR_METRICS:
        values = normalize_string_list(projection_value(projection, metric_key))
        if values:
            updated.setdefault(metric_key, set()).update(values)
    return updated


def actual_value_for_state(
    spec: Dict[str, Any],
    *,
    day_projection: Optional[Dict[str, Any]] = None,
    trip_numeric_totals: Optional[Dict[str, float]] = None,
    presence_values: Optional[Dict[str, Set[str]]] = None,
) -> Any:
    metric_key = spec_metric_key(spec)
    scope = spec_scope(spec)
    if scope == "day":
        return projection_value(day_projection or {}, metric_key)
    if scope == "trip":
        totals = trip_numeric_totals or {}
        if metric_key == "total_trip_budget_cny":
            return totals.get("daily_budget_cny")
        return totals.get(metric_key)
    presence = presence_values or {}
    if metric_key in PRESENCE_ACTIVITY_METRICS:
        return sorted(presence.get("activities", set()))
    if metric_key in PRESENCE_TAG_METRICS:
        return sorted(presence.get("tags", set()))
    if metric_key in PRESENCE_SCALAR_METRICS:
        return sorted(presence.get(metric_key, set()))
    return None


def presence_projection_seed(projection: Dict[str, Any]) -> Dict[str, Set[str]]:
    return update_presence_values({}, projection)


def metrics_for_projection_request(traveler_profile: TravelerProfile) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for spec in iter_profile_specs(traveler_profile):
        metric_key = spec_metric_key(spec)
        if not metric_key:
            continue
        out.append(
            {
                "metric_key": metric_key,
                "dimension": str(spec.get("dimension") or ""),
                "description": str(spec.get("description") or ""),
            }
        )
    return out
