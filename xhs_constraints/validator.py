from __future__ import annotations

from typing import Dict, Set

from itinerary_models import Itinerary
from xhs_travel_graph.models import TravelerProfile

from .models import ItinerarySkeleton, PlanningBudget, ValidationIssue, ValidationReport
from .profile_specs import (
    compare_spec,
    iter_profile_specs,
    normalize_string_list,
    spec_metric_key,
    spec_scope,
    update_presence_values,
    update_trip_numeric_totals,
    actual_value_for_state,
)


REST_LOCATIONS = {"休息缓冲", "轻松缓冲", "休整"}


def validate_final_itinerary(
    *,
    itinerary: Itinerary,
    skeleton: ItinerarySkeleton,
    traveler_profile: TravelerProfile,
    planning_budget: PlanningBudget,
) -> ValidationReport:
    issues = []
    allowed = _skeleton_main_locations(skeleton)
    seen: Set[str] = set()
    rest_count = 0
    full_text = []

    trip_numeric_totals: Dict[str, float] = {}
    presence_values: Dict[str, Set[str]] = {}

    for day in itinerary.days:
        for event in day.events:
            full_text.append(event.description)
            if event.location in REST_LOCATIONS or event.itinerary_role == "rest_buffer":
                rest_count += 1
                continue
            if event.type != "Attraction":
                continue
            if event.location not in allowed:
                issues.append(
                    ValidationIssue(
                        issue_id="added_poi",
                        severity="hard",
                        message=f"最终行程新增了 skeleton 中不存在的主景点：{event.location}",
                    )
                )
            if planning_budget.no_duplicate_main_poi and event.location in seen:
                issues.append(
                    ValidationIssue(
                        issue_id="duplicate_poi",
                        severity="hard",
                        message=f"主景点重复：{event.location}",
                    )
                )
            seen.add(event.location)

    text_blob = "\n".join(full_text)
    for day in skeleton.days:
        attraction_locations = [event.location for event in day.events if event.type == "Attraction"]
        if planning_budget.avoid_cross_scenic_area and len(attraction_locations) > planning_budget.max_core_pois_per_day:
            issues.append(
                ValidationIssue(
                    issue_id="day_density_too_high",
                    severity="hard",
                    message=f"第 {day.day_index} 天景点数量超过当前低负担约束允许的范围。",
                )
            )
        projection = dict(day.projected_metrics or {})
        trip_numeric_totals = update_trip_numeric_totals(trip_numeric_totals, projection)
        presence_values = update_presence_values(presence_values, projection)
        for spec in iter_profile_specs(traveler_profile):
            if not bool(spec.get("hard", False)):
                continue
            scope = spec_scope(spec)
            actual_value = actual_value_for_state(
                spec,
                day_projection=projection,
                trip_numeric_totals=trip_numeric_totals,
                presence_values=presence_values,
            )
            if scope == "presence" and actual_value == []:
                actual_value = normalize_string_list(projection.get(spec_metric_key(spec)))
            passed, reason = compare_spec(spec, actual_value)
            if passed:
                continue
            issues.append(
                ValidationIssue(
                    issue_id=f"traveler_profile_{spec_metric_key(spec)}",
                    severity="hard",
                    message=f"第 {day.day_index} 天未满足画像硬约束：{reason}",
                )
            )
        for event in day.events:
            for action in event.must_do:
                if action and action not in text_blob:
                    issues.append(
                        ValidationIssue(
                            issue_id="missing_required_action",
                            severity="soft",
                            message=f"最终描述缺少必须动作：{action}",
                        )
                    )
            for forbidden in event.must_not_do:
                if forbidden and forbidden in text_blob:
                    issues.append(
                        ValidationIssue(
                            issue_id="forbidden_action_present",
                            severity="hard",
                            message=f"最终描述出现禁止动作：{forbidden}",
                        )
                    )

    if planning_budget.require_rest_buffer and rest_count < max(1, planning_budget.min_rest_blocks_per_day):
        issues.append(
            ValidationIssue(
                issue_id="missing_rest_buffer",
                severity="hard",
                message="当前约束要求休息缓冲，但最终行程未保留对应时段。",
            )
        )
    return ValidationReport(issues=issues)


def _skeleton_main_locations(skeleton: ItinerarySkeleton) -> Set[str]:
    out = set()
    for day in skeleton.days:
        for event in day.events:
            if event.type == "Attraction":
                out.add(event.location)
    return out
