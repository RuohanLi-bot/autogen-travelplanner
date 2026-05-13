from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from xhs_travel_graph.models import TravelerProfile

from .models import ItinerarySkeleton, PlanningBudget, ScoredPlayMode, SkeletonDay, SkeletonEvent
from .profile_specs import (
    compare_spec,
    iter_profile_specs,
    projection_value,
    spec_metric_key,
    spec_scope,
    update_presence_values,
    update_trip_numeric_totals,
    actual_value_for_state,
)


TRANSPORT_LABELS = {
    "cable_car": "索道",
    "elevator": "电梯",
    "escalator": "扶梯",
    "shuttle_bus": "景区接驳车",
    "tourist_train": "观光小火车",
    "walking": "步行短段",
    "walking_stairs": "台阶步行",
}


@dataclass
class _PlanState:
    days: List[SkeletonDay] = field(default_factory=list)
    used_places: Set[str] = field(default_factory=set)
    used_play_modes: Set[str] = field(default_factory=set)
    trip_numeric_totals: Dict[str, float] = field(default_factory=dict)
    presence_values: Dict[str, Set[str]] = field(default_factory=dict)
    score: float = 0.0


def optimize_itinerary_from_play_modes(
    *,
    scored_play_modes: List[ScoredPlayMode],
    traveler_profile: TravelerProfile,
    planning_budget: PlanningBudget,
    constraints: PlanningBudget,
    destination: str,
    trip_days: int,
    beam_width: int = 20,
) -> ItinerarySkeleton:
    feasible = [item for item in scored_play_modes if not item.hard_violations]
    if not feasible:
        print(
            "[行程优化] 无可行玩法簇进入优化器："
            + "；".join(
                f"{item.fit.name}=>{'/'.join(issue.constraint_id for issue in item.hard_violations[:3]) or 'hard_blocked'}"
                for item in scored_play_modes[:8]
            ),
            flush=True,
        )
        return _empty_skeleton(destination, trip_days, constraints)

    print(
        "[行程优化] 可行玩法簇："
        + "；".join(f"{item.fit.name}(score={item.total_score:.2f})" for item in feasible[:8]),
        flush=True,
    )

    beams = [_PlanState()]
    for day_index in range(1, trip_days + 1):
        next_beams: List[_PlanState] = []
        for state in beams:
            for scored in feasible:
                day = _play_mode_to_day(
                    scored=scored,
                    day_index=day_index,
                    destination=destination,
                    planning_budget=planning_budget,
                    used_places=state.used_places,
                )
                if day is None:
                    continue
                trip_numeric_totals = update_trip_numeric_totals(state.trip_numeric_totals, day.projected_metrics)
                presence_values = update_presence_values(state.presence_values, day.projected_metrics)
                if _violates_itinerary_level_hard_specs(
                    traveler_profile=traveler_profile,
                    day_projection=day.projected_metrics,
                    trip_numeric_totals=trip_numeric_totals,
                    presence_values=presence_values,
                ):
                    continue
                used_places = set(state.used_places)
                for event in day.events:
                    if event.type == "Attraction":
                        used_places.add(event.location)
                used_play_modes = set(state.used_play_modes)
                used_play_modes.add(scored.fit.play_mode_id)
                next_beams.append(
                    _PlanState(
                        days=state.days + [day],
                        used_places=used_places,
                        used_play_modes=used_play_modes,
                        trip_numeric_totals=trip_numeric_totals,
                        presence_values=presence_values,
                        score=state.score + scored.total_score - day.daily_load_score,
                    )
                )
        if not next_beams:
            print(
                f"[行程优化] Day {day_index} 未扩展出可行 beam，回退休息缓冲。当前已用景点="
                + (",".join(sorted(state.used_places)) if beams and beams[0].used_places else "n/a"),
                flush=True,
            )
            next_beams = [_append_light_day(state, day_index, destination) for state in beams]
        next_beams.sort(key=lambda item: item.score, reverse=True)
        beams = next_beams[:beam_width]
    best = max(beams, key=lambda item: item.score)
    return ItinerarySkeleton(destination=destination, trip_days=trip_days, days=best.days, constraints_used=constraints)


def _play_mode_to_day(
    *,
    scored: ScoredPlayMode,
    day_index: int,
    destination: str,
    planning_budget: PlanningBudget,
    used_places: Set[str],
) -> Optional[SkeletonDay]:
    fit = scored.fit
    template = fit.representative_route_template or {}
    places = [str(place).strip() for place in template.get("places") or [] if str(place).strip()]
    if not places:
        places = [str(place).strip() for place in fit.representative_places if str(place).strip()]
    places = [place for place in places if place not in used_places]
    if not places:
        print(f"[行程优化] 跳过玩法簇 {fit.name}：无可用景点（可能都已使用或模板缺失）。", flush=True)
        return None
    if planning_budget.avoid_cross_scenic_area and len(fit.cost_vector.scenic_systems) > 1:
        print(
            f"[行程优化] 玩法簇 {fit.name} 跨景区系统 {','.join(fit.cost_vector.scenic_systems)}，仅保留首个景点落地。",
            flush=True,
        )
        places = places[:1]
    places = places[: max(1, planning_budget.max_core_places_per_day)]
    events: List[SkeletonEvent] = []
    selected_option = _transport_summary(fit.dominant_transport_modes)
    for idx, place in enumerate(places):
        evidence = _play_mode_evidence(fit, place=place)
        place_modes = _place_transport_modes(fit, place)
        attraction_option = "、".join(place_modes[:3]) if place_modes else selected_option
        if idx > 0:
            events.append(
                SkeletonEvent(
                    type="Travel",
                    location=f"{places[idx - 1]} -> {place}",
                    city=destination,
                    selected_option=selected_option,
                    description_facts=["按玩法簇对应的代表路线顺序衔接，具体交通以景区当日运营为准。"],
                    source_candidate_id=fit.play_mode_id,
                )
            )
        events.append(
            SkeletonEvent(
                type="Attraction",
                location=place,
                city=destination,
                selected_option=attraction_option,
                description_facts=_play_mode_facts(fit, scored, place=place, primary=(idx == 0)),
                must_do=_play_mode_required_actions(fit, planning_budget),
                must_not_do=[],
                evidence=evidence[:2],
                load_score=scored.fatigue_score,
                source_candidate_id=fit.play_mode_id,
            )
        )
    rest_buffer = ""
    if planning_budget.require_rest_buffer:
        rest_buffer = "当天保留休息缓冲，体力下降时优先删减尾部活动。"
        events.append(
            SkeletonEvent(
                type="Rest",
                location="休息缓冲",
                city=destination,
                description_facts=[rest_buffer],
                source_candidate_id=fit.play_mode_id,
            )
        )
    return SkeletonDay(
        day_index=day_index,
        theme=_theme_from_play_mode(fit),
        events=events,
        daily_load_score=scored.fatigue_score,
        estimated_cost_cny=fit.cost_vector.cost_max_cny,
        rest_buffer=rest_buffer,
        source_module=fit.play_mode_id,
        projected_metrics=dict(fit.constraint_projection or {}),
    )


def _violates_itinerary_level_hard_specs(
    *,
    traveler_profile: TravelerProfile,
    day_projection: Dict[str, Any],
    trip_numeric_totals: Dict[str, float],
    presence_values: Dict[str, Set[str]],
) -> bool:
    for spec in iter_profile_specs(traveler_profile):
        if not bool(spec.get("hard", False)):
            continue
        scope = spec_scope(spec)
        if scope == "presence":
            continue
        actual_value = actual_value_for_state(
            spec,
            day_projection=day_projection,
            trip_numeric_totals=trip_numeric_totals,
            presence_values=presence_values,
        )
        passed, _ = compare_spec(spec, actual_value)
        if not passed:
            return True
    return False


def _append_light_day(
    state: _PlanState,
    day_index: int,
    destination: str,
) -> _PlanState:
    day = SkeletonDay(
        day_index=day_index,
        theme="轻松缓冲日",
        events=[
            SkeletonEvent(
                type="Rest",
                location="休息缓冲",
                city=destination,
                description_facts=["缺少足够稳定的低负担玩法变体，当天按轻松休整或现场短线活动处理。"],
            )
        ],
        rest_buffer="保留整段休息缓冲。",
        source_module="rest_buffer",
    )
    return _PlanState(
        days=state.days + [day],
        used_places=set(state.used_places),
        used_play_modes=set(state.used_play_modes),
        trip_numeric_totals=dict(state.trip_numeric_totals),
        presence_values={key: set(values) for key, values in state.presence_values.items()},
        score=state.score - 1.0,
    )


def _empty_skeleton(destination: str, trip_days: int, constraints: PlanningBudget) -> ItinerarySkeleton:
    base_state = _PlanState()
    days = [_append_light_day(base_state, idx, destination).days[0] for idx in range(1, trip_days + 1)]
    return ItinerarySkeleton(destination=destination, trip_days=trip_days, days=days, constraints_used=constraints)


def _transport_summary(modes: List[str]) -> str:
    if not modes:
        return "交通方式需现场核实"
    return "、".join(TRANSPORT_LABELS.get(mode, mode) for mode in modes[:3])


def _play_mode_facts(fit, scored: ScoredPlayMode, *, place: str, primary: bool) -> List[str]:
    multi_system_landing = len(fit.cost_vector.scenic_systems) > 1
    if primary:
        if multi_system_landing:
            facts = [f"参考玩法落地段：{place} 轻量游览线（来源玩法簇：{fit.name}）。"]
        else:
            facts = [f"参考玩法簇：{fit.name}。"]
        if fit.cost_vector.modules and not multi_system_landing:
            facts.append("涉及模块：" + "、".join(fit.cost_vector.modules[:2]) + "。")
        if multi_system_landing:
            facts.append("原玩法簇跨多个景区系统，当前仅落地首个低负担核心景点。")
    else:
        facts = [f"所属玩法簇：{fit.name}。"]
    soft_issues = list(scored.soft_violations)
    if multi_system_landing:
        soft_issues = [issue for issue in soft_issues if issue.constraint_id == "cross_scenic_system"]
    if soft_issues:
        facts.append("注意：" + "；".join(issue.reason for issue in soft_issues[:2]))
    if primary and not multi_system_landing and fit.cost_vector.walk_distance_km is not None:
        facts.append(f"聚合证据估计该玩法约需步行 {fit.cost_vector.walk_distance_km:g} 公里。")
    if primary and not multi_system_landing and fit.cost_vector.stairs_steps is not None:
        facts.append(f"聚合证据估计该玩法涉及约 {fit.cost_vector.stairs_steps} 级台阶。")
    place_modes = _place_transport_modes(fit, place)
    if place_modes:
        facts.append("当前景点建议交通：" + "、".join(place_modes[:3]) + "。")
    elif primary and fit.dominant_transport_modes:
        facts.append("主要交通方式：" + _transport_summary(fit.dominant_transport_modes) + "。")
    return facts


def _play_mode_required_actions(fit, planning_budget: PlanningBudget) -> List[str]:
    actions = []
    if fit.dominant_transport_modes:
        actions.append("建议优先采用帖子中高频出现的交通方式，不临时改成长距离步行。")
    if planning_budget.require_rest_buffer:
        actions.append("当天保留休息缓冲，不追加高强度景点。")
    return actions


def _theme_from_play_mode(fit) -> str:
    places = [str(place).strip() for place in fit.representative_places if str(place).strip()]
    if places:
        return " / ".join(places[:2])
    return fit.name or "路线日"


def _play_mode_evidence(fit, *, place: str) -> List[str]:
    template = fit.representative_route_template or {}
    evidence: List[str] = []
    if template.get("evidence_span"):
        evidence.append(str(template["evidence_span"]))
    for field in ("constraints", "requirements", "risks", "segments"):
        for item in template.get(field) or []:
            if not isinstance(item, dict):
                continue
            text = item.get("evidence") or item.get("evidence_span")
            if text:
                evidence.append(str(text))
    if place and not evidence:
        evidence.append(f"{place} 的玩法证据需结合原始帖子核实。")
    return _dedupe_strings(evidence)


def _place_transport_modes(fit, place: str) -> List[str]:
    template = fit.representative_route_template or {}
    modes: List[str] = []
    for segment in template.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        place_names = [str(item).strip() for item in segment.get("place_names") or [] if str(item).strip()]
        segment_places = {str(segment.get("from_place") or "").strip(), str(segment.get("to_place") or "").strip(), *place_names}
        if place in segment_places:
            mode = str(segment.get("transport_mode") or "").strip()
            if mode:
                modes.append(mode)
    if not modes:
        modes.extend(fit.dominant_transport_modes)
    return _dedupe_strings(modes)


def _dedupe_strings(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
