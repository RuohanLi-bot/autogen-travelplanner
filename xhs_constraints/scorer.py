from __future__ import annotations

import math
from typing import Any, Dict, List, Set

from xhs_travel_graph.models import TravelerProfile

from .models import ConstraintViolation, PlanningBudget, PlayModeFit, ScoredPlayMode
from .profile_specs import (
    actual_missing,
    compare_spec,
    iter_profile_specs,
    projection_value,
    spec_is_numeric,
    spec_metric_key,
)


def _relation_hit(places: Set[str], lhs: str, rhs: str) -> bool:
    if not lhs or not rhs:
        return False
    return any(lhs in place or place in lhs for place in places) and any(rhs in place or place in rhs for place in places)


def score_play_modes(
    *,
    play_mode_fits: List[PlayModeFit],
    planning_budget: PlanningBudget,
    traveler_profile: TravelerProfile,
) -> List[ScoredPlayMode]:
    scored = [_score_play_mode(fit, planning_budget, traveler_profile) for fit in play_mode_fits]
    scored.sort(key=lambda item: item.total_score, reverse=True)
    return scored


def _score_play_mode(
    fit: PlayModeFit,
    planning_budget: PlanningBudget,
    traveler_profile: TravelerProfile,
) -> ScoredPlayMode:
    hard: List[ConstraintViolation] = []
    soft: List[ConstraintViolation] = []
    vector = fit.cost_vector
    places = set(fit.representative_places or vector.modules)
    tags = {str(tag).strip().lower() for tag in (vector.tags + fit.style_tags) if str(tag).strip()}

    template_places = [str(place).strip() for place in (fit.representative_route_template or {}).get("places") or [] if str(place).strip()]
    if not template_places and not fit.representative_places:
        hard.append(
            ConstraintViolation(
                constraint_id="missing_places",
                severity="hard",
                reason="该玩法簇缺少可落地的景点模板，不能直接生成行程。",
            )
        )

    if planning_budget.require_rest_buffer and len(vector.scenic_systems) > 1:
        hard.append(
            ConstraintViolation(
                constraint_id="cross_scenic_system",
                severity="hard",
                reason="该玩法簇跨多个独立景区系统，不适合作为老人/儿童轻松游的单日主线。",
            )
        )

    preference_bonus = _evaluate_profile_specs(
        fit=fit,
        traveler_profile=traveler_profile,
        hard=hard,
        soft=soft,
    )

    fit_score = round(max(0.0, min(1.0, fit.support_confidence)), 3)
    fatigue_score = _play_mode_fatigue_score(fit)
    cost_score = min(2.5, (vector.cost_max_cny or 0.0) / (_cost_scale(planning_budget.budget_level)))
    evidence_score = round(
        min(1.0, fit.support_confidence * 0.65 + min(1.0, math.log1p(fit.evidence_count) / math.log1p(12)) * 0.35),
        3,
    )
    coherence_score = _play_mode_coherence_score(fit, planning_budget)
    transport_mode_score = min(0.6, 0.10 * len({mode for mode in fit.dominant_transport_modes if mode and mode != "unknown"}))
    tag_bonus = sum(weight for tag, weight in planning_budget.preferred_candidate_tags.items() if tag in tags)

    total = (
        planning_budget.weights.get("fit", 3.0) * fit_score
        + planning_budget.weights.get("evidence", 1.4) * evidence_score
        + planning_budget.weights.get("coherence", 1.2) * coherence_score
        + transport_mode_score
        + tag_bonus
        + preference_bonus
        - 1.0 * fatigue_score
        - 0.8 * cost_score
        - len(soft) * planning_budget.weights.get("soft_violation", 0.35)
        - len(hard) * planning_budget.weights.get("hard_violation", 1000.0)
    )
    return ScoredPlayMode(
        fit=fit,
        fit_score=fit_score,
        fatigue_score=fatigue_score,
        cost_score=round(cost_score, 3),
        evidence_score=evidence_score,
        coherence_score=coherence_score,
        transport_mode_score=transport_mode_score,
        total_score=round(total, 3),
        hard_violations=hard,
        soft_violations=soft,
    )


def summarize_scored_play_modes(items: List[ScoredPlayMode]) -> str:
    hard_blocked = sum(1 for item in items if item.hard_violations)
    soft_warned = sum(1 for item in items if item.soft_violations)
    return f"已评分={len(items)}，硬约束拦截={hard_blocked}，软约束提醒={soft_warned}"


def format_scored_play_mode_details(items: List[ScoredPlayMode]) -> str:
    if not items:
        return "无候选评分明细"
    lines = ["[候选评分明细]"]
    for idx, item in enumerate(items, 1):
        hard_reasons = "；".join(issue.reason for issue in item.hard_violations[:3]) or "n/a"
        soft_reasons = "；".join(issue.reason for issue in item.soft_violations[:3]) or "n/a"
        lines.extend(
            [
                f"{idx}. {item.fit.name}",
                f"   total={item.total_score:.3f}, fit={item.fit_score:.3f}, fatigue={item.fatigue_score:.3f}, cost={item.cost_score:.3f}, evidence={item.evidence_score:.3f}, coherence={item.coherence_score:.3f}, transport={item.transport_mode_score:.3f}",
                f"   hard_violations={len(item.hard_violations)}, soft_violations={len(item.soft_violations)}",
                f"   hard_reasons={hard_reasons}",
                f"   soft_reasons={soft_reasons}",
            ]
        )
    return "\n".join(lines)


def _evaluate_profile_specs(
    *,
    fit: PlayModeFit,
    traveler_profile: TravelerProfile,
    hard: List[ConstraintViolation],
    soft: List[ConstraintViolation],
) -> float:
    bonus = 0.0
    projection = fit.constraint_projection or {}
    for spec in iter_profile_specs(traveler_profile):
        metric_key = spec_metric_key(spec)
        if not metric_key:
            continue
        actual_value = projection_value(projection, metric_key)
        is_hard = bool(spec.get("hard", False))
        if actual_missing(actual_value):
            violation = ConstraintViolation(
                constraint_id=metric_key,
                severity="hard" if is_hard else "soft",
                reason=f"{metric_key} 缺少可比较的路线量化值。",
            )
            (hard if is_hard else soft).append(violation)
            continue
        passed, reason = compare_spec(spec, actual_value)
        if passed:
            if not spec_is_numeric(spec):
                bonus += 0.18 if is_hard else 0.10
            elif not is_hard:
                bonus += 0.05
            continue
        violation = ConstraintViolation(
            constraint_id=metric_key,
            severity="hard" if is_hard else "soft",
            reason=reason,
            value=_render_value(actual_value),
            limit=_render_value(spec.get("value")),
        )
        (hard if is_hard else soft).append(violation)
    return round(bonus, 3)


def _play_mode_fatigue_score(fit: PlayModeFit) -> float:
    vector = fit.cost_vector
    score = 0.0
    if vector.walk_distance_km is not None:
        score += max(0.0, vector.walk_distance_km / 3.5)
    if vector.stairs_steps is not None:
        score += max(0.0, vector.stairs_steps / 500.0)
    if vector.continuous_walk_min is not None:
        score += max(0.0, vector.continuous_walk_min / 90.0)
    if vector.active_hours is not None:
        score += max(0.0, vector.active_hours / 6.0)
    score += 0.5 * max(0, vector.transfer_complexity - 1)
    if vector.physical_load_rank is not None:
        score += max(0.0, (vector.physical_load_rank - 2) * 0.35)
    return round(max(0.0, score), 3)


def _play_mode_coherence_score(fit: PlayModeFit, planning_budget: PlanningBudget) -> float:
    score = 1.0
    if len(fit.representative_places) > planning_budget.max_core_places_per_day:
        score -= 0.18 * (len(fit.representative_places) - planning_budget.max_core_places_per_day)
    if fit.cost_vector.transfer_complexity >= 2:
        score -= 0.12 * (fit.cost_vector.transfer_complexity - 1)
    if fit.selected_scenario != "default":
        score -= 0.08
    return round(max(0.0, min(1.0, score)), 3)


def _cost_scale(level: str) -> float:
    if level == "low":
        return 350.0
    if level == "high":
        return 700.0
    return 500.0


def _render_value(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)
