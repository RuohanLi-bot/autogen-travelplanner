from __future__ import annotations

from typing import Dict

from xhs_travel_graph.models import TravelerProfile
from xhs_travel_graph.profile_parser import (
    profile_budget_level,
    profile_has_preference,
    profile_has_role,
    profile_is_mobility_limited,
)

from .models import PlanningBudget


DEFAULT_WEIGHTS = {
    "fit": 3.0,
    "evidence": 1.4,
    "coherence": 1.2,
    "transport": 0.6,
    "soft_violation": 0.35,
    "hard_violation": 1000.0,
    "missing_metric_soft": 0.2,
}


def build_planning_budget(
    *,
    traveler_profile: TravelerProfile,
) -> PlanningBudget:
    budget_level = profile_budget_level(traveler_profile)
    has_senior = profile_has_role(traveler_profile, "senior")
    has_child = profile_has_role(traveler_profile, "child")
    mobility_limited = profile_is_mobility_limited(traveler_profile)
    relaxed = profile_has_preference(traveler_profile, "relaxed")

    require_rest_buffer = relaxed or has_senior or has_child
    avoid_cross = require_rest_buffer or mobility_limited
    max_core_places = 2 if require_rest_buffer else 3

    preferred_tags: Dict[str, float] = {}
    required_tags = []
    forbidden_tags = []

    if relaxed:
        preferred_tags["low_crowd"] = max(preferred_tags.get("low_crowd", 0.0), 0.35)
    if has_senior or mobility_limited:
        preferred_tags["accessibility"] = max(preferred_tags.get("accessibility", 0.0), 0.75)
    if has_child:
        preferred_tags["family_friendly"] = max(preferred_tags.get("family_friendly", 0.0), 0.55)
    if budget_level == "low":
        preferred_tags["budget_friendly"] = max(preferred_tags.get("budget_friendly", 0.0), 0.40)

    return PlanningBudget(
        budget_id=f"{traveler_profile.profile_id}:budget",
        destination=traveler_profile.destination,
        traveler_profile_id=traveler_profile.profile_id,
        budget_level=budget_level,
        require_rest_buffer=require_rest_buffer,
        avoid_cross_scenic_area=avoid_cross,
        max_core_places_per_day=max_core_places,
        max_core_pois_per_day=max_core_places,
        min_rest_blocks_per_day=1 if require_rest_buffer else 0,
        required_candidate_tags=required_tags,
        forbidden_candidate_tags=forbidden_tags,
        preferred_candidate_tags=preferred_tags,
        weights=dict(DEFAULT_WEIGHTS),
        cost_policy={"budget_level": budget_level, "avoid_unnecessary_paid_options": budget_level == "low"},
        explanations=[
            {"reason": f"当前画像预算={budget_level}"},
            {"reason": f"需要休息缓冲={require_rest_buffer}"},
            {"reason": f"避免跨景区系统={avoid_cross}"},
        ],
    )


def summarize_planning_budget(planning_budget: PlanningBudget) -> str:
    tags = ",".join(sorted(planning_budget.preferred_candidate_tags.keys())) or "无"
    return (
        f"休息缓冲={planning_budget.require_rest_buffer}；"
        f"最大核心点={planning_budget.max_core_places_per_day}；"
        f"预算等级={planning_budget.budget_level}；"
        f"偏好tags={tags}"
    )
