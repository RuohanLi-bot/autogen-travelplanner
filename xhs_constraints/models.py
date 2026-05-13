from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class CostPolicy(BaseModel):
    budget_level: Literal["unknown", "low", "medium", "high"] = "unknown"
    avoid_unnecessary_paid_options: bool = False


class PlanningBudget(BaseModel):
    budget_id: str
    destination: str = ""
    traveler_profile_id: str = ""
    budget_level: Literal["unknown", "low", "medium", "high"] = "unknown"
    no_duplicate_main_poi: bool = True
    require_transport_between_areas: bool = True
    require_rest_buffer: bool = False
    allow_unknown_option: bool = False
    avoid_cross_scenic_area: bool = False
    max_core_places_per_day: int = 3
    max_core_pois_per_day: int = 3
    min_rest_blocks_per_day: int = 0
    max_places_per_option: int = 4

    required_candidate_tags: List[str] = Field(default_factory=list)
    forbidden_candidate_tags: List[str] = Field(default_factory=list)
    preferred_candidate_tags: Dict[str, float] = Field(default_factory=dict)
    weights: Dict[str, float] = Field(default_factory=dict)
    cost_policy: CostPolicy = Field(default_factory=CostPolicy)
    explanations: List[Dict[str, Any]] = Field(default_factory=list)


class CanonicalTransport(BaseModel):
    canonical_id: str = "unknown"
    confidence: float = 0.0
    reason: str = ""


class ConstraintViolation(BaseModel):
    constraint_id: str
    severity: Literal["hard", "soft"]
    reason: str
    value: Optional[str] = None
    limit: Optional[str] = None


class PlayModeCostVector(BaseModel):
    walk_distance_km: Optional[float] = None
    continuous_walk_min: Optional[int] = None
    stairs_steps: Optional[int] = None
    active_hours: Optional[float] = None
    queue_time_min: Optional[int] = None
    transfer_complexity: int = 0
    cost_max_cny: Optional[float] = None
    physical_load_rank: Optional[int] = None
    modules: List[str] = Field(default_factory=list)
    scenic_systems: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class PlayModeFit(BaseModel):
    play_mode_id: str
    name: str
    destination: str = ""
    representative_places: List[str] = Field(default_factory=list)
    dominant_transport_modes: List[str] = Field(default_factory=list)
    style_tags: List[str] = Field(default_factory=list)
    route_variant_ids: List[str] = Field(default_factory=list)
    representative_route_variant_id: str = ""
    representative_route_template: Dict[str, Any] = Field(default_factory=dict)
    support_confidence: float = 0.0
    evidence_count: int = 0
    selected_scenario: str = "default"
    cost_vector: PlayModeCostVector = Field(default_factory=PlayModeCostVector)
    constraint_projection: Dict[str, Any] = Field(default_factory=dict)
    raw: Dict[str, Any] = Field(default_factory=dict)


class ScoredPlayMode(BaseModel):
    fit: PlayModeFit
    fit_score: float = 0.0
    fatigue_score: float = 0.0
    cost_score: float = 0.0
    evidence_score: float = 0.0
    coherence_score: float = 0.0
    transport_mode_score: float = 0.0
    total_score: float = 0.0
    hard_violations: List[ConstraintViolation] = Field(default_factory=list)
    soft_violations: List[ConstraintViolation] = Field(default_factory=list)


class SkeletonEvent(BaseModel):
    type: Literal["Attraction", "Travel", "Rest", "Optional"] = "Attraction"
    location: str
    city: str
    selected_option: str = ""
    description_facts: List[str] = Field(default_factory=list)
    must_do: List[str] = Field(default_factory=list)
    must_not_do: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    load_score: float = 0.0
    source_candidate_id: str = ""


class SkeletonDay(BaseModel):
    day_index: int
    theme: str = ""
    events: List[SkeletonEvent] = Field(default_factory=list)
    daily_load_score: float = 0.0
    estimated_cost_cny: Optional[float] = None
    rest_buffer: str = ""
    source_module: str = ""
    projected_metrics: Dict[str, Any] = Field(default_factory=dict)


class ItinerarySkeleton(BaseModel):
    destination: str
    trip_days: int
    days: List[SkeletonDay]
    constraints_used: PlanningBudget


class ValidationIssue(BaseModel):
    issue_id: str
    severity: Literal["hard", "soft"] = "soft"
    message: str


class ValidationReport(BaseModel):
    issues: List[ValidationIssue] = Field(default_factory=list)

    @property
    def has_hard_failures(self) -> bool:
        return any(issue.severity == "hard" for issue in self.issues)
