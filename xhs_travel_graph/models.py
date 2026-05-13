from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class XHSPostEvidence(BaseModel):
    post_id: str
    run_id: str = "xhs"
    source_file: str
    result_index: int
    result_count: int
    task: str = ""
    query: str = ""
    title: str = ""
    author: str = ""
    body: str
    raw_result: str
    parse_quality: Literal["high", "medium", "low"] = "medium"


class EvidenceRecord(BaseModel):
    evidence_id: str
    run_id: str = "xhs"
    post_id: str
    source_file: str
    result_index: int
    text: str


class ConstraintFact(BaseModel):
    metric: str
    value_num: Optional[float] = None
    value_text: str = ""
    unit: str = ""
    bound: Literal["exact", "min", "max", "range", "unknown"] = "unknown"
    polarity: Literal["positive", "negative", "neutral"] = "neutral"
    evidence_span: str


class RequirementFact(BaseModel):
    requirement_type: str
    demand: str
    magnitude: Optional[float] = None
    unit: str = ""
    evidence_span: str


class RiskFact(BaseModel):
    risk_type: str
    severity: Literal["low", "medium", "high", "unknown"] = "unknown"
    reason: str = ""
    evidence_span: str


class RouteSegmentFact(BaseModel):
    order: int
    from_place: str = ""
    to_place: str = ""
    place_names: List[str] = Field(default_factory=list)
    transport_mode: str = "unknown"
    duration_min: Optional[int] = None
    duration_max_min: Optional[int] = None
    stairs: Optional[int] = None
    extra_cost_cny: Optional[float] = None
    physical_load_rank: Optional[int] = None
    evidence_span: str = ""


class RouteVariantFact(BaseModel):
    route_variant_id: str
    post_id: str
    run_id: str = "xhs"
    name: str
    destination: str = ""
    places: List[str] = Field(default_factory=list)
    segments: List[RouteSegmentFact] = Field(default_factory=list)
    constraints: List[ConstraintFact] = Field(default_factory=list)
    requirements: List[RequirementFact] = Field(default_factory=list)
    risks: List[RiskFact] = Field(default_factory=list)
    style_tags: List[str] = Field(default_factory=list)
    evidence_span: str


class TravelerProfile(BaseModel):
    profile_id: str = ""
    destination: str = ""
    user_query: str = ""
    figure: List[str] = Field(default_factory=list)
    budget: List[Dict[str, Any]] = Field(default_factory=list)
    strength: List[Dict[str, Any]] = Field(default_factory=list)
    activity: List[Dict[str, Any]] = Field(default_factory=list)
    preference: List[Dict[str, Any]] = Field(default_factory=list)
    source: Literal["query_seed", "graph_reuse", "grounded"] = "query_seed"


class FitAssessment(BaseModel):
    assessment_id: str
    profile_hash: str
    route_variant_id: str
    decision: Literal["pass", "conditional", "fail", "unknown"]
    hard_fail: bool = False
    reasons: List[str] = Field(default_factory=list)
    required_actions: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    evidence_used: List[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    play_mode_id: str
    name: str
    assessment: FitAssessment
    route_variant_ids: List[str] = Field(default_factory=list)
    evidence_count: int = 0
    decision_rank: int = 3
    missing_required_evidence_count: int = 0
    unresolved_risk_count: int = 0
    required_action_count: int = 0
    cost_max_cny: Optional[float] = None
    duration_max_min: Optional[int] = None
    blocked_by_safety_floor: bool = False
    raw: Dict = Field(default_factory=dict)
