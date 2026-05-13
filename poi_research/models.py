from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class NormalizedPOI(BaseModel):
    poi_id: str
    place_name: str
    category: str
    city: Optional[str] = None
    region: Optional[str] = None


class ResearchDoc(BaseModel):
    url: str
    title: str = ""
    content: str = ""
    score: float = 0.0


class FieldEvidence(BaseModel):
    value: str
    evidence_span: str
    source_url: str
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedFieldCandidate(BaseModel):
    field_name: str
    candidates: List[FieldEvidence] = Field(default_factory=list)


class FactorPlan(BaseModel):
    factors: List[str] = Field(default_factory=list)


class ReflectDecision(BaseModel):
    needs_secondary_search: bool = False
    reasons: List[str] = Field(default_factory=list)
    additional_queries: List[str] = Field(default_factory=list)


class POIStructuredInfo(BaseModel):
    poi_id: str
    place_name: str
    category: str
    best_visit_time: str = "unknown"
    recommended_duration: str = "unknown"
    opening_hours: str = "unknown"
    closed_days: str = "unknown"
    reservation_need: str = "unknown"
    ticket_need: str = "unknown"
    physical_intensity: Literal["unknown", "low", "medium", "high", "very_high"] = "unknown"
    weather_sensitivity: str = "unknown"
    crowd_level: str = "unknown"
    activity_type: str = "unknown"
    itinerary_role: Literal[
        "unknown",
        "full_day_anchor",
        "half_day_anchor",
        "short_stop",
        "sunset_stop",
        "rain_backup",
        "transit_stop",
    ] = "unknown"
    evidence_sources: List[str] = Field(default_factory=list)
    evidence_spans: Dict[str, List[Dict[str, str]]] = Field(default_factory=dict)
    confidence_by_field: Dict[str, float] = Field(default_factory=dict)
    unresolved_questions: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

