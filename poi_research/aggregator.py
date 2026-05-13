from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .models import FieldEvidence, NormalizedPOI, POIStructuredInfo


FIELDS = [
    "best_visit_time",
    "recommended_duration",
    "opening_hours",
    "closed_days",
    "reservation_need",
    "ticket_need",
    "physical_intensity",
    "weather_sensitivity",
    "crowd_level",
    "activity_type",
    "itinerary_role",
]


class Aggregator:
    def aggregate(self, poi: NormalizedPOI, extracted: Dict[str, List[FieldEvidence]]) -> POIStructuredInfo:
        output = POIStructuredInfo(
            poi_id=poi.poi_id,
            place_name=poi.place_name,
            category=poi.category,
        )
        evidence_sources = set()
        evidence_spans = defaultdict(list)
        confidence_by_field: Dict[str, float] = {}
        unresolved_questions: List[str] = []
        notes: List[str] = []

        for field in FIELDS:
            candidates = extracted.get(field, [])
            if not candidates:
                unresolved_questions.append(f"Need stronger evidence for {field}")
                continue
            candidates = sorted(candidates, key=lambda item: item.confidence, reverse=True)
            top = candidates[0]
            if field in {"recommended_duration", "best_visit_time"} and len(candidates) > 1:
                unique_values = []
                for item in candidates:
                    if item.value not in unique_values:
                        unique_values.append(item.value)
                value = " / ".join(unique_values[:3])
            else:
                value = top.value
            setattr(output, field, value)
            confidence_by_field[field] = top.confidence
            for item in candidates:
                evidence_sources.add(item.source_url)
                evidence_spans[field].append(
                    {
                        "source_url": item.source_url,
                        "evidence_span": item.evidence_span,
                        "value": item.value,
                    }
                )
            if len({item.value for item in candidates}) > 1:
                notes.append(f"Conflicting evidence retained for {field}")

        if output.recommended_duration == "unknown":
            output.recommended_duration = "1-2h"
            notes.append("Fallback duration range used because evidence was insufficient")

        output.evidence_sources = sorted(evidence_sources)
        output.evidence_spans = dict(evidence_spans)
        output.confidence_by_field = confidence_by_field
        output.unresolved_questions = unresolved_questions
        output.notes = notes
        return output

