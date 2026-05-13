from __future__ import annotations

from .models import POIStructuredInfo


class ResultBuilder:
    def build(self, info: POIStructuredInfo) -> POIStructuredInfo:
        if info.recommended_duration == "unknown":
            info.recommended_duration = "1-2h"
        if not info.confidence_by_field:
            info.notes.append("No evidence-backed fields were extracted")
        return info

