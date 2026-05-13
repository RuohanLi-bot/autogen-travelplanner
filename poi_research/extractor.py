from __future__ import annotations

import json
from typing import Dict, Iterable, List

from .llm_client import OpenAILLMClient
from .models import FieldEvidence, NormalizedPOI, ResearchDoc


TARGET_FIELDS = [
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


class EvidenceExtractor:
    def __init__(self, llm_client: OpenAILLMClient):
        self.llm_client = llm_client

    def extract(self, poi: NormalizedPOI, docs: Iterable[ResearchDoc]) -> Dict[str, List[FieldEvidence]]:
        docs = list(docs)
        if not docs:
            return {}
        snippets = []
        for idx, doc in enumerate(docs, 1):
            snippets.append(
                f"[DOC {idx}]\nURL: {doc.url}\nTITLE: {doc.title}\nCONTENT:\n{doc.content[:5000]}"
            )
        payload = self.llm_client.generate_json(
            system_prompt=(
                "Extract evidence-backed POI facts. "
                "Return JSON object mapping each field name to a list of candidates. "
                "Each candidate must contain value, evidence_span, source_url, confidence. "
                "Do not output unsupported fields or claims without evidence."
            ),
            user_prompt=(
                f"POI: {poi.model_dump_json()}\n"
                f"Target fields: {TARGET_FIELDS}\n\n"
                + "\n\n".join(snippets)
            ),
            default={},
        )
        results: Dict[str, List[FieldEvidence]] = {}
        for field_name in TARGET_FIELDS:
            candidates = payload.get(field_name) or []
            parsed: List[FieldEvidence] = []
            for candidate in candidates:
                try:
                    parsed.append(FieldEvidence(**candidate))
                except Exception:
                    continue
            if parsed:
                results[field_name] = parsed
        return results

