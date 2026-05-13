from __future__ import annotations

from typing import Dict, List

from .llm_client import OpenAILLMClient
from .models import NormalizedPOI, ReflectDecision


CORE_FIELDS = ["best_visit_time", "recommended_duration", "opening_hours"]


class ReflectAndRefine:
    def __init__(self, llm_client: OpenAILLMClient):
        self.llm_client = llm_client

    def decide(self, poi: NormalizedPOI, extracted: Dict[str, list]) -> ReflectDecision:
        missing = [field for field in CORE_FIELDS if not extracted.get(field)]
        if missing:
            payload = self.llm_client.generate_json(
                system_prompt=(
                    "Decide whether a POI research pass needs secondary search. "
                    "Return JSON with keys needs_secondary_search, reasons, additional_queries."
                ),
                user_prompt=(
                    f"POI: {poi.model_dump_json()}\n"
                    f"Missing core fields: {missing}\n"
                    f"Extracted fields: {list(extracted.keys())}"
                ),
                default={
                    "needs_secondary_search": True,
                    "reasons": [f"Missing core fields: {', '.join(missing)}"],
                    "additional_queries": [
                        f"{poi.place_name} opening hours",
                        f"{poi.place_name} best time to visit",
                        f"how long to spend at {poi.place_name}",
                    ],
                },
            )
            return ReflectDecision(**payload)
        return ReflectDecision(needs_secondary_search=False, reasons=[], additional_queries=[])

