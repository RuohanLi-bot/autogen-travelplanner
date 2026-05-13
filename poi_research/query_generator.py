from __future__ import annotations

from typing import List

from .llm_client import OpenAILLMClient
from .models import FactorPlan, NormalizedPOI


class QueryGenerator:
    def __init__(self, llm_client: OpenAILLMClient):
        self.llm_client = llm_client

    def generate(self, poi: NormalizedPOI, factor_plan: FactorPlan) -> List[str]:
        base_location = ", ".join(part for part in [poi.city, poi.region] if part)
        queries: List[str] = []
        for factor in factor_plan.factors:
            queries.append(
                " ".join(
                    part for part in [poi.place_name, base_location, factor] if part
                ).strip()
            )
            queries.append(f"how long to spend at {poi.place_name} {base_location}".strip())

        payload = self.llm_client.generate_json(
            system_prompt=(
                "Generate concise web search queries for researching a POI. "
                "Return JSON with key 'queries'. Avoid duplicates."
            ),
            user_prompt=(
                f"POI: {poi.place_name}\n"
                f"Category: {poi.category}\n"
                f"Location: {base_location or 'unknown'}\n"
                f"Factors: {factor_plan.factors}"
            ),
            default={"queries": []},
        )
        llm_queries = [str(q).strip() for q in payload.get("queries", []) if str(q).strip()]
        deduped = []
        seen = set()
        for query in queries + llm_queries:
            key = query.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(query)
        return deduped[:16]

