from __future__ import annotations

from .llm_client import OpenAILLMClient
from .models import FactorPlan, NormalizedPOI


DEFAULT_FACTORS = [
    "best time to visit",
    "recommended duration",
    "opening hours",
    "reservation need",
    "ticket need",
    "crowd level",
]


class FactorPlanner:
    def __init__(self, llm_client: OpenAILLMClient):
        self.llm_client = llm_client

    def plan(self, poi: NormalizedPOI) -> FactorPlan:
        payload = self.llm_client.generate_json(
            system_prompt=(
                "You select 4 to 6 research factors for a travel POI. "
                "Return JSON with key 'factors'. Keep factors concrete and evidence-friendly."
            ),
            user_prompt=(
                f"POI name: {poi.place_name}\n"
                f"Category: {poi.category}\n"
                f"City: {poi.city or 'unknown'}\n"
                f"Region: {poi.region or 'unknown'}"
            ),
            default={"factors": DEFAULT_FACTORS},
        )
        factors = payload.get("factors") or DEFAULT_FACTORS
        factors = [str(f).strip() for f in factors if str(f).strip()]
        factors = factors[:6] if len(factors) > 6 else factors
        if len(factors) < 4:
            factors = DEFAULT_FACTORS
        return FactorPlan(factors=factors)

