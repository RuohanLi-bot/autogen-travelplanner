from __future__ import annotations

from typing import Iterable, List

from .models import NormalizedPOI


class POINormalizer:
    def normalize(self, raw_pois: Iterable[dict]) -> List[NormalizedPOI]:
        seen = set()
        normalized: List[NormalizedPOI] = []
        for raw in raw_pois:
            poi = NormalizedPOI(
                poi_id=str(raw["poi_id"]).strip(),
                place_name=" ".join(str(raw["place_name"]).split()),
                category=" ".join(str(raw["category"]).split()),
                city=(" ".join(str(raw.get("city", "")).split()) or None),
                region=(" ".join(str(raw.get("region", "")).split()) or None),
            )
            dedupe_key = (
                poi.place_name.casefold(),
                poi.category.casefold(),
                (poi.city or "").casefold(),
                (poi.region or "").casefold(),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append(poi)
        return normalized

