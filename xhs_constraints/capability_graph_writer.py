from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

from xhs_travel_graph.graph_repository import QueryRunner
from xhs_travel_graph.models import TravelerProfile
from xhs_travel_graph.normalizer import stable_id


class CapabilityGraphWriter:
    def __init__(self, query_runner: QueryRunner):
        self.query_runner = query_runner

    def write_traveler_profile(self, *, user_id: str, traveler_profile: TravelerProfile) -> None:
        if not user_id or not traveler_profile.figure:
            return
        figure = sorted(str(item).strip().lower() for item in traveler_profile.figure if str(item).strip())
        profile_id = stable_id("traveler_profile", user_id, json.dumps(figure, ensure_ascii=False))
        self.query_runner.query(
            """
            MERGE (u:User {id: $user_id})
            MERGE (tp:TravelerProfile {id: $profile_id})
            SET tp.figure = $figure,
                tp.destination = $destination,
                tp.user_query = $user_query,
                tp.source = $source,
                tp.updated_at = $updated_at
            MERGE (u)-[:HAS_TRAVELER_PROFILE]->(tp)
            """,
            {
                "user_id": user_id,
                "profile_id": profile_id,
                "figure": figure,
                "destination": traveler_profile.destination,
                "user_query": traveler_profile.user_query,
                "source": traveler_profile.source,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        self.query_runner.query(
            """
            MATCH (tp:TravelerProfile {id: $profile_id})-[rel:HAS_STRENGTH|HAS_BUDGET|HAS_ACTIVITY|HAS_PREFERENCE]->(node)
            DETACH DELETE node
            """,
            {"profile_id": profile_id},
        )
        self.query_runner.query(
            """
            MATCH (tp:TravelerProfile {id: $profile_id})
            MERGE (s:Strength {id: $strength_id})
            SET s.items_json = $strength_json
            MERGE (tp)-[:HAS_STRENGTH]->(s)
            MERGE (b:Budget {id: $budget_id})
            SET b.items_json = $budget_json
            MERGE (tp)-[:HAS_BUDGET]->(b)
            MERGE (a:Activity {id: $activity_id})
            SET a.items_json = $activity_json
            MERGE (tp)-[:HAS_ACTIVITY]->(a)
            MERGE (p:Preference {id: $preference_id})
            SET p.items_json = $preference_json
            MERGE (tp)-[:HAS_PREFERENCE]->(p)
            """,
            {
                "profile_id": profile_id,
                "strength_id": f"{profile_id}:strength",
                "strength_json": json.dumps(traveler_profile.strength or [], ensure_ascii=False, sort_keys=True),
                "budget_id": f"{profile_id}:budget",
                "budget_json": json.dumps(traveler_profile.budget or [], ensure_ascii=False, sort_keys=True),
                "activity_id": f"{profile_id}:activity",
                "activity_json": json.dumps(traveler_profile.activity or [], ensure_ascii=False, sort_keys=True),
                "preference_id": f"{profile_id}:preference",
                "preference_json": json.dumps(traveler_profile.preference or [], ensure_ascii=False, sort_keys=True),
            },
        )

    def load_traveler_profile_by_figure(
        self,
        *,
        user_id: str,
        figure: Sequence[str],
    ) -> Optional[TravelerProfile]:
        canonical_figure = sorted(str(item).strip().lower() for item in figure if str(item).strip())
        if not user_id or not canonical_figure:
            return None
        rows = self.query_runner.query(
            """
            MATCH (u:User {id: $user_id})-[:HAS_TRAVELER_PROFILE]->(tp:TravelerProfile)
            WHERE tp.figure = $figure
            OPTIONAL MATCH (tp)-[:HAS_STRENGTH]->(s:Strength)
            OPTIONAL MATCH (tp)-[:HAS_BUDGET]->(b:Budget)
            OPTIONAL MATCH (tp)-[:HAS_ACTIVITY]->(a:Activity)
            OPTIONAL MATCH (tp)-[:HAS_PREFERENCE]->(p:Preference)
            RETURN tp.id AS id,
                   tp.destination AS destination,
                   tp.user_query AS user_query,
                   tp.source AS source,
                   tp.figure AS figure,
                   s.items_json AS strength_json,
                   b.items_json AS budget_json,
                   a.items_json AS activity_json,
                   p.items_json AS preference_json
            LIMIT 1
            """,
            {"user_id": user_id, "figure": canonical_figure},
        )
        if not rows:
            return None
        row = rows[0]
        strength = self._json_list(row.get("strength_json"))
        budget = self._json_list(row.get("budget_json"))
        activity = self._json_list(row.get("activity_json"))
        preference = self._json_list(row.get("preference_json"))
        return TravelerProfile(
            profile_id=str(row.get("id") or ""),
            destination=str(row.get("destination") or ""),
            user_query=str(row.get("user_query") or ""),
            figure=[str(item).strip().lower() for item in row.get("figure") or [] if str(item).strip()],
            budget=budget,
            strength=[item for item in strength if isinstance(item, dict)],
            activity=[item for item in activity if isinstance(item, dict)],
            preference=[item for item in preference if isinstance(item, dict)],
            source=str(row.get("source") or "graph_reuse") or "graph_reuse",
        )

    @staticmethod
    def _json_list(value: Any) -> list[Any]:
        if not value:
            return []
        if isinstance(value, list):
            return value
        try:
            parsed = json.loads(str(value))
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else []
