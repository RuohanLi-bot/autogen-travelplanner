"""
Itinerary Pydantic models and travel-time enrichment.

Travel times are computed via the Google Maps Directions API (mode=walking).
If the walking duration exceeds 1 hour the function retries in mode=driving
and labels the result accordingly.  When no API key is available the code
falls back to Nominatim geocode + Haversine walking estimate.
"""

import json
import logging
import os
from math import asin, cos, radians, sin, sqrt
from typing import Dict, List, Literal, Optional, Tuple

import requests
from pydantic import BaseModel

from autogen.agentchat.group import (
    ReplyResult,
    ContextVariables,
    AgentNameTarget,
    StayTarget,
)

logger = logging.getLogger(__name__)

_GOOGLE_MAP_API_KEY = os.environ.get("GOOGLE_MAP_API_KEY")
_WALKING_SPEED_KMH = 5.0
_DIRECTIONS_ENDPOINT = "https://maps.googleapis.com/maps/api/directions/json"
_NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/search"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Event(BaseModel):
    type: Literal["Attraction", "Travel"]
    location: str
    city: str
    description: str


class Day(BaseModel):
    events: List[Event]


class Itinerary(BaseModel):
    days: List[Day]


# ---------------------------------------------------------------------------
# Haversine (fallback only)
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(a))


def _format_haversine(distance_km: float) -> str:
    walk_hours = distance_km / _WALKING_SPEED_KMH
    if walk_hours < 1:
        time_str = f"{int(walk_hours * 60)} mins"
    else:
        h = int(walk_hours)
        m = int((walk_hours - h) * 60)
        time_str = f"{h} hr {m} mins" if m else f"{h} hr"
    dist_str = f"{int(distance_km * 1000)} m" if distance_km < 1 else f"{distance_km:.1f} km"
    return f"By foot (est.): {time_str} ({dist_str})"


# ---------------------------------------------------------------------------
# Google Maps Directions API
# ---------------------------------------------------------------------------

def _fetch_directions(origin: str, destination: str, mode: str) -> dict:
    """Call the Directions API and return the raw JSON, or {} on failure."""
    if not _GOOGLE_MAP_API_KEY:
        return {}
    try:
        resp = requests.get(
            _DIRECTIONS_ENDPOINT,
            params={
                "origin": origin,
                "destination": destination,
                "mode": mode,
                "key": _GOOGLE_MAP_API_KEY,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.debug("Directions API error (%s→%s, %s): %s", origin, destination, mode, exc)
    return {}


def _parse_leg(data: dict) -> Optional[Tuple[int, str, str]]:
    """Extract (duration_secs, duration_text, distance_text) from a Directions response."""
    try:
        leg = data["routes"][0]["legs"][0]
        return leg["duration"]["value"], leg["duration"]["text"], leg["distance"]["text"]
    except (KeyError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Nominatim fallback
# ---------------------------------------------------------------------------

def _geocode_nominatim(location_str: str) -> Optional[Tuple[float, float]]:
    try:
        resp = requests.get(
            _NOMINATIM_ENDPOINT,
            params={"q": location_str, "format": "json", "limit": 1},
            headers={"User-Agent": "travel-planner/1.0"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data:
                return (float(data[0]["lat"]), float(data[0]["lon"]))
    except Exception as exc:
        logger.debug("Nominatim geocode failed for %s: %s", location_str, exc)
    return None


# ---------------------------------------------------------------------------
# Unified travel estimation
# ---------------------------------------------------------------------------

def _estimate_travel(origin: str, destination: str) -> Tuple[Optional[str], str]:
    """Return ``(description, mode)`` for the trip from *origin* to *destination*.

    Mode selection:
    - If Google API is available: try walking first.
      If walking > 60 min, switch to driving and label accordingly.
    - Fallback: Nominatim geocode + Haversine walking estimate.

    Returns
    -------
    description : str | None
        Human-readable travel summary, or None on total failure.
    mode : str
        One of "walking", "driving", "haversine", or "unknown".
    """
    if _GOOGLE_MAP_API_KEY:
        walk_data = _fetch_directions(origin, destination, "walking")
        parsed = _parse_leg(walk_data)
        if parsed:
            secs, dur_text, dist_text = parsed
            if secs <= 3600:
                return f"By foot: {dur_text} ({dist_text})", "walking"
            # Walking > 1 hour → retry with driving
            drive_data = _fetch_directions(origin, destination, "driving")
            drive_parsed = _parse_leg(drive_data)
            if drive_parsed:
                _, ddur, ddist = drive_parsed
                return f"By car: {ddur} ({ddist})", "driving"
            # Driving API also failed; fall through to Nominatim
        else:
            logger.warning("Directions API returned no route: %s → %s", origin, destination)

    # Fallback: Nominatim + Haversine
    origin_coords = _geocode_nominatim(origin)
    dest_coords = _geocode_nominatim(destination)
    if origin_coords and dest_coords:
        dist = _haversine_km(origin_coords[0], origin_coords[1], dest_coords[0], dest_coords[1])
        return _format_haversine(dist), "haversine"

    logger.warning("Could not estimate travel: %s → %s", origin, destination)
    return None, "unknown"


# ---------------------------------------------------------------------------
# Tool: update_itinerary_with_travel_times
# ---------------------------------------------------------------------------

def update_itinerary_with_travel_times(
    context_variables: ContextVariables,
) -> ReplyResult:
    """Insert Travel events between each consecutive Attraction in the itinerary.

    Uses Google Directions API (walking, falling back to driving if > 1 hr).
    Falls back to Haversine estimate when no API key is configured.
    """
    if context_variables.get("structured_itinerary") is None:
        return ReplyResult(
            message=(
                "Structured itinerary not found. "
                "Please create the structured output via structured_output_agent."
            ),
            context_variables=context_variables,
            target=AgentNameTarget("structured_output_agent"),
        )

    if "timed_itinerary" in context_variables:
        return ReplyResult(
            message="Timed itinerary already done, inform the customer that their itinerary is ready!",
            context_variables=context_variables,
            target=StayTarget(),
        )

    itinerary_object = Itinerary.model_validate(
        json.loads(context_variables["structured_itinerary"])
    )

    for day in itinerary_object.days:
        events = day.events
        new_events: List[Event] = []
        for i, cur_event in enumerate(events):
            if i > 0:
                pre_event = events[i - 1]
                origin = f"{pre_event.location}, {pre_event.city}"
                destination = f"{cur_event.location}, {cur_event.city}"

                travel_desc, mode = _estimate_travel(origin, destination)

                if mode == "driving":
                    travel_location = (
                        f"driving from {pre_event.location} to {cur_event.location}"
                    )
                else:
                    travel_location = (
                        f"walking from {pre_event.location} to {cur_event.location}"
                    )

                new_events.append(
                    Event(
                        type="Travel",
                        location=travel_location,
                        city=cur_event.city,
                        description=travel_desc or "Travel time unavailable",
                    )
                )
            new_events.append(cur_event)
        day.events = new_events

    context_variables["timed_itinerary"] = itinerary_object.model_dump()

    return ReplyResult(
        message="Timed itinerary added to context with travel times",
        context_variables=context_variables,
        target=StayTarget(),
    )
