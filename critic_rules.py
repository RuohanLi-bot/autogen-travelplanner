from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Any, Dict, List, Optional, Tuple


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def _parse_time_hhmm(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%H:%M", "%I:%M %p"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.hour * 60 + dt.minute
        except ValueError:
            continue
    return None


def _parse_duration_range_minutes(value: str) -> Tuple[Optional[int], Optional[int]]:
    if not value or value == "unknown":
        return None, None
    v = value.lower().strip()
    if "half" in v:
        return 180, 300
    if "full" in v:
        return 360, 540
    hour_matches = re.findall(r"(\d+)\s*(?:-|–|to)\s*(\d+)\s*h", v)
    if hour_matches:
        lo, hi = hour_matches[0]
        return int(lo) * 60, int(hi) * 60
    one_hour = re.findall(r"(\d+)\s*h", v)
    if one_hour:
        mins = int(one_hour[0]) * 60
        return mins, mins
    minute_matches = re.findall(r"(\d+)\s*(?:-|–|to)\s*(\d+)\s*m", v)
    if minute_matches:
        lo, hi = minute_matches[0]
        return int(lo), int(hi)
    return None, None


def _extract_opening_bounds(value: str) -> Tuple[Optional[int], Optional[int]]:
    if not value or value == "unknown":
        return None, None
    matches = re.findall(r"(\d{1,2}:\d{2}\s*[APMapm]{0,2})", value)
    if len(matches) >= 2:
        start = _parse_time_hhmm(matches[0].upper().replace("AM", " AM").replace("PM", " PM"))
        end = _parse_time_hhmm(matches[1].upper().replace("AM", " AM").replace("PM", " PM"))
        return start, end
    return None, None


def _day_total_minutes(events: List[Dict[str, Any]]) -> int:
    total = 0
    for event in events:
        if event.get("type") != "Attraction":
            continue
        planned = event.get("planned_duration_minutes")
        if isinstance(planned, int) and planned > 0:
            total += planned
            continue
        start_min = _parse_time_hhmm(event.get("start_time"))
        end_min = _parse_time_hhmm(event.get("end_time"))
        if start_min is not None and end_min is not None and end_min > start_min:
            total += end_min - start_min
    return total


def _parse_duration_text_to_minutes(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    text = str(value).strip().lower()
    total = 0
    matched = False

    hour_matches = re.findall(r"(\d+)\s*(?:hr|hour|hours|h)\b", text)
    minute_matches = re.findall(r"(\d+)\s*(?:min|mins|minute|minutes|m)\b", text)

    for item in hour_matches:
        total += int(item) * 60
        matched = True
    for item in minute_matches:
        total += int(item)
        matched = True
    return total if matched else None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(a))


def _json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def evaluate_itinerary(
    draft_itinerary: str | Dict[str, Any] | None,
    poi_research_results: List[Dict[str, Any]] | None,
    poi_candidates: List[Dict[str, Any]] | None = None,
    *,
    max_iteration: int = 3,
    iteration_count: int = 0,
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    hard_constraints: List[str] = []
    soft_constraints: List[str] = []

    if not draft_itinerary:
        return {
            "status": "revise",
            "summary": "No itinerary draft is available.",
            "hard_constraints": ["A draft itinerary must exist before critique."],
            "soft_constraints": [],
            "issues": [
                {
                    "issue_type": "missing_itinerary",
                    "related_pois": [],
                    "conflict_reason": "draft_itinerary is empty.",
                    "suggested_fix": "Generate an itinerary draft before sending it to the critic.",
                    "severity": "hard",
                }
            ],
        }

    if isinstance(draft_itinerary, str):
        try:
            itinerary = json.loads(draft_itinerary)
        except json.JSONDecodeError:
            return {
                "status": "revise",
                "summary": "Draft itinerary is not valid JSON.",
                "hard_constraints": [
                    "The critic only accepts JSON itinerary drafts matching the Itinerary schema."
                ],
                "soft_constraints": [],
                "issues": [
                    {
                        "issue_type": "invalid_itinerary_format",
                        "related_pois": [],
                        "conflict_reason": "draft_itinerary contains natural-language text instead of structured JSON.",
                        "suggested_fix": (
                            "Reformat the itinerary as JSON with top-level 'days' and nested "
                            "event objects before sending it to the critic."
                        ),
                        "severity": "hard",
                    }
                ],
            }
    else:
        itinerary = draft_itinerary
    poi_research_results = poi_research_results or []
    poi_candidates = poi_candidates or []
    poi_lookup = {_normalize_name(item.get("place_name", "")): item for item in poi_research_results}
    candidate_lookup = {_normalize_name(item.get("name", "")): item for item in poi_candidates}

    days = itinerary.get("days", []) if isinstance(itinerary, dict) else []
    seen_locations = set()
    category_mix = set()

    for day_index, day in enumerate(days, 1):
        events = day.get("events", [])
        attraction_events = [e for e in events if e.get("type") == "Attraction"]

        if not attraction_events:
            issues.append(
                {
                    "issue_type": "empty_day",
                    "related_pois": [],
                    "conflict_reason": f"Day {day_index} does not contain any attraction visit.",
                    "suggested_fix": "Add at least one researched POI or remove the empty day.",
                    "severity": "hard",
                }
            )
            hard_constraints.append("Each itinerary day should contain at least one attraction.")
            continue

        if len(attraction_events) > 5:
            issues.append(
                {
                    "issue_type": "day_overpacked",
                    "related_pois": [e.get("location", "") for e in attraction_events],
                    "conflict_reason": f"Day {day_index} contains {len(attraction_events)} attractions.",
                    "suggested_fix": "Reduce the number of attractions or move some to another day.",
                    "severity": "soft",
                }
            )
            soft_constraints.append(f"Day {day_index} should not feel overpacked.")

        day_minutes = _day_total_minutes(attraction_events)
        if day_minutes > 10 * 60:
            issues.append(
                {
                    "issue_type": "day_duration_too_long",
                    "related_pois": [e.get("location", "") for e in attraction_events],
                    "conflict_reason": f"Day {day_index} schedules {day_minutes} minutes of attraction time.",
                    "suggested_fix": "Move some POIs to another day or shorten non-essential stops.",
                    "severity": "hard",
                }
            )
            hard_constraints.append("Total attraction time per day must stay within a workable window.")

        consecutive_high_intensity = 0
        for event in attraction_events:
            location = event.get("location", "")
            norm_location = _normalize_name(location)
            info = poi_lookup.get(norm_location)
            if not info:
                issues.append(
                    {
                        "issue_type": "missing_poi_research",
                        "related_pois": [location],
                        "conflict_reason": f"No researched profile was found for '{location}'.",
                        "suggested_fix": "Use only researched POIs from graphrag results.",
                        "severity": "hard",
                    }
                )
                hard_constraints.append("Every itinerary POI must come from researched candidates.")
                continue

            if norm_location in seen_locations:
                issues.append(
                    {
                        "issue_type": "duplicate_poi",
                        "related_pois": [location],
                        "conflict_reason": f"'{location}' appears multiple times across the itinerary.",
                        "suggested_fix": "Replace the duplicate with a different researched POI.",
                        "severity": "soft",
                    }
                )
                soft_constraints.append("Avoid duplicate POIs unless the repeat is intentional.")
            seen_locations.add(norm_location)

            category_mix.add((info.get("category") or "unknown").casefold())
            intensity = info.get("physical_intensity", "unknown")
            if intensity in {"high", "very_high"}:
                consecutive_high_intensity += 1
            else:
                consecutive_high_intensity = 0
            if consecutive_high_intensity >= 2:
                issues.append(
                    {
                        "issue_type": "intensity_clash",
                        "related_pois": [e.get("location", "") for e in attraction_events],
                        "conflict_reason": f"Day {day_index} contains back-to-back high intensity POIs.",
                        "suggested_fix": "Swap in a lower intensity stop or insert a shorter break activity.",
                        "severity": "soft",
                    }
                )
                soft_constraints.append("Avoid stacking multiple high-intensity POIs back-to-back.")

            start_min = _parse_time_hhmm(event.get("start_time"))
            end_min = _parse_time_hhmm(event.get("end_time"))
            if start_min is not None and end_min is not None and end_min <= start_min:
                issues.append(
                    {
                        "issue_type": "invalid_time_window",
                        "related_pois": [location],
                        "conflict_reason": f"'{location}' has end_time earlier than start_time.",
                        "suggested_fix": "Correct the visit time window.",
                        "severity": "hard",
                    }
                )
                hard_constraints.append("Each POI must have a valid time window when times are provided.")
            if start_min is not None and start_min < 8 * 60:
                issues.append(
                    {
                        "issue_type": "start_too_early",
                        "related_pois": [location],
                        "conflict_reason": f"'{location}' starts before 08:00.",
                        "suggested_fix": "Shift the visit later unless there is strong evidence for an early start.",
                        "severity": "soft",
                    }
                )
                soft_constraints.append("Avoid very early attraction starts by default.")
            if end_min is not None and end_min > 21 * 60:
                issues.append(
                    {
                        "issue_type": "end_too_late",
                        "related_pois": [location],
                        "conflict_reason": f"'{location}' ends after 21:00.",
                        "suggested_fix": "Move the stop earlier or split the day across more days.",
                        "severity": "soft",
                    }
                )
                soft_constraints.append("Avoid overly late attraction finishes by default.")

            open_start, open_end = _extract_opening_bounds(info.get("opening_hours", ""))
            if start_min is not None and end_min is not None and open_start is not None and open_end is not None:
                if start_min < open_start or end_min > open_end:
                    issues.append(
                        {
                            "issue_type": "opening_hours_conflict",
                            "related_pois": [location],
                            "conflict_reason": (
                                f"'{location}' is scheduled outside its known opening window "
                                f"({info.get('opening_hours', 'unknown')})."
                            ),
                            "suggested_fix": "Move the visit within opening hours or replace the POI.",
                            "severity": "hard",
                        }
                    )
                    hard_constraints.append("Visits must respect known opening hours.")

            planned_duration = event.get("planned_duration_minutes")
            recommended_low, recommended_high = _parse_duration_range_minutes(
                info.get("recommended_duration", "unknown")
            )
            if planned_duration and recommended_low and planned_duration < int(recommended_low * 0.5):
                issues.append(
                    {
                        "issue_type": "duration_too_short",
                        "related_pois": [location],
                        "conflict_reason": (
                            f"'{location}' is planned for {planned_duration} minutes, much shorter than "
                            f"the researched recommendation {info.get('recommended_duration')}."
                        ),
                        "suggested_fix": "Increase the stop duration or use it as a different itinerary role.",
                        "severity": "soft",
                    }
                )
                soft_constraints.append("Planned stays should be reasonably close to researched duration.")
            if (
                planned_duration
                and recommended_high
                and planned_duration > int(recommended_high * 1.5)
            ):
                issues.append(
                    {
                        "issue_type": "duration_too_long",
                        "related_pois": [location],
                        "conflict_reason": (
                            f"'{location}' is planned for {planned_duration} minutes, much longer than "
                            f"the researched recommendation {info.get('recommended_duration')}."
                        ),
                        "suggested_fix": "Shorten the stop or convert part of the time into free exploration.",
                        "severity": "soft",
                    }
                )
                soft_constraints.append("Planned stays should stay close to researched duration ranges.")
            if info.get("reservation_need", "unknown").casefold() == "yes":
                issues.append(
                    {
                        "issue_type": "reservation_risk",
                        "related_pois": [location],
                        "conflict_reason": f"'{location}' may require advance reservation.",
                        "suggested_fix": "Confirm reservation status or keep a backup POI ready.",
                        "severity": "soft",
                    }
                )
                soft_constraints.append("Flag reservation-dependent POIs so the planner can keep a backup.")

        previous_end = None
        previous_location = None
        previous_candidate = None
        for event in attraction_events:
            location = event.get("location", "")
            norm_location = _normalize_name(location)
            current_candidate = candidate_lookup.get(norm_location)
            current_start = _parse_time_hhmm(event.get("start_time"))
            if previous_end is not None and current_start is not None and current_start < previous_end:
                issues.append(
                    {
                        "issue_type": "overlapping_schedule",
                        "related_pois": [event.get("location", "")],
                        "conflict_reason": f"Day {day_index} has overlapping attraction times.",
                        "suggested_fix": "Reorder attractions or shift the start times.",
                        "severity": "hard",
                    }
                )
                hard_constraints.append("Attraction visits within a day must not overlap.")
            if previous_end is not None and current_start is not None:
                transfer_gap = current_start - previous_end
                if transfer_gap < 15:
                    issues.append(
                        {
                            "issue_type": "transfer_gap_too_short",
                            "related_pois": [previous_location or "", event.get("location", "")],
                            "conflict_reason": (
                                f"Only {transfer_gap} minutes are reserved between consecutive attractions "
                                f"on day {day_index}."
                            ),
                            "suggested_fix": "Insert more travel/buffer time or swap to a closer POI.",
                            "severity": "hard",
                        }
                    )
                    hard_constraints.append("Consecutive attractions need a realistic transfer buffer.")
                if previous_candidate and current_candidate:
                    try:
                        prev_lat = float(previous_candidate.get("latitude"))
                        prev_lon = float(previous_candidate.get("longitude"))
                        cur_lat = float(current_candidate.get("latitude"))
                        cur_lon = float(current_candidate.get("longitude"))
                    except (TypeError, ValueError):
                        prev_lat = prev_lon = cur_lat = cur_lon = None
                    if None not in (prev_lat, prev_lon, cur_lat, cur_lon):
                        distance_km = _haversine_km(prev_lat, prev_lon, cur_lat, cur_lon)
                        if distance_km > 80 and transfer_gap < 180:
                            issues.append(
                                {
                                    "issue_type": "geographic_jump",
                                    "related_pois": [previous_location or "", location],
                                    "conflict_reason": (
                                        f"'{previous_location}' and '{location}' are about {distance_km:.1f} km apart "
                                        f"with only {transfer_gap} minutes between them."
                                    ),
                                    "suggested_fix": "Keep the day geographically coherent or split these POIs across days.",
                                    "severity": "hard",
                                }
                            )
                            hard_constraints.append("Same-day POIs must be geographically reachable.")
            end = _parse_time_hhmm(event.get("end_time"))
            if end is not None:
                previous_end = end
            previous_location = event.get("location")
            previous_candidate = current_candidate

    if len(category_mix) <= 1 and len(days) > 1:
        issues.append(
            {
                "issue_type": "low_category_diversity",
                "related_pois": [],
                "conflict_reason": "The itinerary uses only one POI category across multiple days.",
                "suggested_fix": "Swap in researched POIs from different categories to diversify the trip.",
                "severity": "soft",
            }
        )
        soft_constraints.append("Prefer category diversity across a multi-day itinerary.")

    hard_issue_count = sum(1 for issue in issues if issue["severity"] == "hard")
    if hard_issue_count == 0 and not issues:
        status = "pass"
        summary = "No rule violations were detected."
    elif iteration_count >= max_iteration:
        status = "degrade"
        summary = "Hard constraints remain after repeated revisions; degrade to an executable plan."
    elif hard_issue_count > 0:
        status = "revise"
        summary = f"{hard_issue_count} hard issue(s) require local repair."
    else:
        status = "revise"
        summary = "Soft issues detected; local itinerary improvements are recommended."

    return {
        "status": status,
        "summary": summary,
        "hard_constraint_count": len(set(hard_constraints)),
        "soft_constraint_count": len(set(soft_constraints)),
        "hard_constraints": sorted(set(hard_constraints)),
        "soft_constraints": sorted(set(soft_constraints)),
        "issues": issues,
    }


def validate_timed_itinerary(
    timed_itinerary: str | Dict[str, Any] | None,
    poi_research_results: List[Dict[str, Any]] | None,
    poi_candidates: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    hard_constraints: List[str] = []
    soft_constraints: List[str] = []

    if not timed_itinerary:
        return {
            "status": "revise",
            "summary": "Timed itinerary is missing.",
            "hard_constraints": ["A timed itinerary must exist before final execution checks."],
            "soft_constraints": [],
            "issues": [
                {
                    "issue_type": "missing_timed_itinerary",
                    "related_pois": [],
                    "conflict_reason": "timed_itinerary is empty.",
                    "suggested_fix": "Generate travel legs before validating final execution feasibility.",
                    "severity": "hard",
                }
            ],
        }

    if isinstance(timed_itinerary, str):
        try:
            itinerary = json.loads(timed_itinerary)
        except json.JSONDecodeError:
            return {
                "status": "revise",
                "summary": "Timed itinerary is not valid JSON.",
                "hard_constraints": ["The final timed itinerary must be valid JSON."],
                "soft_constraints": [],
                "issues": [
                    {
                        "issue_type": "invalid_timed_itinerary_format",
                        "related_pois": [],
                        "conflict_reason": "timed_itinerary contains invalid JSON.",
                        "suggested_fix": "Persist the timed itinerary as structured JSON before validation.",
                        "severity": "hard",
                    }
                ],
            }
    else:
        itinerary = timed_itinerary

    poi_research_results = poi_research_results or []
    poi_candidates = poi_candidates or []
    poi_lookup = {_normalize_name(item.get("place_name", "")): item for item in poi_research_results}
    candidate_lookup = {_normalize_name(item.get("name", "")): item for item in poi_candidates}

    days = itinerary.get("days", []) if isinstance(itinerary, dict) else []

    for day_index, day in enumerate(days, 1):
        events = day.get("events", [])
        if not events:
            issues.append(
                {
                    "issue_type": "empty_day_after_timing",
                    "related_pois": [],
                    "conflict_reason": f"Day {day_index} became empty after timing enrichment.",
                    "suggested_fix": "Rebuild the day itinerary before final delivery.",
                    "severity": "hard",
                }
            )
            hard_constraints.append("Timed itinerary days must not be empty.")
            continue

        attraction_events = [event for event in events if event.get("type") == "Attraction"]
        travel_events = [event for event in events if event.get("type") == "Travel"]
        if len(attraction_events) >= 2 and len(travel_events) < len(attraction_events) - 1:
            issues.append(
                {
                    "issue_type": "missing_travel_segments",
                    "related_pois": [event.get("location", "") for event in attraction_events],
                    "conflict_reason": (
                        f"Day {day_index} has {len(attraction_events)} attractions but only "
                        f"{len(travel_events)} travel segment(s)."
                    ),
                    "suggested_fix": "Insert travel segments between consecutive attractions.",
                    "severity": "hard",
                }
            )
            hard_constraints.append("Consecutive attractions in the final itinerary need explicit travel segments.")

        total_minutes = 0
        previous_attraction = None
        previous_end = None

        for index, event in enumerate(events):
            event_type = event.get("type")
            location = event.get("location", "")
            if event_type == "Attraction":
                total_minutes += max(0, _day_total_minutes([event]))
                start_min = _parse_time_hhmm(event.get("start_time"))
                end_min = _parse_time_hhmm(event.get("end_time"))
                info = poi_lookup.get(_normalize_name(location))

                if previous_end is not None and start_min is not None and start_min < previous_end:
                    issues.append(
                        {
                            "issue_type": "post_timing_overlap",
                            "related_pois": [location],
                            "conflict_reason": (
                                f"Day {day_index} attraction '{location}' starts before the previous event chain ends."
                            ),
                            "suggested_fix": "Shift the attraction later or shorten earlier events.",
                            "severity": "hard",
                        }
                    )
                    hard_constraints.append("Final timed itinerary must preserve a non-overlapping event chain.")

                if info:
                    open_start, open_end = _extract_opening_bounds(info.get("opening_hours", ""))
                    if (
                        start_min is not None
                        and end_min is not None
                        and open_start is not None
                        and open_end is not None
                        and (start_min < open_start or end_min > open_end)
                    ):
                        issues.append(
                            {
                                "issue_type": "post_timing_opening_hours_conflict",
                                "related_pois": [location],
                                "conflict_reason": (
                                    f"'{location}' falls outside its opening window after travel timing was inserted."
                                ),
                                "suggested_fix": "Move the visit within opening hours or replace the POI.",
                                "severity": "hard",
                            }
                        )
                        hard_constraints.append("Timed itinerary must still respect opening hours after travel insertion.")

                previous_attraction = event
                if end_min is not None:
                    previous_end = end_min
            elif event_type == "Travel":
                total_minutes += max(0, _parse_duration_text_to_minutes(event.get("description")) or 0)
                desc = str(event.get("description") or "").strip().lower()
                minutes = _parse_duration_text_to_minutes(event.get("description"))
                if not desc or "unavailable" in desc or "could not estimate" in desc:
                    issues.append(
                        {
                            "issue_type": "travel_time_unavailable",
                            "related_pois": [location],
                            "conflict_reason": f"Day {day_index} has a travel segment without a usable travel duration.",
                            "suggested_fix": "Re-estimate the route or replace one of the POIs with a closer option.",
                            "severity": "hard",
                        }
                    )
                    hard_constraints.append("Final travel legs need usable duration estimates.")
                elif minutes is None:
                    issues.append(
                        {
                            "issue_type": "travel_duration_unparseable",
                            "related_pois": [location],
                            "conflict_reason": (
                                f"Day {day_index} travel description could not be parsed into minutes: {event.get('description', '')}"
                            ),
                            "suggested_fix": "Normalize the travel description so duration is explicit.",
                            "severity": "soft",
                        }
                    )
                    soft_constraints.append("Travel descriptions should expose machine-checkable durations.")

                if previous_end is not None:
                    previous_end = previous_end + (minutes or 0)

                if previous_attraction is not None and index + 1 < len(events):
                    next_event = events[index + 1]
                    if next_event.get("type") == "Attraction":
                        prev_candidate = candidate_lookup.get(
                            _normalize_name(previous_attraction.get("location", ""))
                        )
                        next_candidate = candidate_lookup.get(
                            _normalize_name(next_event.get("location", ""))
                        )
                        if prev_candidate and next_candidate and minutes is not None:
                            try:
                                prev_lat = float(prev_candidate.get("latitude"))
                                prev_lon = float(prev_candidate.get("longitude"))
                                cur_lat = float(next_candidate.get("latitude"))
                                cur_lon = float(next_candidate.get("longitude"))
                            except (TypeError, ValueError):
                                prev_lat = prev_lon = cur_lat = cur_lon = None
                            if None not in (prev_lat, prev_lon, cur_lat, cur_lon):
                                distance_km = _haversine_km(prev_lat, prev_lon, cur_lat, cur_lon)
                                if distance_km > 80 and minutes < 90:
                                    issues.append(
                                        {
                                            "issue_type": "timed_geographic_jump",
                                            "related_pois": [
                                                previous_attraction.get("location", ""),
                                                next_event.get("location", ""),
                                            ],
                                            "conflict_reason": (
                                                f"'{previous_attraction.get('location', '')}' and "
                                                f"'{next_event.get('location', '')}' are about {distance_km:.1f} km apart "
                                                f"but the final travel segment is only {minutes} minutes."
                                            ),
                                            "suggested_fix": "Keep the day geographically coherent or split these POIs across days.",
                                            "severity": "hard",
                                        }
                                    )
                                    hard_constraints.append(
                                        "Final travel timing must remain geographically plausible."
                                    )

        if total_minutes > 12 * 60:
            issues.append(
                {
                    "issue_type": "timed_day_too_long",
                    "related_pois": [event.get("location", "") for event in attraction_events],
                    "conflict_reason": (
                        f"Day {day_index} totals about {total_minutes} minutes when attractions and travel are combined."
                    ),
                    "suggested_fix": "Move a stop to another day or shorten the day.",
                    "severity": "hard",
                }
            )
            hard_constraints.append("Final itinerary days must remain executable after travel time is included.")
        elif total_minutes > 10 * 60:
            issues.append(
                {
                    "issue_type": "timed_day_tight",
                    "related_pois": [event.get("location", "") for event in attraction_events],
                    "conflict_reason": (
                        f"Day {day_index} totals about {total_minutes} minutes after travel insertion and may feel tight."
                    ),
                    "suggested_fix": "Keep a backup cuttable stop or add more slack.",
                    "severity": "soft",
                }
            )
            soft_constraints.append("Prefer some slack after travel timing is added.")

    hard_issue_count = sum(1 for issue in issues if issue["severity"] == "hard")
    if hard_issue_count == 0 and not issues:
        status = "pass"
        summary = "Timed itinerary passed final execution checks."
    elif hard_issue_count > 0:
        status = "revise"
        summary = f"{hard_issue_count} hard timed-itinerary issue(s) block final delivery."
    else:
        status = "pass"
        summary = "Timed itinerary is executable but has soft execution risks."

    return {
        "status": status,
        "summary": summary,
        "hard_constraint_count": len(set(hard_constraints)),
        "soft_constraint_count": len(set(soft_constraints)),
        "hard_constraints": sorted(set(hard_constraints)),
        "soft_constraints": sorted(set(soft_constraints)),
        "issues": issues,
    }
