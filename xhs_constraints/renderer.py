from __future__ import annotations

from itinerary_models import Day, Event, Itinerary

from .models import ItinerarySkeleton, SkeletonEvent


def render_itinerary_deterministically(skeleton: ItinerarySkeleton) -> Itinerary:
    days = []
    for day in skeleton.days:
        events = []
        for item in day.events:
            events.append(_render_event(item, day.theme, day.rest_buffer))
        days.append(Day(events=events))
    return Itinerary(days=days)


def _render_event(item: SkeletonEvent, theme: str, rest_buffer: str) -> Event:
    event_type = "Travel" if item.type == "Travel" else "Attraction"
    if item.type == "Rest":
        description = "休息缓冲：" + "；".join(item.description_facts or [rest_buffer or "保留低强度休息时间。"])
    else:
        parts = []
        if theme and item.type == "Attraction":
            parts.append(f"当天主题：{theme}。")
        if item.selected_option:
            parts.append(f"建议方式：{item.selected_option}。")
        parts.extend(item.description_facts)
        if item.must_do:
            parts.append("必须执行：" + "；".join(item.must_do[:4]) + "。")
        if item.must_not_do:
            parts.append("原始高体力、长距离或高台阶方案不作为主线。")
        if item.evidence:
            parts.append("依据：" + "；".join(_shorten(text, 80) for text in item.evidence[:2]))
        description = "".join(parts) or "按约束优化后的低体力方案执行。"

    return Event(
        type=event_type,
        location=item.location,
        city=item.city,
        description=description,
        poi_id=item.source_candidate_id or None,
        itinerary_role="constraint_aware_xhs",
    )


def _shorten(text: str, limit: int) -> str:
    raw = " ".join(str(text or "").split())
    return raw if len(raw) <= limit else raw[:limit] + "..."
