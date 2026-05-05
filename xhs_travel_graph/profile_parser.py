from __future__ import annotations

import re
from typing import List

from .models import TravelerProfile


AGE_PATTERN = re.compile(r"(\d{1,3})\s*岁")
CHILD_KEYWORDS = ("孩子", "小孩", "儿童", "带娃", "亲子", "娃")
SENIOR_KEYWORDS = ("老人", "父母", "爸妈", "长辈", "腿脚不便", "老年")
MOBILITY_KEYWORDS = ("腿脚不便", "走不动", "少走路", "不爬山", "不能爬", "轻体力", "轮椅")
BUDGET_LOW_KEYWORDS = ("省钱", "经济", "便宜", "预算低", "穷游")
RELAXED_KEYWORDS = ("轻松", "休闲", "不累", "慢节奏", "轻体力", "少走路")
INTENSIVE_KEYWORDS = ("特种兵", "暴走", "打卡", "高强度", "一天刷")


def parse_traveler_profile(user_text: str) -> TravelerProfile:
    text = user_text or ""
    seniors: List[int] = []
    children: List[int] = []
    for match in AGE_PATTERN.finditer(text):
        age = int(match.group(1))
        window = text[max(0, match.start() - 8) : match.end() + 8]
        if age <= 14 or any(keyword in window for keyword in CHILD_KEYWORDS):
            children.append(age)
        elif age >= 50 or any(keyword in window for keyword in SENIOR_KEYWORDS):
            seniors.append(age)

    mobility_notes = [keyword for keyword in MOBILITY_KEYWORDS if keyword in text]
    if any(keyword in text for keyword in SENIOR_KEYWORDS) and not seniors:
        mobility_notes.append("提到老人但未给出年龄")

    swimming_ability = "unknown"
    if "不会游泳" in text or "不能游泳" in text:
        swimming_ability = "cannot_swim"
    elif "会游泳" in text:
        swimming_ability = "good"

    guardian_available = "yes" if any(keyword in text for keyword in ("家长陪同", "大人陪", "陪同")) else "unknown"
    budget_level = "low" if any(keyword in text for keyword in BUDGET_LOW_KEYWORDS) else "unknown"
    if any(keyword in text for keyword in RELAXED_KEYWORDS):
        pace = "relaxed"
    elif any(keyword in text for keyword in INTENSIVE_KEYWORDS):
        pace = "intensive"
    else:
        pace = "unknown"

    avoid_styles = []
    if any(keyword in text for keyword in ("不要特种兵", "不想特种兵", "不要暴走", "不想暴走")):
        avoid_styles.append("intensive")

    return TravelerProfile(
        seniors_ages=sorted(set(seniors)),
        children_ages=sorted(set(children)),
        mobility_notes=sorted(set(mobility_notes)),
        swimming_ability=swimming_ability,
        guardian_available=guardian_available,
        budget_level=budget_level,
        pace=pace,
        avoid_styles=avoid_styles,
        raw_text=text,
    )
