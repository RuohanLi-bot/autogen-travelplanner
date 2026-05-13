from __future__ import annotations

import re
from typing import Dict, Iterable

from .models import CanonicalTransport


TRANSPORT_PATTERNS: Dict[str, tuple[str, ...]] = {
    "tourist_train": ("小火车", "观光小火车", "观光车轨道", "小火车往返"),
    "shuttle_bus": ("环保车", "景交车", "景区车", "接驳车", "景区大巴", "摆渡车"),
    "walking_stairs": ("爬台阶", "上台阶", "下台阶", "台阶步行", "楼梯"),
    "walking": ("步行", "徒步", "走路", "暴走", "walk"),
    "cable_car": ("索道", "缆车", "吊厢"),
    "elevator": ("电梯", "天梯", "百龙电梯"),
    "escalator": ("扶梯",),
}

def normalize_transport_mode(raw_text: str = "", field_value: str = "") -> CanonicalTransport:
    text = f"{field_value or ''}\n{raw_text or ''}".strip()
    if not text:
        return CanonicalTransport(canonical_id="unknown", confidence=0.0, reason="交通字段为空。")

    exact = _exact_match(text)
    if exact:
        return CanonicalTransport(canonical_id=exact, confidence=0.95, reason="命中高置信交通关键词。")

    scored = []
    for mode_id, patterns in TRANSPORT_PATTERNS.items():
        score = _pattern_score(text, patterns)
        if score > 0:
            scored.append((score, mode_id))

    if not scored:
        return CanonicalTransport(canonical_id="unknown", confidence=0.0, reason="未找到可映射的交通方式。")

    scored.sort(reverse=True)
    score, mode_id = scored[0]
    confidence = min(0.9, 0.45 + score * 0.08)
    if confidence < 0.65:
        return CanonicalTransport(
            canonical_id="unknown",
            confidence=confidence,
            reason=f"最相近候选为 {mode_id}，但证据不足。",
        )
    return CanonicalTransport(
        canonical_id=mode_id,
        confidence=confidence,
        reason=f"原文与 {mode_id} 的关键词最接近。",
    )


def _exact_match(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    if any(token in compact for token in ("不用走路", "少走路", "减少步行", "省力")):
        for mode_id in ("tourist_train", "shuttle_bus", "cable_car", "elevator", "escalator"):
            if any(pattern in compact for pattern in TRANSPORT_PATTERNS.get(mode_id, ())):
                return mode_id
    for mode_id, patterns in TRANSPORT_PATTERNS.items():
        if any(pattern in compact for pattern in patterns):
            return mode_id
    return ""


def _pattern_score(text: str, patterns: Iterable[str]) -> float:
    score = 0.0
    for pattern in patterns:
        if pattern and pattern in text:
            score += 3.0
    for token in _char_ngrams(text):
        if any(token in pattern for pattern in patterns):
            score += 0.4
    return score


def _char_ngrams(text: str) -> Iterable[str]:
    compact = re.sub(r"\s+", "", text)
    for size in (2, 3):
        for idx in range(max(0, len(compact) - size + 1)):
            yield compact[idx : idx + size]
