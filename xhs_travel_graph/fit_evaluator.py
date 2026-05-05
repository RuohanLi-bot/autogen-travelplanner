from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from .models import FitAssessment, TravelerProfile
from .normalizer import stable_id


DECISIONS = {"pass", "conditional", "fail", "unknown"}
SAFETY_CRITICAL_RISKS = {"water_safety", "height_exposure", "traffic_safety"}
SAFETY_MITIGATION_TYPES = {
    "coach",
    "safety_equipment",
    "shallow_water",
    "lifeguard",
    "official_service",
    "guardrail",
    "transport_substitution",
}


class FitEvaluator:
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def evaluate_route_variant(self, profile: TravelerProfile, route_payload: Dict[str, Any]) -> FitAssessment:
        assessment = None
        if self.llm_client is not None and getattr(self.llm_client, "available", lambda: False)():
            assessment = self._llm_assess(profile, route_payload)
        if assessment is None:
            assessment = self._fallback_assess(profile, route_payload)
        return self._apply_safety_floor(profile, route_payload, assessment)

    def profile_hash(self, profile: TravelerProfile) -> str:
        raw = json.dumps(profile.model_dump(), ensure_ascii=False, sort_keys=True)
        return stable_id("profile", raw)

    def _llm_assess(self, profile: TravelerProfile, route_payload: Dict[str, Any]) -> Optional[FitAssessment]:
        system_prompt = (
            "你是旅行玩法适配评估器。只能基于输入的结构化 facts 和 evidence 评估，"
            "不要补造缺失证据。输出 JSON object: "
            '{"decision":"pass|conditional|fail|unknown","hard_fail":false,'
            '"reasons":[],"required_actions":[],"missing_evidence":[],"evidence_used":[]}'
        )
        payload = {
            "profile": profile.model_dump(),
            "route": route_payload,
            "rule": "老人、低龄儿童、水域/高空/交通安全风险需要证据支持；缺证据时输出 unknown 或 fail。",
        }
        result = self.llm_client.generate_json(
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            temperature=0.0,
            default={},
        )
        if not isinstance(result, dict) or result.get("decision") not in DECISIONS:
            return None
        return FitAssessment(
            assessment_id=stable_id("fit", self.profile_hash(profile), route_payload.get("play_mode_id") or route_payload.get("route_variant_id")),
            profile_hash=self.profile_hash(profile),
            route_variant_id=str(route_payload.get("play_mode_id") or route_payload.get("route_variant_id") or ""),
            decision=result.get("decision", "unknown"),
            hard_fail=bool(result.get("hard_fail", False)),
            reasons=_string_list(result.get("reasons")),
            required_actions=_string_list(result.get("required_actions")),
            missing_evidence=_string_list(result.get("missing_evidence")),
            evidence_used=_string_list(result.get("evidence_used")),
        )

    def _fallback_assess(self, profile: TravelerProfile, route_payload: Dict[str, Any]) -> FitAssessment:
        route_id = str(route_payload.get("play_mode_id") or route_payload.get("route_variant_id") or route_payload.get("id") or "")
        profile_hash = self.profile_hash(profile)
        reasons: List[str] = []
        required_actions: List[str] = []
        missing_evidence: List[str] = []
        evidence_used = _evidence_used(route_payload)
        decision = "pass" if evidence_used else "unknown"
        hard_fail = False

        seniors_max = max(profile.seniors_ages) if profile.seniors_ages else None
        child_min = min(profile.children_ages) if profile.children_ages else None
        mitigations = _dicts(route_payload.get("mitigations"))
        has_transport_substitution = _has_available_mitigation(mitigations, {"transport_substitution"})

        for requirement in _dicts(route_payload.get("requirements")):
            if requirement.get("demand") == "climb_stairs":
                steps = _as_float(requirement.get("magnitude"))
                if steps is not None and steps >= 500 and (seniors_max is not None or profile.pace == "relaxed"):
                    if has_transport_substitution:
                        decision = _max_decision(decision, "conditional")
                        required_actions.append("选择帖子中有证据的省力交通替代，不走高台阶方案。")
                        reasons.append(f"{int(steps)}级台阶对老人或轻松游画像体力风险高，但存在交通替代证据。")
                    else:
                        decision = "fail"
                        hard_fail = seniors_max is not None and seniors_max >= 70
                        reasons.append(f"{int(steps)}级台阶对老人或轻松游画像体力风险高，且缺少替代方案证据。")

        for risk in _dicts(route_payload.get("risks")):
            risk_type = risk.get("risk_type")
            severity = risk.get("severity") or "unknown"
            if risk_type == "water_safety" and child_min is not None and child_min <= 6:
                if _has_available_mitigation(mitigations, {"coach", "safety_equipment", "shallow_water", "lifeguard"}):
                    decision = _max_decision(decision, "conditional")
                    required_actions.append("只选择有教练/救生装备/浅水区等证据的水上活动。")
                    reasons.append("低龄儿童参与水上活动需要明确安全保障。")
                else:
                    decision = _max_decision(decision, "unknown")
                    hard_fail = True
                    missing_evidence.extend(["child_age_min", "coach_available", "safety_equipment", "shallow_water_area"])
                    reasons.append("帖子缺少低龄儿童水上活动安全证据，不能判为适合。")
            elif risk_type in SAFETY_CRITICAL_RISKS and severity in {"high", "unknown"} and (child_min is not None or seniors_max is not None):
                if not _has_available_mitigation(mitigations, SAFETY_MITIGATION_TYPES):
                    decision = _max_decision(decision, "unknown")
                    hard_fail = True
                    missing_evidence.append(f"{risk_type}_mitigation")
                    reasons.append(f"{risk_type} 风险缺少针对老人或儿童的缓解证据。")
            elif risk_type == "fatigue" and severity == "high" and (seniors_max is not None or child_min is not None or profile.pace == "relaxed"):
                if has_transport_substitution:
                    decision = _max_decision(decision, "conditional")
                    required_actions.append("避开高体力段或采用交通替代。")
                    reasons.append("高体力玩法与老人、儿童或轻松游画像冲突，需要替代动作。")
                else:
                    decision = _max_decision(decision, "fail")
                    reasons.append("高体力玩法与老人、儿童或轻松游画像冲突。")

        for style in profile.avoid_styles:
            if style in _string_list(route_payload.get("style_tags")):
                decision = _max_decision(decision, "fail")
                reasons.append(f"用户明确规避 {style} 风格。")

        if not reasons and decision == "pass":
            reasons.append("结构化证据中未发现与用户画像冲突的要求或高风险。")
        if not evidence_used:
            missing_evidence.append("route_evidence")
            reasons.append("缺少可追溯原文证据。")

        return FitAssessment(
            assessment_id=stable_id("fit", profile_hash, route_id),
            profile_hash=profile_hash,
            route_variant_id=route_id,
            decision=decision,
            hard_fail=hard_fail,
            reasons=_dedupe(reasons),
            required_actions=_dedupe(required_actions),
            missing_evidence=_dedupe(missing_evidence),
            evidence_used=_dedupe(evidence_used),
        )

    def _apply_safety_floor(
        self,
        profile: TravelerProfile,
        route_payload: Dict[str, Any],
        assessment: FitAssessment,
    ) -> FitAssessment:
        child_min = min(profile.children_ages) if profile.children_ages else None
        seniors_max = max(profile.seniors_ages) if profile.seniors_ages else None
        risks = _dicts(route_payload.get("risks"))
        mitigations = _dicts(route_payload.get("mitigations"))
        for risk in risks:
            risk_type = risk.get("risk_type")
            severity = risk.get("severity") or "unknown"
            if risk_type in SAFETY_CRITICAL_RISKS and severity in {"high", "unknown"} and (child_min is not None or seniors_max is not None):
                if not _has_available_mitigation(mitigations, SAFETY_MITIGATION_TYPES):
                    assessment.hard_fail = True
                    if assessment.decision == "pass":
                        assessment.decision = "unknown"
                    key = f"{risk_type}_mitigation"
                    if key not in assessment.missing_evidence:
                        assessment.missing_evidence.append(key)
                    reason = f"{risk_type} 对老人或儿童属于安全关键风险，缺少缓解证据，不能作为主推荐。"
                    if reason not in assessment.reasons:
                        assessment.reasons.append(reason)
        return assessment


def _dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict) and any(v is not None for v in item.values())]


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _has_available_mitigation(mitigations: List[Dict[str, Any]], allowed_types: Iterable[str]) -> bool:
    allowed = set(allowed_types)
    return any(
        item.get("mitigation_type") in allowed and item.get("status", "unknown") == "available"
        for item in mitigations
    )


def _evidence_used(route_payload: Dict[str, Any]) -> List[str]:
    evidence: List[str] = []
    for field in ("constraints", "requirements", "risks", "mitigations"):
        for item in _dicts(route_payload.get(field)):
            text = item.get("evidence") or item.get("evidence_span")
            if text:
                evidence.append(str(text))
    if route_payload.get("evidence_span"):
        evidence.append(str(route_payload["evidence_span"]))
    return _dedupe(evidence)


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _max_decision(current: str, candidate: str) -> str:
    rank = {"pass": 0, "conditional": 1, "unknown": 2, "fail": 3}
    return candidate if rank[candidate] > rank[current] else current
