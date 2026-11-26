"""Guardrailed Incident Summary service with 3-tier fallback.

Tier order:
1. Local Ollama (`OLLAMA_URL`)
2. Cloud LLM (`GROQ_API_KEY` or `GEMINI_API_KEY`)
3. Deterministic template fallback (zero external dependencies)

Every LLM response is strictly validated for:
- Pydantic v2 schema conformance (`IncidentSummary`)
- Exact `risk_level` parity with the deterministic Risk Engine
- Constant `is_lab_confirmed == False`
- Rejection of unsupported pollution-confirmation or illegal-dumping claims
- Rejection of invented numeric claims not grounded in the input evidence
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Literal, Mapping, Sequence

import httpx
from pydantic import BaseModel, ConfigDict, Field


EventType = Literal[
    "algal_bloom_proxy",
    "turbidity_sediment_proxy",
    "combined",
    "elevated_anomaly_uncorroborated",
    "baseline_screening",
    "insufficient_evidence",
]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
GeneratedBy = Literal["ollama", "groq", "gemini", "deterministic_fallback"]

LLM_TIMEOUT_SECONDS = 8.0

_FORBIDDEN_CLAIM_PATTERNS = (
    r"\bconfirmed\s+(?:water\s+)?pollution\b",
    r"\bconfirmed\s+illegal\s+dumping\b",
    r"\billegal\s+dumping\s+(?:is\s+)?confirmed\b",
    r"\bverified\s+illegal\s+dumping\b",
    r"\bproven\s+(?:pollution|contamination|dumping)\b",
    r"\bdefinitive\s+(?:pollution|contamination|dumping)\b",
    r"(?<!not\s)(?<!non-)\blab(?:oratory)?[- ]confirmed\b",
)

_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])(?:(?<!\d)-)?\d+(?:\.\d+)?(?![A-Za-z_])")


class IncidentSummary(BaseModel):
    """Structured incident summary matching SPEC.md Section 8.4."""

    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    risk_level: RiskLevel
    affected_location: str = Field(min_length=1)
    supporting_observations: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)
    is_lab_confirmed: Literal[False] = False
    generated_by: GeneratedBy = "deterministic_fallback"

    @property
    def summary_source(self) -> Literal["llm", "deterministic_fallback"]:
        return "deterministic_fallback" if self.generated_by == "deterministic_fallback" else "llm"

    def to_spec_dict(self) -> dict[str, Any]:
        """Return exact SPEC.md Section 8.4 `incident_summary` fields."""
        return {
            "event_type": self.event_type,
            "risk_level": self.risk_level,
            "affected_location": self.affected_location,
            "supporting_observations": self.supporting_observations,
            "recommended_action": self.recommended_action,
            "is_lab_confirmed": False,
        }


def _is_finite(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _infer_event_type(
    risk_assessment: Mapping[str, Any],
    hotspots: Sequence[Mapping[str, Any]],
) -> EventType:
    rule = str(risk_assessment.get("risk_rule_triggered") or "")
    if "MISSING" in rule:
        return "insufficient_evidence"
    if rule == "LOW_UNCORROBORATED_NO_HOTSPOTS":
        return "elevated_anomaly_uncorroborated"

    signals = {str(h.get("dominant_signal")) for h in hotspots if h.get("dominant_signal")}
    if "combined" in signals or (
        "algal_bloom_proxy" in signals and "turbidity_sediment_proxy" in signals
    ):
        return "combined"
    if "algal_bloom_proxy" in signals:
        return "algal_bloom_proxy"
    if "turbidity_sediment_proxy" in signals:
        return "turbidity_sediment_proxy"
    return "baseline_screening"


def _collect_allowed_numbers(
    observation_date: str,
    risk_assessment: Mapping[str, Any],
    hotspots: Sequence[Mapping[str, Any]],
    proxy_estimates: Mapping[str, Any],
) -> set[str]:
    """Collect exact and standard rounded string forms of all input numbers."""
    raw_values: list[float] = []

    for key in ("risk_score", "anomaly_score", "hotspot_count"):
        val = risk_assessment.get(key)
        if _is_finite(val):
            raw_values.append(float(val))

    for ev in risk_assessment.get("supporting_evidence") or []:
        for match in _NUMBER_RE.findall(str(ev)):
            if _is_finite(match):
                raw_values.append(float(match))

    for hs in hotspots:
        for key in ("lat", "lon", "pixel_count", "peak_ndci", "peak_ndti", "severity_score"):
            val = hs.get(key)
            if _is_finite(val):
                raw_values.append(float(val))

    for val in proxy_estimates.values():
        if _is_finite(val):
            raw_values.append(float(val))

    allowed: set[str] = set()
    for num in raw_values:
        for decimals in (0, 1, 2, 3, 4, 5, 6):
            rounded = round(num, decimals)
            if decimals == 0:
                allowed.add(str(int(rounded)))
            else:
                allowed.add(f"{rounded:.{decimals}f}")
                allowed.add(str(rounded))
        allowed.add(str(num))

    for part in str(observation_date).split("-"):
        if part.isdigit():
            allowed.add(part)
            allowed.add(str(int(part)))

    return allowed


def _contains_forbidden_claim(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in _FORBIDDEN_CLAIM_PATTERNS)


def _contains_ungrounded_numbers(text: str, allowed_numbers: set[str]) -> bool:
    for match in _NUMBER_RE.findall(text):
        normalized = match
        if normalized in allowed_numbers:
            continue
        if _is_finite(normalized):
            val = float(normalized)
            if str(val) in allowed_numbers:
                continue
            if val.is_integer() and str(int(val)) in allowed_numbers:
                continue
        return True
    return False


def _validate_llm_payload(
    payload: Mapping[str, Any],
    *,
    expected_risk_level: str,
    generated_by: GeneratedBy,
    allowed_numbers: set[str],
) -> IncidentSummary:
    """Validate LLM JSON strictly against schema, risk parity, and scientific guardrails."""
    data = dict(payload)
    data["generated_by"] = generated_by

    if data.get("risk_level") != expected_risk_level:
        raise ValueError("LLM risk_level does not match deterministic Risk Engine output.")
    if data.get("is_lab_confirmed") is not False:
        raise ValueError("LLM attempted to mark screening output as lab-confirmed.")

    summary = IncidentSummary.model_validate(data)
    combined_text = " ".join(
        [
            summary.affected_location,
            summary.supporting_observations,
            summary.recommended_action,
        ]
    )

    if _contains_forbidden_claim(combined_text):
        raise ValueError("LLM output contains unsupported pollution-confirmation claim.")
    if _contains_ungrounded_numbers(combined_text, allowed_numbers):
        raise ValueError("LLM output contains numeric values absent from verified evidence.")

    return summary


def build_deterministic_summary(
    *,
    lake_name: str,
    observation_date: str,
    risk_assessment: Mapping[str, Any],
    hotspots: Sequence[Mapping[str, Any]] | None = None,
    proxy_estimates: Mapping[str, Any] | None = None,
) -> IncidentSummary:
    """Build a safe, strictly grounded incident summary without external LLM calls."""
    hs_list = list(hotspots or [])
    proxies = dict(proxy_estimates or {})
    risk_level = str(risk_assessment.get("risk_level") or "LOW")
    if risk_level not in ("LOW", "MEDIUM", "HIGH"):
        risk_level = "LOW"

    event_type = _infer_event_type(risk_assessment, hs_list)
    hotspot_count = risk_assessment.get("hotspot_count")
    anomaly_score = risk_assessment.get("anomaly_score")
    risk_score = risk_assessment.get("risk_score")

    if hs_list:
        coords = ", ".join(
            f"{float(h['lat']):.4f} N, {float(h['lon']):.4f} E"
            for h in hs_list[:2]
            if _is_finite(h.get("lat")) and _is_finite(h.get("lon"))
        )
        affected_location = (
            f"{lake_name} ({len(hs_list)} spatial hotspot cluster(s) near {coords})"
            if coords
            else f"{lake_name} ({len(hs_list)} spatial hotspot cluster(s))"
        )
    else:
        affected_location = f"{lake_name} (no localized spatial hotspot clusters detected on {observation_date})"

    obs_parts: list[str] = []
    if _is_finite(anomaly_score):
        obs_parts.append(f"Satellite anomaly score was {float(anomaly_score):.2f}")
    else:
        obs_parts.append("Satellite anomaly score was unavailable")

    if hotspot_count is not None:
        obs_parts.append(f"with {int(hotspot_count)} spatial hotspot cluster(s)")
    if hs_list:
        max_px = max((int(h["pixel_count"]) for h in hs_list if _is_finite(h.get("pixel_count"))), default=0)
        max_sev = max((float(h["severity_score"]) for h in hs_list if _is_finite(h.get("severity_score"))), default=0.0)
        if max_px > 0:
            obs_parts.append(f"(largest cluster {max_px} pixels, max severity {max_sev:.2f})")

    proxy_bits: list[str] = []
    chla = proxies.get("pred_chla_proxy")
    tss = proxies.get("pred_tss_proxy")
    if _is_finite(chla):
        proxy_bits.append(f"Chla proxy {float(chla):.2f} mg/m3")
    if _is_finite(tss):
        proxy_bits.append(f"TSS proxy {float(tss):.2f} mg/L")
    if _is_finite(risk_score):
        proxy_bits.append(f"screening score {float(risk_score):.2f}")

    supporting_observations = " ".join(obs_parts) + "."
    if proxy_bits:
        supporting_observations += " Satellite-derived estimates (not lab-confirmed): " + ", ".join(proxy_bits) + "."

    if risk_level == "HIGH":
        recommended_action = (
            "Prioritize field water sampling and in-situ inspection near the identified cluster coordinates "
            "to verify whether satellite optical anomalies reflect elevated turbidity or algal biomass."
        )
    elif risk_level == "MEDIUM":
        recommended_action = (
            "Schedule routine field verification and visual shoreline screening near the flagged hotspot "
            "zones; continue monitoring subsequent cloud-free Sentinel-2 passes."
        )
    else:
        recommended_action = (
            "Maintain routine satellite monitoring; no immediate escalated field intervention is indicated "
            "from current screening evidence."
        )

    return IncidentSummary(
        event_type=event_type,
        risk_level=risk_level,  # type: ignore[arg-type]
        affected_location=affected_location,
        supporting_observations=supporting_observations,
        recommended_action=recommended_action,
        is_lab_confirmed=False,
        generated_by="deterministic_fallback",
    )


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response.")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object.")
    return parsed


def _build_prompt(
    *,
    lake_name: str,
    observation_date: str,
    risk_assessment: Mapping[str, Any],
    hotspots: Sequence[Mapping[str, Any]],
    proxy_estimates: Mapping[str, Any],
) -> str:
    evidence_payload = {
        "lake_name": lake_name,
        "observation_date": observation_date,
        "risk_assessment": {
            "risk_score": risk_assessment.get("risk_score"),
            "risk_level": risk_assessment.get("risk_level"),
            "anomaly_score": risk_assessment.get("anomaly_score"),
            "hotspot_count": risk_assessment.get("hotspot_count"),
            "risk_rule_triggered": risk_assessment.get("risk_rule_triggered"),
            "supporting_evidence": risk_assessment.get("supporting_evidence", []),
        },
        "hotspots": list(hotspots),
        "proxy_estimates": dict(proxy_estimates),
    }
    return (
        "You are a cautious satellite water-quality screening assistant. "
        "Summarize ONLY the verified JSON evidence below into a single JSON object with keys: "
        "event_type, risk_level, affected_location, supporting_observations, recommended_action, is_lab_confirmed.\n"
        "Strict rules:\n"
        "1. risk_level MUST exactly match risk_assessment.risk_level.\n"
        "2. is_lab_confirmed MUST be false.\n"
        "3. event_type MUST be one of: algal_bloom_proxy, turbidity_sediment_proxy, combined, "
        "elevated_anomaly_uncorroborated, baseline_screening, insufficient_evidence.\n"
        "4. NEVER invent numbers, dates, or locations not present in the JSON.\n"
        "5. NEVER claim confirmed pollution, confirmed illegal dumping, or laboratory-confirmed measurements.\n"
        "Return JSON only.\n\n"
        f"EVIDENCE:\n{json.dumps(evidence_payload, default=str)}"
    )


def _call_ollama(ollama_url: str, prompt: str) -> dict[str, Any]:
    url = ollama_url.rstrip("/")
    if not url.endswith("/api/generate") and not url.endswith("/api/chat"):
        url = f"{url}/api/generate"
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    body = {"model": model, "prompt": prompt, "stream": False, "format": "json"}
    resp = httpx.post(url, json=body, timeout=LLM_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("response") or data.get("message", {}).get("content") or ""
    return _extract_json_object(str(raw))


def _call_groq(api_key: str, prompt: str) -> dict[str, Any]:
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return _extract_json_object(str(content))


def _call_gemini(api_key: str, prompt: str) -> dict[str, Any]:
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }
    resp = httpx.post(
        url,
        params={"key": api_key},
        json=body,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return _extract_json_object(str(text))


def generate_incident_summary(
    *,
    lake_name: str,
    observation_date: str,
    risk_assessment: Mapping[str, Any],
    hotspots: Sequence[Mapping[str, Any]] | None = None,
    proxy_estimates: Mapping[str, Any] | None = None,
    allow_llm: bool = True,
) -> IncidentSummary:
    """Generate a guardrailed incident summary across Ollama -> Cloud LLM -> Fallback."""
    hs_list = list(hotspots or [])
    proxies = dict(proxy_estimates or {})
    expected_risk_level = str(risk_assessment.get("risk_level") or "LOW")

    if not allow_llm:
        return build_deterministic_summary(
            lake_name=lake_name,
            observation_date=observation_date,
            risk_assessment=risk_assessment,
            hotspots=hs_list,
            proxy_estimates=proxies,
        )

    allowed_numbers = _collect_allowed_numbers(
        observation_date=observation_date,
        risk_assessment=risk_assessment,
        hotspots=hs_list,
        proxy_estimates=proxies,
    )
    prompt = _build_prompt(
        lake_name=lake_name,
        observation_date=observation_date,
        risk_assessment=risk_assessment,
        hotspots=hs_list,
        proxy_estimates=proxies,
    )

    ollama_url = os.getenv("OLLAMA_URL", "").strip()
    if ollama_url:
        try:
            payload = _call_ollama(ollama_url, prompt)
            return _validate_llm_payload(
                payload,
                expected_risk_level=expected_risk_level,
                generated_by="ollama",
                allowed_numbers=allowed_numbers,
            )
        except Exception:
            pass

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            payload = _call_groq(groq_key, prompt)
            return _validate_llm_payload(
                payload,
                expected_risk_level=expected_risk_level,
                generated_by="groq",
                allowed_numbers=allowed_numbers,
            )
        except Exception:
            pass

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            payload = _call_gemini(gemini_key, prompt)
            return _validate_llm_payload(
                payload,
                expected_risk_level=expected_risk_level,
                generated_by="gemini",
                allowed_numbers=allowed_numbers,
            )
        except Exception:
            pass

    return build_deterministic_summary(
        lake_name=lake_name,
        observation_date=observation_date,
        risk_assessment=risk_assessment,
        hotspots=hs_list,
        proxy_estimates=proxies,
    )
