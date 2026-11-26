"""Automated tests for `shared.llm_summary` guardrails and 3-tier fallback."""

from __future__ import annotations

import json
from typing import Any

import httpx

from shared.llm_summary import IncidentSummary, generate_incident_summary


SAMPLE_RISK = {
    "risk_score": 46.66,
    "risk_level": "MEDIUM",
    "anomaly_score": 3.61,
    "hotspot_count": 1,
    "risk_rule_triggered": "MEDIUM_SCORE_WITH_HOTSPOTS",
    "supporting_evidence": [
        "Anomaly score 3.61 in [2.50, 4.50] corroborated by 1 spatial hotspot cluster(s)."
    ],
}

SAMPLE_HOTSPOTS = [
    {
        "hotspot_id": "bellandur_2026-11-23_h01",
        "lat": 12.9361,
        "lon": 77.6659,
        "pixel_count": 13,
        "peak_ndci": 0.48,
        "peak_ndti": -0.13,
        "dominant_signal": "algal_bloom_proxy",
        "severity_score": 0.61,
    }
]

SAMPLE_PROXIES = {
    "pred_chla_proxy": 51.75,
    "pred_tss_proxy": 10.66,
    "NDCI_med": 0.22,
    "NDTI_med": -0.11,
    "NDWI_med": 0.36,
}


def _valid_llm_json(risk_level: str = "MEDIUM", **overrides: Any) -> str:
    payload = {
        "event_type": "algal_bloom_proxy",
        "risk_level": risk_level,
        "affected_location": "Bellandur Lake near 12.9361 N, 77.6659 E",
        "supporting_observations": (
            "Anomaly score reached 3.61 with 1 cluster (13 pixels, severity 0.61). "
            "Satellite proxy estimates: Chla 51.75 mg/m3, TSS 10.66 mg/L."
        ),
        "recommended_action": "Schedule routine shoreline screening and field water sampling.",
        "is_lab_confirmed": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_deterministic_fallback_when_no_llm_configured(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    res = generate_incident_summary(
        lake_name="Bellandur Lake",
        observation_date="2026-11-23",
        risk_assessment=SAMPLE_RISK,
        hotspots=SAMPLE_HOTSPOTS,
        proxy_estimates=SAMPLE_PROXIES,
    )
    assert isinstance(res, IncidentSummary)
    assert res.generated_by == "deterministic_fallback"
    assert res.summary_source == "deterministic_fallback"
    assert res.risk_level == "MEDIUM"
    assert res.is_lab_confirmed is False
    assert "Bellandur Lake" in res.affected_location


def test_valid_ollama_summary_and_risk_parity(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fake_post(url: str, json: dict, timeout: float):
        req = httpx.Request("POST", url)
        return httpx.Response(200, json={"response": _valid_llm_json("MEDIUM")}, request=req)

    monkeypatch.setattr(httpx, "post", fake_post)

    res = generate_incident_summary(
        lake_name="Bellandur Lake",
        observation_date="2026-11-23",
        risk_assessment=SAMPLE_RISK,
        hotspots=SAMPLE_HOTSPOTS,
        proxy_estimates=SAMPLE_PROXIES,
    )
    assert res.generated_by == "ollama"
    assert res.summary_source == "llm"
    assert res.risk_level == "MEDIUM"
    assert res.is_lab_confirmed is False


def test_rejects_mismatched_risk_level_and_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fake_post(url: str, json: dict, timeout: float):
        req = httpx.Request("POST", url)
        return httpx.Response(200, json={"response": _valid_llm_json("HIGH")}, request=req)

    monkeypatch.setattr(httpx, "post", fake_post)

    res = generate_incident_summary(
        lake_name="Bellandur Lake",
        observation_date="2026-11-23",
        risk_assessment=SAMPLE_RISK,
        hotspots=SAMPLE_HOTSPOTS,
        proxy_estimates=SAMPLE_PROXIES,
    )
    assert res.generated_by == "deterministic_fallback"
    assert res.risk_level == "MEDIUM"


def test_rejects_invented_numeric_claims(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    bad_json = _valid_llm_json(
        supporting_observations="Chlorophyll-a measured at 999.9 mg/m3 with 1 cluster."
    )

    def fake_post(url: str, headers: dict, json: dict, timeout: float):
        req = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": bad_json}}]},
            request=req,
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    res = generate_incident_summary(
        lake_name="Bellandur Lake",
        observation_date="2026-11-23",
        risk_assessment=SAMPLE_RISK,
        hotspots=SAMPLE_HOTSPOTS,
        proxy_estimates=SAMPLE_PROXIES,
    )
    assert res.generated_by == "deterministic_fallback"
    assert "999.9" not in res.supporting_observations


def test_rejects_unsupported_pollution_confirmation_claims(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    bad_json = _valid_llm_json(
        supporting_observations="Confirmed illegal dumping detected at Bellandur Lake with anomaly score 3.61."
    )

    def fake_post(url: str, params: dict, json: dict, timeout: float):
        req = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": bad_json}]}}]},
            request=req,
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    res = generate_incident_summary(
        lake_name="Bellandur Lake",
        observation_date="2026-11-23",
        risk_assessment=SAMPLE_RISK,
        hotspots=SAMPLE_HOTSPOTS,
        proxy_estimates=SAMPLE_PROXIES,
    )
    assert res.generated_by == "deterministic_fallback"
    assert res.is_lab_confirmed is False


def test_ollama_unavailable_falls_back_to_cloud_then_deterministic(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    calls: list[str] = []

    def fake_post(url: str, **kwargs):
        calls.append(url)
        if "localhost" in url:
            raise httpx.ConnectError("Ollama offline")
        if "groq" in url:
            req = httpx.Request("POST", url)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": _valid_llm_json("MEDIUM")}}]},
                request=req,
            )
        raise AssertionError("Should not reach Gemini when Groq succeeds")

    monkeypatch.setattr(httpx, "post", fake_post)

    res = generate_incident_summary(
        lake_name="Bellandur Lake",
        observation_date="2026-11-23",
        risk_assessment=SAMPLE_RISK,
        hotspots=SAMPLE_HOTSPOTS,
        proxy_estimates=SAMPLE_PROXIES,
    )
    assert len(calls) == 2
    assert res.generated_by == "groq"

    # Now make all three fail -> deterministic fallback
    def all_fail(url: str, **kwargs):
        raise httpx.TimeoutException("Network timeout")

    monkeypatch.setattr(httpx, "post", all_fail)
    res_fallback = generate_incident_summary(
        lake_name="Bellandur Lake",
        observation_date="2026-11-23",
        risk_assessment=SAMPLE_RISK,
        hotspots=SAMPLE_HOTSPOTS,
        proxy_estimates=SAMPLE_PROXIES,
    )
    assert res_fallback.generated_by == "deterministic_fallback"
    assert res_fallback.is_lab_confirmed is False


def test_rejects_invalid_json_from_llm(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fake_post(url: str, **kwargs):
        req = httpx.Request("POST", url)
        return httpx.Response(200, json={"response": "not-json-at-all"}, request=req)

    monkeypatch.setattr(httpx, "post", fake_post)

    res = generate_incident_summary(
        lake_name="Bellandur Lake",
        observation_date="2026-11-23",
        risk_assessment=SAMPLE_RISK,
        hotspots=SAMPLE_HOTSPOTS,
        proxy_estimates=SAMPLE_PROXIES,
    )
    assert res.generated_by == "deterministic_fallback"
