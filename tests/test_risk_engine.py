"""Automated tests for `shared.risk_engine`."""

from __future__ import annotations

import math

import pytest

from shared.risk_engine import RiskConfig, calculate_risk


def test_low_risk_conditions() -> None:
    obs = {
        "baseline_anomaly_score": 1.1,
        "hotspot_count": 0,
        "anomaly_drivers": "none",
        "pred_chla_proxy": 18.0,
        "pred_tss_proxy": 12.0,
    }
    res = calculate_risk(obs, historical_obs=[obs] * 5, config=RiskConfig())
    assert res.risk_level == "LOW"
    assert 0.0 <= res.risk_score < 20.0
    assert res.risk_rule_triggered == "LOW_BELOW_THRESHOLD_OR_NO_HOTSPOTS"
    assert res.triggered_rules


def test_medium_risk_conditions() -> None:
    obs = {
        "baseline_anomaly_score": 3.4,
        "hotspot_count": 2,
        "max_hotspot_severity": 0.55,
        "dominant_signals": ["algal_bloom_proxy"],
        "anomaly_drivers": "NDCI_p90(3.40)",
    }
    res = calculate_risk(obs, historical_obs=[obs] * 5, config=RiskConfig())
    assert res.risk_level == "MEDIUM"
    assert 20.0 <= res.risk_score <= 100.0
    assert res.risk_rule_triggered == "MEDIUM_SCORE_WITH_HOTSPOTS"
    assert any("spatial hotspot" in ev.lower() for ev in res.supporting_evidence)


def test_high_risk_conditions() -> None:
    obs = {
        "baseline_anomaly_score": 6.5,
        "hotspot_count": 3,
        "max_hotspot_severity": 0.85,
        "dominant_signals": ["combined"],
        "anomaly_drivers": "NDTI_p90(6.50)",
    }
    res = calculate_risk(obs, historical_obs=[obs] * 6, config=RiskConfig())
    assert res.risk_level == "HIGH"
    assert res.risk_score > 50.0
    assert res.risk_rule_triggered == "HIGH_SCORE_WITH_HOTSPOTS"


def test_uncorroborated_spike_never_claims_confirmed_pollution() -> None:
    obs = {"baseline_anomaly_score": 8.0, "hotspot_count": 0}
    res = calculate_risk(obs, config=RiskConfig())
    assert res.risk_level == "LOW"
    assert res.risk_rule_triggered == "LOW_UNCORROBORATED_NO_HOTSPOTS"
    assert any("not confirmed pollution" in ev.lower() for ev in res.supporting_evidence)


def test_missing_observations_safe_low() -> None:
    res_none = calculate_risk(None)
    assert res_none.risk_level == "LOW"
    assert res_none.risk_score == 0.0
    assert res_none.risk_rule_triggered == "LOW_MISSING_ALL_EVIDENCE"

    res_warmup = calculate_risk(
        {"baseline_anomaly_score": None, "hotspot_count": 2},
        historical_obs=[{"date": "2025-01-01"}],
    )
    assert res_warmup.risk_level == "LOW"
    assert res_warmup.risk_rule_triggered == "LOW_MISSING_ANOMALY_SCORE"
    assert any("historical context" in ev.lower() for ev in res_warmup.supporting_evidence)


def test_invalid_values_handled_safely() -> None:
    res_nan = calculate_risk(
        {"baseline_anomaly_score": float("nan"), "hotspot_count": -3}
    )
    assert res_nan.risk_level == "LOW"
    assert math.isfinite(res_nan.risk_score)
    assert res_nan.anomaly_score is None
    assert res_nan.hotspot_count is None

    res_inf = calculate_risk(
        {"baseline_anomaly_score": float("inf"), "hotspot_count": True}  # type: ignore[dict-item]
    )
    assert res_inf.risk_level == "LOW"
    assert res_inf.anomaly_score is None
    assert res_inf.hotspot_count is None


def test_threshold_changes_and_proxy_notes() -> None:
    obs = {
        "baseline_anomaly_score": 2.2,
        "hotspot_count": 1,
        "pred_chla_proxy": 120.0,
        "pred_tss_proxy": 80.0,
    }
    default_res = calculate_risk(obs, config=RiskConfig())
    assert default_res.risk_level == "LOW"

    custom_cfg = RiskConfig(anomaly_thresh=2.0, chla_bloom=100.0, tss_high=75.0)
    custom_res = calculate_risk(obs, config=custom_cfg)
    assert custom_res.risk_level == "MEDIUM"
    assert any("chla proxy" in r.lower() for r in custom_res.triggered_rules)
    assert any("tss proxy" in r.lower() for r in custom_res.triggered_rules)

    with pytest.raises(ValueError):
        RiskConfig(anomaly_thresh=-1.0).to_thresholds()


def test_deterministic_output() -> None:
    obs = {
        "baseline_anomaly_score": 3.7,
        "hotspot_count": 2,
        "max_hotspot_severity": 0.62,
        "dominant_signals": ["turbidity_sediment_proxy"],
    }
    r1 = calculate_risk(obs, historical_obs=[obs] * 4, config=RiskConfig())
    r2 = calculate_risk(obs, historical_obs=[obs] * 4, config=RiskConfig())
    assert r1 == r2
    assert r1.to_dict() == r2.to_dict()
