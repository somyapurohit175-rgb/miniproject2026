"""Shared deterministic Risk Engine interface wrapping `backend.risk_engine`.

Preserves the verified SPEC.md Section 5.3 conservative rule precedence and
continuous risk scoring while exposing the `RiskConfig` and `calculate_risk`
interface required by both backend services and downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from backend.risk_engine import RiskThresholds, classify_lake_risk


@dataclass(frozen=True)
class RiskConfig:
    """Configuration for deterministic satellite screening risk evaluation.

    Note: `anomaly_medium_min` (2.5) and `anomaly_high_min` (4.5) are the
    SPEC.md Section 5.3 demo screening rules. `tss_high` and `chla_bloom` are
    optional caller-supplied proxy reference thresholds (not defined in SPEC.md)
    that add informational screening notes when exceeded, without overriding
    conservative spatial-corroboration precedence.
    """

    anomaly_medium_min: float = 2.5
    anomaly_high_min: float = 4.5
    min_hotspots_for_elevation: int = 1
    enable_cnn_lstm_rule: bool = False
    anomaly_thresh: float | None = None
    tss_high: float | None = None
    chla_bloom: float | None = None

    def to_thresholds(self) -> RiskThresholds:
        med_min = (
            float(self.anomaly_thresh)
            if self.anomaly_thresh is not None and _is_finite(self.anomaly_thresh)
            else float(self.anomaly_medium_min)
        )
        if med_min < 0.0:
            raise ValueError("anomaly_thresh / anomaly_medium_min must be >= 0.")
        high_min = float(self.anomaly_high_min)
        if high_min < med_min:
            high_min = med_min + 2.0
        if self.min_hotspots_for_elevation < 0:
            raise ValueError("min_hotspots_for_elevation must be >= 0.")
        return RiskThresholds(
            anomaly_medium_min=med_min,
            anomaly_high_min=high_min,
            min_hotspots_for_elevation=int(self.min_hotspots_for_elevation),
            enable_cnn_lstm_rule=bool(self.enable_cnn_lstm_rule),
        )


@dataclass(frozen=True)
class RiskResult:
    """Structured deterministic output from `calculate_risk`."""

    risk_score: float
    risk_level: str
    triggered_rules: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    risk_rule_triggered: str = ""
    anomaly_score: float | None = None
    hotspot_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "triggered_rules": list(self.triggered_rules),
            "supporting_evidence": list(self.supporting_evidence),
            "risk_rule_triggered": self.risk_rule_triggered,
            "anomaly_score": self.anomaly_score,
            "hotspot_count": self.hotspot_count,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


def _is_finite(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _extract_value(obs: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in obs and obs[key] is not None:
            return obs[key]
    return None


def calculate_risk(
    latest_obs: Mapping[str, Any] | None,
    historical_obs: Sequence[Mapping[str, Any]] | None = None,
    config: RiskConfig | None = None,
) -> RiskResult:
    """Evaluate deterministic screening risk from latest and historical observations.

    Parameters
    ----------
    latest_obs:
        Mapping containing observation fields such as `baseline_anomaly_score` or
        `anomaly_score`, `hotspot_count` (or `hotspots` list), `anomaly_drivers`,
        `max_hotspot_severity`, `dominant_signals`, `pred_chla_proxy`/`pred_chla`,
        and `pred_tss_proxy`/`pred_tss`.
    historical_obs:
        Optional chronological observation sequence used to verify sufficient
        history context without fabricating missing values.
    config:
        Optional `RiskConfig` overriding thresholds.
    """
    cfg = config or RiskConfig()
    thresholds = cfg.to_thresholds()

    if not latest_obs:
        res = classify_lake_risk(None, None, thresholds=thresholds)
        return RiskResult(
            risk_score=res.risk_score,
            risk_level=res.risk_level,
            triggered_rules=[res.risk_rule_triggered],
            supporting_evidence=list(res.supporting_evidence),
            risk_rule_triggered=res.risk_rule_triggered,
            anomaly_score=res.anomaly_score,
            hotspot_count=res.hotspot_count,
        )

    anomaly_raw = _extract_value(latest_obs, "baseline_anomaly_score", "anomaly_score")
    hotspots_list = latest_obs.get("hotspots") if isinstance(latest_obs.get("hotspots"), list) else None

    hotspot_count = _extract_value(latest_obs, "hotspot_count")
    if hotspot_count is None and hotspots_list is not None:
        hotspot_count = len(hotspots_list)

    max_sev = _extract_value(latest_obs, "max_hotspot_severity")
    if max_sev is None and hotspots_list:
        sevs = [float(h["severity_score"]) for h in hotspots_list if _is_finite(h.get("severity_score"))]
        max_sev = max(sevs) if sevs else None

    dom_signals = latest_obs.get("dominant_signals")
    if dom_signals is None and hotspots_list:
        dom_signals = [
            str(h["dominant_signal"]) for h in hotspots_list if h.get("dominant_signal")
        ]

    anomaly_drivers = _extract_value(latest_obs, "anomaly_drivers")
    if anomaly_drivers is not None:
        anomaly_drivers = str(anomaly_drivers)

    cnn_lstm_flag = latest_obs.get("cnn_lstm_anomaly")

    res = classify_lake_risk(
        anomaly_score=anomaly_raw,
        hotspot_count=hotspot_count,
        anomaly_drivers=anomaly_drivers,
        max_hotspot_severity=max_sev,
        dominant_signals=list(dom_signals) if dom_signals else None,
        cnn_lstm_anomaly=cnn_lstm_flag,
        thresholds=thresholds,
    )

    evidence = list(res.supporting_evidence)
    triggered_rules = [
        f"{res.risk_rule_triggered}: "
        + (evidence[0] if evidence else f"Classified as {res.risk_level}.")
    ]

    if historical_obs is not None:
        hist_count = len(historical_obs)
        if hist_count < 3 and res.anomaly_score is None:
            evidence.append(
                f"Historical context has {hist_count} observation(s) (< 3 required for trailing baseline)."
            )

    # Optional user-configured proxy thresholds (informational screening notes only)
    chla_val = _extract_value(latest_obs, "pred_chla_proxy", "pred_chla")
    if cfg.chla_bloom is not None and _is_finite(cfg.chla_bloom) and _is_finite(chla_val):
        if float(chla_val) >= float(cfg.chla_bloom):
            note = (
                f"Satellite Chla proxy ({float(chla_val):.2f} mg/m3) >= configured "
                f"screening threshold ({float(cfg.chla_bloom):.2f} mg/m3); uncalibrated proxy, not lab-confirmed."
            )
            evidence.append(note)
            triggered_rules.append(f"PROXY_CHLA_SCREENING_NOTE: {note}")

    tss_val = _extract_value(latest_obs, "pred_tss_proxy", "pred_tss")
    if cfg.tss_high is not None and _is_finite(cfg.tss_high) and _is_finite(tss_val):
        if float(tss_val) >= float(cfg.tss_high):
            note = (
                f"Satellite TSS proxy ({float(tss_val):.2f} mg/L) >= configured "
                f"screening threshold ({float(cfg.tss_high):.2f} mg/L); uncalibrated proxy, not lab-confirmed."
            )
            evidence.append(note)
            triggered_rules.append(f"PROXY_TSS_SCREENING_NOTE: {note}")

    return RiskResult(
        risk_score=res.risk_score,
        risk_level=res.risk_level,
        triggered_rules=triggered_rules,
        supporting_evidence=evidence,
        risk_rule_triggered=res.risk_rule_triggered,
        anomaly_score=res.anomaly_score,
        hotspot_count=res.hotspot_count,
    )
