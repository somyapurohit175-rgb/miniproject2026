"""Deterministic conservative Risk Engine for satellite water-quality screening.

This module intentionally sits downstream of the statistical anomaly detector
(`ml/03b_anomaly_hotspots.ipynb`) and spatial hotspot extractor (`spatial_hotspots.json`).
It combines continuous anomaly magnitude with spatial hotspot corroboration under
configurable demo thresholds without altering the underlying baseline detector.

IMPORTANT LIMITATIONS:
- Initial thresholds (anomaly_low_max=2.5, anomaly_high_min=4.5) are configurable
  demo screening rules, NOT scientifically calibrated regulatory pollution thresholds.
- CNN-LSTM corroboration is reserved as an explicit, disabled future integration hook.
- Missing evidence or uncorroborated single-modality spikes are handled conservatively
  and never emit confirmed pollution claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


VALID_RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")


@dataclass(frozen=True)
class RiskThresholds:
    """Configurable thresholds and scoring weights for risk classification.

    Rule precedence (evaluated in strict order):
    1. Missing/non-finite anomaly_score or missing/negative hotspot_count -> LOW
       (conservative guard so incomplete inputs never generate an unsupported alert).
    2. anomaly_score < anomaly_medium_min (2.5) OR hotspot_count == 0 -> LOW
       (uncorroborated spikes without spatial hotspot evidence remain LOW, even when
       anomaly_score > 4.5, though the continuous risk_score still reflects elevation).
    3. hotspot_count >= 1 AND anomaly_score > anomaly_high_min (4.5) -> HIGH.
       (Optional future hook: when enable_cnn_lstm_rule=True and cnn_lstm_anomaly=True
       with hotspot_count >= 1, elevate to HIGH; disabled until Day 3 CNN-LSTM exists.)
    4. hotspot_count >= 1 AND 2.5 <= anomaly_score <= 4.5 -> MEDIUM.
    """

    anomaly_medium_min: float = 2.5
    anomaly_high_min: float = 4.5
    min_hotspots_for_elevation: int = 1
    enable_cnn_lstm_rule: bool = False
    anomaly_score_cap_for_risk_score: float = 10.0
    hotspot_count_cap_for_risk_score: int = 5
    anomaly_weight: float = 60.0
    hotspot_weight: float = 40.0


@dataclass(frozen=True)
class RiskAssessmentResult:
    """Structured deterministic output from the conservative risk engine."""

    risk_score: float
    risk_level: str
    anomaly_score: float | None
    hotspot_count: int | None
    supporting_evidence: list[str] = field(default_factory=list)
    risk_rule_triggered: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "anomaly_score": self.anomaly_score,
            "hotspot_count": self.hotspot_count,
            "supporting_evidence": list(self.supporting_evidence),
            "risk_rule_triggered": self.risk_rule_triggered,
        }


def _is_valid_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        num = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(num)


def _compute_continuous_risk_score(
    anomaly_score: float | None,
    hotspot_count: int | None,
    max_hotspot_severity: float | None,
    cfg: RiskThresholds,
) -> float:
    """Compute an independent continuous [0.0, 100.0] screening score from available evidence.

    Kept strictly separate from `risk_level` tier assignment. Missing components
    contribute 0.0 without fabricating placeholder measurements.
    """
    anom_component = 0.0
    if anomaly_score is not None and anomaly_score > 0.0:
        anom_ratio = min(anomaly_score / cfg.anomaly_score_cap_for_risk_score, 1.0)
        anom_component = cfg.anomaly_weight * anom_ratio

    hs_component = 0.0
    if hotspot_count is not None and hotspot_count > 0:
        count_ratio = min(hotspot_count / float(cfg.hotspot_count_cap_for_risk_score), 1.0)
        if _is_valid_number(max_hotspot_severity):
            sev_ratio = max(0.0, min(float(max_hotspot_severity), 1.0))
            spatial_ratio = 0.6 * count_ratio + 0.4 * sev_ratio
        else:
            spatial_ratio = count_ratio
        hs_component = cfg.hotspot_weight * spatial_ratio

    return round(max(0.0, min(anom_component + hs_component, 100.0)), 2)


def classify_lake_risk(
    anomaly_score: float | None,
    hotspot_count: int | None,
    *,
    anomaly_drivers: str | None = None,
    max_hotspot_severity: float | None = None,
    dominant_signals: list[str] | None = None,
    cnn_lstm_anomaly: bool | None = None,
    thresholds: RiskThresholds | None = None,
) -> RiskAssessmentResult:
    """Classify deterministic lake screening risk from anomaly and spatial evidence.

    Parameters
    ----------
    anomaly_score:
        Continuous positive robust Z-score (`baseline_anomaly_score`) or `None`/NaN.
    hotspot_count:
        Count of validated spatial clusters (>= 3 pixels inside lake polygon) or `None`.
    anomaly_drivers:
        Optional string describing triggering statistical indicators.
    max_hotspot_severity:
        Optional peak cluster severity score in [0, 1] for continuous risk scoring.
    dominant_signals:
        Optional list of spatial `dominant_signal` labels for evidence reporting.
    cnn_lstm_anomaly:
        Future Day 3 integration flag; ignored unless `thresholds.enable_cnn_lstm_rule=True`.
    thresholds:
        Optional `RiskThresholds` override; defaults to `RiskThresholds()`.
    """
    cfg = thresholds or RiskThresholds()

    clean_anom: float | None = float(anomaly_score) if _is_valid_number(anomaly_score) else None
    if clean_anom is not None and clean_anom < 0.0:
        clean_anom = 0.0

    clean_hs: int | None = None
    if hotspot_count is not None and not isinstance(hotspot_count, bool):
        try:
            hs_int = int(hotspot_count)
            if float(hotspot_count) == hs_int and hs_int >= 0:
                clean_hs = hs_int
        except (TypeError, ValueError):
            clean_hs = None

    risk_score = _compute_continuous_risk_score(clean_anom, clean_hs, max_hotspot_severity, cfg)
    evidence: list[str] = []

    if anomaly_drivers and anomaly_drivers not in {"none", "insufficient_history"}:
        evidence.append(f"Temporal drivers: {anomaly_drivers}.")
    if dominant_signals:
        unique_signals = sorted(set(dominant_signals))
        evidence.append(f"Spatial hotspot signals: {', '.join(unique_signals)}.")
    if cnn_lstm_anomaly is not None and not cfg.enable_cnn_lstm_rule:
        evidence.append(
            "CNN-LSTM corroboration input provided but disabled (Day 3 model integration pending)."
        )

    # Precedence 1: Missing anomaly score or missing hotspot data -> safe LOW fallback
    if clean_anom is None and clean_hs is None:
        evidence.insert(0, "Both anomaly_score and hotspot_count are missing; defaulting to LOW screening risk.")
        return RiskAssessmentResult(
            risk_score=risk_score,
            risk_level="LOW",
            anomaly_score=None,
            hotspot_count=None,
            supporting_evidence=evidence,
            risk_rule_triggered="LOW_MISSING_ALL_EVIDENCE",
        )
    if clean_anom is None:
        evidence.insert(
            0,
            f"Anomaly score unavailable (warm-up or missing) with hotspot_count={clean_hs}; defaulting conservatively to LOW.",
        )
        return RiskAssessmentResult(
            risk_score=risk_score,
            risk_level="LOW",
            anomaly_score=None,
            hotspot_count=clean_hs,
            supporting_evidence=evidence,
            risk_rule_triggered="LOW_MISSING_ANOMALY_SCORE",
        )
    if clean_hs is None:
        evidence.insert(
            0,
            f"Spatial hotspot data unavailable with anomaly_score={clean_anom:.2f}; uncorroborated observations remain LOW.",
        )
        return RiskAssessmentResult(
            risk_score=risk_score,
            risk_level="LOW",
            anomaly_score=clean_anom,
            hotspot_count=None,
            supporting_evidence=evidence,
            risk_rule_triggered="LOW_MISSING_HOTSPOTS",
        )

    # Precedence 2: Below anomaly threshold OR zero spatial hotspots -> LOW
    if clean_hs < cfg.min_hotspots_for_elevation and clean_anom > cfg.anomaly_high_min:
        evidence.insert(
            0,
            f"Elevated anomaly score ({clean_anom:.2f} > {cfg.anomaly_high_min:.2f}) has 0 spatial hotspots; "
            "uncorroborated statistical spike classified conservatively as LOW (not confirmed pollution).",
        )
        return RiskAssessmentResult(
            risk_score=risk_score,
            risk_level="LOW",
            anomaly_score=clean_anom,
            hotspot_count=clean_hs,
            supporting_evidence=evidence,
            risk_rule_triggered="LOW_UNCORROBORATED_NO_HOTSPOTS",
        )

    if clean_anom < cfg.anomaly_medium_min or clean_hs < cfg.min_hotspots_for_elevation:
        evidence.insert(
            0,
            f"Anomaly score {clean_anom:.2f} (threshold >= {cfg.anomaly_medium_min:.2f}) and "
            f"{clean_hs} hotspot cluster(s) (required >= {cfg.min_hotspots_for_elevation}) yield LOW screening risk.",
        )
        return RiskAssessmentResult(
            risk_score=risk_score,
            risk_level="LOW",
            anomaly_score=clean_anom,
            hotspot_count=clean_hs,
            supporting_evidence=evidence,
            risk_rule_triggered="LOW_BELOW_THRESHOLD_OR_NO_HOTSPOTS",
        )

    # Precedence 3: HIGH (anomaly > 4.5 with hotspot corroboration, or gated future CNN-LSTM hook)
    if clean_anom > cfg.anomaly_high_min:
        evidence.insert(
            0,
            f"Anomaly score {clean_anom:.2f} > {cfg.anomaly_high_min:.2f} corroborated by {clean_hs} "
            "spatial hotspot cluster(s); satellite-derived HIGH screening alert (lab confirmation required).",
        )
        return RiskAssessmentResult(
            risk_score=risk_score,
            risk_level="HIGH",
            anomaly_score=clean_anom,
            hotspot_count=clean_hs,
            supporting_evidence=evidence,
            risk_rule_triggered="HIGH_SCORE_WITH_HOTSPOTS",
        )

    if cfg.enable_cnn_lstm_rule and bool(cnn_lstm_anomaly) and clean_hs >= cfg.min_hotspots_for_elevation:
        evidence.insert(
            0,
            f"CNN-LSTM anomaly independently corroborated by {clean_hs} spatial hotspot cluster(s).",
        )
        return RiskAssessmentResult(
            risk_score=risk_score,
            risk_level="HIGH",
            anomaly_score=clean_anom,
            hotspot_count=clean_hs,
            supporting_evidence=evidence,
            risk_rule_triggered="HIGH_CNN_LSTM_CORROBORATED",
        )

    # Precedence 4: MEDIUM (2.5 <= anomaly_score <= 4.5 AND hotspot_count >= 1)
    evidence.insert(
        0,
        f"Anomaly score {clean_anom:.2f} in [{cfg.anomaly_medium_min:.2f}, {cfg.anomaly_high_min:.2f}] "
        f"corroborated by {clean_hs} spatial hotspot cluster(s).",
    )
    return RiskAssessmentResult(
        risk_score=risk_score,
        risk_level="MEDIUM",
        anomaly_score=clean_anom,
        hotspot_count=clean_hs,
        supporting_evidence=evidence,
        risk_rule_triggered="MEDIUM_SCORE_WITH_HOTSPOTS",
    )
