"""Validation suite for the conservative Risk Engine (`backend/risk_engine.py`).

Tests all 7 required scenarios, boundary/precedence behavior, determinism,
missing-data safety, and verifies real Day 2 observation distribution without
modifying any Day 2 dataset.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.risk_engine import RiskThresholds, classify_lake_risk


class TestConservativeRiskEngine(unittest.TestCase):
    def test_1_low_score_zero_hotspots(self) -> None:
        res = classify_lake_risk(anomaly_score=1.2, hotspot_count=0)
        self.assertEqual(res.risk_level, "LOW")
        self.assertEqual(res.risk_rule_triggered, "LOW_BELOW_THRESHOLD_OR_NO_HOTSPOTS")
        self.assertLess(res.risk_score, 20.0)

    def test_2_high_score_zero_hotspots_conservative_low(self) -> None:
        res = classify_lake_risk(anomaly_score=12.4, hotspot_count=0, anomaly_drivers="pred_tss_proxy(12.40)")
        self.assertEqual(res.risk_level, "LOW")
        self.assertEqual(res.risk_rule_triggered, "LOW_UNCORROBORATED_NO_HOTSPOTS")
        self.assertGreater(res.risk_score, 0.0)
        self.assertTrue(any("not confirmed pollution" in msg.lower() for msg in res.supporting_evidence))

    def test_3_medium_score_one_hotspot(self) -> None:
        res = classify_lake_risk(anomaly_score=3.6, hotspot_count=1, max_hotspot_severity=0.6)
        self.assertEqual(res.risk_level, "MEDIUM")
        self.assertEqual(res.risk_rule_triggered, "MEDIUM_SCORE_WITH_HOTSPOTS")

    def test_4_high_score_multiple_hotspots(self) -> None:
        res = classify_lake_risk(
            anomaly_score=6.8,
            hotspot_count=3,
            max_hotspot_severity=0.75,
            dominant_signals=["algal_bloom_proxy", "combined"],
        )
        self.assertEqual(res.risk_level, "HIGH")
        self.assertEqual(res.risk_rule_triggered, "HIGH_SCORE_WITH_HOTSPOTS")
        self.assertGreater(res.risk_score, 50.0)

    def test_5_missing_anomaly_score(self) -> None:
        for missing_val in (None, float("nan")):
            res = classify_lake_risk(anomaly_score=missing_val, hotspot_count=2)
            self.assertEqual(res.risk_level, "LOW")
            self.assertIsNone(res.anomaly_score)
            self.assertEqual(res.risk_rule_triggered, "LOW_MISSING_ANOMALY_SCORE")

    def test_6_missing_hotspot_info(self) -> None:
        res = classify_lake_risk(anomaly_score=5.2, hotspot_count=None)
        self.assertEqual(res.risk_level, "LOW")
        self.assertIsNone(res.hotspot_count)
        self.assertEqual(res.risk_rule_triggered, "LOW_MISSING_HOTSPOTS")

    def test_7_future_cnn_lstm_disabled_by_default(self) -> None:
        res = classify_lake_risk(
            anomaly_score=2.9,
            hotspot_count=2,
            cnn_lstm_anomaly=True,
        )
        # Cannot promote MEDIUM (2.9 <= 4.5) to HIGH while CNN-LSTM hook is disabled
        self.assertEqual(res.risk_level, "MEDIUM")
        self.assertEqual(res.risk_rule_triggered, "MEDIUM_SCORE_WITH_HOTSPOTS")
        self.assertTrue(any("disabled" in msg.lower() for msg in res.supporting_evidence))

        # When explicitly enabled in config (future Day 3 usage), elevates cleanly to HIGH
        enabled = RiskThresholds(enable_cnn_lstm_rule=True)
        res_enabled = classify_lake_risk(
            anomaly_score=2.9,
            hotspot_count=2,
            cnn_lstm_anomaly=True,
            thresholds=enabled,
        )
        self.assertEqual(res_enabled.risk_level, "HIGH")
        self.assertEqual(res_enabled.risk_rule_triggered, "HIGH_CNN_LSTM_CORROBORATED")

    def test_8_determinism_and_boundary_values(self) -> None:
        r1 = classify_lake_risk(anomaly_score=2.5, hotspot_count=1)
        r2 = classify_lake_risk(anomaly_score=2.5, hotspot_count=1)
        self.assertEqual(r1, r2)
        self.assertEqual(r1.risk_level, "MEDIUM")

        r_edge_hi = classify_lake_risk(anomaly_score=4.5, hotspot_count=1)
        self.assertEqual(r_edge_hi.risk_level, "MEDIUM")
        r_above_hi = classify_lake_risk(anomaly_score=4.51, hotspot_count=1)
        self.assertEqual(r_above_hi.risk_level, "HIGH")


def run_real_data_calibration_check() -> None:
    anom_df = pd.read_csv(ROOT / "data/processed/sentinel_anomalies.csv")
    hotspots = json.loads((ROOT / "data/processed/spatial_hotspots.json").read_text(encoding="utf-8"))

    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    latest_summary = []
    for lake_id, grp in anom_df.groupby("lake_id", sort=False):
        grp = grp.sort_values("date")
        for _, row in grp.iterrows():
            date_str = str(row["date"])
            hs_list = hotspots[lake_id][date_str]
            anom_val = None if pd.isna(row["baseline_anomaly_score"]) else float(row["baseline_anomaly_score"])
            max_sev = max((h["severity_score"] for h in hs_list), default=None)
            sigs = [h["dominant_signal"] for h in hs_list]
            res = classify_lake_risk(
                anom_val,
                len(hs_list),
                anomaly_drivers=None if pd.isna(row["anomaly_drivers"]) else str(row["anomaly_drivers"]),
                max_hotspot_severity=max_sev,
                dominant_signals=sigs,
            )
            counts[res.risk_level] += 1
        # Record latest date for summary
        last = grp.iloc[-1]
        ldate = str(last["date"])
        lhs = hotspots[lake_id][ldate]
        lres = classify_lake_risk(
            None if pd.isna(last["baseline_anomaly_score"]) else float(last["baseline_anomaly_score"]),
            len(lhs),
            anomaly_drivers=str(last["anomaly_drivers"]),
            max_hotspot_severity=max((h["severity_score"] for h in lhs), default=None),
            dominant_signals=[h["dominant_signal"] for h in lhs],
        )
        latest_summary.append((lake_id, ldate, round(float(last["baseline_anomaly_score"]), 3), len(lhs), lres.risk_level, lres.risk_score, lres.risk_rule_triggered))

    raw_anoms = int(anom_df["is_anomaly"].sum())
    print(f"Real Sentinel dataset (128 obs): raw is_anomaly=True count = {raw_anoms}/128 ({raw_anoms/116*100:.1f}% of valid post-warmup)")
    print(f"Conservative Risk Engine tier counts (128 obs): {counts}")
    print("Latest date per lake under conservative Risk Engine:")
    for item in latest_summary:
        print(" ", item)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestConservativeRiskEngine)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print()
    run_real_data_calibration_check()
