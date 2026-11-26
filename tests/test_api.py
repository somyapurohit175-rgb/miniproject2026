"""Automated API & inference service tests using FastAPI TestClient."""

from __future__ import annotations

from pathlib import Path
import shutil

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from backend.main import ROOT_DIR, create_app
from backend.services.inference import ModelInferenceService


def test_health_endpoint() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["backend_ready"] is True
        assert body["models_ready"] is True
        assert body["data_ready"] is True
        assert body["water_bodies_loaded"] == 4
        assert body["models"]["components"]["cnn_lstm_fp32"]["loaded"] is True
        assert body["models"]["components"]["cnn_lstm_int8"]["loaded"] is True
        assert body["models"]["components"]["baseline_chla"]["loaded"] is True
        assert body["models"]["components"]["baseline_tss"]["loaded"] is True
        assert "not laboratory-confirmed" in body["disclaimer"].lower()


def test_water_bodies_endpoint() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/water-bodies")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["water_bodies"]) == 4
        ids = [wb["lake_id"] for wb in body["water_bodies"]]
        assert ids == ["bellandur", "hussain_sagar", "yamuna_okhla", "sukhna"]
        for wb in body["water_bodies"]:
            assert wb["boundary"]["type"] in ("Polygon", "MultiPolygon")
            assert wb["latest_risk_status"]["risk_level"] in ("LOW", "MEDIUM", "HIGH")
            assert 0.0 <= wb["latest_risk_status"]["risk_score"] <= 100.0
            assert wb["latest_observation"] is not None


def test_timeseries_endpoints_and_unknown_lake_404() -> None:
    app = create_app()
    with TestClient(app) as client:
        # Prompt route (sentinel_final_predictions.csv)
        resp = client.get("/api/water-bodies/bellandur/timeseries")
        assert resp.status_code == 200
        body = resp.json()
        assert body["water_body_id"] == "bellandur"
        assert body["count"] == 32
        dates = [r["date"] for r in body["records"]]
        assert dates == sorted(dates)
        assert body["records"][0]["pred_chla"] is None  # Warm-up NaN preserved as null

        # SPEC.md Section 8.2 route (sentinel_anomalies.csv)
        spec_resp = client.get("/api/timeseries/bellandur")
        assert spec_resp.status_code == 200
        spec_body = spec_resp.json()
        assert spec_body["lake_id"] == "bellandur"
        assert spec_body["count"] == 32
        assert spec_body["records"][0]["baseline_anomaly_score"] is None

        # Unknown lake ID returns 404
        assert client.get("/api/water-bodies/unknown_lake/timeseries").status_code == 404
        assert client.get("/api/timeseries/unknown_lake").status_code == 404

        # Date filtering works and invalid date range returns 400
        filtered = client.get(
            "/api/water-bodies/bellandur/timeseries",
            params={"start_date": "2026-01-01", "end_date": "2026-12-31"},
        )
        assert filtered.status_code == 200
        f_body = filtered.json()
        assert 0 < f_body["count"] < 32
        assert all("2026-01-01" <= r["date"] <= "2026-12-31" for r in f_body["records"])

        assert (
            client.get(
                "/api/water-bodies/bellandur/timeseries",
                params={"start_date": "2026-12-01", "end_date": "2026-01-01"},
            ).status_code
            == 400
        )


def test_hotspots_and_report_spec_endpoints() -> None:
    app = create_app()
    with TestClient(app) as client:
        hs_resp = client.get("/api/hotspots/bellandur")
        assert hs_resp.status_code == 200
        hs_body = hs_resp.json()
        assert hs_body["crs"] == "EPSG:4326"
        assert hs_body["hotspot_count"] == len(hs_body["hotspots"])

        # Empty hotspots date returns 200 with [] and informative message
        empty_hs = client.get("/api/hotspots/hussain_sagar?date=2026-11-18")
        assert empty_hs.status_code == 200
        assert empty_hs.json()["hotspot_count"] == 0
        assert empty_hs.json()["hotspots"] == []
        assert empty_hs.json()["message"] is not None

        # Bad date format -> 400
        assert client.get("/api/hotspots/bellandur?date=23-11-2026").status_code == 400

        # SPEC.md POST /api/report/{lake_id}
        rep_resp = client.post(
            "/api/report/bellandur",
            json={"include_llm_summary": False},
        )
        assert rep_resp.status_code == 200
        rep_body = rep_resp.json()
        assert rep_body["risk_assessment"]["risk_level"] == rep_body["incident_summary"]["risk_level"]
        assert rep_body["incident_summary"]["is_lab_confirmed"] is False
        assert rep_body["summary_source"] == "deterministic_fallback"


def test_analysis_endpoint_and_summary_cache(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    app = create_app()
    with TestClient(app) as client:
        r1 = client.get("/api/water-bodies/bellandur/analysis")
        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["water_body_id"] == "bellandur"
        assert b1["historical_count"] == 32
        assert b1["cached_summary"] is False
        assert b1["incident_summary"]["is_lab_confirmed"] is False
        assert b1["incident_summary"]["risk_level"] == b1["risk_assessment"]["risk_level"]

        # Second call uses cached summary
        r2 = client.get("/api/water-bodies/bellandur/analysis")
        assert r2.status_code == 200
        assert r2.json()["cached_summary"] is True

        # regenerate=true bypasses cache
        r3 = client.get(
            "/api/water-bodies/bellandur/analysis",
            params={"regenerate": "true", "anomaly_thresh": 2.0, "chla_bloom": 40.0, "tss_high": 10.0},
        )
        assert r3.status_code == 200
        b3 = r3.json()
        assert b3["cached_summary"] is False
        assert any("chla proxy" in ev.lower() for ev in b3["risk_assessment"]["supporting_evidence"])

        # Changing threshold enough to change risk_level misses cache even when regenerate=false
        r4 = client.get(
            "/api/water-bodies/bellandur/analysis",
            params={"anomaly_thresh": 10.0, "regenerate": "false"},
        )
        assert r4.status_code == 200
        b4 = r4.json()
        assert b4["risk_assessment"]["risk_level"] == "LOW"
        assert b4["incident_summary"]["risk_level"] == "LOW"
        assert b4["cached_summary"] is False


def test_metrics_endpoint() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "day2_gloria_baselines" in body
        assert "day3_cnn_lstm" in body
        assert "deployment_inference_formats" in body


def test_refresh_endpoint_handles_network_success_and_failure(monkeypatch) -> None:
    app = create_app()
    with TestClient(app) as client:
        # Mock network failure -> graceful 200 with status="unavailable"
        def fail_stac(bbox: list[float], latest_local_date: str):
            raise TimeoutError("Simulated STAC timeout")

        monkeypatch.setattr(
            "backend.routers.water_bodies._query_planetary_computer_newer_scenes",
            fail_stac,
        )
        r_fail = client.post("/api/water-bodies/sukhna/refresh")
        assert r_fail.status_code == 200
        b_fail = r_fail.json()
        assert b_fail["status"] == "unavailable"
        assert b_fail["datasets_modified"] is False

        # Mock newer scenes available
        def ok_stac(bbox: list[float], latest_local_date: str):
            return [{"scene_id": "S2A_TEST", "datetime": "2026-11-26T05:00:00Z", "eo_cloud_cover": 2.5}]

        monkeypatch.setattr(
            "backend.routers.water_bodies._query_planetary_computer_newer_scenes",
            ok_stac,
        )
        r_ok = client.post("/api/water-bodies/sukhna/refresh")
        assert r_ok.status_code == 200
        b_ok = r_ok.json()
        assert b_ok["status"] == "newer_data_available"
        assert b_ok["newer_scene_count"] == 1
        assert b_ok["datasets_modified"] is False

        # Unknown lake ID -> 404
        assert client.post("/api/water-bodies/invalid_lake/refresh").status_code == 404


def test_model_inference_service_real_models() -> None:
    svc = ModelInferenceService(root_dir=ROOT_DIR, preferred_onnx="fp32")
    svc.load_all()
    assert svc.is_ready()

    # Baseline prediction using first row of sentinel_anomalies.csv
    df_anom = pd.read_csv(ROOT_DIR / "data" / "processed" / "sentinel_anomalies.csv")
    row0 = df_anom.iloc[0].to_dict()
    preds = svc.predict_quality(row0)
    assert preds["pred_chla_proxy"] > 0.0
    assert preds["pred_tss_proxy"] > 0.0
    # Verify exact match with Day 2 stored proxy on row 0
    assert np.isclose(preds["pred_chla_proxy"], float(row0["pred_chla_proxy"]), rtol=1e-4)
    assert np.isclose(preds["pred_tss_proxy"], float(row0["pred_tss_proxy"]), rtol=1e-4)

    # CNN-LSTM ONNX anomaly scoring on real bellandur patch sequence t=0..3 -> target t=4
    npz = np.load(ROOT_DIR / "data" / "processed" / "sentinel_patches" / "bellandur.npz")
    patch_seq = npz["patches"][0:4]  # (4, 9, 64, 64)
    row4 = df_anom[(df_anom["lake_id"] == "bellandur")].sort_values("date").iloc[4]
    actual_vec = [
        float(row4["NDWI_med"]),
        float(row4["NDCI_med"]),
        float(row4["NDTI_med"]),
        float(row4["pred_chla_proxy"]),
        float(row4["pred_tss_proxy"]),
    ]

    fp32_out = svc.score_sequence_anomaly(patch_seq, actual_vec, variant="fp32")
    assert fp32_out["available"] is True
    assert fp32_out["anomaly_score"] >= 0.0
    # Compare with sentinel_final_predictions.csv row 4 for bellandur
    df_final = pd.read_csv(ROOT_DIR / "data" / "processed" / "sentinel_final_predictions.csv")
    f_row4 = df_final[df_final["water_body_id"] == "bellandur"].sort_values("date").iloc[4]
    assert np.isclose(fp32_out["anomaly_score"], float(f_row4["anomaly_score"]), atol=1e-4)
    assert np.isclose(fp32_out["predicted_raw"]["Chla"], float(f_row4["pred_chla"]), rtol=1e-4)

    # INT8 variant also works
    int8_out = svc.score_sequence_anomaly(patch_seq, actual_vec, variant="int8")
    assert int8_out["available"] is True

    # Missing inputs return unavailable without fabricating
    unavail = svc.score_sequence_anomaly(None, None)
    assert unavail["available"] is False
    assert unavail["anomaly_score"] is None

    # Invalid tensor shape raises ValueError
    with pytest.raises(ValueError):
        svc.score_sequence_anomaly(np.zeros((2, 9, 64, 64), dtype=np.float32), actual_vec)

    # Missing feature raises ValueError
    with pytest.raises(ValueError):
        svc.predict_quality({"B02": 0.05})


def test_missing_and_malformed_data_and_model_failure_degraded(tmp_path: Path) -> None:
    # Create an isolated temporary root with missing models and malformed CSV
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "lake_boundaries").mkdir(parents=True)
    (tmp_path / "backend").mkdir(parents=True)
    shutil.copy2(
        ROOT_DIR / "backend" / "cnn_lstm_norm_stats.json",
        tmp_path / "backend" / "cnn_lstm_norm_stats.json",
    )
    # Write malformed CSV
    (tmp_path / "data" / "processed" / "sentinel_anomalies.csv").write_text("bad_col\n1\n", encoding="utf-8")

    app = create_app(root_dir=tmp_path)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        hb = health.json()
        assert hb["status"] == "degraded"
        assert hb["backend_ready"] is True
        assert hb["models_ready"] is False
        assert hb["data_ready"] is False
        assert hb["water_bodies_loaded"] == 0
        assert hb["models"]["components"]["cnn_lstm_fp32"]["loaded"] is False

        # Endpoints return 503 when artifacts are missing/malformed
        assert client.get("/api/metrics").status_code == 503
        assert client.get("/api/water-bodies").status_code == 503
        assert client.get("/api/timeseries/bellandur").status_code == 503
