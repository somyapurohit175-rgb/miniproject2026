"""FastAPI router for water bodies, time series, hotspots, analysis, metrics, and STAC refresh."""

from __future__ import annotations

from datetime import date as dt_date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request
import pystac_client
from pystac_client.stac_api_io import StacApiIO

from backend.schemas import (
    AnalysisResponse,
    HotspotsResponse,
    IncidentSummaryOut,
    LatestObservation,
    LatestRiskStatus,
    PredictionTimeseriesRecord,
    ProxyEstimatesOut,
    RefreshResponse,
    ReportRequest,
    ReportResponse,
    RiskAssessmentOut,
    SpatialHotspot,
    SpecTimeseriesRecord,
    SpecTimeseriesResponse,
    TimeseriesResponse,
    WaterBodiesResponse,
    WaterBodyItem,
)
from backend.services.data_loader import (
    DataLoader,
    ObservationDateNotFoundError,
    SCIENTIFIC_DISCLAIMER,
    UnknownWaterBodyError,
)
from shared.llm_summary import IncidentSummary, generate_incident_summary
from shared.risk_engine import RiskConfig, calculate_risk


router = APIRouter(tags=["water-bodies"])

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
STAC_MAX_CLOUD_COVER = 20
STAC_TIMEOUT_SECONDS = 12
EMPTY_HOTSPOTS_MESSAGE = (
    "No spatial hotspots exceeded the historical baseline for this observation date."
)


def _get_loader(request: Request) -> DataLoader:
    loader = getattr(request.app.state, "data_loader", None)
    if loader is None:
        raise HTTPException(status_code=503, detail="DataLoader is not initialized.")
    return loader


def _parse_iso_date(date_str: str) -> dt_date:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.",
        ) from exc


def _handle_data_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (UnknownWaterBodyError, ObservationDateNotFoundError)):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (FileNotFoundError, ValueError)):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=500, detail="Unexpected backend data error.")


def _build_latest_observation_model(obs: dict[str, Any]) -> LatestObservation:
    return LatestObservation(
        date=obs["date"],
        cloud_cover=obs.get("cloud_cover"),
        water_pixel_count=obs.get("water_pixel_count"),
        NDWI_med=obs.get("NDWI_med"),
        NDCI_med=obs.get("NDCI_med"),
        NDTI_med=obs.get("NDTI_med"),
        pred_chla_proxy=obs.get("pred_chla_proxy"),
        pred_tss_proxy=obs.get("pred_tss_proxy"),
        baseline_anomaly_score=obs.get("baseline_anomaly_score"),
        is_anomaly=bool(obs.get("is_anomaly", False)),
        open_water_pct=obs.get("open_water_pct"),
        veg_cover_pct=obs.get("veg_cover_pct"),
        cnn_lstm_pred_chla=obs.get("cnn_lstm_pred_chla"),
        cnn_lstm_pred_tss=obs.get("cnn_lstm_pred_tss"),
        cnn_lstm_anomaly_score=obs.get("cnn_lstm_anomaly_score"),
        cnn_lstm_is_anomaly=obs.get("cnn_lstm_is_anomaly"),
    )


def _evaluate_lake_assessment(
    loader: DataLoader,
    lake_id: str,
    *,
    observation_date: str | None = None,
    config: RiskConfig | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    meta = loader.validate_water_body_id(lake_id)
    history = loader.get_anomaly_observations(lake_id)
    latest_obs = loader.get_latest_observation(lake_id, date=observation_date)
    selected_date = latest_obs["date"]

    try:
        _, hotspots = loader.get_hotspots(lake_id, date=selected_date)
    except ObservationDateNotFoundError:
        hotspots = []

    obs_for_risk = {**latest_obs, "hotspots": hotspots, "hotspot_count": len(hotspots)}
    risk_result = calculate_risk(obs_for_risk, history, config=config)

    proxy_estimates = {
        "pred_chla_proxy": latest_obs.get("pred_chla_proxy"),
        "pred_tss_proxy": latest_obs.get("pred_tss_proxy"),
        "NDCI_med": latest_obs.get("NDCI_med"),
        "NDTI_med": latest_obs.get("NDTI_med"),
        "NDWI_med": latest_obs.get("NDWI_med"),
    }
    risk_assessment = {
        "risk_score": risk_result.risk_score,
        "risk_level": risk_result.risk_level,
        "anomaly_score": risk_result.anomaly_score,
        "hotspot_count": risk_result.hotspot_count,
        "supporting_evidence": risk_result.supporting_evidence,
        "risk_rule_triggered": risk_result.risk_rule_triggered,
        "triggered_rules": risk_result.triggered_rules,
        "supporting_hotspots": hotspots,
        "proxy_estimates": proxy_estimates,
    }
    return meta.lake_name, latest_obs, history, hotspots, proxy_estimates, risk_assessment


def _get_or_generate_summary(
    loader: DataLoader,
    *,
    lake_id: str,
    lake_name: str,
    observation_date: str,
    risk_assessment: dict[str, Any],
    hotspots: list[dict[str, Any]],
    proxy_estimates: dict[str, Any],
    allow_llm: bool,
    regenerate: bool = False,
) -> tuple[IncidentSummary, bool]:
    risk_level = str(risk_assessment["risk_level"])
    if not regenerate:
        cached = loader.get_cached_summary(lake_id, observation_date, risk_level)
        if cached is not None:
            return cached, True

    summary = generate_incident_summary(
        lake_name=lake_name,
        observation_date=observation_date,
        risk_assessment=risk_assessment,
        hotspots=hotspots,
        proxy_estimates=proxy_estimates,
        allow_llm=allow_llm,
    )
    loader.set_cached_summary(lake_id, observation_date, risk_level, summary)
    return summary, False


@router.get("/water-bodies", response_model=WaterBodiesResponse)
def list_water_bodies(request: Request) -> WaterBodiesResponse:
    """Return the four approved water bodies with boundaries and latest risk status."""
    loader = _get_loader(request)
    try:
        items: list[WaterBodyItem] = []
        for meta in loader.get_all_water_bodies():
            boundary = loader.get_boundary(meta.lake_id)
            history = loader.get_anomaly_observations(meta.lake_id)
            latest_obs = loader.get_latest_observation(meta.lake_id)
            obs_date = latest_obs["date"]
            try:
                _, hotspots = loader.get_hotspots(meta.lake_id, date=obs_date)
            except ObservationDateNotFoundError:
                hotspots = []

            risk_res = calculate_risk(
                {**latest_obs, "hotspots": hotspots, "hotspot_count": len(hotspots)},
                history,
                config=RiskConfig(),
            )
            active_anomalies = sum(1 for row in history if row.get("is_anomaly"))

            items.append(
                WaterBodyItem(
                    lake_id=meta.lake_id,
                    lake_name=meta.lake_name,
                    city=meta.city,
                    center_lat=meta.center_lat,
                    center_lon=meta.center_lon,
                    bbox=list(meta.bbox),
                    boundary=boundary,
                    latest_risk_status=LatestRiskStatus(
                        observation_date=obs_date,
                        risk_level=risk_res.risk_level,
                        risk_score=risk_res.risk_score,
                        anomaly_score=risk_res.anomaly_score,
                        hotspot_count=int(risk_res.hotspot_count or 0),
                        active_anomalies=active_anomalies,
                        supporting_evidence=risk_res.supporting_evidence,
                        risk_rule_triggered=risk_res.risk_rule_triggered,
                    ),
                    latest_observation=_build_latest_observation_model(latest_obs),
                )
            )
        return WaterBodiesResponse(water_bodies=items, disclaimer=SCIENTIFIC_DISCLAIMER)
    except Exception as exc:
        raise _handle_data_error(exc) from exc


def _filter_records_by_date(
    records: list[dict[str, Any]],
    start_date: str | None,
    end_date: str | None,
) -> list[dict[str, Any]]:
    start_dt = _parse_iso_date(start_date) if start_date is not None else None
    end_dt = _parse_iso_date(end_date) if end_date is not None else None
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        raise HTTPException(
            status_code=400,
            detail=f"start_date '{start_date}' must not be after end_date '{end_date}'.",
        )
    filtered: list[dict[str, Any]] = []
    for row in records:
        row_date = row["date"]
        if start_date is not None and row_date < start_date:
            continue
        if end_date is not None and row_date > end_date:
            continue
        filtered.append(row)
    return filtered


@router.get("/water-bodies/{water_body_id}/timeseries", response_model=TimeseriesResponse)
def get_water_body_timeseries(
    water_body_id: str,
    request: Request,
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
) -> TimeseriesResponse:
    """Return chronological observations from `sentinel_final_predictions.csv`."""
    loader = _get_loader(request)
    try:
        meta = loader.validate_water_body_id(water_body_id)
        records = _filter_records_by_date(
            loader.get_final_predictions(water_body_id),
            start_date,
            end_date,
        )
        return TimeseriesResponse(
            water_body_id=meta.lake_id,
            lake_name=meta.lake_name,
            count=len(records),
            records=[PredictionTimeseriesRecord(**row) for row in records],
            disclaimer=SCIENTIFIC_DISCLAIMER,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _handle_data_error(exc) from exc


@router.get("/timeseries/{lake_id}", response_model=SpecTimeseriesResponse)
def get_spec_timeseries(
    lake_id: str,
    request: Request,
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
) -> SpecTimeseriesResponse:
    """SPEC.md Section 8.2: Return chronological observations from `sentinel_anomalies.csv`."""
    loader = _get_loader(request)
    try:
        meta = loader.validate_water_body_id(lake_id)
        records = _filter_records_by_date(
            loader.get_anomaly_observations(lake_id),
            start_date,
            end_date,
        )
        spec_records = [
            SpecTimeseriesRecord(
                date=r["date"],
                NDWI_med=r["NDWI_med"],
                NDCI_med=r["NDCI_med"],
                NDTI_med=r["NDTI_med"],
                pred_chla_proxy=r["pred_chla_proxy"],
                pred_tss_proxy=r["pred_tss_proxy"],
                baseline_anomaly_score=r["baseline_anomaly_score"],
                is_anomaly=r["is_anomaly"],
                open_water_pct=r["open_water_pct"],
                veg_cover_pct=r["veg_cover_pct"],
                anomaly_drivers=r["anomaly_drivers"],
                sufficient_history=r["sufficient_history"],
            )
            for r in records
        ]
        return SpecTimeseriesResponse(
            lake_id=meta.lake_id,
            lake_name=meta.lake_name,
            count=len(spec_records),
            records=spec_records,
            disclaimer=SCIENTIFIC_DISCLAIMER,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _handle_data_error(exc) from exc


@router.get("/hotspots/{lake_id}", response_model=HotspotsResponse)
def get_hotspots(
    lake_id: str,
    request: Request,
    date: str | None = Query(default=None),
) -> HotspotsResponse:
    """SPEC.md Section 8.3: Return spatial hotspot clusters from `spatial_hotspots.json`."""
    loader = _get_loader(request)
    if date is not None:
        _parse_iso_date(date)
    try:
        selected_date, clusters = loader.get_hotspots(lake_id, date=date)
        return HotspotsResponse(
            lake_id=lake_id,
            date=selected_date,
            crs="EPSG:4326",
            hotspot_count=len(clusters),
            hotspots=[SpatialHotspot(**c) for c in clusters],
            message=EMPTY_HOTSPOTS_MESSAGE if not clusters else None,
            disclaimer=SCIENTIFIC_DISCLAIMER,
        )
    except Exception as exc:
        raise _handle_data_error(exc) from exc


@router.get("/water-bodies/{water_body_id}/analysis", response_model=AnalysisResponse)
def get_water_body_analysis(
    water_body_id: str,
    request: Request,
    tss_high: float | None = Query(default=None, ge=0.0),
    chla_bloom: float | None = Query(default=None, ge=0.0),
    anomaly_thresh: float | None = Query(default=None, ge=0.0),
    regenerate: bool = Query(default=False),
) -> AnalysisResponse:
    """Return latest/historical observations, deterministic risk, hotspots, and summary."""
    loader = _get_loader(request)
    try:
        config = RiskConfig(
            anomaly_thresh=anomaly_thresh,
            tss_high=tss_high,
            chla_bloom=chla_bloom,
        )
        lake_name, latest_obs, history, hotspots, proxy_estimates, risk_assessment = (
            _evaluate_lake_assessment(loader, water_body_id, config=config)
        )
        selected_date = latest_obs["date"]
        summary, cached = _get_or_generate_summary(
            loader,
            lake_id=water_body_id,
            lake_name=lake_name,
            observation_date=selected_date,
            risk_assessment=risk_assessment,
            hotspots=hotspots,
            proxy_estimates=proxy_estimates,
            allow_llm=True,
            regenerate=regenerate,
        )
        spec_history = [
            SpecTimeseriesRecord(
                date=r["date"],
                NDWI_med=r["NDWI_med"],
                NDCI_med=r["NDCI_med"],
                NDTI_med=r["NDTI_med"],
                pred_chla_proxy=r["pred_chla_proxy"],
                pred_tss_proxy=r["pred_tss_proxy"],
                baseline_anomaly_score=r["baseline_anomaly_score"],
                is_anomaly=r["is_anomaly"],
                open_water_pct=r["open_water_pct"],
                veg_cover_pct=r["veg_cover_pct"],
                anomaly_drivers=r["anomaly_drivers"],
                sufficient_history=r["sufficient_history"],
            )
            for r in history
        ]
        return AnalysisResponse(
            water_body_id=water_body_id,
            lake_name=lake_name,
            observation_date=selected_date,
            latest_observation=_build_latest_observation_model(latest_obs),
            historical_count=len(spec_history),
            historical_observations=spec_history,
            risk_assessment=RiskAssessmentOut(
                **{
                    **risk_assessment,
                    "supporting_hotspots": [SpatialHotspot(**h) for h in hotspots],
                    "proxy_estimates": ProxyEstimatesOut(**proxy_estimates),
                }
            ),
            hotspots=[SpatialHotspot(**h) for h in hotspots],
            incident_summary=IncidentSummaryOut(**summary.model_dump()),
            summary_source=summary.summary_source,
            cached_summary=cached,
            disclaimer=SCIENTIFIC_DISCLAIMER,
        )
    except Exception as exc:
        raise _handle_data_error(exc) from exc


@router.post("/report/{lake_id}", response_model=ReportResponse)
def generate_report(
    lake_id: str,
    request: Request,
    payload: ReportRequest | None = Body(default=None),
) -> ReportResponse:
    """SPEC.md Section 8.4: Generate deterministic risk report + guardrailed summary."""
    loader = _get_loader(request)
    req = payload or ReportRequest()
    if req.date is not None:
        _parse_iso_date(req.date)

    try:
        lake_name, latest_obs, _, hotspots, proxy_estimates, risk_assessment = (
            _evaluate_lake_assessment(
                loader,
                lake_id,
                observation_date=req.date,
                config=RiskConfig(),
            )
        )
        selected_date = latest_obs["date"]
        summary, _ = _get_or_generate_summary(
            loader,
            lake_id=lake_id,
            lake_name=lake_name,
            observation_date=selected_date,
            risk_assessment=risk_assessment,
            hotspots=hotspots,
            proxy_estimates=proxy_estimates,
            allow_llm=req.include_llm_summary,
            regenerate=False,
        )
        return ReportResponse(
            lake_id=lake_id,
            lake_name=lake_name,
            observation_date=selected_date,
            risk_assessment=RiskAssessmentOut(
                **{
                    **risk_assessment,
                    "supporting_hotspots": [SpatialHotspot(**h) for h in hotspots],
                    "proxy_estimates": ProxyEstimatesOut(**proxy_estimates),
                }
            ),
            incident_summary=IncidentSummaryOut(**summary.model_dump()),
            summary_source=summary.summary_source,
            disclaimer=SCIENTIFIC_DISCLAIMER,
        )
    except Exception as exc:
        raise _handle_data_error(exc) from exc


@router.get("/metrics")
def get_metrics(request: Request) -> dict[str, Any]:
    """Return verified model evaluation metrics from `models/metrics_comparison.json`."""
    loader = _get_loader(request)
    try:
        metrics = loader.get_metrics()
        return {**metrics, "disclaimer": SCIENTIFIC_DISCLAIMER}
    except Exception as exc:
        raise _handle_data_error(exc) from exc


def _query_planetary_computer_newer_scenes(
    bbox: list[float],
    latest_local_date: str,
) -> list[dict[str, Any]]:
    """Query Microsoft Planetary Computer STAC API with strict 12s timeout (read-only)."""
    start_dt = datetime.strptime(latest_local_date, "%Y-%m-%d").date() + timedelta(days=1)
    end_dt = max(dt_date.today(), start_dt)
    stac_io = StacApiIO(timeout=STAC_TIMEOUT_SECONDS)
    catalog = pystac_client.Client.open(STAC_API_URL, stac_io=stac_io)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_dt.isoformat()}/{end_dt.isoformat()}",
        query={"eo:cloud_cover": {"lt": STAC_MAX_CLOUD_COVER}},
        max_items=10,
    )
    scenes: list[dict[str, Any]] = []
    for item in search.items():
        scenes.append(
            {
                "scene_id": item.id,
                "datetime": item.datetime.isoformat() if item.datetime else None,
                "eo_cloud_cover": item.properties.get("eo:cloud_cover"),
            }
        )
    scenes.sort(key=lambda s: str(s.get("datetime") or ""), reverse=True)
    return scenes


@router.post("/water-bodies/{water_body_id}/refresh", response_model=RefreshResponse)
def refresh_water_body_availability(
    water_body_id: str,
    request: Request,
) -> RefreshResponse:
    """Check Planetary Computer STAC for newer Sentinel-2 L2A scenes without modifying data."""
    loader = _get_loader(request)
    try:
        meta = loader.validate_water_body_id(water_body_id)
        latest_obs = loader.get_latest_observation(water_body_id)
        latest_local_date = latest_obs["date"]
    except Exception as exc:
        raise _handle_data_error(exc) from exc

    try:
        scenes = _query_planetary_computer_newer_scenes(meta.bbox, latest_local_date)
        if scenes:
            latest_remote = str(scenes[0].get("datetime") or "")[:10] or None
            return RefreshResponse(
                water_body_id=meta.lake_id,
                lake_name=meta.lake_name,
                status="newer_data_available",
                latest_local_date=latest_local_date,
                latest_remote_date=latest_remote,
                newer_scene_count=len(scenes),
                candidate_scenes=scenes[:5],
                datasets_modified=False,
                message=(
                    f"Found {len(scenes)} Sentinel-2 L2A scene(s) with <{STAC_MAX_CLOUD_COVER}% "
                    f"tile cloud cover newer than {latest_local_date}. Local datasets were not modified."
                ),
                disclaimer=SCIENTIFIC_DISCLAIMER,
            )
        return RefreshResponse(
            water_body_id=meta.lake_id,
            lake_name=meta.lake_name,
            status="no_new_data",
            latest_local_date=latest_local_date,
            latest_remote_date=None,
            newer_scene_count=0,
            candidate_scenes=[],
            datasets_modified=False,
            message=(
                f"No Sentinel-2 L2A scenes with <{STAC_MAX_CLOUD_COVER}% tile cloud cover "
                f"found after {latest_local_date}."
            ),
            disclaimer=SCIENTIFIC_DISCLAIMER,
        )
    except Exception:
        return RefreshResponse(
            water_body_id=meta.lake_id,
            lake_name=meta.lake_name,
            status="unavailable",
            latest_local_date=latest_local_date,
            latest_remote_date=None,
            newer_scene_count=0,
            candidate_scenes=[],
            datasets_modified=False,
            message=(
                "Microsoft Planetary Computer STAC catalog is currently unreachable or timed out; "
                "continuing to serve verified local observations."
            ),
            disclaimer=SCIENTIFIC_DISCLAIMER,
        )
