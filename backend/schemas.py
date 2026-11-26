"""Strict Pydantic v2 response and request schemas matching SPEC.md Section 8."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
EventType = Literal[
    "algal_bloom_proxy",
    "turbidity_sediment_proxy",
    "combined",
    "elevated_anomaly_uncorroborated",
    "baseline_screening",
    "insufficient_evidence",
]


class SpatialHotspot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hotspot_id: str
    lat: float
    lon: float
    pixel_count: int = Field(ge=3)
    peak_ndci: float
    peak_ndti: float
    dominant_signal: Literal["algal_bloom_proxy", "turbidity_sediment_proxy", "combined"]
    severity_score: float = Field(ge=0.0, le=1.0)


class LatestRiskStatus(BaseModel):
    observation_date: str
    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=100.0)
    anomaly_score: float | None
    hotspot_count: int = Field(ge=0)
    active_anomalies: int = Field(ge=0)
    supporting_evidence: list[str]
    risk_rule_triggered: str


class LatestObservation(BaseModel):
    date: str
    cloud_cover: float | None = None
    water_pixel_count: int | None = None
    NDWI_med: float | None = None
    NDCI_med: float | None = None
    NDTI_med: float | None = None
    pred_chla_proxy: float | None = None
    pred_tss_proxy: float | None = None
    baseline_anomaly_score: float | None = None
    is_anomaly: bool
    open_water_pct: float | None = None
    veg_cover_pct: float | None = None
    cnn_lstm_pred_chla: float | None = None
    cnn_lstm_pred_tss: float | None = None
    cnn_lstm_anomaly_score: float | None = None
    cnn_lstm_is_anomaly: bool | None = None


class GeoJSONGeometry(BaseModel):
    type: Literal["Polygon", "MultiPolygon"]
    coordinates: list[Any]


class WaterBodyItem(BaseModel):
    lake_id: str
    lake_name: str
    city: str
    center_lat: float
    center_lon: float
    bbox: list[float] = Field(min_length=4, max_length=4)
    boundary: GeoJSONGeometry
    latest_risk_status: LatestRiskStatus
    latest_observation: LatestObservation | None = None


class WaterBodiesResponse(BaseModel):
    water_bodies: list[WaterBodyItem] = Field(min_length=4, max_length=4)
    disclaimer: str


class SpecTimeseriesRecord(BaseModel):
    date: str
    NDWI_med: float | None
    NDCI_med: float | None
    NDTI_med: float | None
    pred_chla_proxy: float | None
    pred_tss_proxy: float | None
    baseline_anomaly_score: float | None
    is_anomaly: bool
    open_water_pct: float | None
    veg_cover_pct: float | None
    anomaly_drivers: str | None = None
    sufficient_history: bool | None = None


class SpecTimeseriesResponse(BaseModel):
    """SPEC.md Section 8.2 response for GET /api/timeseries/{lake_id}."""

    lake_id: str
    lake_name: str
    count: int = Field(ge=0)
    records: list[SpecTimeseriesRecord]
    disclaimer: str


class PredictionTimeseriesRecord(BaseModel):
    water_body_id: str
    date: str
    cloud_cover: float | None
    water_pixel_count: int | None
    ndwi_mean: float | None
    ndci_mean: float | None
    ndti_mean: float | None
    pred_chla: float | None
    pred_tss: float | None
    anomaly_score: float | None
    is_anomaly: bool


class TimeseriesResponse(BaseModel):
    """Response for GET /api/water-bodies/{water_body_id}/timeseries."""

    water_body_id: str
    lake_name: str
    count: int = Field(ge=0)
    records: list[PredictionTimeseriesRecord]
    disclaimer: str


class HotspotsResponse(BaseModel):
    """SPEC.md Section 8.3 response for GET /api/hotspots/{lake_id}."""

    lake_id: str
    date: str
    crs: Literal["EPSG:4326"] = "EPSG:4326"
    hotspot_count: int = Field(ge=0)
    hotspots: list[SpatialHotspot]
    message: str | None = None
    disclaimer: str


class ProxyEstimatesOut(BaseModel):
    pred_chla_proxy: float | None
    pred_tss_proxy: float | None
    NDCI_med: float | None
    NDTI_med: float | None
    NDWI_med: float | None


class RiskAssessmentOut(BaseModel):
    risk_score: float = Field(ge=0.0, le=100.0)
    risk_level: RiskLevel
    anomaly_score: float | None
    hotspot_count: int | None
    supporting_evidence: list[str]
    risk_rule_triggered: str
    triggered_rules: list[str] = Field(default_factory=list)
    supporting_hotspots: list[SpatialHotspot]
    proxy_estimates: ProxyEstimatesOut


class IncidentSummaryOut(BaseModel):
    event_type: EventType
    risk_level: RiskLevel
    affected_location: str
    supporting_observations: str
    recommended_action: str
    is_lab_confirmed: Literal[False] = False
    generated_by: Literal["ollama", "groq", "gemini", "deterministic_fallback"] = (
        "deterministic_fallback"
    )


class ReportRequest(BaseModel):
    date: str | None = None
    include_llm_summary: bool = True


class ReportResponse(BaseModel):
    """SPEC.md Section 8.4 response for POST /api/report/{lake_id}."""

    lake_id: str
    lake_name: str
    observation_date: str
    risk_assessment: RiskAssessmentOut
    incident_summary: IncidentSummaryOut
    summary_source: Literal["llm", "deterministic_fallback"]
    disclaimer: str


class AnalysisResponse(BaseModel):
    """Response for GET /api/water-bodies/{water_body_id}/analysis."""

    water_body_id: str
    lake_name: str
    observation_date: str
    latest_observation: LatestObservation
    historical_count: int = Field(ge=0)
    historical_observations: list[SpecTimeseriesRecord]
    risk_assessment: RiskAssessmentOut
    hotspots: list[SpatialHotspot]
    incident_summary: IncidentSummaryOut
    summary_source: Literal["llm", "deterministic_fallback"]
    cached_summary: bool = False
    disclaimer: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    backend_ready: bool
    models_ready: bool
    data_ready: bool
    water_bodies_loaded: int = Field(ge=0, le=4)
    models: dict[str, Any]
    datasets: dict[str, Any]
    disclaimer: str


class RefreshResponse(BaseModel):
    water_body_id: str
    lake_name: str
    status: Literal["no_new_data", "newer_data_available", "unavailable"]
    latest_local_date: str
    latest_remote_date: str | None = None
    newer_scene_count: int = Field(ge=0)
    candidate_scenes: list[dict[str, Any]] = Field(default_factory=list)
    datasets_modified: Literal[False] = False
    message: str
    disclaimer: str
