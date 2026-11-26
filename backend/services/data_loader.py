"""DataLoader for processed satellite CSVs, GeoJSON boundaries, hotspots, and metrics.

Reads actual persisted repository files relative to repository root (`ROOT_DIR`)
and maintains an in-memory incident summary cache keyed by
`(water_body_id, observation_date, risk_level)`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
SCIENTIFIC_DISCLAIMER = (
    "All values are satellite-derived screening estimates, not laboratory-confirmed measurements."
)


@dataclass(frozen=True)
class WaterBodyMeta:
    lake_id: str
    lake_name: str
    city: str
    center_lat: float
    center_lon: float
    bbox: list[float]
    geometry_type: str


WATER_BODY_REGISTRY: dict[str, WaterBodyMeta] = {
    "bellandur": WaterBodyMeta(
        lake_id="bellandur",
        lake_name="Bellandur Lake",
        city="Bengaluru",
        center_lat=12.9342,
        center_lon=77.6628,
        bbox=[77.645, 12.920, 77.685, 12.946],
        geometry_type="MultiPolygon",
    ),
    "hussain_sagar": WaterBodyMeta(
        lake_id="hussain_sagar",
        lake_name="Hussain Sagar",
        city="Hyderabad",
        center_lat=17.4248,
        center_lon=78.4733,
        bbox=[78.455, 17.408, 78.492, 17.442],
        geometry_type="Polygon",
    ),
    "yamuna_okhla": WaterBodyMeta(
        lake_id="yamuna_okhla",
        lake_name="Yamuna River (Okhla Stretch)",
        city="Delhi / Noida",
        center_lat=28.5585,
        center_lon=77.3076,
        bbox=[77.292, 28.545, 77.322, 28.575],
        geometry_type="Polygon",
    ),
    "sukhna": WaterBodyMeta(
        lake_id="sukhna",
        lake_name="Sukhna Lake",
        city="Chandigarh",
        center_lat=30.7421,
        center_lon=76.8188,
        bbox=[76.803, 30.733, 76.835, 30.752],
        geometry_type="Polygon",
    ),
}

OFFICIAL_LAKE_IDS: tuple[str, ...] = tuple(WATER_BODY_REGISTRY.keys())


class UnknownWaterBodyError(KeyError):
    """Raised when a requested water_body_id / lake_id is not in the official 4-lake registry."""


class ObservationDateNotFoundError(KeyError):
    """Raised when a valid formatted date is absent for a lake."""


def _clean_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        if pd.isna(val):
            return None
        num = float(val)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _clean_int(val: Any) -> int | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        if pd.isna(val):
            return None
        return int(val)
    except (TypeError, ValueError):
        return None


def _clean_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    try:
        if pd.isna(val):
            return False
    except TypeError:
        pass
    return bool(val)


def _clean_str(val: Any) -> str | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except TypeError:
        pass
    s = str(val).strip()
    return s if s else None


class DataLoader:
    """Loads and serves verified project datasets and boundaries."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else ROOT_DIR
        self.processed_dir = self.root_dir / "data" / "processed"
        self.models_dir = self.root_dir / "models"

        self.anomalies_df: pd.DataFrame | None = None
        self.predictions_df: pd.DataFrame | None = None
        self.hotspots: dict[str, dict[str, list[dict[str, Any]]]] | None = None
        self.boundaries: dict[str, dict[str, Any]] = {}
        self.metrics_comparison: dict[str, Any] | None = None

        self.load_errors: dict[str, str] = {}
        self.summary_cache: dict[tuple[str, str, str], Any] = {}

    def load_all(self) -> None:
        """Load all required CSV, JSON, and GeoJSON artifacts into memory."""
        self.load_errors.clear()
        self._load_anomalies_csv()
        self._load_predictions_csv()
        self._load_hotspots_json()
        self._load_boundaries()
        self._load_metrics_json()

    def _load_anomalies_csv(self) -> None:
        path = self.processed_dir / "sentinel_anomalies.csv"
        try:
            if not path.is_file():
                raise FileNotFoundError(f"Missing dataset: {path}")
            df = pd.read_csv(path)
            required = {
                "lake_id",
                "date",
                "NDWI_med",
                "NDCI_med",
                "NDTI_med",
                "pred_chla_proxy",
                "pred_tss_proxy",
                "baseline_anomaly_score",
                "is_anomaly",
                "open_water_pct",
                "veg_cover_pct",
            }
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"sentinel_anomalies.csv missing columns: {sorted(missing)}")
            df["date"] = df["date"].astype(str)
            self.anomalies_df = df.sort_values(["lake_id", "date"]).reset_index(drop=True)
        except Exception as exc:
            self.anomalies_df = None
            self.load_errors["sentinel_anomalies"] = str(exc)

    def _load_predictions_csv(self) -> None:
        path = self.processed_dir / "sentinel_final_predictions.csv"
        try:
            if not path.is_file():
                raise FileNotFoundError(f"Missing dataset: {path}")
            df = pd.read_csv(path)
            required = {
                "water_body_id",
                "date",
                "cloud_cover",
                "water_pixel_count",
                "ndwi_mean",
                "ndci_mean",
                "ndti_mean",
                "pred_chla",
                "pred_tss",
                "anomaly_score",
                "is_anomaly",
            }
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"sentinel_final_predictions.csv missing columns: {sorted(missing)}")
            df["date"] = df["date"].astype(str)
            self.predictions_df = df.sort_values(["water_body_id", "date"]).reset_index(drop=True)
        except Exception as exc:
            self.predictions_df = None
            self.load_errors["sentinel_final_predictions"] = str(exc)

    def _load_hotspots_json(self) -> None:
        path = self.processed_dir / "spatial_hotspots.json"
        try:
            if not path.is_file():
                raise FileNotFoundError(f"Missing hotspots JSON: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("spatial_hotspots.json must be a JSON object keyed by lake_id.")
            self.hotspots = data
        except Exception as exc:
            self.hotspots = None
            self.load_errors["spatial_hotspots"] = str(exc)

    def _load_boundaries(self) -> None:
        self.boundaries = {}
        bdir = self.processed_dir / "lake_boundaries"
        for lake_id in OFFICIAL_LAKE_IDS:
            path = bdir / f"{lake_id}.geojson"
            try:
                if not path.is_file():
                    raise FileNotFoundError(f"Missing boundary GeoJSON: {path}")
                gj = json.loads(path.read_text(encoding="utf-8"))
                geom = gj["features"][0]["geometry"]
                if not isinstance(geom, dict) or "type" not in geom or "coordinates" not in geom:
                    raise ValueError(f"Invalid GeoJSON geometry in {path.name}")
                self.boundaries[lake_id] = geom
            except Exception as exc:
                self.load_errors[f"boundary_{lake_id}"] = str(exc)

    def _load_metrics_json(self) -> None:
        path = self.models_dir / "metrics_comparison.json"
        try:
            if not path.is_file():
                raise FileNotFoundError(f"Missing metrics JSON: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("metrics_comparison.json must be a JSON object.")
            self.metrics_comparison = data
        except Exception as exc:
            self.metrics_comparison = None
            self.load_errors["metrics_comparison"] = str(exc)

    def is_ready(self) -> bool:
        return (
            self.anomalies_df is not None
            and self.predictions_df is not None
            and self.hotspots is not None
            and len(self.boundaries) == len(OFFICIAL_LAKE_IDS)
            and self.metrics_comparison is not None
        )

    def get_status(self) -> dict[str, Any]:
        return {
            "ready": self.is_ready(),
            "datasets": {
                "sentinel_anomalies": {
                    "loaded": self.anomalies_df is not None,
                    "rows": int(len(self.anomalies_df)) if self.anomalies_df is not None else 0,
                    "error": self.load_errors.get("sentinel_anomalies"),
                },
                "sentinel_final_predictions": {
                    "loaded": self.predictions_df is not None,
                    "rows": int(len(self.predictions_df)) if self.predictions_df is not None else 0,
                    "error": self.load_errors.get("sentinel_final_predictions"),
                },
                "spatial_hotspots": {
                    "loaded": self.hotspots is not None,
                    "lakes": len(self.hotspots) if self.hotspots is not None else 0,
                    "error": self.load_errors.get("spatial_hotspots"),
                },
                "lake_boundaries": {
                    "loaded": len(self.boundaries) == len(OFFICIAL_LAKE_IDS),
                    "count": len(self.boundaries),
                    "errors": {
                        k: v for k, v in self.load_errors.items() if k.startswith("boundary_")
                    },
                },
                "metrics_comparison": {
                    "loaded": self.metrics_comparison is not None,
                    "error": self.load_errors.get("metrics_comparison"),
                },
            },
        }

    @staticmethod
    def validate_water_body_id(water_body_id: str) -> WaterBodyMeta:
        if water_body_id not in WATER_BODY_REGISTRY:
            raise UnknownWaterBodyError(
                f"Unknown lake_id '{water_body_id}'. Valid lake_ids: {list(OFFICIAL_LAKE_IDS)}."
            )
        return WATER_BODY_REGISTRY[water_body_id]

    def get_all_water_bodies(self) -> list[WaterBodyMeta]:
        return [WATER_BODY_REGISTRY[lid] for lid in OFFICIAL_LAKE_IDS]

    def get_boundary(self, water_body_id: str) -> dict[str, Any]:
        self.validate_water_body_id(water_body_id)
        if water_body_id not in self.boundaries:
            raise FileNotFoundError(
                f"Boundary GeoJSON unavailable for '{water_body_id}': "
                f"{self.load_errors.get(f'boundary_{water_body_id}', 'not loaded')}"
            )
        return self.boundaries[water_body_id]

    def get_anomaly_observations(self, water_body_id: str) -> list[dict[str, Any]]:
        """Return chronological observations from `sentinel_anomalies.csv`."""
        self.validate_water_body_id(water_body_id)
        if self.anomalies_df is None:
            raise FileNotFoundError(
                f"sentinel_anomalies.csv unavailable: {self.load_errors.get('sentinel_anomalies')}"
            )
        sub = self.anomalies_df[self.anomalies_df["lake_id"] == water_body_id].sort_values("date")
        records: list[dict[str, Any]] = []
        for _, row in sub.iterrows():
            records.append(
                {
                    "date": str(row["date"]),
                    "NDWI_med": _clean_float(row.get("NDWI_med")),
                    "NDCI_med": _clean_float(row.get("NDCI_med")),
                    "NDTI_med": _clean_float(row.get("NDTI_med")),
                    "pred_chla_proxy": _clean_float(row.get("pred_chla_proxy")),
                    "pred_tss_proxy": _clean_float(row.get("pred_tss_proxy")),
                    "baseline_anomaly_score": _clean_float(row.get("baseline_anomaly_score")),
                    "is_anomaly": _clean_bool(row.get("is_anomaly")),
                    "open_water_pct": _clean_float(row.get("open_water_pct")),
                    "veg_cover_pct": _clean_float(row.get("veg_cover_pct")),
                    "anomaly_drivers": _clean_str(row.get("anomaly_drivers")),
                    "sufficient_history": _clean_bool(row.get("sufficient_history")),
                    "tile_cloud_pct": _clean_float(row.get("tile_cloud_pct")),
                    "selected_water_px": _clean_int(row.get("selected_water_px")),
                }
            )
        return records

    def get_final_predictions(self, water_body_id: str) -> list[dict[str, Any]]:
        """Return chronological observations from `sentinel_final_predictions.csv`."""
        self.validate_water_body_id(water_body_id)
        if self.predictions_df is None:
            raise FileNotFoundError(
                "sentinel_final_predictions.csv unavailable: "
                f"{self.load_errors.get('sentinel_final_predictions')}"
            )
        sub = self.predictions_df[
            self.predictions_df["water_body_id"] == water_body_id
        ].sort_values("date")
        records: list[dict[str, Any]] = []
        for _, row in sub.iterrows():
            records.append(
                {
                    "water_body_id": str(row["water_body_id"]),
                    "date": str(row["date"]),
                    "cloud_cover": _clean_float(row.get("cloud_cover")),
                    "water_pixel_count": _clean_int(row.get("water_pixel_count")),
                    "ndwi_mean": _clean_float(row.get("ndwi_mean")),
                    "ndci_mean": _clean_float(row.get("ndci_mean")),
                    "ndti_mean": _clean_float(row.get("ndti_mean")),
                    "pred_chla": _clean_float(row.get("pred_chla")),
                    "pred_tss": _clean_float(row.get("pred_tss")),
                    "anomaly_score": _clean_float(row.get("anomaly_score")),
                    "is_anomaly": _clean_bool(row.get("is_anomaly")),
                }
            )
        return records

    def get_latest_observation(
        self,
        water_body_id: str,
        *,
        date: str | None = None,
    ) -> dict[str, Any]:
        """Return the latest (or requested date) combined observation record for a lake."""
        records = self.get_anomaly_observations(water_body_id)
        if not records:
            raise ObservationDateNotFoundError(f"No observations available for '{water_body_id}'.")

        if date is None:
            anom_obs = records[-1]
        else:
            matches = [r for r in records if r["date"] == date]
            if not matches:
                raise ObservationDateNotFoundError(
                    f"Observation date '{date}' not found for '{water_body_id}'."
                )
            anom_obs = matches[-1]

        pred_obs: dict[str, Any] = {}
        if self.predictions_df is not None:
            preds = self.get_final_predictions(water_body_id)
            pred_matches = [p for p in preds if p["date"] == anom_obs["date"]]
            if pred_matches:
                pred_obs = pred_matches[-1]

        return {
            **anom_obs,
            "cnn_lstm_pred_chla": pred_obs.get("pred_chla"),
            "cnn_lstm_pred_tss": pred_obs.get("pred_tss"),
            "cnn_lstm_anomaly_score": pred_obs.get("anomaly_score"),
            "cnn_lstm_is_anomaly": pred_obs.get("is_anomaly"),
            "cloud_cover": pred_obs.get("cloud_cover", anom_obs.get("tile_cloud_pct")),
            "water_pixel_count": pred_obs.get(
                "water_pixel_count", anom_obs.get("selected_water_px")
            ),
        }

    def get_hotspots(
        self,
        water_body_id: str,
        *,
        date: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return `(observation_date, hotspots_list)` from `spatial_hotspots.json`."""
        self.validate_water_body_id(water_body_id)
        if self.hotspots is None:
            raise FileNotFoundError(
                f"spatial_hotspots.json unavailable: {self.load_errors.get('spatial_hotspots')}"
            )
        lake_hotspots = self.hotspots.get(water_body_id)
        if not isinstance(lake_hotspots, dict) or not lake_hotspots:
            raise ObservationDateNotFoundError(f"No hotspot entries for '{water_body_id}'.")

        if date is None:
            selected_date = sorted(lake_hotspots.keys())[-1]
        else:
            if date not in lake_hotspots:
                raise ObservationDateNotFoundError(
                    f"Hotspot data for date '{date}' not found for '{water_body_id}'."
                )
            selected_date = date

        clusters = lake_hotspots.get(selected_date)
        if not isinstance(clusters, list):
            raise ValueError(f"Malformed hotspot list for '{water_body_id}' on '{selected_date}'.")
        return selected_date, [dict(item) for item in clusters]

    def get_metrics(self) -> dict[str, Any]:
        if self.metrics_comparison is None:
            raise FileNotFoundError(
                f"metrics_comparison.json unavailable: {self.load_errors.get('metrics_comparison')}"
            )
        return dict(self.metrics_comparison)

    def get_cached_summary(
        self,
        water_body_id: str,
        observation_date: str,
        risk_level: str,
    ) -> Any | None:
        return self.summary_cache.get((water_body_id, observation_date, risk_level))

    def set_cached_summary(
        self,
        water_body_id: str,
        observation_date: str,
        risk_level: str,
        summary: Any,
    ) -> None:
        self.summary_cache[(water_body_id, observation_date, risk_level)] = summary
