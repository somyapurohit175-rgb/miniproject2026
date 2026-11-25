# Project Specification & Shared Contract (`SPEC.md`)

**Project:** Spatio-Temporal Satellite Water Contamination Analysis & Illegal Dumping Detection Engine  
**Team ID:** 30 | **Repo:** `miniproject2026`

---

## 1. Agent & Collaboration Rules (For Google Antigravity)
- **Strict Contract:** Never rename JSON keys, endpoint paths, or water body IDs without updating this file first.
- **Folder Ownership:**
  - Member A owns: `/data`, `/ml`, `/backend`
  - Member B owns: `/frontend`, `/docs`
- **ML Scope:** Do NOT auto-generate or overwrite notebooks inside `/ml`. ML training and evaluation are performed manually.
- **Architectural Guardrail:** Risk scores and risk levels (`LOW`, `MEDIUM`, `HIGH`) MUST be computed strictly by deterministic rules in the Risk Engine. The Language Model ONLY summarizes validated metrics into structured JSON and must never invent measurements or override the risk level.
- **Scientific Disclaimer:** Every API response and UI view must clearly state that values are satellite-derived screening estimates, not laboratory-confirmed measurements.

---

## 2. Target Water Bodies (Verified WGS84 / EPSG:4326)

Bounding box format strictly follows STAC/GeoJSON standard: `[min_lon, min_lat, max_lon, max_lat]`.

| `id` | `name` | `city` | `center_lat` | `center_lon` | `bbox` `[W, S, E, N]` | Role in Evaluation |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| `bellandur` | Bellandur Lake | Bengaluru | 12.9342 | 77.6628 | `[77.645, 12.920, 77.685, 12.946]` | High pollution / foam & turbidity spikes |
| `hussain_sagar` | Hussain Sagar | Hyderabad | 17.4248 | 78.4733 | `[78.455, 17.408, 78.492, 17.442]` | Urban eutrophication / seasonal Chl-a |
| `yamuna_okhla` | Yamuna River (Okhla Stretch) | Delhi / Noida | 28.5585 | 77.3076 | `[77.292, 28.545, 77.322, 28.575]` | Industrial / sewage discharge anomalies |
| `sukhna` | Sukhna Lake | Chandigarh | 30.7421 | 76.8188 | `[76.803, 30.733, 76.835, 30.752]` | Cleaner baseline / control reference |

---

## 3. Spectral Indices & Preprocessing Constants
- **Sentinel-2 Product:** `sentinel-2-l2a` via Microsoft Planetary Computer STAC API.
- **Bands Used:** `B02` (490nm), `B03` (560nm), `B04` (665nm), `B05` (705nm), `B06` (740nm), `B07` (783nm), `B08` (842nm), `SCL` (Scene Classification Layer).
- **Scale Factor:** Divide raw Sentinel-2 L2A uint16 reflectance by `10000.0`.
- **Cloud Mask:** Exclude pixels where `SCL` is in `{3, 8, 9, 10}` (cloud shadow, medium/high cloud, cirrus).
- **Water Mask:** `NDWI = (B03 - B08) / (B03 + B08 + 1e-6)`; valid water pixels satisfy `NDWI > 0.1` and `SCL == 6`.
- **Quality Proxies:**
  - `NDCI = (B05 - B04) / (B05 + B04 + 1e-6)` (Chlorophyll-a proxy)
  - `NDTI = (B04 - B03) / (B04 + B03 + 1e-6)` (Turbidity / TSS proxy)

---

## 4. FastAPI JSON Contract (Base URL: `/api`)

> **Schema Consistency & Alignment Note:** The initial Day 1 scaffold ended prematurely inside Section 4.1 with incomplete `id`/`name` fields. This finalized contract unifies water body naming (`lake_id`, `lake_name`), includes full GeoJSON `boundary` geometries and structured `latest_risk_status` from the Risk Engine, and matches the persisted files `data/processed/sentinel_anomalies.csv`, `data/processed/spatial_hotspots.json`, and `data/processed/lake_boundaries/*.geojson`.
>
> **Mandatory Scientific Disclaimer:** Every API response must include a disclaimer stating that reported metrics, anomaly scores, proxy concentrations, hotspots, and risk tiers are satellite-derived screening estimates and **not** laboratory-confirmed pollution measurements.

Valid `lake_id` path values across all endpoints:
- `"bellandur"` (`Bellandur Lake`)
- `"hussain_sagar"` (`Hussain Sagar`)
- `"yamuna_okhla"` (`Yamuna River (Okhla Stretch)`)
- `"sukhna"` (`Sukhna Lake`)

### Common Error Responses
- `400 Bad Request`: Malformed query parameter (e.g., `date` not formatted as `YYYY-MM-DD`).
- `404 Not Found`: Unknown `lake_id` (not in official 4-lake registry) or requested `date` unavailable for that lake.
- `422 Unprocessable Entity`: Invalid FastAPI/Pydantic request body or parameter validation failure.
- `500 Internal Server Error` / `503 Service Unavailable`: Unreadable processed data artifact or downstream failure.

```json
{
  "detail": "Unknown lake_id 'varthur'. Valid lake_ids: ['bellandur', 'hussain_sagar', 'yamuna_okhla', 'sukhna']."
}
```

---

### 4.1 `GET /api/water-bodies`
Returns all four official monitored water bodies with registry metadata, GeoJSON boundary geometry (`Polygon` or `MultiPolygon` from `data/processed/lake_boundaries/{lake_id}.geojson`), and latest deterministic `Risk Engine` status.

**Response Fields (`200 OK`):**
- `water_bodies` (`array[object]`, required, exact length `4`):
  - `lake_id` (`string`, required): Canonical identifier (`bellandur`, `hussain_sagar`, `yamuna_okhla`, `sukhna`).
  - `lake_name` (`string`, required): Human-readable name.
  - `city` (`string`, required): City / metropolitan region.
  - `center_lat` (`number`, required): Nominal WGS84 (`EPSG:4326`) center latitude.
  - `center_lon` (`number`, required): Nominal WGS84 (`EPSG:4326`) center longitude.
  - `bbox` (`array[number]`, required): `[min_lon, min_lat, max_lon, max_lat]` in WGS84.
  - `boundary` (`object`, required): GeoJSON Geometry object (`{"type": "Polygon" | "MultiPolygon", "coordinates": [...]}`) loaded from `features[0].geometry` of `data/processed/lake_boundaries/{lake_id}.geojson` in `EPSG:4326`. Note: `bellandur` is `MultiPolygon`; `hussain_sagar`, `yamuna_okhla`, `sukhna` are `Polygon`.
  - `latest_risk_status` (`object`, required): Deterministic output of `backend/risk_engine.py` evaluated on the latest chronological observation:
    - `observation_date` (`string`, required, `YYYY-MM-DD`)
    - `risk_level` (`string`, required, enum: `"LOW" | "MEDIUM" | "HIGH"`)
    - `risk_score` (`number`, required, range `0.0–100.0`)
    - `anomaly_score` (`number | null`, required): Latest `baseline_anomaly_score` from `sentinel_anomalies.csv` (`null` if warm-up/missing).
    - `hotspot_count` (`integer`, required, `>= 0`): Count of extracted hotspot clusters on `observation_date` from `spatial_hotspots.json`.
    - `active_anomalies` (`integer`, required, `>= 0`): Count of flagged anomaly observations (`is_anomaly == true`) for that lake across the monitored series.
    - `supporting_evidence` (`array[string]`, required): Human-readable list of deterministic evidence strings.
    - `risk_rule_triggered` (`string`, required): Identifier of the risk rule branch applied.
- `disclaimer` (`string`, required)

**Example Response (`200 OK`):**
```json
{
  "water_bodies": [
    {
      "lake_id": "bellandur",
      "lake_name": "Bellandur Lake",
      "city": "Bengaluru",
      "center_lat": 12.9342,
      "center_lon": 77.6628,
      "bbox": [77.645, 12.920, 77.685, 12.946],
      "boundary": {
        "type": "MultiPolygon",
        "coordinates": [[[[77.6438, 12.9293], [77.6442, 12.9288], [77.6438, 12.9293]]]]
      },
      "latest_risk_status": {
        "observation_date": "2026-11-23",
        "risk_level": "MEDIUM",
        "risk_score": 46.66,
        "anomaly_score": 3.6148,
        "hotspot_count": 3,
        "active_anomalies": 20,
        "supporting_evidence": [
          "Anomaly score 3.61 in [2.50, 4.50] corroborated by 3 spatial hotspot cluster(s).",
          "Temporal drivers: NDCI_p90(3.61).",
          "Spatial hotspot signals: algal_bloom_proxy, combined, turbidity_sediment_proxy."
        ],
        "risk_rule_triggered": "MEDIUM_SCORE_WITH_HOTSPOTS"
      }
    }
  ],
  "disclaimer": "All values are satellite-derived screening estimates, not laboratory-confirmed measurements."
}
```

---

### 4.2 `GET /api/timeseries/{lake_id}`
Reads chronological Sentinel-2 observations, baseline RF proxy estimates, and trailing robust Z-score anomaly metrics directly from `data/processed/sentinel_anomalies.csv` for `{lake_id}` sorted ascending by `date`.

**Path Parameter:**
- `lake_id` (`string`, required): One of `bellandur`, `hussain_sagar`, `yamuna_okhla`, `sukhna`. Any other value returns `404 Not Found`.

**Response Fields (`200 OK`):**
- `lake_id` (`string`, required)
- `lake_name` (`string`, required)
- `count` (`integer`, required): Number of chronological observations (`32` in current dataset).
- `records` (`array[object]`, required): Ordered oldest to newest by `date`:
  - `date` (`string`, required, ISO-8601 `YYYY-MM-DD`)
  - `NDWI_med` (`number | null`, required): Scene median NDWI across valid water pixels.
  - `NDCI_med` (`number | null`, required): Scene median NDCI (Chl-a proxy index).
  - `NDTI_med` (`number | null`, required): Scene median NDTI (TSS/turbidity proxy index).
  - `pred_chla_proxy` (`number | null`, required): Baseline RF Chlorophyll-a estimate (`mg/m³`, `>= 0`).
  - `pred_tss_proxy` (`number | null`, required): Baseline RF Total Suspended Solids estimate (`mg/L`, `>= 0`).
  - `baseline_anomaly_score` (`number | null`, required): Composite positive robust Z-score across the 6 indicators (`null` during first 3 warm-up observations where `hist_ref_count < 3`).
  - `is_anomaly` (`boolean`, required): `true` iff `baseline_anomaly_score > 2.0`; `false` otherwise (including warm-up rows).
  - `open_water_pct` (`number | null`, required): Percentage (`0.0–100.0`) of lake pixels classified as open water.
  - `veg_cover_pct` (`number | null`, required): Percentage (`0.0–100.0`) of lake pixels with aquatic/emergent vegetation cover.
  - `anomaly_drivers` (`string`, optional): Driver breakdown string from `sentinel_anomalies.csv`.
  - `sufficient_history` (`boolean`, optional): `true` when `hist_ref_count >= 3`.
- `disclaimer` (`string`, required)

**Missing-Value Behaviour:** CSV `NaN` entries (such as warm-up `baseline_anomaly_score` on dates `0..2`, or missing spectral fields) MUST serialize as JSON `null`, never coerced to `0.0` or interpolated.

**Example Response (`200 OK`):**
```json
{
  "lake_id": "bellandur",
  "lake_name": "Bellandur Lake",
  "count": 32,
  "records": [
    {
      "date": "2024-06-01",
      "NDWI_med": 0.2535,
      "NDCI_med": 0.0887,
      "NDTI_med": -0.1726,
      "pred_chla_proxy": 383.7538,
      "pred_tss_proxy": 130.8418,
      "baseline_anomaly_score": null,
      "is_anomaly": false,
      "open_water_pct": 14.32,
      "veg_cover_pct": 71.16,
      "anomaly_drivers": "insufficient_history",
      "sufficient_history": false
    },
    {
      "date": "2026-11-23",
      "NDWI_med": 0.363,
      "NDCI_med": 0.2215,
      "NDTI_med": -0.1058,
      "pred_chla_proxy": 51.7524,
      "pred_tss_proxy": 10.6634,
      "baseline_anomaly_score": 3.6148,
      "is_anomaly": true,
      "open_water_pct": 5.61,
      "veg_cover_pct": 85.88,
      "anomaly_drivers": "NDCI_p90(3.61)",
      "sufficient_history": true
    }
  ],
  "disclaimer": "All values are satellite-derived screening estimates, not laboratory-confirmed measurements."
}
```

---

### 4.3 `GET /api/hotspots/{lake_id}?date=YYYY-MM-DD`
Reads cluster-level spatial pollution hotspots directly from `data/processed/spatial_hotspots.json` (which is stored as `{lake_id: {YYYY-MM-DD: [SpatialHotspot, ...]}}`). All coordinates are WGS84 (`EPSG:4326`), derived from north-up `64x64` pixel centroids verified inside `data/processed/lake_boundaries/{lake_id}.geojson`.

**Parameters:**
- `lake_id` (`string`, path, required): Valid official `lake_id` (`404 Not Found` otherwise).
- `date` (`string`, query, optional, format `YYYY-MM-DD`): If omitted, selects the latest chronological observation date available in `spatial_hotspots.json[lake_id]`. If malformed, returns `400 Bad Request`; if formatted cleanly but absent for that lake, returns `404 Not Found`.

**Response Fields (`200 OK`):**
- `lake_id` (`string`, required)
- `date` (`string`, required, `YYYY-MM-DD`)
- `crs` (`string`, required, constant `"EPSG:4326"`)
- `hotspot_count` (`integer`, required, `>= 0`)
- `hotspots` (`array[object]`, required; empty list `[]` when zero clusters meet `>= 3` px threshold):
  - `hotspot_id` (`string`, required, e.g. `"bellandur_2026-11-23_h01"`)
  - `lat` (`number`, required, WGS84 latitude)
  - `lon` (`number`, required, WGS84 longitude)
  - `pixel_count` (`integer`, required, `>= 3`)
  - `peak_ndci` (`number`, required): Maximum pixel `NDCI` within the cluster.
  - `peak_ndti` (`number`, required): Maximum pixel `NDTI` within the cluster.
  - `dominant_signal` (`string`, required, enum: `"algal_bloom_proxy" | "turbidity_sediment_proxy" | "combined"`)
  - `severity_score` (`number`, required, range `[0.0, 1.0]`)
- `message` (`string | null`, optional): Informative note when `hotspot_count == 0`.
- `disclaimer` (`string`, required)

**Empty Hotspots Behaviour:** When zero connected water-pixel clusters (`>= 3` pixels) exceed the historical pixel baseline on a valid date (e.g., `hussain_sagar` on `2026-11-18` or `sukhna` on `2026-11-24`), return `200 OK` with `hotspot_count: 0`, `hotspots: []`, and `message: "No spatial hotspots exceeded the historical baseline for this observation date."` Never fabricate placeholder markers.

**Example Response (`200 OK` — hotspots present):**
```json
{
  "lake_id": "bellandur",
  "date": "2026-11-23",
  "crs": "EPSG:4326",
  "hotspot_count": 3,
  "hotspots": [
    {
      "hotspot_id": "bellandur_2026-11-23_h01",
      "lat": 12.936078,
      "lon": 77.665856,
      "pixel_count": 3,
      "peak_ndci": 0.48543,
      "peak_ndti": 0.04911,
      "dominant_signal": "combined",
      "severity_score": 0.6604
    },
    {
      "hotspot_id": "bellandur_2026-11-23_h02",
      "lat": 12.942216,
      "lon": 77.662524,
      "pixel_count": 13,
      "peak_ndci": 0.48121,
      "peak_ndti": -0.12714,
      "dominant_signal": "algal_bloom_proxy",
      "severity_score": 0.6147
    },
    {
      "hotspot_id": "bellandur_2026-11-23_h03",
      "lat": 12.932598,
      "lon": 77.665371,
      "pixel_count": 3,
      "peak_ndci": 0.30267,
      "peak_ndti": 0.0,
      "dominant_signal": "turbidity_sediment_proxy",
      "severity_score": 0.4732
    }
  ],
  "message": null,
  "disclaimer": "All values are satellite-derived screening estimates, not laboratory-confirmed measurements."
}
```

---

### 4.4 `POST /api/report/{lake_id}`
Generates a structured screening incident report combining deterministic `Risk Engine` calculations (`backend/risk_engine.py`) with an LLM-formatted textual incident summary. **Strict separation guardrail:** `risk_assessment` is computed solely by deterministic Python rules from validated CSV/JSON artifacts; the LLM only formats `incident_summary` from `risk_assessment` and MUST NOT alter `risk_level`, invent metrics, or claim laboratory-confirmed pollution.

**Path Parameter:**
- `lake_id` (`string`, required): Official lake ID (`404 Not Found` if invalid).

**Request Body (`application/json`, optional — defaults apply if omitted or `{}`):**
```json
{
  "date": "2026-11-23",
  "include_llm_summary": true
}
```
- `date` (`string | null`, optional, `YYYY-MM-DD`): Observation date to report. Defaults to the lake's latest chronological date. Returns `400 Bad Request` on invalid date format or `404 Not Found` if missing for that lake.
- `include_llm_summary` (`boolean`, optional, default `true`): Uses the configured LLM summarizer when `true`, falling back to a deterministic structured template if the LLM service is unreachable.

**Response Fields (`200 OK`):**
- `lake_id` (`string`, required)
- `lake_name` (`string`, required)
- `observation_date` (`string`, required, `YYYY-MM-DD`)
- `risk_assessment` (`object`, required — strictly rule-based):
  - `risk_score` (`number`, required, `0.0–100.0`)
  - `risk_level` (`string`, required, `"LOW" | "MEDIUM" | "HIGH"`)
  - `anomaly_score` (`number | null`, required)
  - `hotspot_count` (`integer | null`, required)
  - `supporting_evidence` (`array[string]`, required)
  - `risk_rule_triggered` (`string`, required)
  - `supporting_hotspots` (`array[object]`, required): Matching hotspot objects from `spatial_hotspots.json`.
  - `proxy_estimates` (`object`, required): Actual row values (`pred_chla_proxy`, `pred_tss_proxy`, `NDCI_med`, `NDTI_med`, `NDWI_med`; nullable floats).
- `incident_summary` (`object`, required — LLM-formatted narrative grounded strictly in `risk_assessment`):
  - `event_type` (`string`, required, enum: `"algal_bloom_proxy" | "turbidity_sediment_proxy" | "combined" | "elevated_anomaly_uncorroborated" | "baseline_screening" | "insufficient_evidence"`)
  - `risk_level` (`string`, required): MUST exactly match `risk_assessment.risk_level`.
  - `affected_location` (`string`, required): Human-readable location reference derived strictly from `lake_name` and validated hotspot coordinates (when present).
  - `supporting_observations` (`string`, required): Concise summary quoting only verified numeric values from `risk_assessment`.
  - `recommended_action` (`string`, required): Non-accusatory follow-up screening/sampling guidance commensurate with `risk_level`.
  - `is_lab_confirmed` (`boolean`, required, **constant `false`**): Must always be `false`.
- `summary_source` (`string`, required, enum: `"llm" | "deterministic_fallback"`)
- `disclaimer` (`string`, required)

**Example Response (`200 OK`):**
```json
{
  "lake_id": "bellandur",
  "lake_name": "Bellandur Lake",
  "observation_date": "2026-11-23",
  "risk_assessment": {
    "risk_score": 46.66,
    "risk_level": "MEDIUM",
    "anomaly_score": 3.6148,
    "hotspot_count": 3,
    "supporting_evidence": [
      "Anomaly score 3.61 in [2.50, 4.50] corroborated by 3 spatial hotspot cluster(s).",
      "Temporal drivers: NDCI_p90(3.61).",
      "Spatial hotspot signals: algal_bloom_proxy, combined, turbidity_sediment_proxy."
    ],
    "risk_rule_triggered": "MEDIUM_SCORE_WITH_HOTSPOTS",
    "supporting_hotspots": [
      {
        "hotspot_id": "bellandur_2026-11-23_h01",
        "lat": 12.936078,
        "lon": 77.665856,
        "pixel_count": 3,
        "peak_ndci": 0.48543,
        "peak_ndti": 0.04911,
        "dominant_signal": "combined",
        "severity_score": 0.6604
      }
    ],
    "proxy_estimates": {
      "pred_chla_proxy": 51.7524,
      "pred_tss_proxy": 10.6634,
      "NDCI_med": 0.2215,
      "NDTI_med": -0.1058,
      "NDWI_med": 0.363
    }
  },
  "incident_summary": {
    "event_type": "combined",
    "risk_level": "MEDIUM",
    "affected_location": "Bellandur Lake (3 spatial clusters centered near 12.9361 N, 77.6659 E and 12.9422 N, 77.6625 E)",
    "supporting_observations": "Trailing robust Z-score reached 3.61 (driven by NDCI_p90) alongside 3 water-pixel hotspot clusters up to 13 pixels (max severity 0.66). Satellite proxy estimates: Chla 51.75 mg/m3, TSS 10.66 mg/L.",
    "recommended_action": "Schedule routine field verification and inspect storm-water/northern inlet zones for surface scum or localized sediment discharge.",
    "is_lab_confirmed": false
  },
  "summary_source": "llm",
  "disclaimer": "All values are satellite-derived screening estimates, not laboratory-confirmed measurements."
}
```
