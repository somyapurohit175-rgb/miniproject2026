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

### 4.1 `GET /api/water-bodies`
Returns all monitored water bodies with their latest screening status.
```json
{
  "water_bodies": [
    {
      "id": "bellandur",
      "name": "Bellandur Lake",
      "city": "Bengaluru",
      "center_lat": 12.9342,
      "center_lon": 77.6628,
      "bbox": [77.645, 12.920, 77.685, 12.946],
      "latest_observation_date": "2026-11-18",
      "latest_risk_level": "HIGH",
      "latest_risk_score": 82.5,
      "active_anomalies": 2
    }
  ]
}