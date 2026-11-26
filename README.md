# Spatio-Temporal Satellite Water Contamination Analysis & Illegal Dumping Detection Engine

**Team 30 | Repository:** `miniproject2026`

A satellite water-quality screening and anomaly-detection pipeline combining in-situ hyperspectral water-quality matchups (**GLORIA**), multi-temporal **Sentinel-2 Level-2A** imagery over four Indian water bodies, classical Random Forest proxy models, a spatio-temporal **CNN-LSTM** (`WaterCNNLSTM`) exported to **ONNX**, and a deterministic **Risk Engine**.

> **Scientific Disclaimer:** No laboratory-confirmed Chlorophyll-a, TSS, or pollution ground-truth labels exist for the four Indian Sentinel-2 water bodies. All proxy concentrations, anomaly scores, spatial hotspots, and risk tiers in this repository are **satellite-derived screening estimates**, not laboratory-confirmed measurements.

---

## Project Status

| Component | Status | Details |
|---|---|---|
| **Day 1:** GLORIA EDA & Sentinel-2 L2A Ingestion | **Completed** | Cleaned GLORIA dataset (`5,008` rows), `4` lake GeoJSON boundaries, `128` Sentinel-2 scenes (`32`/lake), `(32, 9, 64, 64)` NPZ patch archives (`ml/01`, `ml/02`) |
| **Day 2:** Baseline RF, Anomaly Detector, Hotspots & Risk Engine | **Completed** | GLORIA RF models (`Chla`, `TSS`), trailing robust Z-score anomalies (`128` rows), semi-synthetic benchmark (`n=64`), `170` spatial hotspot clusters, deterministic `backend/risk_engine.py` (`8/8` unit tests passing) |
| **Day 3:** CNN-LSTM Training, ONNX Export, INT8 & Benchmarking | **Completed** | Trained `WaterCNNLSTM` (`cnn_lstm_best.pt`), exported/validated `cnn_lstm_fp32.onnx` & `cnn_lstm_int8.onnx`, CPU benchmarked, generated validation residual stats & `sentinel_final_predictions.csv` |
| **Day 4:** FastAPI REST API (`backend/main.py`) & LLM Summary | **Planned (Scaffold)** | API contract and risk engine are finalized in `SPEC.md`; `backend/main.py` is currently an empty scaffold |
| **Day 4:** Streamlit Geospatial Dashboard (`frontend/app.py`) | **Planned (Scaffold)** | Dependencies defined in `frontend/requirements.txt`; `frontend/app.py` is currently an empty scaffold |

---

## Problem & Proposed Solution

### Problem
Field water sampling across urban lakes and rivers is infrequent and spatially sparse, leaving algal blooms, sediment spikes, and localized discharge events undetected between surveys.

### Proposed Solution
1. **Optical Proxy Calibration (GLORIA -> Sentinel-2):** Harmonize quality-controlled hyperspectral GLORIA measurements into Sentinel-2 reflectance bands and spectral indices (`NDCI`, `NDTI`) to train Random Forest estimators for `Chla` and `TSS`.
2. **Multi-Temporal Sentinel-2 Pipeline:** Fetch cloud-screened, polygon-clipped Sentinel-2 L2A imagery across four Indian water bodies (`bellandur`, `hussain_sagar`, `yamuna_okhla`, `sukhna`) via Microsoft Planetary Computer STAC.
3. **Dual-Modality Anomaly & Hotspot Screening:** Combine per-lake trailing robust MAD Z-scores with connected-component spatial hotspot clustering (`>= 3` water pixels) and a deterministic conservative Risk Engine (`LOW`, `MEDIUM`, `HIGH`).
4. **Spatio-Temporal Forecasting & ONNX Deployment:** Train a compact `WaterCNNLSTM` on 4-frame `(4, 9, 64, 64)` patch windows to forecast next-scene indices/proxies and export to ONNX Runtime for low-latency CPU serving.

---

## Key Features & System Workflow

```text
GLORIA (5,008 clean matchups) ──► RF Baselines (Chla, TSS) ──────────────────────┐
                                                                                 ▼
Sentinel-2 L2A STAC (4 lakes × 32 dates) ──► Scene Indices + 64×64×9 NPZ Patches ──► Trailing Robust Z-Score + Spatial Hotspots
                                                        │                                │
                                                        ▼                                ▼
                                          WaterCNNLSTM (4-frame window)         Conservative Risk Engine
                                          ──► ONNX FP32 / INT8 + Residuals      (backend/risk_engine.py)
                                                        │                                │
                                                        └──────────► Day 4 Planned: FastAPI + Streamlit ◄──┘
```

- **Completed:**
  - Leakage-resistant GLORIA preprocessing (`scratch/preprocess_gloria.py`, `ml/01_gloria_eda.ipynb`).
  - Cloud-screened (`local_cloud_pct < 15%`), polygon-eroded Sentinel-2 extraction (`ml/02_sentinel_fetch.ipynb`).
  - `GroupKFold`-selected Random Forest models (`ml/03_baseline_models.ipynb`).
  - Trailing robust MAD temporal anomaly scoring, semi-synthetic plume benchmark (`n=64`), and `>= 3`-pixel spatial hotspot extraction (`ml/03b_anomaly_hotspots.ipynb`).
  - Compact `WaterCNNLSTM` training, chronological evaluation, ONNX FP32/INT8 export, CPU latency benchmarking, and final prediction CSV generation (`ml/04_cnn_lstm_colab.ipynb`).
  - Deterministic rule-based `Risk Engine` with missing-data guards (`backend/risk_engine.py`, `backend/test_risk_engine.py`).
- **Planned:**
  - Serving `GET /api/water-bodies`, `GET /api/timeseries/{lake_id}`, `GET /api/hotspots/{lake_id}`, and `POST /api/report/{lake_id}` in `backend/main.py` (with optional LLM incident formatting).
  - Interactive map and time-series dashboard in `frontend/app.py`.

---

## Technologies & Model Architecture

- **ML & Data:** Python `3.11`, PyTorch (`2.15.0+cpu`), scikit-learn, NumPy, pandas, Matplotlib, Seaborn, Rasterio, `pystac-client`, `planetary-computer`.
- **Deployment & Quantization:** ONNX (`1.23.2`), ONNX Runtime (`1.32.0`), `onnxscript`.
- **Backend / Frontend (Target Stack):** FastAPI, Uvicorn, Pydantic, Streamlit, Folium, `streamlit-folium`.
- **`WaterCNNLSTM` Architecture (`25,029` parameters):**
  - **Input:** `(B, 4, 9, 64, 64)` — 4 preceding Sentinel-2 frames, 9 channels (`B02`, `B03`, `B04`, `B05`, `B06`, `B8A`, `NDCI`, `NDTI`, `water_mask`).
  - **Spatial Encoder:** 3× `Conv2d` (`9->16->32->32`, `k=3`, `p=1`) + `BatchNorm2d` + `ReLU`, two `MaxPool2d(2)` layers, global average pooling, `Dropout(0.25) -> Linear(32->32) -> ReLU`.
  - **Temporal Encoder:** 1-layer `LSTM(input_size=32, hidden_size=32, batch_first=True)` taking the final hidden state.
  - **Head:** `Dropout(0.20) -> Linear(32->5)` predicting standardized `[NDWI_med, NDCI_med, NDTI_med, log1p(Chla_proxy), log1p(TSS_proxy)]`.

---

## Dataset Sources & Preprocessing

1. **GLORIA (`data/raw/gloria/` — not tracked in Git due to size):**
   - `GLORIA_meta_and_lab.csv`, `GLORIA_qc_flags.csv`, `GLORIA_Rrs.csv`.
   - QC-filtered (`Flagged == 0`), converted via $\rho_w = \pi R_{rs}$, and exported to `data/processed/gloria_cleaned.csv` (`5,008` rows: `3,746` valid `Chla`, `3,460` valid `TSS`).
2. **Sentinel-2 L2A (`data/processed/`):**
   - `4` approved Indian water bodies (`bellandur`, `hussain_sagar`, `yamuna_okhla`, `sukhna`), `32` scenes each (`2024-06` to `2026-11`), boundaries in `data/processed/lake_boundaries/*.geojson`.
   - `data/processed/sentinel_timeseries.csv` (`128` rows × `36` cols) and `data/processed/sentinel_patches/{lake_id}.npz` (`(32, 9, 64, 64)` `float32`).
   - `data/processed/sentinel_anomalies.csv` (`128` rows × `49` cols), `data/processed/anomaly_benchmark.csv` / `.npz` (`64` rows), `data/processed/spatial_hotspots.json` (`170` clusters), and `data/processed/sentinel_final_predictions.csv` (`128` rows).

---

## Repository Structure

```text
miniproject/
├── SPEC.md                               # Technical specification & API/data contracts
├── README.md                             # Project overview & setup guide
├── requirements.txt                      # Composite requirements (backend, frontend, ml)
├── backend/
│   ├── main.py                           # [Planned] FastAPI server scaffold (currently empty)
│   ├── risk_engine.py                    # [Completed] Deterministic conservative Risk Engine
│   ├── test_risk_engine.py               # [Completed] Risk Engine unit tests & 128-row check
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── app.py                            # [Planned] Streamlit + Folium app (currently empty)
│   ├── requirements.txt
│   └── README.md
├── ml/
│   ├── 01_gloria_eda.ipynb               # Day 1: GLORIA EDA & preprocessing
│   ├── 02_sentinel_fetch.ipynb           # Day 1: Sentinel-2 L2A STAC extraction & patches
│   ├── 03_baseline_models.ipynb          # Day 2: GLORIA RF models & zero-shot Sentinel inference
│   ├── 03b_anomaly_hotspots.ipynb        # Day 2: Trailing MAD anomalies, benchmark & hotspots
│   ├── 04_cnn_lstm_colab.ipynb           # Day 3: CNN-LSTM training, ONNX/INT8 export & CSV export
│   ├── *_verification.png                # Day 1 lake mask verification plots (4 files)
│   └── requirements.txt
├── models/
│   ├── baseline_chla.pkl                 # Day 2 Random Forest Chla model
│   ├── baseline_tss.pkl                  # Day 2 Random Forest TSS model
│   ├── baseline_metrics.json             # Day 2 GLORIA & MAD benchmark metrics
│   ├── cnn_lstm_best.pt                  # Day 3 trained WaterCNNLSTM checkpoint (epoch 41)
│   ├── cnn_lstm_fp32.onnx                # Day 3 recommended ONNX FP32 model (opset 18)
│   ├── cnn_lstm_int8.onnx                # Day 3 dynamic QInt8 quantized ONNX model
│   ├── cnn_lstm_best.onnx                # Day 3 legacy Phase E ONNX FP32 export (opset 17)
│   ├── cnn_lstm_metrics.json             # Day 3 evaluation & anomaly comparison metrics
│   ├── cnn_lstm_residual_stats.json      # Validation-only (n=20) residual Z-score stats
│   ├── onnx_benchmark_results.json       # PyTorch vs ONNX FP32 vs INT8 CPU benchmarks
│   └── metrics_comparison.json           # Combined Day 2, Day 3 & deployment summary
├── data/
│   ├── raw/gloria/                       # Raw GLORIA CSVs (local only, gitignored)
│   └── processed/                        # Cleaned CSVs, GeoJSONs, NPZ patches, hotspots, predictions
├── docs/
│   ├── README.md
│   └── figures/                          # 15 generated evaluation, verification & loss plots
└── scratch/                              # Preprocessing & notebook helper scripts
```

---

## Setup & Installation

```bash
# 1. Clone and enter repository
git clone https://github.com/somyapurohit175-rgb/miniproject2026.git
cd miniproject2026

# 2. Create and activate Python 3.11 virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# 3. Install core project dependencies
pip install -r requirements.txt

# 4. (Optional, for Day 3 CNN-LSTM / ONNX workflows)
pip install torch onnx onnxruntime onnxscript
```

---

## How to Run

### 1. Run Deterministic Risk Engine Tests (Ready Now)
```bash
python backend/test_risk_engine.py
```
Runs 8 unit tests plus a calibration summary across all `128` observations in `data/processed/sentinel_anomalies.csv` and `data/processed/spatial_hotspots.json`.

### 2. Inspect or Re-run ML Notebooks
Launch Jupyter inside `.venv` (`jupyter notebook`) and open notebooks in order:
1. `ml/01_gloria_eda.ipynb` — requires raw GLORIA CSVs in `data/raw/gloria/`.
2. `ml/02_sentinel_fetch.ipynb` — requires network access to Microsoft Planetary Computer STAC API.
3. `ml/03_baseline_models.ipynb` — trains Day 2 Random Forest models on `data/processed/gloria_cleaned.csv`.
4. `ml/03b_anomaly_hotspots.ipynb` — computes trailing MAD anomaly scores, semi-synthetic benchmark, and spatial hotspots.
5. `ml/04_cnn_lstm_colab.ipynb` — trains/evaluates `WaterCNNLSTM` on `data/processed/sentinel_patches/*.npz`, exports ONNX FP32 & INT8 models, runs CPU benchmarks, and exports `data/processed/sentinel_final_predictions.csv`. *(Colab fallback automatically mounts Google Drive if `SPEC.md` is not in the working directory.)*

### 3. Backend & Frontend Apps (Pending Day 4)
`backend/main.py` and `frontend/app.py` are empty scaffolds; `uvicorn` and `streamlit` entrypoints will become runnable once Day 4 integration is implemented.

---

## Model Artifacts & Evaluation Results

### Day 2 GLORIA Random Forest Baselines (`models/baseline_metrics.json`)
| Model | Dataset / Test Split | `R²_log` | `RMSE_log` | `MAE_log` | `R²_orig` | `MAE_orig` | `RMSE_orig` |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline_chla.pkl` (RF) | GLORIA lab (`n = 908`) | `0.8754` | `0.4238` | `0.2942` | `0.3966` | `6.34 mg/m³` | `38.52 mg/m³` |
| `baseline_tss.pkl` (RF) | GLORIA lab (`n = 904`) | `0.4981` | `0.7471` | `0.5877` | `0.5625` | `13.50 mg/L` | `61.69 mg/L` |

### Day 3 CNN-LSTM vs Same-32 Persistence Baseline (`models/cnn_lstm_metrics.json`)
Evaluated on `32` held-out chronological Sentinel-2 test scenes (`t = 24..31`); `Chla` and `TSS` targets are Day 2 RF proxy estimates:

| Target | CNN-LSTM `R²_orig` | Persistence `R²_orig` | CNN-LSTM `MAE_orig` | Persistence `MAE_orig` | CNN-LSTM `RMSE_orig` | Persistence `RMSE_orig` |
|---|---:|---:|---:|---:|---:|---:|
| `NDWI` | `0.0926` | `-0.6723` | `0.1187` | `0.1767` | `0.1681` | `0.2282` |
| `NDCI` | `-0.1182` | `-0.8009` | `0.0714` | `0.0906` | `0.1029` | `0.1306` |
| `NDTI` | `0.1312` | `-0.5064` | `0.0616` | `0.0759` | `0.0806` | `0.1062` |
| `Chla` (RF proxy) | `0.3146` | `0.2804` | `71.92 mg/m³` | `77.61 mg/m³` | `107.49 mg/m³` | `110.14 mg/m³` |
| `TSS` (RF proxy) | `-0.0692` | `-1.1144` | `50.27 mg/L` | `82.50 mg/L` | `149.51 mg/L` | `210.25 mg/L` |

### Anomaly Benchmark Comparison (`n = 64`, 32 Normal + 32 Simulated Plumes)
- **Day 2 Rolling MAD:** Precision `0.6444`, Recall `0.9063`, F1 `0.7532`, ROC-AUC `0.7964`.
- **Day 3 CNN-LSTM Residual (`> 2.0`):** Precision `0.5000`, Recall `0.1563`, F1 `0.2381`, ROC-AUC `0.4868`.

### ONNX Export & CPU Benchmark (`models/onnx_benchmark_results.json`)
Measured on CPU (Windows 10, `Intel64 Family 6 Model 170 Stepping 4`, PyTorch `2.15.0+cpu`, ONNX Runtime `1.32.0`):

| Model | File Size (MB) | Batch 1 Mean / Median | Batch 32 Mean / Median | Max Scaled Diff vs PyTorch (`n=32` / `n=112`) |
|---|---:|---:|---:|---:|
| `cnn_lstm_best.pt` (PyTorch FP32) | `0.1068` | `3.647 / 3.318 ms` | `61.629 / 61.307 ms` | Reference |
| **`cnn_lstm_fp32.onnx` (Selected)** | `0.1312` | **`1.664 / 0.911 ms`** | **`21.790 / 18.791 ms`** | `3.58e-07` / `5.36e-07` |
| `cnn_lstm_int8.onnx` (Dynamic QInt8) | `0.0340` | `5.444 / 5.116 ms` | `313.788 / 302.156 ms` | `0.0863` / `0.6961` |

`models/cnn_lstm_fp32.onnx` is selected for backend deployment because it runs 2.2×–2.8× faster than PyTorch FP32 on CPU with machine-precision parity, whereas dynamic INT8 quantization overhead degrades CPU speed on this compact network and introduces up to `28.93 mg/m³` Chla drift on `n = 112`.

---

## Current Limitations & Future Improvements

### Current Limitations
- **Proxy Labels Only:** No in-situ laboratory measurements exist for the 4 Indian water bodies; CNN-LSTM `Chla`/`TSS` targets are Day 2 GLORIA RF estimates.
- **Small & Irregular Time Series:** Only `112` temporal sequences (`60` train, `20` val, `32` test) exist across `2`–`185` day monsoon cloud gaps, limiting deep spatio-temporal generalization and fixed-step LSTM assumptions.
- **Channel / Summary Differences:** `64x64` NPZ patches store `B8A` (omitting `B07`/`B08`), and `sentinel_final_predictions.csv` maps `ndwi_mean`/`ndci_mean`/`ndti_mean` to native-10m robust scene medians (`NDWI_med`, `NDCI_med`, `NDTI_med`).
- **Screening Thresholds:** `RiskThresholds` (`2.5` / `4.5`) are uncalibrated demo screening rules, and `enable_cnn_lstm_rule` is `False` by default.
- **Pending Apps:** `backend/main.py` and `frontend/app.py` are not yet implemented.

### Planned Next Steps
1. Implement `backend/main.py` FastAPI endpoints per `SPEC.md` Section 8, integrating `models/cnn_lstm_fp32.onnx` and `backend/risk_engine.py`.
2. Build `frontend/app.py` Streamlit + Folium dashboard for interactive lake boundary maps, hotspot inspection, time-series plots, and screening reports.
3. Acquire field validation matchups for Indian lakes and incorporate time-gap embeddings (`delta_t`) into future spatio-temporal models.
