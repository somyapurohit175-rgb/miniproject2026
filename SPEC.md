# Project Specification & Shared Contract (`SPEC.md`)

**Project:** Spatio-Temporal Satellite Water Contamination Analysis & Illegal Dumping Detection Engine  
**Team ID:** 30 | **Repo:** `miniproject2026`  
**Branch:** `feat/data-ml` | **Status:** Day 1–3 Data, ML, ONNX Export & Risk Engine Complete; FastAPI & Streamlit Integration Pending

---

## 1. Problem Statement, Objectives & Proposed Solution

### 1.1 Problem Statement
Urban lakes and river stretches in India experience rapid eutrophication, algal blooms, sediment resuspension, and localized industrial/sewage discharge events. Routine laboratory water-quality sampling is sparse in both space and time, creating blind spots between field campaigns.

### 1.2 Objectives
1. Harmonize global in-situ hyperspectral water-quality matchups (**GLORIA**) with Sentinel-2 Level-2A optical bands to train interpretable baseline water-quality proxy estimators (`Chla`, `TSS`).
2. Build a reproducible, leakage-resistant Sentinel-2 L2A multi-temporal ingestion pipeline over four Indian water bodies (`2024-06` to `2026-11`).
3. Detect temporal spectral/proxy anomalies and localize sub-lake spatial pollution hotspot clusters inside verified polygon boundaries.
4. Train a compact spatio-temporal **CNN-LSTM** (`WaterCNNLSTM`) to forecast next-observation water-quality indices and proxy estimates, derive validation-calibrated residual anomaly scores, export/benchmark ONNX FP32 and INT8 deployment formats, and enforce conservative deterministic risk scoring separated from LLM summarization.

### 1.3 Proposed Solution & Scientific Guardrail
- **Two-Stage Screening Architecture:**
  - **Stage A (Baseline & Spatial Screening — Day 2):** GLORIA-trained Random Forest regressors generate Sentinel-2 proxy estimates (`pred_chla_proxy`, `pred_tss_proxy`), paired with a trailing robust MAD Z-score temporal detector (`sentinel_anomalies.csv`), spatial hotspot clustering (`spatial_hotspots.json`), and a deterministic conservative Risk Engine (`backend/risk_engine.py`).
  - **Stage B (Spatio-Temporal Forecasting & Deployment — Day 3):** A compact `WaterCNNLSTM` processes 4-frame `(4, 9, 64, 64)` Sentinel-2 patch sequences to forecast `[NDWI, NDCI, NDTI, Chla_proxy, TSS_proxy]` and compute two-sided validation-calibrated residual anomaly scores, deployed via ONNX Runtime (`models/cnn_lstm_fp32.onnx`).
- **Mandatory Scientific Disclaimer:** No laboratory-confirmed Chla/TSS or pollution ground-truth labels exist for the four Indian Sentinel-2 water bodies. All reported concentrations, anomaly scores, hotspots, and risk tiers are **satellite-derived screening estimates**, not laboratory-confirmed measurements.

---

## 2. Agent & Collaboration Rules
- **Strict Contract:** Never rename JSON keys, endpoint paths, or water body IDs without updating this file first.
- **Folder Ownership:**
  - Member A owns: `/data`, `/ml`, `/backend`
  - Member B owns: `/frontend`, `/docs`
- **ML Scope:** Notebooks inside `/ml` (`01`–`04`) document manual/verified ML training, evaluation, and Day 3 ONNX export/benchmarking/prediction pipelines. Do not overwrite trained checkpoints or Day 2 baseline metrics without explicit instruction.
- **Architectural Guardrail:** Risk scores and risk levels (`LOW`, `MEDIUM`, `HIGH`) MUST be computed strictly by deterministic rules in `backend/risk_engine.py`. Any Language Model integration ONLY summarizes validated `risk_assessment` JSON into structured `incident_summary` JSON and must never invent measurements or override `risk_level`.

---

## 3. Target Water Bodies (Verified WGS84 / EPSG:4326)

Bounding box format strictly follows STAC/GeoJSON standard: `[min_lon, min_lat, max_lon, max_lat]`. Polygon boundaries are persisted in `data/processed/lake_boundaries/{lake_id}.geojson`.

| `id` | `name` | `city` | `center_lat` | `center_lon` | `bbox` `[W, S, E, N]` | Geometry Type | Role in Evaluation |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| `bellandur` | Bellandur Lake | Bengaluru | 12.9342 | 77.6628 | `[77.645, 12.920, 77.685, 12.946]` | `MultiPolygon` | High pollution / foam & turbidity spikes |
| `hussain_sagar` | Hussain Sagar | Hyderabad | 17.4248 | 78.4733 | `[78.455, 17.408, 78.492, 17.442]` | `Polygon` | Urban eutrophication / seasonal Chl-a |
| `yamuna_okhla` | Yamuna River (Okhla Stretch) | Delhi / Noida | 28.5585 | 77.3076 | `[77.292, 28.545, 77.322, 28.575]` | `Polygon` | Industrial / sewage discharge anomalies |
| `sukhna` | Sukhna Lake | Chandigarh | 30.7421 | 76.8188 | `[76.803, 30.733, 76.835, 30.752]` | `Polygon` | Cleaner baseline / control reference |

---

## 4. Data Sources, Preprocessing & Spectral Indices

### 4.1 GLORIA In-Situ Hyperspectral Dataset (Day 1 — `ml/01_gloria_eda.ipynb`)
- **Raw Sources:** `data/raw/gloria/GLORIA_meta_and_lab.csv`, `GLORIA_qc_flags.csv`, `GLORIA_Rrs.csv` (Lehmann et al., *Scientific Data*, 2023).
- **Preprocessing (`scratch/preprocess_gloria.py` & `ml/01_gloria_eda.ipynb`):**
  - Sanitized trailing empty `GLORIA_ID` row, merged tables on `GLORIA_ID`, filtered QC-passed observations (`Flagged == 0`).
  - Converted hyperspectral remote-sensing reflectance to dimensionless surface water reflectance: $\rho_w(\lambda) = \pi \cdot R_{rs}(\lambda)$.
  - Computed Sentinel-2 band equivalents (nominal center wavelengths and FWHM spectral-response-function averages) and aquatic indices (`NDCI`, `NDTI`, `NDWI_B8`, `NDWI_B8A`, `2BDA`).
  - Independent target preservation retained **5,008** clean rows (`3,746` valid `Chla`, `3,460` valid `TSS`) in `data/processed/gloria_cleaned.csv` (`62` columns).

### 4.2 Sentinel-2 L2A Ingestion & Preprocessing (Day 1 — `ml/02_sentinel_fetch.ipynb`)
- **Sentinel-2 Product:** `sentinel-2-l2a` via Microsoft Planetary Computer STAC API.
- **Reference Spectral Definitions:**
  - **Bands Used in Scene Extraction:** `B02` (490 nm), `B03` (560 nm), `B04` (665 nm), `B05` (705 nm), `B06` (740 nm), `B07` (783 nm), `B08` (842 nm), `B8A` (865 nm), `SCL` (Scene Classification Layer).
  - **Scale Factor:** Divide raw Sentinel-2 L2A `uint16` reflectance by `10000.0`.
  - **Cloud Mask:** Exclude pixels where `SCL` is in `{3, 8, 9, 10}` (cloud shadow, medium/high cloud, cirrus); scenes require `> 95%` valid lake non-nodata coverage and `< 15%` local cloud/shadow (`local_cloud_pct`).
  - **Water Mask:**
    - Reference specification: `NDWI = (B03 - B08) / (B03 + B08 + 1e-6)`, open water `SCL == 6` / `NDWI > 0.1`.
    - Implemented quality-gated core mask (`ml/02_sentinel_fetch.ipynb`): `in_poly & v_mask & ((scl_arr == 6) | (ndwi > 0.05))` with morphological erosion (`PASS_ERODED`, requiring `selected_water_px >= 100`).
  - **Quality Proxies:**
    - `NDCI = (B05 - B04) / (B05 + B04 + 1e-6)` (Chlorophyll-a proxy)
    - `NDTI = (B04 - B03) / (B04 + B03 + 1e-6)` (Turbidity / TSS proxy)
    - `2BDA = B05 / (B04 + 1e-6)` (Auxiliary 2-band Chl-a ratio)

### 4.3 Temporal Tabular & Spatial Patch Datasets
- **Scene Time Series:** `data/processed/sentinel_timeseries.csv` (`128` rows × `36` columns; `32` chronological observations per lake from `2024-06` to `2026-11`; all `128` scenes pass `PASS_ERODED`). Inter-observation gaps range from `2` to `185` days due to monsoon cloud cover.
- **Spatial Patches:** `data/processed/sentinel_patches/{lake_id}.npz` (`4` archives), each storing:
  - `patches`: `float32` tensor of shape `(32, 9, 64, 64)` (zero `NaN`/`Inf`).
  - `channel_names`: `['B02', 'B03', 'B04', 'B05', 'B06', 'B8A', 'NDCI', 'NDTI', 'water_mask']` (Note: `B07` and `B08` are omitted from the 9-channel NPZ stack; `B8A` is used as the NIR band).
  - `dates`: `32` ISO-8601 date strings aligned 1:1 with `sentinel_timeseries.csv` and `sentinel_anomalies.csv`.
  - `bounds`: `[W, S, E, N]` in `EPSG:4326`.

---

## 5. Baseline ML Models & Anomaly Detection (Day 2)

### 5.1 GLORIA Regression Baselines (`ml/03_baseline_models.ipynb`)
- **Models Saved:** `models/baseline_chla.pkl` (`Chla`, `mg/m³`) and `models/baseline_tss.pkl` (`TSS`, `mg/L`), both `RandomForest` (`n_estimators=300`, `max_depth=12`, `min_samples_leaf=5`) selected by lowest 5-fold `GroupKFold` CV mean `RMSE_log` on spatial `group_key` splits (`test_size=0.2`, `random_state=42`).
- **11-Feature Schema:** `['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'NDCI', 'NDTI', 'ratio_r_g', 'ratio_re1_r', 'ratio_g_b']`. (`B08`/`NDWI` omitted due to missingness in GLORIA/Sentinel alignment.)
- **Target Transform:** `log1p` forward, `expm1` clipped `>= 0` inverse.
- **GLORIA Held-Out Test Metrics (`models/baseline_metrics.json`):**
  - `Chla` (`n_test = 908`): `R²_log = 0.8754`, `RMSE_log = 0.4238`, `MAE_log = 0.2942`, `R²_orig = 0.3966`, `MAE_orig = 6.34 mg/m³`, `RMSE_orig = 38.52 mg/m³`, `MdAPE = 25.81%`.
  - `TSS` (`n_test = 904`): `R²_log = 0.4981`, `RMSE_log = 0.7471`, `MAE_log = 0.5877`, `R²_orig = 0.5625`, `MAE_orig = 13.50 mg/L`, `RMSE_orig = 61.69 mg/L`, `MdAPE = 59.62%`.
- **Zero-Shot Sentinel Inference:** Applied to `sentinel_timeseries.csv` to populate `pred_chla_proxy` and `pred_tss_proxy`.

### 5.2 Trailing Robust Z-Score Anomaly Detector & Spatial Hotspots (`ml/03b_anomaly_hotspots.ipynb`)
- **Temporal Anomaly Detector (`data/processed/sentinel_anomalies.csv`, `128` rows × `49` cols):**
  - Uses per-lake trailing window (`trailing_window = 5`, `min_history = 3`; first 3 rows `t = 0..2` per lake have `baseline_anomaly_score = NaN` and `is_anomaly = False`).
  - Positive robust Z-score across 6 indicators (`NDCI_med`, `NDTI_med`, `NDCI_p90`, `NDTI_p90`, `pred_chla_proxy`, `pred_tss_proxy`) with `epsilon_formula = max(base_floor, 0.25 * MAD_lake_ref)`.
  - Rule: `is_anomaly = (baseline_anomaly_score > 2.0)`. Flags `63 / 128` observations (`bellandur: 20`, `yamuna_okhla: 17`, `sukhna: 14`, `hussain_sagar: 12`).
- **Semi-Synthetic Benchmark (`data/processed/anomaly_benchmark.csv`, `.npz`, `n = 64`):**
  - `32` held-out normal scenes (`t = 24..31`) + `32` localized simulated plume injections (`16` `turbidity_sediment` modifying `B04`, `16` `algal_bloom` modifying `B05`).
  - Day 2 Rolling MAD metrics: `Precision = 0.6444`, `Recall = 0.9063`, `F1 = 0.7532`, `ROC-AUC = 0.7964` (`TP=29, FP=16, TN=16, FN=3`).
- **Spatial Hotspot Extraction (`data/processed/spatial_hotspots.json`):**
  - Connected water-pixel clusters (`>= 3` pixels) exceeding historical pixel baseline inside verified lake boundaries (`170` total clusters across `128` lake-dates; empty list `[]` when no cluster meets threshold).

### 5.3 Conservative Deterministic Risk Engine (`backend/risk_engine.py`)
- **Module Status:** Implemented and unit-tested (`8/8` tests pass in `backend/test_risk_engine.py`).
- **Thresholds (`RiskThresholds`):** `anomaly_medium_min = 2.5`, `anomaly_high_min = 4.5`, `min_hotspots_for_elevation = 1`, `enable_cnn_lstm_rule = False` (future CNN-LSTM hook disabled by default).
- **Precedence Rules:**
  1. Missing/non-finite `anomaly_score` or `hotspot_count` -> `LOW`.
  2. `anomaly_score < 2.5` OR `hotspot_count == 0` -> `LOW` (uncorroborated statistical spikes without spatial hotspots remain `LOW`).
  3. `hotspot_count >= 1` AND `anomaly_score > 4.5` -> `HIGH`.
  4. `hotspot_count >= 1` AND `2.5 <= anomaly_score <= 4.5` -> `MEDIUM`.
- **Continuous Screening Score (`[0.0, 100.0]`):** `60 * min(anomaly_score / 10.0, 1.0) + 40 * (0.6 * min(hotspot_count / 5.0, 1.0) + 0.4 * clamp(max_severity, 0, 1))`.
- **Calibration Across Current 128 Observations:** `LOW: 88`, `MEDIUM: 21`, `HIGH: 19`.

---

## 6. CNN-LSTM Architecture, Training, Evaluation & Deployment (Day 3)

### 6.1 Model Architecture (`WaterCNNLSTM` in `ml/04_cnn_lstm_colab.ipynb`)
- **Verified Input Tensor:** `(B, 4, 9, 64, 64)` (`batch`, 4 chronological frames `[t-4..t-1]`, 9 channels `['B02', 'B03', 'B04', 'B05', 'B06', 'B8A', 'NDCI', 'NDTI', 'water_mask']`, `64x64` spatial patch). *(Adapted from initial `(B, 4, 8, 32, 32)` plan to match actual NPZ tensors without dropping bands or cropping.)*
- **Network Layers (`25,029` trainable parameters):**
  - Spatial CNN Encoder (applied per frame over `B*4`): `Conv2d(9->16, k=3, p=1) + BN + ReLU + MaxPool2d(2)` -> `Conv2d(16->32, k=3, p=1) + BN + ReLU + MaxPool2d(2)` -> `Conv2d(32->32, k=3, p=1) + BN + ReLU`.
  - Global Average Pooling (`mean(dim=(-2, -1))`) -> `(B*4, 32)`.
  - Spatial Projection: `Dropout(0.25) -> Linear(32->32) -> ReLU` -> reshape `(B, 4, 32)`.
  - Temporal Encoder: `LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)` -> last hidden step `(B, 32)`.
  - Prediction Head: `Dropout(0.20) -> Linear(32->5)`.
- **Output Tensor:** `(B, 5)` predicting standardized `[NDWI_med, NDCI_med, NDTI_med, log1p(pred_chla_proxy), log1p(pred_tss_proxy)]`.

### 6.2 Training Methodology & Preprocessing
- **Sequence Construction:** `28` sliding windows (`t = 4..31`) per lake = `112` total sequences (never crossing lake boundaries).
- **Chronological Split:** Train `t = 4..18` (`60` seqs, 15/lake), Validation `t = 19..23` (`20` seqs, 5/lake), Held-out Test `t = 24..31` (`32` seqs, 8/lake — aligned 1:1 with Day 2 benchmark dates).
- **Normalization (Train-Only Fit):** Channel standardization (`x_mean_8`, `x_std_8`) on channels `0..7` with binary `water_mask` (ch `8`) unscaled in `{0, 1}`; `log1p` applied to `Chla` and `TSS` followed by target standardization (`y_mean_5`, `y_std_5`) stored inside `models/cnn_lstm_best.pt`.
- **Augmentation (Train-Only):** Uniform horizontal/vertical flips and 90°/180°/270° rotations across all 4 frames and 9 channels.
- **Optimization:** `SmoothL1Loss`, `AdamW(lr=1e-3, weight_decay=1e-4)`, `batch_size=8`, `max_epochs=120`, `patience=15`, `random_seed=42`. Best checkpoint saved at **epoch 41** (`best_val_loss = 0.448638`; early stopping at epoch `56`; loss curve saved at `docs/figures/cnn_lstm_loss_curve.png`).

### 6.3 Evaluation & Baseline Comparisons (`models/cnn_lstm_metrics.json`, `models/metrics_comparison.json`)
- **Held-Out Test (`n = 32`, proxy-target evaluation for Chla/TSS) vs Same-32 Persistence Baseline (`y_hat(t) = y(t-1)`):**

| Target | Unit / Source | CNN-LSTM `R²_orig` | Persist `R²_orig` | CNN-LSTM `MAE_orig` | Persist `MAE_orig` | CNN-LSTM `RMSE_orig` | Persist `RMSE_orig` | CNN-LSTM `R²_log` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `NDWI` | unitless (S2 index) | `0.0926` | `-0.6723` | `0.1187` | `0.1767` | `0.1681` | `0.2282` | — |
| `NDCI` | unitless (S2 index) | `-0.1182` | `-0.8009` | `0.0714` | `0.0906` | `0.1029` | `0.1306` | — |
| `NDTI` | unitless (S2 index) | `0.1312` | `-0.5064` | `0.0616` | `0.0759` | `0.0806` | `0.1062` | — |
| `Chla` | mg/m³ (Day 2 RF proxy) | `0.3146` | `0.2804` | `71.9151` | `77.6143` | `107.4930` | `110.1418` | `0.3596` |
| `TSS` | mg/L (Day 2 RF proxy) | `-0.0692` | `-1.1144` | `50.2683` | `82.4997` | `149.5109` | `210.2545` | `-0.0460` |

- **Residual Anomaly Detection (`models/cnn_lstm_residual_stats.json`):**
  - Fitted strictly on validation residuals (`t = 19..23`, `n = 20`, zero test leakage) using two-sided standardized Z-score: `anomaly_score = max_k |residual_k - mu_val_k| / max(std_val_k, 1e-6)`, `is_anomaly = (anomaly_score > 2.0)`.
  - Held-out real scenes (`n = 32`, no ground-truth labels): flagged `6 / 32` vs Day 2 Rolling MAD `16 / 32` (agreement `18 / 32`).
  - Semi-synthetic benchmark (`n = 64`, 4 shared targets): `Precision = 0.5000`, `Recall = 0.1563`, `F1 = 0.2381`, `ROC-AUC = 0.4868` (`TP=5, FP=5, TN=27, FN=27`), reflecting that localized single-frame plume perturbations are smoothed out by 4-frame global average pooling compared to tabular `NDCI_p90`/`NDTI_p90` MAD scoring.

### 6.4 ONNX Export, INT8 Quantization & Benchmarking (`models/onnx_benchmark_results.json`)
- **Models Generated & Validated (`onnx.checker` + ORT `1.32.0` `CPUExecutionProvider` across `B in {1, 8, 20, 32, 112}`):**
  - `models/cnn_lstm_fp32.onnx` (`137,611` bytes / `0.1312` MB, opset `18`, `dynamo=True, external_data=False`): machine-precision parity (`max_abs_diff = 3.58e-07` on test `n=32`, `5.36e-07` on `n=112` in scaled space).
  - `models/cnn_lstm_int8.onnx` (`35,666` bytes / `0.0340` MB, `74.08%` smaller): `quant_pre_process` + `quantize_dynamic(QuantType.QInt8)`. Quantized `Conv`->`ConvInteger` (3), `Gemm`->`MatMulInteger` (2), and `LSTM`->`DynamicQuantizeLSTM` (1), leaving `Relu`, `MaxPool`, `ReduceMean`, and shape ops in FP32. Introduces `0.0863` scaled (`3.87 mg/m³` Chla) max drift on `n=32` and `0.6961` scaled (`28.93 mg/m³` Chla) max drift on `n=112`.
  - *(Legacy Day 3 Phase E export `models/cnn_lstm_best.onnx`, opset `17`, `106,135` bytes, is preserved for reference.)*
- **Measured CPU Latency (Windows 10, Intel64 Family 6 Model 170 Stepping 4, 18 logical CPUs, PyTorch `2.15.0+cpu` 14 threads):**

| Batch Size | Model | Size (MB) | Mean Latency | Median Latency | P95 Latency | Throughput |
|---|---|---:|---:|---:|---:|---:|
| `1` (300 iters, 50 warmup) | PyTorch FP32 | 0.1068 | 3.647 ms | 3.318 ms | 6.310 ms | 274.2 samples/s |
| `1` (300 iters, 50 warmup) | **ONNX FP32** | **0.1312** | **1.664 ms** | **0.911 ms** | **3.707 ms** | **601.0 samples/s** |
| `1` (300 iters, 50 warmup) | ONNX INT8 | 0.0340 | 5.444 ms | 5.116 ms | 7.596 ms | 183.7 samples/s |
| `32` (150 iters, 30 warmup) | PyTorch FP32 | 0.1068 | 61.629 ms | 61.307 ms | 86.029 ms | 519.2 samples/s |
| `32` (150 iters, 30 warmup) | **ONNX FP32** | **0.1312** | **21.790 ms** | **18.791 ms** | **35.497 ms** | **1468.6 samples/s** |
| `32` (150 iters, 30 warmup) | ONNX INT8 | 0.0340 | 313.788 ms | 302.156 ms | 551.501 ms | 102.0 samples/s |

- **Selected Deployment Candidate:** `models/cnn_lstm_fp32.onnx` (fastest CPU latency, exact FP32 accuracy, already `< 0.14` MB; INT8 dynamic quantization overhead slows CPU execution on this 25k-parameter model).

---

## 7. Final Prediction CSV Schema (`data/processed/sentinel_final_predictions.csv`)

Generated from verified `models/cnn_lstm_fp32.onnx` inference and validation-only `models/cnn_lstm_residual_stats.json` over all `128` valid source observations (`32` per lake; `112` CNN-LSTM forecasts for `t = 4..31` plus `16` warm-up rows for `t = 0..3` where `pred_chla`, `pred_tss`, `anomaly_score` are `NaN` and `is_anomaly = False`).

| Column | Type | Source / Definition | Missing Policy |
| :--- | :--- | :--- | :--- |
| `water_body_id` | `string` | Official lake identifier (`bellandur`, `hussain_sagar`, `yamuna_okhla`, `sukhna`) | Required, non-null |
| `date` | `string` | Actual Sentinel-2 observation date (`YYYY-MM-DD`) | Required, non-null |
| `cloud_cover` | `float` | STAC scene metadata `eo:cloud_cover` (`tile_cloud_pct` from `sentinel_anomalies.csv`, `%`) | Required, non-null |
| `water_pixel_count` | `int` | Valid eroded water-mask pixel count (`selected_water_px`, `>= 100`) | Required, non-null |
| `ndwi_mean` | `float` | Native-10m valid-water-pixel scene median `NDWI_med` (`sentinel_anomalies.csv`) | Non-null (`NaN` -> `null` in JSON) |
| `ndci_mean` | `float` | Native-10m valid-water-pixel scene median `NDCI_med` (`sentinel_anomalies.csv`) | Non-null (`NaN` -> `null` in JSON) |
| `ndti_mean` | `float` | Native-10m valid-water-pixel scene median `NDTI_med` (`sentinel_anomalies.csv`) | Non-null (`NaN` -> `null` in JSON) |
| `pred_chla` | `float \| null` | CNN-LSTM Chlorophyll-a RF proxy forecast (`mg/m³`, `>= 0`, **not lab-confirmed**) | `NaN` on `t = 0..3` warm-up (`16` rows) |
| `pred_tss` | `float \| null` | CNN-LSTM TSS RF proxy forecast (`mg/L`, `>= 0`, **not lab-confirmed**) | `NaN` on `t = 0..3` warm-up (`16` rows) |
| `anomaly_score` | `float \| null` | Validation-calibrated two-sided max standardized residual `z_k` (`>= 0`) | `NaN` on `t = 0..3` warm-up (`16` rows) |
| `is_anomaly` | `bool` | `True` iff `anomaly_score > 2.0`; `False` otherwise (including warm-up rows) | Required, non-null (`15` flagged) |

> **Field Mapping Note:** Native-10m arithmetic means were not exported in Day 1/2 tabular summaries and `64x64` NPZ patches omit `B08`; therefore `ndwi_mean`, `ndci_mean`, and `ndti_mean` map to the project's approved native-10m robust scene medians (`NDWI_med`, `NDCI_med`, `NDTI_med`).

---

## 8. FastAPI JSON Contract (Base URL: `/api`) — Target Contract (Integration Pending)

> **Implementation Status:** `backend/risk_engine.py` and `backend/test_risk_engine.py` are complete and verified. `backend/main.py` and `frontend/app.py` are currently empty scaffolds awaiting Day 4 integration.
>
> **Mandatory Scientific Disclaimer:** Every API response must include a disclaimer stating that reported metrics, anomaly scores, proxy concentrations, hotspots, and risk tiers are satellite-derived screening estimates and **not** laboratory-confirmed pollution measurements.

Valid `lake_id` path values across all endpoints: `"bellandur"`, `"hussain_sagar"`, `"yamuna_okhla"`, `"sukhna"`.

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

### 8.1 `GET /api/water-bodies`
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

### 8.2 `GET /api/timeseries/{lake_id}`
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
      "NDWI_med": -0.110724,
      "NDCI_med": 0.209907,
      "NDTI_med": -0.094552,
      "pred_chla_proxy": 383.7538,
      "pred_tss_proxy": 130.8418,
      "baseline_anomaly_score": null,
      "is_anomaly": false,
      "open_water_pct": 18.0449,
      "veg_cover_pct": 37.1174,
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

### 8.3 `GET /api/hotspots/{lake_id}?date=YYYY-MM-DD`
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

### 8.4 `POST /api/report/{lake_id}`
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

---

## 9. Persisted Artifacts, File Paths & Data Dependencies

| Artifact Path | Producer | Status | Purpose / Dependencies |
| :--- | :--- | :--- | :--- |
| `data/raw/gloria/*.csv` | GLORIA dataset | Present (gitignored) | Raw lab matchups, QC flags, hyperspectral `Rrs` (`350–900 nm`) |
| `data/processed/gloria_cleaned.csv` | `ml/01_gloria_eda.ipynb` | Completed | `5,008` QC-passed rows (`3,746` Chla, `3,460` TSS) |
| `data/processed/lake_boundaries/*.geojson` | Day 1 acquisition | Completed | Verified `EPSG:4326` lake polygons (`4` files) |
| `data/processed/sentinel_timeseries.csv` | `ml/02_sentinel_fetch.ipynb` | Completed | `128` Sentinel-2 L2A scene summaries (`32`/lake) |
| `data/processed/sentinel_patches/*.npz` | `ml/02_sentinel_fetch.ipynb` | Completed | `(32, 9, 64, 64)` `float32` patch stacks (`4` lakes) |
| `data/processed/sentinel_anomalies.csv` | `ml/03b_anomaly_hotspots.ipynb` | Completed | `128` scenes with RF proxies & trailing MAD anomaly scores |
| `data/processed/anomaly_benchmark.csv`, `.npz` | `ml/03b_anomaly_hotspots.ipynb` | Completed | `64`-row semi-synthetic plume benchmark (`32` normal + `32` injected) |
| `data/processed/spatial_hotspots.json` | `ml/03b_anomaly_hotspots.ipynb` | Completed | `170` WGS84 hotspot clusters across `128` lake-dates |
| `data/processed/sentinel_final_predictions.csv` | `ml/04_cnn_lstm_colab.ipynb` | Completed | `128` rows (`112` CNN-LSTM forecasts + `16` warm-up `NaN` rows) |
| `models/baseline_chla.pkl`, `baseline_tss.pkl` | `ml/03_baseline_models.ipynb` | Completed | Day 2 GLORIA Random Forest regressors |
| `models/baseline_metrics.json` | `ml/03_baseline_models.ipynb`, `03b` | Completed | Day 2 GLORIA CV/test metrics & MAD benchmark metrics |
| `models/cnn_lstm_best.pt` | `ml/04_cnn_lstm_colab.ipynb` | Completed | Best epoch (`41`) PyTorch weights + train normalization stats |
| `models/cnn_lstm_fp32.onnx` | `ml/04_cnn_lstm_colab.ipynb` | Completed | Recommended ONNX FP32 deployment model (opset `18`, `0.1312` MB) |
| `models/cnn_lstm_int8.onnx` | `ml/04_cnn_lstm_colab.ipynb` | Completed | Dynamic QInt8 quantized model (`0.0340` MB; slower on CPU) |
| `models/cnn_lstm_best.onnx` | `ml/04_cnn_lstm_colab.ipynb` | Completed | Legacy Phase E opset-17 FP32 export (`0.1012` MB) |
| `models/cnn_lstm_metrics.json` | `ml/04_cnn_lstm_colab.ipynb` | Completed | Day 3 test metrics, persistence comparison, anomaly comparison |
| `models/cnn_lstm_residual_stats.json` | `ml/04_cnn_lstm_colab.ipynb` | Completed | Validation-only (`n=20`) two-sided residual Z-score parameters |
| `models/onnx_benchmark_results.json` | `ml/04_cnn_lstm_colab.ipynb` | Completed | PyTorch vs ONNX FP32 vs ONNX INT8 CPU benchmarks & operator audit |
| `models/metrics_comparison.json` | `ml/04_cnn_lstm_colab.ipynb` | Completed | Combined Day 2, Day 3, and ONNX deployment comparison |
| `docs/figures/*.png` | `ml/01`–`04` | Completed | `15` evaluation, verification, and training loss figures |
| `backend/risk_engine.py`, `test_risk_engine.py` | Day 2 backend | Completed | Deterministic risk engine + `8/8` passing unit tests |
| `backend/main.py` | Day 4 backend | Planned (empty) | FastAPI REST server implementing Section 8 contract |
| `frontend/app.py` | Day 4 frontend | Planned (empty) | Streamlit + Folium geospatial screening dashboard |
