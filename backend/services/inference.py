"""CPU-only model inference service using ONNX Runtime and Joblib.

Loads:
- `models/cnn_lstm_fp32.onnx` (primary deployment model per SPEC.md Section 6.4)
- `models/cnn_lstm_int8.onnx` (quantized fallback/comparison model)
- `models/baseline_chla.pkl` (GLORIA Random Forest Chla estimator)
- `models/baseline_tss.pkl` (GLORIA Random Forest TSS estimator)
- `models/cnn_lstm_residual_stats.json` (validation-only residual Z-score params)
- `backend/cnn_lstm_norm_stats.json` (train-split channel & target normalization stats)

Never imports PyTorch; all inference executes on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class ComponentState:
    name: str
    path: str
    loaded: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "loaded": self.loaded,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class ModelInferenceService:
    """Thread-safe CPU inference service for baseline RF and CNN-LSTM ONNX models."""

    def __init__(
        self,
        root_dir: Path | None = None,
        *,
        preferred_onnx: str = "fp32",
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else ROOT_DIR
        self.models_dir = self.root_dir / "models"
        self.preferred_onnx = preferred_onnx if preferred_onnx in ("fp32", "int8") else "fp32"

        self.chla_bundle: dict[str, Any] | None = None
        self.tss_bundle: dict[str, Any] | None = None
        self.onnx_sessions: dict[str, ort.InferenceSession] = {}
        self.onnx_io_specs: dict[str, dict[str, Any]] = {}
        self.residual_stats: dict[str, Any] | None = None
        self.norm_stats: dict[str, Any] | None = None

        self.component_states: dict[str, ComponentState] = {}

    def load_all(self) -> None:
        """Load all model bundles, ONNX sessions, and statistics once at startup."""
        self._load_baseline_bundle("baseline_chla", self.models_dir / "baseline_chla.pkl")
        self._load_baseline_bundle("baseline_tss", self.models_dir / "baseline_tss.pkl")
        self._load_onnx_model("cnn_lstm_fp32", self.models_dir / "cnn_lstm_fp32.onnx", key="fp32")
        self._load_onnx_model("cnn_lstm_int8", self.models_dir / "cnn_lstm_int8.onnx", key="int8")
        self._load_residual_stats(self.models_dir / "cnn_lstm_residual_stats.json")
        self._load_norm_stats(self.root_dir / "backend" / "cnn_lstm_norm_stats.json")

    def _load_baseline_bundle(self, name: str, path: Path) -> None:
        state = ComponentState(name=name, path=str(path.relative_to(self.root_dir)))
        try:
            if not path.is_file():
                raise FileNotFoundError(f"Missing model file: {path}")
            bundle = joblib.load(path)
            if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
                raise ValueError(f"Invalid baseline model bundle structure in {path.name}")
            if name == "baseline_chla":
                self.chla_bundle = bundle
            else:
                self.tss_bundle = bundle
            state.loaded = True
            state.metadata = {
                "algorithm": bundle.get("algorithm"),
                "model_name": bundle.get("model_name"),
                "features": list(bundle["features"]),
                "target_transform": bundle.get("target_transform"),
            }
        except Exception as exc:
            state.loaded = False
            state.error = str(exc)
        self.component_states[name] = state

    def _load_onnx_model(self, name: str, path: Path, *, key: str) -> None:
        state = ComponentState(name=name, path=str(path.relative_to(self.root_dir)))
        try:
            if not path.is_file():
                raise FileNotFoundError(f"Missing ONNX model file: {path}")
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(
                str(path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            inputs = [
                {"name": inp.name, "shape": list(inp.shape), "type": inp.type}
                for inp in sess.get_inputs()
            ]
            outputs = [
                {"name": out.name, "shape": list(out.shape), "type": out.type}
                for out in sess.get_outputs()
            ]
            if not inputs or not outputs:
                raise ValueError(f"ONNX model {path.name} missing inputs or outputs")

            self.onnx_sessions[key] = sess
            self.onnx_io_specs[key] = {
                "input_name": inputs[0]["name"],
                "input_shape": inputs[0]["shape"],
                "input_type": inputs[0]["type"],
                "output_name": outputs[0]["name"],
                "output_shape": outputs[0]["shape"],
                "output_type": outputs[0]["type"],
                "providers": sess.get_providers(),
            }
            state.loaded = True
            state.metadata = dict(self.onnx_io_specs[key])
        except Exception as exc:
            state.loaded = False
            state.error = str(exc)
        self.component_states[name] = state

    def _load_residual_stats(self, path: Path) -> None:
        state = ComponentState(name="cnn_lstm_residual_stats", path=str(path.relative_to(self.root_dir)))
        try:
            if not path.is_file():
                raise FileNotFoundError(f"Missing residual stats file: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
            targets = data.get("targets")
            if not isinstance(targets, list) or len(targets) != 5:
                raise ValueError("Residual stats JSON must define exactly 5 targets.")
            self.residual_stats = data
            state.loaded = True
            state.metadata = {
                "anomaly_threshold": data.get("anomaly_threshold", 2.0),
                "targets": [t.get("target") for t in targets],
            }
        except Exception as exc:
            state.loaded = False
            state.error = str(exc)
        self.component_states["cnn_lstm_residual_stats"] = state

    def _load_norm_stats(self, path: Path) -> None:
        state = ComponentState(name="cnn_lstm_norm_stats", path=str(path.relative_to(self.root_dir)))
        try:
            if not path.is_file():
                raise FileNotFoundError(f"Missing normalization stats file: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
            for req in ("x_mean_8", "x_std_8", "y_mean_5", "y_std_5", "input_shape"):
                if req not in data:
                    raise ValueError(f"Missing '{req}' in {path.name}")
            self.norm_stats = {
                "input_shape": [int(v) for v in data["input_shape"]],
                "x_mean_8": np.asarray(data["x_mean_8"], dtype=np.float32).reshape(1, 1, 8, 1, 1),
                "x_std_8": np.maximum(np.asarray(data["x_std_8"], dtype=np.float32).reshape(1, 1, 8, 1, 1), 1e-6),
                "y_mean_5": np.asarray(data["y_mean_5"], dtype=np.float32).reshape(1, 5),
                "y_std_5": np.maximum(np.asarray(data["y_std_5"], dtype=np.float32).reshape(1, 5), 1e-6),
                "log_target_indices": [int(i) for i in data.get("log_target_indices", [3, 4])],
                "target_names": list(data.get("target_names", ["NDWI", "NDCI", "NDTI", "Chla", "TSS"])),
            }
            state.loaded = True
            state.metadata = {
                "input_shape": self.norm_stats["input_shape"],
                "target_names": self.norm_stats["target_names"],
            }
        except Exception as exc:
            state.loaded = False
            state.error = str(exc)
        self.component_states["cnn_lstm_norm_stats"] = state

    def is_ready(self) -> bool:
        """True only when baseline models, at least one ONNX model, and stats are loaded."""
        required_ok = all(
            self.component_states.get(k, ComponentState(k, "")).loaded
            for k in (
                "baseline_chla",
                "baseline_tss",
                "cnn_lstm_residual_stats",
                "cnn_lstm_norm_stats",
            )
        )
        onnx_ok = self.preferred_onnx in self.onnx_sessions or bool(self.onnx_sessions)
        return required_ok and onnx_ok

    def get_status(self) -> dict[str, Any]:
        return {
            "ready": self.is_ready(),
            "preferred_onnx": self.preferred_onnx,
            "components": {k: v.to_dict() for k, v in self.component_states.items()},
        }

    @staticmethod
    def _extract_feature_value(row: Mapping[str, Any], name: str) -> float:
        candidates = (name, f"{name}_med")
        for key in candidates:
            if key in row and row[key] is not None:
                val = row[key]
                if isinstance(val, bool):
                    raise ValueError(f"Feature '{key}' must be numeric, got bool.")
                fval = float(val)
                if not math.isfinite(fval):
                    raise ValueError(f"Feature '{key}' is non-finite ({val}).")
                return fval
        raise ValueError(f"Missing required feature '{name}'.")

    @classmethod
    def _prepare_baseline_features(
        cls,
        feature_row: Mapping[str, Any],
        required_features: list[str],
    ) -> pd.DataFrame:
        if not isinstance(feature_row, Mapping):
            raise ValueError("feature_row must be a mapping of feature names to numeric values.")

        values: dict[str, float] = {}
        base_bands = ("B02", "B03", "B04", "B05", "B06", "B07", "NDCI", "NDTI")
        for col in base_bands:
            if col in required_features or col in ("B02", "B03", "B04", "B05"):
                if col in feature_row or f"{col}_med" in feature_row:
                    values[col] = cls._extract_feature_value(feature_row, col)

        ratio_defs = {
            "ratio_r_g": ("B04", "B03"),
            "ratio_re1_r": ("B05", "B04"),
            "ratio_g_b": ("B03", "B02"),
        }
        for ratio_col, (num_col, den_col) in ratio_defs.items():
            if ratio_col in feature_row and feature_row[ratio_col] is not None:
                values[ratio_col] = cls._extract_feature_value(feature_row, ratio_col)
            elif num_col in values and den_col in values:
                num, den = values[num_col], values[den_col]
                if num <= 0.0 or den <= 0.0:
                    raise ValueError(
                        f"Cannot compute '{ratio_col}': '{num_col}' ({num}) and '{den_col}' ({den}) must be > 0."
                    )
                values[ratio_col] = num / den

        ordered = {}
        for feat in required_features:
            if feat not in values:
                ordered[feat] = cls._extract_feature_value(feature_row, feat)
            else:
                ordered[feat] = values[feat]

        df = pd.DataFrame([ordered], columns=required_features)
        if not np.isfinite(df.to_numpy(dtype=float)).all():
            raise ValueError("Non-finite value encountered in prepared baseline features.")
        return df

    @staticmethod
    def _predict_from_bundle(bundle: dict[str, Any], feature_row: Mapping[str, Any]) -> float:
        features = list(bundle["features"])
        X_df = ModelInferenceService._prepare_baseline_features(feature_row, features)
        raw_pred = float(bundle["model"].predict(X_df)[0])
        if not math.isfinite(raw_pred):
            raise RuntimeError("Baseline model produced a non-finite prediction.")
        if bundle.get("target_transform") == "log1p":
            return float(max(0.0, math.expm1(raw_pred)))
        return raw_pred

    def predict_quality(self, feature_row: Mapping[str, Any]) -> dict[str, float]:
        """Predict Chla (mg/m3) and TSS (mg/L) proxy values from a feature row."""
        if self.chla_bundle is None or self.tss_bundle is None:
            raise RuntimeError("Baseline Chla/TSS models are not loaded.")

        chla_pred = self._predict_from_bundle(self.chla_bundle, feature_row)
        tss_pred = self._predict_from_bundle(self.tss_bundle, feature_row)
        return {
            "pred_chla_proxy": chla_pred,
            "pred_tss_proxy": tss_pred,
        }

    def _get_onnx_session(self, variant: str | None = None) -> tuple[str, ort.InferenceSession, dict[str, Any]]:
        selected = variant or self.preferred_onnx
        if selected not in self.onnx_sessions:
            if variant is None and self.onnx_sessions:
                selected = next(iter(self.onnx_sessions.keys()))
            else:
                raise RuntimeError(f"ONNX model variant '{selected}' is not loaded.")
        return selected, self.onnx_sessions[selected], self.onnx_io_specs[selected]

    def score_sequence_anomaly(
        self,
        patch_seq: np.ndarray | None,
        actual_vec: np.ndarray | list[float] | None,
        *,
        variant: str | None = None,
    ) -> dict[str, Any]:
        """Run CNN-LSTM ONNX inference and score residuals against validation stats.

        `actual_vec` must contain original-scale `[NDWI_med, NDCI_med, NDTI_med,
        pred_chla_proxy, pred_tss_proxy]`.
        """
        if patch_seq is None or actual_vec is None:
            return {
                "available": False,
                "reason": "insufficient_data",
                "anomaly_score": None,
                "is_anomaly": None,
            }

        if self.norm_stats is None or self.residual_stats is None:
            raise RuntimeError("CNN-LSTM normalization or residual statistics are not loaded.")

        selected_variant, sess, io_spec = self._get_onnx_session(variant)

        seq = np.asarray(patch_seq, dtype=np.float32)
        if seq.ndim == 4:
            seq = seq[np.newaxis, ...]
        expected_tail = tuple(self.norm_stats["input_shape"])  # (4, 9, 64, 64)
        if seq.ndim != 5 or seq.shape[1:] != expected_tail:
            raise ValueError(
                f"Invalid patch_seq shape {tuple(seq.shape)}; expected (B, {', '.join(map(str, expected_tail))})."
            )
        if not np.isfinite(seq).all():
            raise ValueError("patch_seq contains NaN or Inf values.")

        actual = np.asarray(actual_vec, dtype=np.float32)
        if actual.ndim == 1:
            actual = actual[np.newaxis, ...]
        if actual.ndim != 2 or actual.shape != (seq.shape[0], 5):
            raise ValueError(f"Invalid actual_vec shape {tuple(actual.shape)}; expected ({seq.shape[0]}, 5).")
        if not np.isfinite(actual).all():
            raise ValueError("actual_vec contains NaN or Inf values.")

        # Normalize channels 0..7; preserve channel 8 (water_mask) unscaled
        seq_norm = seq.copy()
        seq_norm[:, :, :8, :, :] = (
            seq_norm[:, :, :8, :, :] - self.norm_stats["x_mean_8"]
        ) / self.norm_stats["x_std_8"]

        try:
            pred_scaled = sess.run(
                [io_spec["output_name"]],
                {io_spec["input_name"]: seq_norm},
            )[0]
        except Exception as exc:
            raise RuntimeError(f"ONNX Runtime inference failed: {exc}") from exc

        pred_scaled = np.asarray(pred_scaled, dtype=np.float32)
        if pred_scaled.shape != (seq.shape[0], 5) or not np.isfinite(pred_scaled).all():
            raise RuntimeError("ONNX model returned invalid or non-finite output.")

        pred_trans = pred_scaled * self.norm_stats["y_std_5"] + self.norm_stats["y_mean_5"]
        pred_raw = pred_trans.copy()
        actual_trans = actual.copy()
        for idx in self.norm_stats["log_target_indices"]:
            pred_raw[:, idx] = np.maximum(0.0, np.expm1(pred_raw[:, idx]))
            actual_trans[:, idx] = np.log1p(np.maximum(actual_trans[:, idx], 0.0))

        residuals = actual_trans - pred_trans
        targets_stats = self.residual_stats["targets"]
        res_mu = np.asarray([float(t["residual_mean"]) for t in targets_stats], dtype=np.float32)
        res_std = np.maximum(
            np.asarray([float(t["residual_std"]) for t in targets_stats], dtype=np.float32),
            1e-6,
        )
        z_scores = np.abs(residuals - res_mu) / res_std
        threshold = float(self.residual_stats.get("anomaly_threshold", 2.0))

        # Return single-item dict when batch size == 1 (standard API usage), or batch summary
        anom_scores = z_scores.max(axis=1)
        is_anoms = anom_scores > threshold
        target_names = self.norm_stats["target_names"]

        if seq.shape[0] == 1:
            return {
                "available": True,
                "model_variant": selected_variant,
                "anomaly_score": float(anom_scores[0]),
                "is_anomaly": bool(is_anoms[0]),
                "threshold": threshold,
                "per_target_z": {
                    tname: float(z_scores[0, j]) for j, tname in enumerate(target_names)
                },
                "predicted_raw": {
                    tname: float(pred_raw[0, j]) for j, tname in enumerate(target_names)
                },
            }

        return {
            "available": True,
            "model_variant": selected_variant,
            "anomaly_score": [float(v) for v in anom_scores],
            "is_anomaly": [bool(v) for v in is_anoms],
            "threshold": threshold,
            "predicted_raw": pred_raw.tolist(),
        }
