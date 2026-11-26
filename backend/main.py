"""FastAPI application entrypoint for satellite water-quality screening backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.water_bodies import router as water_bodies_router
from backend.schemas import HealthResponse
from backend.services.data_loader import DataLoader, SCIENTIFIC_DISCLAIMER
from backend.services.inference import ModelInferenceService


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DataLoader and CPU-only ModelInferenceService once at startup."""
    root_dir = Path(getattr(app.state, "root_dir", ROOT_DIR))

    data_loader = DataLoader(root_dir=root_dir)
    data_loader.load_all()

    inference_service = ModelInferenceService(root_dir=root_dir, preferred_onnx="fp32")
    inference_service.load_all()

    app.state.data_loader = data_loader
    app.state.inference_service = inference_service
    yield
    data_loader.summary_cache.clear()


def create_app(root_dir: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="Satellite Water-Quality Screening & Anomaly Detection API",
        version="1.0.0",
        description=(
            "CPU-only FastAPI backend serving Sentinel-2 proxy predictions, "
            "ONNX CNN-LSTM residual scoring, spatial hotspots, and deterministic risk screening."
        ),
        lifespan=lifespan,
    )
    if root_dir is not None:
        app.state.root_dir = Path(root_dir)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(water_bodies_router, prefix="/api")

    @app.get("/api/health", response_model=HealthResponse, tags=["health"])
    def health_check(request: Request) -> HealthResponse:
        loader: DataLoader | None = getattr(request.app.state, "data_loader", None)
        inference: ModelInferenceService | None = getattr(
            request.app.state, "inference_service", None
        )
        models_ready = bool(inference is not None and inference.is_ready())
        data_ready = bool(loader is not None and loader.is_ready())
        overall_status = "ok" if (models_ready and data_ready) else "degraded"
        water_bodies_loaded = 0
        if loader is not None:
            for meta in loader.get_all_water_bodies():
                has_boundary = meta.lake_id in loader.boundaries
                has_anom = (
                    loader.anomalies_df is not None
                    and not loader.anomalies_df[loader.anomalies_df["lake_id"] == meta.lake_id].empty
                )
                has_pred = (
                    loader.predictions_df is not None
                    and not loader.predictions_df[loader.predictions_df["water_body_id"] == meta.lake_id].empty
                )
                if has_boundary and has_anom and has_pred:
                    water_bodies_loaded += 1

        return HealthResponse(
            status=overall_status,
            backend_ready=True,
            models_ready=models_ready,
            data_ready=data_ready,
            water_bodies_loaded=water_bodies_loaded,
            models=inference.get_status() if inference is not None else {"ready": False},
            datasets=loader.get_status() if loader is not None else {"ready": False},
            disclaimer=SCIENTIFIC_DISCLAIMER,
        )

    return app


app = create_app()
