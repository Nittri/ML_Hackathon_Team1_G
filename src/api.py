"""FastAPI service for evaluator trip selection and server-side scoring."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score

from .data_io import find_trips, load_trip
from .decoder import classifier_only, viterbi_decode
from .preprocessing import (
    _reference_arcs_for_observations,
    build_trip_data,
)

LOGGER = logging.getLogger("pathsnap.api")
logging.basicConfig(level=logging.INFO)
ROOT = Path(__file__).resolve().parents[1]
UI_FILE = ROOT / "ui" / "index.html"


class TripSelection(BaseModel):
    trip_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")


class PredictionService:
    def __init__(self, data_dir: Path, model_path: Path, model_name: str):
        self.data_dir = data_dir
        self.model_path = model_path
        self.model_name = model_name
        self._model: Any | None = None

    def available_trips(self) -> list[str]:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Evaluator partition directory not found: {self.data_dir}")
        return find_trips(str(self.data_dir))

    def _load_model(self) -> Any:
        if self._model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Trained model artifact not found: {self.model_path}")
            self._model = joblib.load(self.model_path)
        return self._model

    def predict(self, trip_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", trip_id):
            raise ValueError("Invalid trip ID")
        allowed = set(self.available_trips())
        if trip_id not in allowed:
            raise KeyError(f"Unknown evaluator trip ID: {trip_id}")

        model = self._load_model()
        raw_trip = load_trip(str(self.data_dir), trip_id)
        if raw_trip.get("route") is None:
            raise ValueError(f"Evaluator trip {trip_id} is missing its reference route")
        reference_route = pd_numeric_route(raw_trip["route"]["arc_id"])

        unlabeled_trip = dict(raw_trip)
        unlabeled_trip["route"] = None
        trip = build_trip_data(unlabeled_trip, include_reference=False)
        if np.any(reference_route < 0) or np.any(reference_route >= len(trip.arcs)):
            raise ValueError(f"Evaluator trip {trip_id} contains out-of-range reference arcs")
        started = time.perf_counter()
        if self.model_name in {"Graph-kernel SVM", "GNN"}:
            scores = np.asarray(model.predict_trip(trip), dtype=np.float64)
        else:
            scores = np.asarray(
                model.predict_proba(trip.edge_features.to_numpy(dtype=np.float32))[:, 1],
                dtype=np.float64,
            )
        classifier_route = classifier_only(trip, scores)
        decoded_route = viterbi_decode(trip, scores)
        inference_seconds = time.perf_counter() - started

        reference_observations = _reference_arcs_for_observations(
            trip.track_xy, trip.node_xy, trip.arcs, reference_route
        )
        route_labels = np.zeros(len(trip.arcs), dtype=np.uint8)
        route_labels[np.unique(reference_route)] = 1
        predicted_labels = np.zeros(len(trip.arcs), dtype=np.uint8)
        predicted_labels[np.unique(decoded_route)] = 1
        classifier_top1 = accuracy_score(reference_observations, classifier_route)
        viterbi_top1 = accuracy_score(reference_observations, decoded_route)
        route_macro_f1 = f1_score(route_labels, predicted_labels, average="macro", zero_division=0)
        route_continuity = (
            float(
                np.mean(
                    (decoded_route[:-1] == decoded_route[1:])
                    | (
                        trip.arcs[decoded_route[:-1], 1]
                        == trip.arcs[decoded_route[1:], 0]
                    )
                )
            )
            if len(decoded_route) > 1
            else 1.0
        )

        response = {
            "trip_id": trip_id,
            "model": self.model_name,
            "metrics": {
                "classifier_observation_top1_accuracy": float(classifier_top1),
                "viterbi_observation_top1_accuracy": float(viterbi_top1),
                "route_membership_macro_f1": float(route_macro_f1),
                "route_continuity": route_continuity,
                "candidate_miss_rate": float(
                    np.mean(
                        [
                            reference not in set(candidates.tolist())
                            for reference, candidates in zip(
                                reference_observations, trip.candidates
                            )
                        ]
                    )
                ),
                "inference_seconds": inference_seconds,
            },
            "gps_points": [
                {"lat": float(row.lat), "lon": float(row.lon)}
                for row in trip.track.itertuples(index=False)
            ],
            "predicted_arc_ids": [int(arc_id) for arc_id in decoded_route],
            "predicted_route_coordinates": _route_coordinates(trip, decoded_route),
        }
        LOGGER.info(
            "Completed trip=%s points=%d arcs=%d inference_seconds=%.4f",
            trip_id,
            len(trip.track),
            len(decoded_route),
            inference_seconds,
        )
        return response


def pd_numeric_route(route: Any) -> np.ndarray:
    numeric = np.asarray(route, dtype=float)
    if (
        numeric.ndim != 1
        or len(numeric) == 0
        or not np.isfinite(numeric).all()
        or not np.equal(numeric, np.floor(numeric)).all()
    ):
        raise ValueError("Reference route must contain at least one arc ID")
    return numeric.astype(np.int64)


def _route_coordinates(trip: Any, arc_ids: np.ndarray) -> list[list[float]]:
    coordinates = []
    for arc_id in arc_ids:
        start, end = trip.arcs[int(arc_id)]
        coordinates.append(
            [
                [float(trip.node_xy[start, 1]), float(trip.node_xy[start, 0])],
                [float(trip.node_xy[end, 1]), float(trip.node_xy[end, 0])],
            ]
        )
    return coordinates


def _service_from_environment() -> PredictionService:
    data_dir = Path(os.environ.get("PATHSNAP_EVALUATOR_DIR", ROOT / "data" / "evaluator_test"))
    summary_path = Path(
        os.environ.get("PATHSNAP_RUN_SUMMARY", ROOT / "results" / "experiments" / "run_summary.json")
    )
    explicit_model_path = os.environ.get("PATHSNAP_MODEL_PATH")
    if explicit_model_path:
        model_path = Path(explicit_model_path)
        model_name = os.environ.get("PATHSNAP_MODEL_NAME", "XGBoost (binary logistic)")
    else:
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            model_name = summary["selected_model"]
            tuned_path = ROOT / "results" / "experiments" / "models" / "selected_tuned_model.joblib"
            model_slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            baseline_path = ROOT / "results" / "experiments" / "models" / f"{model_slug}.joblib"
            model_path = tuned_path if tuned_path.is_file() else baseline_path
        else:
            model_name = os.environ.get("PATHSNAP_MODEL_NAME", "XGBoost (binary logistic)")
            model_path = ROOT / "results" / "experiments" / "models" / "selected_tuned_model.joblib"
    return PredictionService(data_dir, model_path, model_name)


app = FastAPI(title="PathSnap route reconstruction API", version="1.0.0")
app.state.prediction_service = _service_from_environment()


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    if not UI_FILE.is_file():
        raise HTTPException(status_code=500, detail="Evaluator UI file is missing")
    return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))


@app.get("/api/trips")
def list_trips() -> dict[str, list[str]]:
    try:
        return {"trip_ids": app.state.prediction_service.available_trips()}
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/predict")
def predict_trip(selection: TripSelection) -> dict[str, Any]:
    try:
        return app.state.prediction_service.predict(selection.trip_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
