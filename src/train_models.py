"""Grouped model comparison, Bayesian tuning, and student-test evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from scipy.special import expit

from .decoder import classifier_only, viterbi_decode
from .models import GraphKernelSVM, GraphNeuralNetwork, build_tabular_models
from .preprocessing import (
    FEATURE_COLUMNS,
    TripData,
    balanced_edge_indices,
    load_partition,
)
from .report import generate_report

SEED = 42
MODEL_NAMES = (
    "XGBoost (binary logistic)",
    "Random Forest",
    "Graph-kernel SVM",
    "GNN",
    "LightGBM",
)


def _selected_features(
    trips: Sequence[TripData], selected: Sequence[tuple[int, int]]
) -> np.ndarray:
    return np.vstack(
        [trips[trip_index].edge_features.iloc[edge_index].to_numpy() for trip_index, edge_index in selected]
    ).astype(np.float32)


def _fit_model(
    name: str,
    trips: Sequence[TripData],
    random_state: int = SEED,
    params: dict[str, Any] | None = None,
    negative_multiplier: int = 10,
    gnn_epochs: int = 8,
) -> Any:
    selected, labels = balanced_edge_indices(
        trips,
        max_negatives_per_positive=(5 if name == "Graph-kernel SVM" else negative_multiplier),
        random_state=random_state,
    )
    if not len(labels) or len(np.unique(labels)) < 2:
        raise ValueError(f"Training partition for {name} must include positive and negative arcs")

    if name == "Graph-kernel SVM":
        return GraphKernelSVM(
            random_state=random_state, C=float((params or {}).get("C", 1.0))
        ).fit(trips, selected, labels)
    if name == "GNN":
        gnn_config = dict(params or {})
        epochs = int(gnn_config.pop("epochs", gnn_epochs))
        return GraphNeuralNetwork(
            random_state=random_state, epochs=epochs, **gnn_config
        ).fit(trips, selected, labels)

    models = build_tabular_models(
        random_state=random_state,
        xgb_params=params if name == "XGBoost (binary logistic)" else None,
    )
    model = models[name]
    if params and name != "XGBoost (binary logistic)":
        model.set_params(**params)
    features = _selected_features(trips, selected)
    if name == "XGBoost (binary logistic)":
        model.set_params(
            scale_pos_weight=max(float((labels == 0).sum()) / max((labels == 1).sum(), 1), 1.0)
        )
    model.fit(features, labels)
    return model


def _predict_trip(model: Any, name: str, trip: TripData) -> tuple[np.ndarray, bool]:
    if name in {"Graph-kernel SVM", "GNN"}:
        scores = model.predict_trip(trip)
        return np.asarray(scores, dtype=np.float64), name == "GNN"
    probabilities = model.predict_proba(trip.edge_features.to_numpy(dtype=np.float32))[:, 1]
    return probabilities.astype(np.float64), True


def _route_continuity(trip: TripData, arc_ids: np.ndarray) -> float:
    if len(arc_ids) < 2:
        return 1.0
    same_segment = arc_ids[:-1] == arc_ids[1:]
    connected = trip.arcs[arc_ids[:-1], 1] == trip.arcs[arc_ids[1:], 0]
    return float(np.mean(same_segment | connected))


def _evaluate(
    trips: Sequence[TripData],
    scores_by_trip: Sequence[np.ndarray],
    probabilities: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    truth = np.concatenate([trip.edge_labels for trip in trips])
    scores = np.concatenate(scores_by_trip)
    decision_threshold = 0.5 if probabilities else 0.0
    edge_predictions = (scores >= decision_threshold).astype(np.uint8)
    metric_probabilities = np.clip(scores, 1e-7, 1 - 1e-7) if probabilities else None
    observations: list[dict[str, Any]] = []
    pointwise_predictions = []
    decoded_predictions = []
    pointwise_truth = []
    miss_rates = []
    classifier_continuity = []
    decoded_continuity = []
    decode_started = time.perf_counter()

    for trip, trip_scores in zip(trips, scores_by_trip):
        classifier_route = classifier_only(trip, trip_scores)
        decoded_route = viterbi_decode(trip, trip_scores)
        pointwise_predictions.extend(classifier_route.tolist())
        decoded_predictions.extend(decoded_route.tolist())
        pointwise_truth.extend(trip.observation_arc_ids.tolist())
        miss_rates.append(trip.candidate_miss_rate)
        classifier_continuity.append(_route_continuity(trip, classifier_route))
        decoded_continuity.append(_route_continuity(trip, decoded_route))
        observations.append(
            {
                "trip": trip.trip_id,
                "arc_scores": trip_scores,
                "classifier_route": classifier_route,
                "decoded_route": decoded_route,
            }
        )

    metrics: dict[str, Any] = {
        "arc_accuracy": float(accuracy_score(truth, edge_predictions)),
        "arc_precision": float(precision_score(truth, edge_predictions, zero_division=0)),
        "arc_recall": float(recall_score(truth, edge_predictions, zero_division=0)),
        "macro_f1": float(f1_score(truth, edge_predictions, average="macro", zero_division=0)),
        "observation_top1_accuracy": float(accuracy_score(pointwise_truth, pointwise_predictions)),
        "hmm_observation_accuracy": float(accuracy_score(pointwise_truth, decoded_predictions)),
        "candidate_miss_rate": float(np.mean(miss_rates)),
        "classifier_route_continuity": float(np.mean(classifier_continuity)),
        "hmm_route_continuity": float(np.mean(decoded_continuity)),
        "route_decoding_seconds": time.perf_counter() - decode_started,
        "log_loss": (
            float(log_loss(truth, metric_probabilities, labels=[0, 1]))
            if metric_probabilities is not None
            else None
        ),
    }
    return metrics, observations


def _predict_partition(
    model: Any,
    name: str,
    trips: Sequence[TripData],
) -> tuple[list[np.ndarray], bool, float]:
    scores: list[np.ndarray] = []
    probability_output = name != "Graph-kernel SVM"
    started = time.perf_counter()
    for trip in trips:
        trip_scores, trip_probabilities = _predict_trip(model, name, trip)
        scores.append(trip_scores)
        probability_output = probability_output and trip_probabilities
    return scores, probability_output, time.perf_counter() - started


def _cross_validate(
    name: str,
    trips: Sequence[TripData],
    folds: int,
    random_state: int,
    gnn_epochs: int,
) -> tuple[dict[str, Any], float]:
    splitter = GroupKFold(n_splits=folds)
    groups = np.asarray([trip.trip_id for trip in trips])
    out_of_fold_scores: list[np.ndarray | None] = [None] * len(trips)
    fit_seconds = 0.0
    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(trips)), groups=groups), start=1
    ):
        training_trips = [trips[index] for index in train_indices]
        validation_trips = [trips[index] for index in validation_indices]
        started = time.perf_counter()
        model = _fit_model(
            name,
            training_trips,
            random_state=random_state + fold_number,
            gnn_epochs=gnn_epochs,
        )
        fit_seconds += time.perf_counter() - started
        validation_scores, _, _ = _predict_partition(model, name, validation_trips)
        for index, scores in zip(validation_indices, validation_scores):
            out_of_fold_scores[int(index)] = scores
    complete_scores = [scores for scores in out_of_fold_scores if scores is not None]
    if len(complete_scores) != len(trips):
        raise RuntimeError(f"Grouped CV did not produce out-of-fold predictions for {name}")
    cv_probabilities = name != "Graph-kernel SVM"
    metrics, _ = _evaluate(trips, complete_scores, cv_probabilities)
    return metrics, fit_seconds


def _model_params(name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if name == "XGBoost (binary logistic)":
        return {
            "n_estimators": 250,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            **(params or {}),
        }
    if name == "LightGBM":
        return {
            "n_estimators": 250,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "objective": "binary",
            "class_weight": "balanced",
            **(params or {}),
        }
    if name == "Graph-kernel SVM":
        return {
            "kernel": "explicit Weisfeiler-Lehman subtree kernel",
            "base_svm": "LinearSVC",
            "C": 1.0,
            "class_weight": "balanced",
        }
    if name == "GNN":
        return {
            "architecture": "two directed message-passing layers and edge MLP",
            "hidden_channels": 32,
            "learning_rate": 0.003,
            "epochs": 8,
            "loss": "weighted binary cross-entropy with logits",
            "feature_scaling": "StandardScaler fitted on training graphs",
            **(params or {}),
        }
    if name == "Random Forest":
        return {
            "n_estimators": 250,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "criterion": "gini",
        }
    raise ValueError(f"Unknown model: {name}")


def _tuning_search_space(name: str) -> dict[str, Any]:
    if name == "XGBoost (binary logistic)":
        return {
            "max_depth": "integer 3-8",
            "learning_rate": "log-uniform 0.02-0.2",
            "n_estimators": "integer 100-350, step 50",
            "subsample": "uniform 0.65-1.0",
            "colsample_bytree": "uniform 0.65-1.0",
            "min_child_weight": "integer 1-8",
        }
    if name == "LightGBM":
        return {
            "num_leaves": "integer 15-63",
            "learning_rate": "log-uniform 0.02-0.2",
            "n_estimators": "integer 100-350, step 50",
            "max_depth": "integer 3-10",
            "min_child_samples": "integer 10-50",
        }
    if name == "Random Forest":
        return {
            "n_estimators": "integer 100-350, step 50",
            "max_depth": "integer 8-24",
            "min_samples_leaf": "integer 1-8",
        }
    if name == "Graph-kernel SVM":
        return {"C": "log-uniform 0.01-10.0"}
    if name == "GNN":
        return {
            "hidden_channels": "categorical {16, 32, 48}",
            "learning_rate": "log-uniform 0.001-0.01",
        }
    raise ValueError(f"Unknown model: {name}")


def _fit_with_params(
    name: str,
    trips: Sequence[TripData],
    random_state: int,
    params: dict[str, Any] | None,
    gnn_epochs: int,
) -> Any:
    return _fit_model(
        name, trips, random_state, params=params, gnn_epochs=gnn_epochs
    )


def _tune_model(
    name: str,
    trips: Sequence[TripData],
    folds: int,
    trials: int,
    random_state: int,
    gnn_epochs: int,
) -> tuple[dict[str, Any], float, pd.DataFrame, float]:
    import optuna

    splitter = GroupKFold(n_splits=folds)
    groups = np.asarray([trip.trip_id for trip in trips])

    def objective(trial: Any) -> float:
        if name == "XGBoost (binary logistic)":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            }
        elif name == "LightGBM":
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            }
        elif name == "Random Forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
                "max_depth": trial.suggest_int("max_depth", 8, 24),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            }
        elif name == "Graph-kernel SVM":
            params = {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
        else:
            params = {
                "hidden_channels": trial.suggest_categorical("hidden_channels", [16, 32, 48]),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01, log=True),
            }

        fold_scores = []
        for fold_index, (train_indices, validation_indices) in enumerate(
            splitter.split(np.zeros(len(trips)), groups=groups), start=1
        ):
            model = _fit_with_params(
                name,
                [trips[index] for index in train_indices],
                random_state + fold_index,
                params,
                gnn_epochs,
            )
            validation_trips = [trips[index] for index in validation_indices]
            validation_scores, probabilities, _ = _predict_partition(model, name, validation_trips)
            truth = np.concatenate([trip.edge_labels for trip in validation_trips])
            scores = np.concatenate(validation_scores)
            predictions = scores >= (0.5 if probabilities else 0.0)
            fold_scores.append(
                float(f1_score(truth, predictions, average="macro", zero_division=0))
            )
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    started = time.perf_counter()
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    tuning_seconds = time.perf_counter() - started
    trial_rows = [
        {
            "trial": trial.number,
            "value_macro_f1": trial.value,
            "state": trial.state.name,
            **trial.params,
        }
        for trial in study.trials
    ]
    return study.best_params, float(study.best_value), pd.DataFrame(trial_rows), tuning_seconds


def _write_prediction_files(
    output_dir: Path,
    model_name: str,
    trips: Sequence[TripData],
    observations: Sequence[dict[str, Any]],
) -> None:
    model_slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    rows = []
    for trip, result in zip(trips, observations):
        for index, (predicted, decoded, reference) in enumerate(
            zip(
                result["classifier_route"],
                result["decoded_route"],
                trip.observation_arc_ids,
            )
        ):
            rows.append(
                {
                    "trip_id": trip.trip_id,
                    "observation_index": index,
                    "classifier_arc_id": int(predicted),
                    "viterbi_arc_id": int(decoded),
                    "reference_arc_id": int(reference),
                    "candidate_miss": bool(
                        reference not in set(trip.candidates[index].tolist())
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / f"predictions_{model_slug}_student_test.csv", index=False)


def _save_map(
    output_dir: Path,
    trip: TripData,
    prediction: dict[str, Any],
    road_limit: int = 4000,
) -> None:
    import folium
    from folium import FeatureGroup, LayerControl, PolyLine, CircleMarker

    candidate_ids = np.unique(np.concatenate(trip.candidates))
    if len(candidate_ids) > road_limit:
        candidate_ids = candidate_ids[:road_limit]
    point_coordinates = list(zip(trip.track["lat"], trip.track["lon"]))
    map_view = folium.Map(
        location=[float(trip.track["lat"].mean()), float(trip.track["lon"].mean())],
        zoom_start=15,
        control_scale=True,
    )

    road_layer = FeatureGroup(name="Candidate road network")
    for arc_id in candidate_ids:
        start, end = trip.arcs[arc_id]
        coordinates = [
            [float(trip.node_xy[start, 1]), float(trip.node_xy[start, 0])],
            [float(trip.node_xy[end, 1]), float(trip.node_xy[end, 0])],
        ]
        PolyLine(coordinates, color="#8c8c8c", weight=1, opacity=0.4).add_to(road_layer)
    road_layer.add_to(map_view)

    reference_layer = FeatureGroup(name="Reference route (student-test only)")
    _draw_route(reference_layer, trip, trip.route_arc_ids, "#2ca02c", 5)
    reference_layer.add_to(map_view)

    predicted_layer = FeatureGroup(name="Classifier-only route")
    _draw_route(predicted_layer, trip, prediction["classifier_route"], "#ff7f0e", 4)
    predicted_layer.add_to(map_view)

    decoded_layer = FeatureGroup(name="HMM/Viterbi route")
    _draw_route(decoded_layer, trip, prediction["decoded_route"], "#d62728", 4)
    decoded_layer.add_to(map_view)

    gps_layer = FeatureGroup(name="GPS observations")
    for index, (lat, lon) in enumerate(point_coordinates):
        CircleMarker(
            [float(lat), float(lon)],
            radius=2,
            color="#1f77b4",
            fill=True,
            fill_opacity=0.7,
            tooltip=f"Observation {index}",
        ).add_to(gps_layer)
    gps_layer.add_to(map_view)
    LayerControl(collapsed=False).add_to(map_view)
    map_view.save(output_dir / f"student_test_trip_{trip.trip_id}.html")


def _draw_route(layer: Any, trip: TripData, arc_ids: np.ndarray, color: str, weight: int) -> None:
    from folium import PolyLine

    if len(arc_ids) == 0:
        return
    starts = trip.arcs[arc_ids, 0]
    ends = trip.arcs[arc_ids, 1]
    coordinates = [
        [float(trip.node_xy[start, 1]), float(trip.node_xy[start, 0])]
        for start in starts
    ]
    coordinates.append([float(trip.node_xy[ends[-1], 1]), float(trip.node_xy[ends[-1], 0])])
    PolyLine(coordinates, color=color, weight=weight, opacity=0.8).add_to(layer)


def _plot_comparison(results: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("cv_macro_f1", "Grouped CV macro F1"),
        ("cv_observation_top1_accuracy", "Grouped CV observation Top-1"),
        ("student_arc_macro_f1", "Student-test arc macro F1"),
        ("student_observation_top1_accuracy", "Student-test observation Top-1"),
        ("student_hmm_route_continuity", "Student-test HMM route continuity"),
    ]
    for column, title in metrics:
        if column not in results:
            continue
        values = results[column].astype(float)
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.bar(results["model"], values, color="#3977a8")
        axis.set_ylabel("Score")
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure.savefig(output_dir / f"{column}.png", dpi=160)
        plt.close(figure)


def run_experiments(
    train_path: str,
    student_test_path: str,
    output_path: str,
    folds: int = 4,
    tuning_trials: int = 8,
    gnn_epochs: int = 8,
    random_state: int = SEED,
    skip_tuning: bool = False,
) -> pd.DataFrame:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)

    print(f"Preprocessing training trips from {train_path}")
    train_trips = load_partition(train_path)
    print(f"Preprocessing student-test trips from {student_test_path}")
    student_trips = load_partition(student_test_path)
    if folds < 2 or folds > len(train_trips):
        raise ValueError("folds must be between 2 and the number of training trips")

    configs: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    cv_metrics: dict[str, dict[str, Any]] = {}
    student_observations_by_model: dict[str, list[dict[str, Any]]] = {}
    for name in MODEL_NAMES:
        print(f"Grouped CV: {name}")
        cv, cv_seconds = _cross_validate(
            name, train_trips, folds, random_state, gnn_epochs
        )
        cv_metrics[name] = cv
        configs[name] = _model_params(name)
        if name == "GNN":
            configs[name]["epochs"] = gnn_epochs
        started = time.perf_counter()
        model = _fit_model(
            name,
            train_trips,
            random_state=random_state,
            params=configs[name] if name.startswith("XGBoost") else None,
            gnn_epochs=gnn_epochs,
        )
        training_seconds = time.perf_counter() - started
        student_scores, probabilities, inference_seconds = _predict_partition(
            model, name, student_trips
        )
        student, observations = _evaluate(student_trips, student_scores, probabilities)
        student_observations_by_model[name] = observations
        if isinstance(model, GraphNeuralNetwork):
            history = pd.DataFrame(
                {"epoch": np.arange(1, len(model.training_history) + 1), "loss": model.training_history}
            )
            history.to_csv(output_dir / "gnn_training_history.csv", index=False)
            figure, axis = plt.subplots(figsize=(7, 4))
            axis.plot(history["epoch"], history["loss"], marker="o")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Weighted binary cross-entropy")
            axis.set_title("GNN training loss")
            figure.tight_layout()
            figure.savefig(output_dir / "gnn_training_loss.png", dpi=160)
            plt.close(figure)
        if isinstance(model, GraphNeuralNetwork):
            model.to_cpu()
        joblib.dump(model, model_dir / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib")
        _write_prediction_files(predictions_dir, name, student_trips, observations)
        rows.append(
            {
                "model": name,
                "configuration": json.dumps(configs[name], sort_keys=True),
                "cv_macro_f1": cv["macro_f1"],
                "cv_observation_top1_accuracy": cv["observation_top1_accuracy"],
                "cv_hmm_observation_accuracy": cv["hmm_observation_accuracy"],
                "cv_arc_precision": cv["arc_precision"],
                "cv_arc_recall": cv["arc_recall"],
                "cv_log_loss": cv["log_loss"],
                "cv_candidate_miss_rate": cv["candidate_miss_rate"],
                "cv_hmm_route_continuity": cv["hmm_route_continuity"],
                "student_arc_macro_f1": student["macro_f1"],
                "student_observation_top1_accuracy": student["observation_top1_accuracy"],
                "student_hmm_observation_accuracy": student["hmm_observation_accuracy"],
                "student_arc_precision": student["arc_precision"],
                "student_arc_recall": student["arc_recall"],
                "student_log_loss": student["log_loss"],
                "student_candidate_miss_rate": student["candidate_miss_rate"],
                "student_classifier_route_continuity": student["classifier_route_continuity"],
                "student_hmm_route_continuity": student["hmm_route_continuity"],
                "student_route_decoding_seconds": student["route_decoding_seconds"],
                "cv_fit_seconds": cv_seconds,
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
            }
        )
        print(
            f"  CV macro F1={cv['macro_f1']:.4f}; student Top-1="
            f"{student['observation_top1_accuracy']:.4f}; "
            f"HMM continuity={student['hmm_route_continuity']:.4f}"
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    _plot_comparison(comparison, output_dir)

    best_model_name = str(comparison.loc[comparison["cv_macro_f1"].idxmax(), "model"])
    _save_map(
        predictions_dir,
        student_trips[0],
        student_observations_by_model[best_model_name][0],
    )
    tuned_metrics: dict[str, Any] | None = None
    best_params: dict[str, Any] = {}
    tuned_cv_macro_f1: float | None = None
    tuning_seconds = 0.0
    if not skip_tuning and tuning_trials > 0:
        print(f"Bayesian tuning of {best_model_name} on grouped training folds")
        tuning_folds = min(folds, 3)
        best_params, tuned_cv_macro_f1, trials_frame, tuning_seconds = _tune_model(
            best_model_name,
            train_trips,
            folds=tuning_folds,
            trials=tuning_trials,
            random_state=random_state,
            gnn_epochs=gnn_epochs,
        )
        trials_frame.to_csv(output_dir / "bayesian_trials.csv", index=False)
        tuned_model = _fit_with_params(
            best_model_name, train_trips, random_state, best_params, gnn_epochs
        )
        tuned_scores, tuned_probabilities, tuned_inference = _predict_partition(
            tuned_model, best_model_name, student_trips
        )
        tuned_metrics, tuned_observations = _evaluate(
            student_trips, tuned_scores, tuned_probabilities
        )
        if isinstance(tuned_model, GraphNeuralNetwork):
            tuned_model.to_cpu()
        joblib.dump(tuned_model, model_dir / "selected_tuned_model.joblib")
        _write_prediction_files(
            predictions_dir, f"{best_model_name}_tuned", student_trips, tuned_observations
        )
        baseline_row = comparison.loc[comparison["model"] == best_model_name].iloc[0]
        tuned_comparison = {
            "model": best_model_name,
            "cv_macro_f1_untuned": float(baseline_row["cv_macro_f1"]),
            "cv_macro_f1_tuned": tuned_cv_macro_f1,
            "student_observation_top1_accuracy_untuned": float(
                baseline_row["student_observation_top1_accuracy"]
            ),
            "student_observation_top1_accuracy_tuned": tuned_metrics[
                "observation_top1_accuracy"
            ],
            "student_arc_macro_f1_untuned": float(baseline_row["student_arc_macro_f1"]),
            "student_arc_macro_f1_tuned": tuned_metrics["macro_f1"],
            "student_log_loss_untuned": baseline_row["student_log_loss"],
            "student_log_loss_tuned": tuned_metrics["log_loss"],
            "inference_seconds_untuned": float(baseline_row["inference_seconds"]),
            "inference_seconds_tuned": tuned_inference,
        }
        pd.DataFrame([tuned_comparison]).to_csv(
            output_dir / "tuning_comparison.csv", index=False
        )
        config = {
            "selected_model": best_model_name,
            "selection_metric": "grouped training cross-validation macro F1",
            "best_params": best_params,
            "best_cv_macro_f1": tuned_cv_macro_f1,
            "objective": "maximize mean grouped-CV edge macro F1",
            "group_cv_folds": tuning_folds,
            "search_space": _tuning_search_space(best_model_name),
            "tuning_trials": tuning_trials,
            "tuning_seconds": tuning_seconds,
            "tuned_student_test_metrics": tuned_metrics,
            "tuned_inference_seconds": tuned_inference,
            "comparison": tuned_comparison,
        }
        (output_dir / "tuning_summary.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        _save_map(predictions_dir, student_trips[0], tuned_observations[0])

    run_summary = {
        "seed": random_state,
        "train_trip_count": len(train_trips),
        "student_test_trip_count": len(student_trips),
        "group_cv_folds": folds,
        "selected_model": best_model_name,
        "selected_model_cv_macro_f1": cv_metrics[best_model_name]["macro_f1"],
        "tuning_trials": tuning_trials if not skip_tuning else 0,
        "tuning_objective": "maximize mean grouped-CV edge macro F1",
        "tuning_group_cv_folds": min(folds, 3) if not skip_tuning else 0,
        "tuning_search_space": (
            _tuning_search_space(best_model_name) if not skip_tuning else {}
        ),
        "best_params": best_params,
        "tuned_cv_macro_f1": tuned_cv_macro_f1,
        "tuning_seconds": tuning_seconds,
        "gnn_epochs": gnn_epochs,
        "candidate_radius_m": 50,
        "candidate_nearest_count": 20,
        "feature_columns": list(FEATURE_COLUMNS),
        "cv_metrics": cv_metrics,
        "tuned_student_test_metrics": tuned_metrics,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    report_path = generate_report(output_dir)
    print(f"Experiment report: {report_path}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Path to the 80-trip training partition")
    parser.add_argument(
        "--student-test", required=True, help="Path to the 10-trip student-testing partition"
    )
    parser.add_argument("--output", default="results/experiments")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--gnn-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()
    run_experiments(
        args.train,
        args.student_test,
        args.output,
        folds=args.folds,
        tuning_trials=args.trials,
        gnn_epochs=args.gnn_epochs,
        random_state=args.seed,
        skip_tuning=args.skip_tuning,
    )


if __name__ == "__main__":
    main()
