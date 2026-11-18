"""Model implementations for directed road-segment classification."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn
from torch.nn import functional as F

from .preprocessing import FEATURE_COLUMNS, NODE_FEATURE_COLUMNS, TripData


def _require_package(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError as error:
        raise RuntimeError(
            f"{name} is required for this model. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from error


def _node_wl_labels(trip: TripData, iterations: int = 2) -> tuple[list[list[str]], np.ndarray]:
    """Create directed Weisfeiler-Lehman labels and length bins for a road graph."""
    incoming, outgoing = defaultdict(list), defaultdict(list)
    lengths = trip.edge_features["arc_length_m"].to_numpy()
    median_length = max(float(np.median(lengths)), 1.0)
    bins = np.clip(np.floor(np.log2(np.maximum(lengths, 1.0) / median_length) + 3), 0, 7).astype(int)

    incident_lengths: list[list[float]] = [[] for _ in range(len(trip.node_xy))]
    for (start, end), length in zip(trip.arcs, lengths):
        outgoing[int(start)].append(int(end))
        incoming[int(end)].append(int(start))
        incident_lengths[int(start)].append(float(length))
        incident_lengths[int(end)].append(float(length))

    labels = []
    for node, incident in enumerate(incident_lengths):
        in_degree = len(incoming[node])
        out_degree = len(outgoing[node])
        length_bin = int(np.clip(np.floor(np.log2(max(float(np.median(incident)) if len(incident) else 1.0, 1.0) / median_length) + 3), 0, 7))
        labels.append(f"{in_degree}:{out_degree}:{length_bin}")

    node_labels = [labels]
    for _ in range(iterations):
        previous = node_labels[-1]
        refined = []
        for node in range(len(trip.node_xy)):
            in_labels = ",".join(sorted(previous[neighbor] for neighbor in incoming[node]))
            out_labels = ",".join(sorted(previous[neighbor] for neighbor in outgoing[node]))
            signature = f"{previous[node]}|in:{in_labels}|out:{out_labels}"
            refined.append(hashlib.blake2b(signature.encode("utf-8"), digest_size=8).hexdigest())
        node_labels.append(refined)

    return node_labels, bins


def _wl_edge_feature(
    trip: TripData,
    arc_id: int,
    node_labels: list[list[str]],
    length_bins: np.ndarray,
) -> dict[str, float]:
    start, end = trip.arcs[arc_id]
    feature: dict[str, float] = {}
    for depth, labels_at_depth in enumerate(node_labels):
        feature[f"wl{depth}:src:{labels_at_depth[int(start)]}"] = 1.0
        feature[f"wl{depth}:dst:{labels_at_depth[int(end)]}"] = 1.0
        feature[f"wl{depth}:edge:{labels_at_depth[int(start)]}>{labels_at_depth[int(end)]}"] = 1.0
    feature[f"edge_length_bin:{length_bins[arc_id]}"] = 1.0
    return feature


class GraphKernelSVM:
    """Linear SVM over an explicit Weisfeiler-Lehman graph-kernel feature map."""

    def __init__(self, random_state: int = 42, C: float = 1.0):
        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier = LinearSVC(
            C=C,
            class_weight="balanced",
            random_state=random_state,
            max_iter=5000,
        )
        self._labels: dict[int, tuple[list[list[str]], np.ndarray]] = {}

    @staticmethod
    def _supported_sparse_indices(matrix: Any) -> Any:
        limit = np.iinfo(np.int32).max
        if matrix.shape[0] > limit or matrix.shape[1] > limit or matrix.nnz > limit:
            raise ValueError("The graph-kernel feature matrix exceeds scikit-learn's sparse index limit")
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
        return matrix

    def fit(
        self,
        trips: Sequence[TripData],
        selected: Sequence[tuple[int, int]],
        labels: np.ndarray,
    ) -> "GraphKernelSVM":
        rows = []
        for trip_index, edge_index in selected:
            trip = trips[trip_index]
            key = id(trip)
            if key not in self._labels:
                self._labels[key] = _node_wl_labels(trip)
            node_labels, length_bins = self._labels[key]
            rows.append(_wl_edge_feature(trip, edge_index, node_labels, length_bins))
        matrix = self._supported_sparse_indices(self.vectorizer.fit_transform(rows))
        self.classifier.fit(matrix, labels)
        return self

    def predict_trip(self, trip: TripData) -> np.ndarray:
        key = id(trip)
        if key not in self._labels:
            self._labels[key] = _node_wl_labels(trip)
        node_labels, length_bins = self._labels[key]
        scores = np.empty(len(trip.arcs), dtype=np.float32)
        for start in range(0, len(trip.arcs), 8192):
            stop = min(start + 8192, len(trip.arcs))
            rows = [
                _wl_edge_feature(trip, arc_id, node_labels, length_bins)
                for arc_id in range(start, stop)
            ]
            matrix = self._supported_sparse_indices(self.vectorizer.transform(rows))
            scores[start:stop] = self.classifier.decision_function(matrix)
        return scores


def build_tabular_models(random_state: int = 42, xgb_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the requested tree-based models with explicit binary objectives."""
    xgboost = _require_package("xgboost")
    lightgbm = _require_package("lightgbm")
    xgb_config = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "n_jobs": -1,
        "random_state": random_state,
        "verbosity": 0,
    }
    if xgb_params:
        xgb_config.update(xgb_params)
    return {
        "XGBoost (binary logistic)": xgboost.XGBClassifier(**xgb_config),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "LightGBM": lightgbm.LGBMClassifier(
            objective="binary",
            n_estimators=250,
            num_leaves=31,
            learning_rate=0.05,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
            verbosity=-1,
        ),
    }


class _DirectedGraphNetwork(nn.Module):
    """Small directed message-passing network with an edge classifier head."""

    def __init__(self, node_features: int, edge_features: int, hidden: int = 32):
        super().__init__()
        self.node_input = nn.Linear(node_features, hidden)
        self.message_1 = nn.Linear(hidden, hidden, bias=False)
        self.update_1 = nn.Linear(hidden * 2, hidden)
        self.message_2 = nn.Linear(hidden, hidden, bias=False)
        self.update_2 = nn.Linear(hidden * 2, hidden)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 2 + edge_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _aggregate(node_state: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        aggregated = torch.zeros_like(node_state)
        aggregated.index_add_(0, target, node_state[source])
        degree = torch.zeros(node_state.shape[0], device=node_state.device, dtype=node_state.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=node_state.dtype))
        return aggregated / degree.clamp_min(1.0).unsqueeze(1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        hidden = F.relu(self.node_input(node_features))
        neighbor_state = self._aggregate(hidden, edge_index)
        hidden = F.relu(self.update_1(torch.cat((hidden, self.message_1(neighbor_state)), dim=1)))
        neighbor_state = self._aggregate(hidden, edge_index)
        hidden = F.relu(self.update_2(torch.cat((hidden, self.message_2(neighbor_state)), dim=1)))
        return self.edge_head(torch.cat((hidden[source], hidden[target], edge_features), dim=1)).squeeze(-1)


class GraphNeuralNetwork:
    """Train a directed graph neural network to classify route-membership arcs."""

    def __init__(
        self,
        random_state: int = 42,
        epochs: int = 8,
        hidden_channels: int = 32,
        learning_rate: float = 0.003,
    ):
        self.random_state = random_state
        self.epochs = epochs
        self.hidden_channels = hidden_channels
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(random_state)
        self.model = _DirectedGraphNetwork(
            len(NODE_FEATURE_COLUMNS), len(FEATURE_COLUMNS), hidden_channels
        ).to(self.device)
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.training_history: list[float] = []

    def to_cpu(self) -> None:
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def fit(
        self,
        trips: Sequence[TripData],
        selected: Sequence[tuple[int, int]],
        labels: np.ndarray,
    ) -> "GraphNeuralNetwork":
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        for trip in trips:
            self.node_scaler.partial_fit(trip.node_features)
            self.edge_scaler.partial_fit(trip.edge_features.to_numpy(dtype=np.float32))
        grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for (trip_index, edge_index), label in zip(selected, labels):
            grouped[trip_index].append((edge_index, int(label)))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        for _ in range(self.epochs):
            epoch_losses = []
            self.model.train()
            for trip_index, selected_edges in grouped.items():
                trip = trips[trip_index]
                node_features, edge_index, edge_features = self._tensor_graph(trip)
                edge_ids = torch.as_tensor(
                    [item[0] for item in selected_edges], dtype=torch.long, device=self.device
                )
                target = torch.as_tensor(
                    [item[1] for item in selected_edges], dtype=torch.float32, device=self.device
                )
                positive_count = max(float(target.sum().item()), 1.0)
                negative_count = max(float(len(target) - target.sum().item()), 1.0)
                positive_weight = torch.tensor(
                    min(negative_count / positive_count, 20.0), device=self.device
                )
                logits = self.model(node_features, edge_index, edge_features)[edge_ids]
                loss = F.binary_cross_entropy_with_logits(
                    logits, target, pos_weight=positive_weight
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            self.training_history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        return self

    def _tensor_graph(self, trip: TripData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = self.node_scaler.transform(trip.node_features).astype(
            np.float32, copy=False
        )
        edge_features = self.edge_scaler.transform(
            trip.edge_features.to_numpy(dtype=np.float32)
        ).astype(np.float32, copy=False)
        node_features = torch.as_tensor(node_features, dtype=torch.float32, device=self.device)
        edge_index = torch.as_tensor(trip.edge_index, dtype=torch.long, device=self.device)
        edge_features = torch.as_tensor(edge_features, dtype=torch.float32, device=self.device)
        return node_features, edge_index, edge_features

    def predict_trip(self, trip: TripData) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            node_features, edge_index, edge_features = self._tensor_graph(trip)
            logits = self.model(node_features, edge_index, edge_features)
            return torch.sigmoid(logits).cpu().numpy()
