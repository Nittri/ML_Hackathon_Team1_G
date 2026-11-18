"""Cleaning, feature engineering, and candidate generation for PathSnap trips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import STRtree, linestrings, points

from .data_io import find_trips, load_trip, make_projector

FEATURE_COLUMNS = (
    "arc_length_m",
    "arc_heading_sin",
    "arc_heading_cos",
    "midpoint_x_km",
    "midpoint_y_km",
    "start_in_degree",
    "start_out_degree",
    "end_in_degree",
    "end_out_degree",
    "has_reverse_arc",
    "gps_midpoint_distance_m",
    "gps_start_distance_m",
    "gps_end_distance_m",
    "nearest_track_progress",
    "nearest_track_speed_mps",
    "gps_heading_alignment",
)
NODE_FEATURE_COLUMNS = (
    "relative_x_km",
    "relative_y_km",
    "incoming_degree_scaled",
    "outgoing_degree_scaled",
    "nearest_track_distance_log_m",
    "nearest_track_progress",
)


@dataclass(frozen=True)
class TripData:
    """Prepared directed road graph and GPS observations for one trip."""

    trip_id: str
    track: pd.DataFrame
    track_xy: np.ndarray
    node_xy: np.ndarray
    arcs: np.ndarray
    route_arc_ids: np.ndarray | None
    observation_arc_ids: np.ndarray | None
    edge_features: pd.DataFrame
    edge_labels: np.ndarray | None
    node_features: np.ndarray
    edge_index: np.ndarray
    candidates: tuple[np.ndarray, ...]
    candidate_distances: tuple[np.ndarray, ...]
    candidate_miss_rate: float | None
    epsg: int


def clean_track(track: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid GPS rows and stably sort observations by timestamp."""
    required = {"lon", "lat", "t"}
    missing = required.difference(track.columns)
    if missing:
        raise ValueError(f"Track is missing required columns: {sorted(missing)}")

    cleaned = track.loc[:, ["lon", "lat", "t"]].copy()
    for column in ("lon", "lat", "t"):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    valid = (
        np.isfinite(cleaned[["lon", "lat", "t"]]).all(axis=1)
        & cleaned["lon"].between(-180, 180)
        & cleaned["lat"].between(-90, 90)
    )
    cleaned = cleaned.loc[valid].sort_values("t", kind="stable").reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("Track contains no valid GPS observations")
    return cleaned


def _segment_distances(point_xy: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact point-to-segment distances and fractions along each segment."""
    vectors = ends - starts
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    offsets = point_xy - starts
    fractions = np.divide(
        np.einsum("ij,ij->i", offsets, vectors),
        lengths_squared,
        out=np.zeros(len(starts), dtype=float),
        where=lengths_squared > 0,
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    nearest = starts + fractions[:, None] * vectors
    distances = np.linalg.norm(nearest - point_xy, axis=1)
    return distances, fractions


def _directed_degrees(arcs: np.ndarray, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    outgoing = np.bincount(arcs[:, 0], minlength=node_count)
    incoming = np.bincount(arcs[:, 1], minlength=node_count)
    return incoming, outgoing


def _reference_arcs_for_observations(
    track_xy: np.ndarray,
    node_xy: np.ndarray,
    arcs: np.ndarray,
    route_arc_ids: np.ndarray,
) -> np.ndarray:
    """Align each GPS point to its nearest route occurrence, preserving route order."""
    route_starts = node_xy[arcs[route_arc_ids, 0]]
    route_ends = node_xy[arcs[route_arc_ids, 1]]
    route_lines = linestrings(np.stack((route_starts, route_ends), axis=1))
    tree = STRtree(route_lines)
    match_pairs = tree.query_nearest(points(track_xy), all_matches=True)

    matches_by_observation: list[list[int]] = [[] for _ in range(len(track_xy))]
    for observation_index, route_position in zip(match_pairs[0], match_pairs[1]):
        matches_by_observation[int(observation_index)].append(int(route_position))

    aligned_positions = np.empty(len(track_xy), dtype=np.int64)
    previous_position = 0
    for index, positions in enumerate(matches_by_observation):
        if not positions:
            aligned_positions[index] = previous_position
            continue
        positions.sort()
        forward = [position for position in positions if position >= previous_position]
        if forward:
            previous_position = forward[0]
        aligned_positions[index] = previous_position
    return route_arc_ids[aligned_positions]


def _candidate_arcs(
    track_xy: np.ndarray,
    node_xy: np.ndarray,
    arcs: np.ndarray,
    reference_arc_ids: np.ndarray | None,
    radius_m: float,
    nearest_count: int,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float | None]:
    starts = node_xy[arcs[:, 0]]
    ends = node_xy[arcs[:, 1]]
    lines = linestrings(np.stack((starts, ends), axis=1))
    line_tree = STRtree(lines)
    midpoints = (starts + ends) / 2.0
    midpoint_tree = cKDTree(midpoints)
    candidate_sets: list[np.ndarray] = []
    candidate_distances: list[np.ndarray] = []
    misses = 0

    for observation_index, point_xy in enumerate(track_xy):
        reference_arc_id = (
            int(reference_arc_ids[observation_index]) if reference_arc_ids is not None else -1
        )
        point_geometry = points(point_xy.reshape(1, 2))[0]
        within_radius = line_tree.query(point_geometry, predicate="dwithin", distance=radius_m)
        search_count = min(max(nearest_count * 4, nearest_count), len(arcs))
        _, midpoint_candidates = midpoint_tree.query(point_xy, k=search_count)
        midpoint_candidates = np.atleast_1d(midpoint_candidates).astype(np.int64)
        midpoint_distances, _ = _segment_distances(
            point_xy, starts[midpoint_candidates], ends[midpoint_candidates]
        )
        nearest_segments = midpoint_candidates[
            np.argsort(midpoint_distances)[:nearest_count]
        ]
        candidate_ids = np.unique(np.concatenate((within_radius, nearest_segments))).astype(np.int64)
        distances, _ = _segment_distances(point_xy, starts[candidate_ids], ends[candidate_ids])
        order = np.argsort(candidate_ids)
        candidate_ids = candidate_ids[order]
        distances = distances[order]
        candidate_sets.append(candidate_ids)
        candidate_distances.append(distances.astype(np.float32))
        if reference_arc_ids is not None and not np.any(candidate_ids == reference_arc_id):
            misses += 1

    miss_rate = misses / len(track_xy) if reference_arc_ids is not None else None
    return tuple(candidate_sets), tuple(candidate_distances), miss_rate


def build_trip_data(
    raw_trip: dict,
    radius_m: float = 50.0,
    nearest_count: int = 20,
    include_reference: bool = True,
) -> TripData:
    """Clean one raw trip and build edge/node features and GPS candidates."""
    trip_id = str(raw_trip["id"])
    required = ("track", "nodes", "arcs")
    if include_reference:
        required = (*required, "route")
    absent = [name for name in required if raw_trip.get(name) is None]
    if absent:
        raise ValueError(f"Trip {trip_id} is missing required files: {absent}")

    track = clean_track(raw_trip["track"])
    nodes_frame = raw_trip["nodes"].apply(pd.to_numeric, errors="coerce")
    node_values = nodes_frame.to_numpy(dtype=float)
    if (
        not np.isfinite(node_values).all()
        or not nodes_frame.iloc[:, 0].between(-180, 180).all()
        or not nodes_frame.iloc[:, 1].between(-90, 90).all()
    ):
        raise ValueError(f"Trip {trip_id} contains invalid road-node coordinates")
    arcs_frame = raw_trip["arcs"].apply(pd.to_numeric, errors="coerce")
    arc_values = arcs_frame.to_numpy(dtype=float)
    if not np.isfinite(arc_values).all() or not np.equal(arc_values, np.floor(arc_values)).all():
        raise ValueError(f"Trip {trip_id} contains invalid road-arc endpoints")
    arcs = arc_values.astype(np.int64)
    node_count = len(nodes_frame)
    if len(arcs) == 0 or np.any(arcs < 0) or np.any(arcs >= node_count):
        raise ValueError(f"Trip {trip_id} contains out-of-range road-arc endpoints")

    route_arc_ids = None
    if include_reference:
        route_values = pd.to_numeric(raw_trip["route"]["arc_id"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(route_values).all():
            raise ValueError(f"Trip {trip_id} contains invalid route arc IDs")
        if not np.equal(route_values, np.floor(route_values)).all():
            raise ValueError(f"Trip {trip_id} contains non-integer route arc IDs")
        route_arc_ids = route_values.astype(np.int64)
        if len(route_arc_ids) == 0 or np.any(route_arc_ids < 0) or np.any(route_arc_ids >= len(arcs)):
            raise ValueError(f"Trip {trip_id} contains out-of-range route arc IDs")

    project, epsg = make_projector(float(track["lon"].mean()), float(track["lat"].mean()))
    track_xy = project(track["lon"], track["lat"])
    node_xy = project(nodes_frame.iloc[:, 0], nodes_frame.iloc[:, 1])
    observation_arc_ids = (
        _reference_arcs_for_observations(track_xy, node_xy, arcs, route_arc_ids)
        if route_arc_ids is not None
        else None
    )

    starts, ends = node_xy[arcs[:, 0]], node_xy[arcs[:, 1]]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    midpoint = (starts + ends) / 2.0
    incoming, outgoing = _directed_degrees(arcs, node_count)
    arc_pairs = set(map(tuple, arcs.tolist()))
    has_reverse = np.fromiter(
        ((int(end), int(start)) in arc_pairs for start, end in arcs),
        dtype=np.float32,
        count=len(arcs),
    )

    track_tree = cKDTree(track_xy)
    midpoint_distances, nearest_track = track_tree.query(midpoint, k=1)
    start_distances, _ = track_tree.query(starts, k=1)
    end_distances, _ = track_tree.query(ends, k=1)
    track_steps = np.diff(track_xy, axis=0, prepend=track_xy[:1])
    track_step_lengths = np.linalg.norm(track_steps, axis=1)
    time_gaps = track["t"].diff().to_numpy(copy=True)
    time_gaps[0] = 0.0
    speeds = np.divide(
        track_step_lengths,
        time_gaps,
        out=np.zeros_like(track_step_lengths),
        where=time_gaps > 0,
    )
    track_headings = np.arctan2(track_steps[:, 1], track_steps[:, 0])
    arc_headings = np.arctan2(vectors[:, 1], vectors[:, 0])
    heading_delta = arc_headings - track_headings[nearest_track]

    track_center = np.median(track_xy, axis=0)
    edge_features = pd.DataFrame(
        {
            "arc_length_m": lengths,
            "arc_heading_sin": np.sin(arc_headings),
            "arc_heading_cos": np.cos(arc_headings),
            "midpoint_x_km": (midpoint[:, 0] - track_center[0]) / 1000.0,
            "midpoint_y_km": (midpoint[:, 1] - track_center[1]) / 1000.0,
            "start_in_degree": incoming[arcs[:, 0]],
            "start_out_degree": outgoing[arcs[:, 0]],
            "end_in_degree": incoming[arcs[:, 1]],
            "end_out_degree": outgoing[arcs[:, 1]],
            "has_reverse_arc": has_reverse,
            "gps_midpoint_distance_m": midpoint_distances,
            "gps_start_distance_m": start_distances,
            "gps_end_distance_m": end_distances,
            "nearest_track_progress": nearest_track / max(len(track) - 1, 1),
            "nearest_track_speed_mps": speeds[nearest_track],
            "gps_heading_alignment": np.cos(heading_delta),
        },
        columns=FEATURE_COLUMNS,
    ).replace([np.inf, -np.inf], np.nan)
    edge_features = edge_features.fillna(0.0).astype(np.float32)

    node_midpoint_distance, nearest_node_track = track_tree.query(node_xy, k=1)
    node_features = np.column_stack(
        (
            (node_xy[:, 0] - track_center[0]) / 1000.0,
            (node_xy[:, 1] - track_center[1]) / 1000.0,
            incoming / max(float(incoming.max()), 1.0),
            outgoing / max(float(outgoing.max()), 1.0),
            np.log1p(node_midpoint_distance),
            nearest_node_track / max(len(track) - 1, 1),
        )
    ).astype(np.float32)

    candidates, candidate_distances, miss_rate = _candidate_arcs(
        track_xy, node_xy, arcs, observation_arc_ids, radius_m, nearest_count
    )
    labels = None
    if route_arc_ids is not None:
        labels = np.zeros(len(arcs), dtype=np.uint8)
        labels[np.unique(route_arc_ids)] = 1

    return TripData(
        trip_id=trip_id,
        track=track,
        track_xy=track_xy,
        node_xy=node_xy,
        arcs=arcs,
        route_arc_ids=route_arc_ids,
        observation_arc_ids=observation_arc_ids,
        edge_features=edge_features,
        edge_labels=labels,
        node_features=node_features,
        edge_index=arcs.T.copy(),
        candidates=candidates,
        candidate_distances=candidate_distances,
        candidate_miss_rate=miss_rate,
        epsg=epsg,
    )


def load_partition(folder: str, radius_m: float = 50.0, nearest_count: int = 20) -> list[TripData]:
    """Read and preprocess every complete trip in a partition folder."""
    trip_ids = find_trips(folder)
    if not trip_ids:
        raise ValueError(f"No PathSnap trips found in {folder!r}")
    return [
        build_trip_data(load_trip(folder, trip_id), radius_m=radius_m, nearest_count=nearest_count)
        for trip_id in trip_ids
    ]


def balanced_edge_indices(
    trips: Iterable[TripData],
    max_negatives_per_positive: int = 10,
    random_state: int = 42,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Return per-trip edge indices sampled without discarding positive route arcs."""
    rng = np.random.default_rng(random_state)
    selected: list[tuple[int, int]] = []
    labels: list[int] = []
    for trip_index, trip in enumerate(trips):
        if trip.edge_labels is None:
            raise ValueError(f"Trip {trip.trip_id} has no reference labels for training")
        positive = np.flatnonzero(trip.edge_labels == 1)
        negative = np.flatnonzero(trip.edge_labels == 0)
        keep_count = min(len(negative), max_negatives_per_positive * max(len(positive), 1))
        candidate_negative = np.unique(np.concatenate(trip.candidates))
        candidate_negative = candidate_negative[trip.edge_labels[candidate_negative] == 0]
        hard_count = min(keep_count, len(candidate_negative))
        hard_negative = (
            rng.choice(candidate_negative, size=hard_count, replace=False)
            if hard_count
            else np.empty(0, dtype=np.int64)
        )
        remaining_negative = np.setdiff1d(negative, hard_negative, assume_unique=False)
        fallback_count = min(keep_count - hard_count, len(remaining_negative))
        fallback_negative = (
            rng.choice(remaining_negative, size=fallback_count, replace=False)
            if fallback_count
            else np.empty(0, dtype=np.int64)
        )
        sampled_negative = np.concatenate((hard_negative, fallback_negative))
        edge_indices = np.concatenate((positive, sampled_negative))
        selected.extend((trip_index, int(edge_index)) for edge_index in edge_indices)
        labels.extend(trip.edge_labels[edge_indices].tolist())
    return selected, np.asarray(labels, dtype=np.uint8)
