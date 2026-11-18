"""Directed-road HMM/Viterbi decoding for per-observation candidate scores."""

from __future__ import annotations

import heapq
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp

from .preprocessing import TripData


def classifier_only(trip: TripData, arc_scores: np.ndarray) -> np.ndarray:
    """Choose the highest-scoring candidate arc independently at each observation."""
    return np.asarray(
        [
            candidates[int(np.argmax(arc_scores[candidates]))]
            for candidates in trip.candidates
        ],
        dtype=np.int64,
    )


def viterbi_decode(trip: TripData, arc_scores: np.ndarray, max_candidates: int = 8) -> np.ndarray:
    """Decode a connected route, relaxing impossible transitions with an explicit penalty."""
    if len(arc_scores) != len(trip.arcs):
        raise ValueError("arc_scores must contain one score per directed road arc")

    states: list[np.ndarray] = []
    emissions: list[np.ndarray] = []
    for candidates, distances in zip(trip.candidates, trip.candidate_distances):
        order = np.argsort(arc_scores[candidates])[::-1][:max_candidates]
        selected = candidates[order]
        selected_distances = distances[order]
        local_scores = arc_scores[selected].astype(float)
        local_scores -= logsumexp(local_scores)
        gps_sigma = 25.0
        local_scores -= (selected_distances ** 2) / (2.0 * gps_sigma**2)
        states.append(selected)
        emissions.append(local_scores)

    lengths = trip.edge_features["arc_length_m"].to_numpy()
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(trip.node_xy))]
    for (start, end), length in zip(trip.arcs, lengths):
        adjacency[int(start)].append((int(end), float(length)))

    @lru_cache(maxsize=4096)
    def paths_between(
        source: int, targets: tuple[int, ...], max_distance: int
    ) -> tuple[tuple[int, float], ...]:
        remaining_targets = set(targets)
        remaining_targets.discard(source)
        distances = {source: 0.0}
        frontier = [(0.0, source)]
        while frontier:
            distance, node = heapq.heappop(frontier)
            if distance != distances[node]:
                continue
            if distance > max_distance:
                break
            remaining_targets.discard(node)
            if not remaining_targets:
                break
            for neighbor, weight in adjacency[node]:
                candidate_distance = distance + weight
                if candidate_distance <= max_distance and candidate_distance < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate_distance
                    heapq.heappush(frontier, (candidate_distance, neighbor))
        return tuple(distances.items())

    backpointers: list[np.ndarray] = [np.zeros(len(states[0]), dtype=np.int64)]
    previous_scores = emissions[0].copy()
    for observation_index in range(1, len(states)):
        previous_states = states[observation_index - 1]
        current_states = states[observation_index]
        previous_nodes = trip.arcs[previous_states, 1]
        current_nodes = trip.arcs[current_states, 0]
        target_nodes = tuple(sorted(set(map(int, current_nodes))))
        displacement = float(
            np.linalg.norm(trip.track_xy[observation_index] - trip.track_xy[observation_index - 1])
        )
        time_gap = float(trip.track.iloc[observation_index]["t"] - trip.track.iloc[observation_index - 1]["t"])
        max_distance = min(max(displacement * 3.0 + 100.0, 100.0), 5000.0)
        transition_scores = np.full(
            (len(previous_states), len(current_states)), -12.0, dtype=np.float64
        )
        for previous_index, source_node in enumerate(previous_nodes):
            reachable = dict(
                paths_between(int(source_node), target_nodes, int(max_distance))
            )
            for current_index, (start_node, arc_id) in enumerate(zip(current_nodes, current_states)):
                if previous_states[previous_index] == arc_id:
                    route_distance = min(displacement, float(lengths[arc_id]))
                else:
                    path_distance = reachable.get(int(start_node))
                    if path_distance is None:
                        continue
                    route_distance = path_distance + float(lengths[arc_id])
                if time_gap <= 0.0 and route_distance > 0.0:
                    continue
                if time_gap > 60.0 and displacement < 10.0 and previous_states[previous_index] == arc_id:
                    transition_scores[previous_index, current_index] = 0.0
                    continue
                scale = max(20.0, displacement * 0.5 + 15.0)
                score = -abs(route_distance - displacement) / scale
                if time_gap > 0.0 and route_distance / time_gap > 50.0:
                    score -= 4.0
                transition_scores[previous_index, current_index] = score

        current_scores = emissions[observation_index] + np.max(
            previous_scores[:, None] + transition_scores, axis=0
        )
        backpointers.append(
            np.argmax(previous_scores[:, None] + transition_scores, axis=0).astype(np.int64)
        )
        previous_scores = current_scores

    decoded_positions = np.empty(len(states), dtype=np.int64)
    decoded_positions[-1] = int(np.argmax(previous_scores))
    for index in range(len(states) - 1, 0, -1):
        decoded_positions[index - 1] = backpointers[index][decoded_positions[index]]
    return np.asarray(
        [states[index][decoded_positions[index]] for index in range(len(states))],
        dtype=np.int64,
    )
