import unittest

import numpy as np
import pandas as pd

from src.decoder import classifier_only, viterbi_decode
from src.models import GraphKernelSVM, GraphNeuralNetwork
from src.preprocessing import balanced_edge_indices, build_trip_data, clean_track


def small_trip():
    return {
        "id": "sample",
        "track": pd.DataFrame(
            {
                "lon": [0.00005, 0.00095, 0.00195],
                "lat": [0.0, 0.0, 0.0],
                "t": [1.0, 2.0, 3.0],
            }
        ),
        "nodes": pd.DataFrame(
            {"lon": [0.0, 0.001, 0.002], "lat": [0.0, 0.0, 0.0]}
        ),
        "arcs": pd.DataFrame(
            {"from_node": [0, 1, 1], "to_node": [1, 2, 0]}
        ),
        "route": pd.DataFrame({"arc_id": [0, 1]}),
    }


class PreprocessingTests(unittest.TestCase):
    def test_clean_track_removes_invalid_rows_and_sorts_stably(self):
        track = pd.DataFrame(
            {
                "lon": [0.0, 1.0, 2.0, 181.0],
                "lat": [0.0, 1.0, np.nan, 0.0],
                "t": [2.0, 1.0, 3.0, 4.0],
            }
        )
        result = clean_track(track)
        self.assertEqual(result["t"].tolist(), [1.0, 2.0])
        self.assertEqual(result["lon"].tolist(), [1.0, 0.0])

    def test_trip_features_keep_directed_route_labels(self):
        trip = build_trip_data(small_trip(), radius_m=50.0, nearest_count=2)
        self.assertEqual(trip.edge_labels.tolist(), [1, 1, 0])
        self.assertEqual(trip.observation_arc_ids.tolist(), [0, 0, 1])
        self.assertEqual(len(trip.edge_features.columns), 16)
        self.assertEqual(trip.candidate_miss_rate, 0.0)
        self.assertTrue(all(candidates.size for candidates in trip.candidates))

    def test_inference_features_do_not_require_reference_labels(self):
        raw_trip = small_trip()
        raw_trip["route"] = None
        trip = build_trip_data(raw_trip, include_reference=False)
        self.assertIsNone(trip.edge_labels)
        self.assertIsNone(trip.observation_arc_ids)
        self.assertIsNone(trip.candidate_miss_rate)
        self.assertEqual(len(trip.edge_features), 3)

    def test_viterbi_uses_directed_connectivity(self):
        trip = build_trip_data(small_trip(), nearest_count=2)
        scores = np.asarray([0.9, 0.9, 0.1])
        self.assertEqual(classifier_only(trip, scores).tolist()[0], 0)
        decoded = viterbi_decode(trip, scores, max_candidates=3)
        self.assertTrue(
            np.all(
                (decoded[:-1] == decoded[1:])
                | (trip.arcs[decoded[:-1], 1] == trip.arcs[decoded[1:], 0])
            )
        )

    def test_gnn_fits_and_predicts_with_node_features(self):
        trip = build_trip_data(small_trip(), nearest_count=2)
        selected, labels = balanced_edge_indices([trip], random_state=1)
        model = GraphNeuralNetwork(random_state=1, epochs=1).fit(
            [trip], selected, labels
        )
        scores = model.predict_trip(trip)
        self.assertEqual(scores.shape, (len(trip.arcs),))
        self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))

    def test_graph_kernel_svm_fits_sparse_features(self):
        trip = build_trip_data(small_trip(), nearest_count=2)
        selected, labels = balanced_edge_indices([trip], random_state=1)
        model = GraphKernelSVM(random_state=1).fit([trip], selected, labels)
        scores = model.predict_trip(trip)
        self.assertEqual(scores.shape, (len(trip.arcs),))


if __name__ == "__main__":
    unittest.main()
