import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.api import PredictionService


class PositiveArcModel:
    def predict_proba(self, features):
        scores = np.where(features[:, 0] > 100.0, 0.9, 0.1)
        return np.column_stack((1.0 - scores, scores))


class ApiTests(unittest.TestCase):
    def test_prediction_scores_without_passing_reference_labels_to_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trip_dir = root / "trips" / "sample"
            trip_dir.mkdir(parents=True)
            (trip_dir / "sample.track").write_text(
                "0.00005 0.0 1\n0.00095 0.0 2\n0.00195 0.0 3\n",
                encoding="utf-8",
            )
            (trip_dir / "sample.nodes").write_text(
                "0.0 0.0\n0.001 0.0\n0.002 0.0\n", encoding="utf-8"
            )
            (trip_dir / "sample.arcs").write_text(
                "0 1\n1 2\n1 0\n", encoding="utf-8"
            )
            (trip_dir / "sample.route").write_text("0\n1\n", encoding="utf-8")
            model_path = root / "model.joblib"
            joblib.dump(PositiveArcModel(), model_path)
            service = PredictionService(root / "trips", model_path, "Random Forest")

            response = service.predict("sample")

            self.assertEqual(response["trip_id"], "sample")
            self.assertIn("viterbi_observation_top1_accuracy", response["metrics"])
            self.assertEqual(len(response["predicted_arc_ids"]), 3)
            self.assertNotIn("reference_arc_ids", response)
            self.assertNotIn("reference_route", response)


if __name__ == "__main__":
    unittest.main()
