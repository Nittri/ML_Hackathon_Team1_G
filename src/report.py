"""Create a concise PDF report from measured PathSnap experiment outputs."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _format(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def generate_report(output_dir: str | Path) -> Path:
    """Build a PDF using only the actual experiment CSV/JSON outputs."""
    output_dir = Path(output_dir)
    comparison = pd.read_csv(output_dir / "model_comparison.csv")
    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    report_path = output_dir / "PathSnap_model_report.pdf"

    with PdfPages(report_path) as report:
        figure = plt.figure(figsize=(11.7, 8.3))
        figure.text(0.08, 0.9, "PathSnap Map-Matching Model Report", fontsize=24, weight="bold")
        overview = [
            f"Training trips: {summary['train_trip_count']} | Student-test trips: {summary['student_test_trip_count']}",
            f"Grouped CV folds: {summary['group_cv_folds']} | Random seed: {summary['seed']}",
            f"Selected model: {summary['selected_model']} (training-CV macro F1: "
            f"{summary['selected_model_cv_macro_f1']:.4f})",
            f"Candidate generation: {summary['candidate_radius_m']} m radius plus "
            f"{summary['candidate_nearest_count']} nearest directed arcs",
            f"Bayesian tuning trials: {summary['tuning_trials']} | "
            f"Best tuned CV macro F1: {_format(summary.get('tuned_cv_macro_f1'), 4)}",
            "",
            "Method",
            "Raw GPS points are validated and stably ordered by timestamp. Road geometry is "
            "projected to local UTM coordinates; directed edge and node features combine "
            "geometry, topology, and GPS proximity/progress/speed/heading. The GNN feature "
            "scalers are fit within each training fold only.",
            "",
            "Evaluation",
            "Model selection and Bayesian optimization use trip-grouped folds from the "
            "training partition. Student-test results are reported for development comparison "
            "only. The evaluator partition is not used by this experiment. Classifier-only "
            "predictions are compared with directed HMM/Viterbi decoding.",
            "",
            "The interactive Folium route visualization is supplied separately as an HTML "
            "file in the predictions directory. All values in this report are read from the "
            "experiment outputs; no metrics are estimated or fabricated.",
        ]
        y = 0.82
        for line in overview:
            if not line:
                y -= 0.025
                continue
            wrapped = textwrap.fill(line, width=125)
            line_count = wrapped.count("\n") + 1
            figure.text(0.08, y, wrapped, fontsize=10, va="top", linespacing=1.3)
            y -= line_count * 0.035 + 0.02
        report.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        metric_columns = [
            ("model", "Model"),
            ("cv_macro_f1", "CV macro F1"),
            ("cv_observation_top1_accuracy", "CV Top-1"),
            ("student_arc_macro_f1", "Student arc F1"),
            ("student_observation_top1_accuracy", "Student Top-1"),
            ("student_hmm_observation_accuracy", "HMM Top-1"),
            ("cv_hmm_route_continuity", "CV HMM continuity"),
            ("student_hmm_route_continuity", "HMM continuity"),
            ("student_candidate_miss_rate", "Candidate miss"),
            ("student_arc_precision", "Precision"),
            ("student_arc_recall", "Recall"),
        ]
        table_data = [
            [label for _, label in metric_columns],
            *[
                [_format(row[column], 4) if column != "model" else str(row[column])
                 for column, _ in metric_columns]
                for _, row in comparison.iterrows()
            ],
        ]
        figure, axis = plt.subplots(figsize=(12, 4))
        axis.axis("off")
        axis.set_title("Model comparison: grouped training CV and student-test development metrics", pad=18)
        table = axis.table(cellText=table_data[1:], colLabels=table_data[0], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        report.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        timing_columns = [
            ("model", "Model"),
            ("cv_fit_seconds", "CV fit (s)"),
            ("training_seconds", "Final fit (s)"),
            ("inference_seconds", "Inference (s)"),
            ("student_route_decoding_seconds", "Route decode (s)"),
            ("cv_log_loss", "CV log loss"),
            ("student_log_loss", "Student log loss"),
        ]
        timing_data = [
            [label for _, label in timing_columns],
            *[
                [_format(row[column], 3) if column != "model" else str(row[column])
                 for column, _ in timing_columns]
                for _, row in comparison.iterrows()
            ],
        ]
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.axis("off")
        axis.set_title("Training, inference, and sequence-decoding measurements", pad=18)
        table = axis.table(cellText=timing_data[1:], colLabels=timing_data[0], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        report.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        if (output_dir / "tuning_comparison.csv").is_file():
            tuning = pd.read_csv(output_dir / "tuning_comparison.csv")
            tuning_summary_path = output_dir / "tuning_summary.json"
            tuning_summary = (
                json.loads(tuning_summary_path.read_text(encoding="utf-8"))
                if tuning_summary_path.is_file()
                else {}
            )
            figure, axis = plt.subplots(figsize=(10, 4))
            axis.axis("off")
            details = (
                f"Objective: {tuning_summary.get('objective', 'maximize grouped-CV macro F1')}\n"
                f"Group CV folds: {tuning_summary.get('group_cv_folds', 'see run configuration')}\n"
                f"Search space: {json.dumps(tuning_summary.get('search_space', {}), sort_keys=True)}"
            )
            axis.set_title(
                "Bayesian tuning: same training-CV and student-test comparison\n" + details,
                pad=25,
                fontsize=9,
            )
            table = axis.table(
                cellText=[[ _format(value, 4) for value in tuning.iloc[0].tolist()]],
                colLabels=list(tuning.columns),
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.8)
            report.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    return report_path
