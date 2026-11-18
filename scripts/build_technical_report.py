"""Build the PathSnap technical report from measured development outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as PdfImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table as PdfTable,
    TableStyle,
)
from sklearn.metrics import f1_score

from src.data_io import load_trip, make_projector
from src.preprocessing import clean_track


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results"
EXPERIMENTS = RESULTS / "experiments"
PREDICTIONS = EXPERIMENTS / "predictions"
FIGURES = REPORTS / "figures"
DATA_ROOT = ROOT / "data" / "Student folder"

MODEL_FILES = {
    "XGBoost (binary logistic)": "predictions_xgboost_binary_logistic_student_test.csv",
    "Random Forest": "predictions_random_forest_student_test.csv",
    "Graph-kernel SVM": "predictions_graph-kernel_svm_student_test.csv",
    "GNN": "predictions_gnn_student_test.csv",
    "LightGBM": "predictions_lightgbm_student_test.csv",
}
MODEL_FAMILY = {
    "XGBoost (binary logistic)": "Gradient-boosted decision trees",
    "Random Forest": "Bagged decision trees",
    "Graph-kernel SVM": "Graph-kernel linear SVM",
    "GNN": "Directed message-passing neural network",
    "LightGBM": "Gradient-boosted decision trees",
}
MODEL_DISCUSSION = {
    "XGBoost (binary logistic)": {
        "why": "Boosted trees can learn nonlinear interactions among road geometry, network topology, and GPS proximity without requiring feature scaling. The logistic objective supplies route-membership probabilities for candidate ranking and the sequence decoder.",
        "strengths": "CV edge macro F1 was 0.8897 and student-test edge macro F1 was 0.8926, a close result across these partitions. Student-test precision was 0.6687 and recall 0.9643, reflecting a recall-heavy route-edge classifier.",
        "limits": "Independent arc scores do not enforce a continuous route. Student-test observation Top-1 was 0.2529 before decoding; candidate misses and ambiguous parallel roads remain difficult.",
        "objective": "Binary logistic loss (`binary:logistic`; `eval_metric=logloss`) on sampled positive route arcs and sampled negatives. `scale_pos_weight` is calculated from sampled training labels.",
    },
    "Random Forest": {
        "why": "A forest is a robust nonlinear baseline for mixed geometric and topological tabular features and can model thresholds/interactions without feature scaling.",
        "strengths": "It led grouped-CV edge macro F1 at 0.9137 and student-test edge macro F1 at 0.9260. It also had the highest student-test edge precision (0.8434) among the five.",
        "limits": "It did not lead observation Top-1: student-test classifier-only Top-1 was 0.2055, below GNN's 0.2780. Its per-arc probabilities do not by themselves guarantee route continuity.",
        "objective": "Random Forest Gini-impurity splits with `balanced_subsample` class weights; each tree uses bootstrap sampling (scikit-learn defaults).",
    },
    "Graph-kernel SVM": {
        "why": "A Weisfeiler-Lehman (WL) feature map represents directed local graph neighborhoods, allowing a linear SVM to use road-network structure rather than only local geometry.",
        "strengths": "The explicit WL map is an interpretable way to incorporate graph structure; the SVM margins can rank candidate edges and feed the decoder.",
        "limits": "It was weakest on these experiments: student-test edge macro F1 0.5062 and observation Top-1 0.0549. The sparse structural features may not capture enough GPS-specific signal. Its margin output is not calibrated probability, so log loss is N/A.",
        "objective": "Class-balanced `LinearSVC` with the default squared-hinge objective; WL neighborhood features use two refinement iterations. The model was fit on sampled route/non-route edges.",
    },
    "GNN": {
        "why": "Directed message passing can combine node context with each candidate edge's geometry and GPS features, naturally representing directed road connectivity.",
        "strengths": "It achieved the highest student-test classifier-only observation Top-1 (0.2780) and student-test edge recall (0.9898). It directly uses the trip graph during prediction.",
        "limits": "Its CV edge macro F1 (0.6502) was much lower than its student-test edge macro F1 (0.7532), and student-test edge precision was only 0.3552. This suggests unstable class trade-offs; one training epoch is too short to judge convergence or generalization confidently.",
        "objective": "Weighted binary cross-entropy with logits, AdamW (learning rate 0.003, weight decay 1e-4); positive weight is calculated per trip and capped at 20. The network has two directed message-passing layers and an edge MLP with dropout 0.15.",
    },
    "LightGBM": {
        "why": "Gradient boosting is effective on tabular engineered features and provides candidate-edge probabilities with relatively fast inference.",
        "strengths": "CV and student-test edge macro F1 were close (0.8889 and 0.8921). Student-test HMM Top-1 was the best among models (0.5604), narrowly above XGBoost (0.5560).",
        "limits": "Student-test classifier-only observation Top-1 was 0.2547, and route quality still depends on candidate coverage and sequence decoding. Its edge precision/recall show the same recall-heavy operating point as XGBoost.",
        "objective": "Binary log-loss objective with balanced class weights; LightGBM's default binary log-loss is minimized.",
    },
}


def _read_inputs():
    comparison = pd.read_csv(EXPERIMENTS / "model_comparison.csv")
    summary = json.loads((EXPERIMENTS / "run_summary.json").read_text(encoding="utf-8"))
    tuning = json.loads((EXPERIMENTS / "tuning_summary.json").read_text(encoding="utf-8"))
    notes = (RESULTS / "data_quality_notes.md").read_text(encoding="utf-8")
    inventories = {
        "train": pd.read_csv(RESULTS / "inventory_train.csv"),
        "student_test": pd.read_csv(RESULTS / "inventory_student_test.csv"),
    }
    checks = {
        "train": pd.read_csv(RESULTS / "checks_train.csv"),
        "student_test": pd.read_csv(RESULTS / "checks_student_test.csv"),
    }
    return comparison, summary, tuning, notes, inventories, checks


def _observation_macro_f1() -> pd.DataFrame:
    rows = []
    files = dict(MODEL_FILES)
    files["Random Forest (tuned)"] = "predictions_random_forest_tuned_student_test.csv"
    for model, filename in files.items():
        predictions = pd.read_csv(PREDICTIONS / filename)
        per_trip = []
        for trip_id, trip in predictions.groupby("trip_id", sort=True):
            reference = trip["reference_arc_id"].to_numpy(dtype=np.int64)
            for prediction_column in ("classifier_arc_id", "viterbi_arc_id"):
                prediction = trip[prediction_column].to_numpy(dtype=np.int64)
                labels = np.unique(np.concatenate((reference, prediction)))
                score = f1_score(
                    reference, prediction, labels=labels, average="macro", zero_division=0
                )
                per_trip.append((trip_id, prediction_column, score))
        for method in ("classifier_arc_id", "viterbi_arc_id"):
            values = [score for _, pred_method, score in per_trip if pred_method == method]
            rows.append(
                {
                    "model": model,
                    "method": "Classifier-only" if method == "classifier_arc_id" else "HMM/Viterbi",
                    "student_observation_macro_f1_trip_mean": float(np.mean(values)),
                }
            )
    return pd.DataFrame(rows)


def _save_figures(comparison: pd.DataFrame, inventories, checks, obs_f1: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors_by_model = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    plots = [
        ("cv_macro_f1", "Grouped CV edge macro F1", "CV macro F1"),
        ("student_observation_top1_accuracy", "Student-test observation Top-1", "Top-1 accuracy"),
        ("student_arc_macro_f1", "Student-test edge-membership macro F1", "Edge macro F1"),
        ("student_log_loss", "Student-test edge-membership log loss", "Log loss"),
    ]
    for metric, title, ylabel in plots:
        values = pd.to_numeric(comparison[metric], errors="coerce")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(comparison["model"], values, color=colors_by_model[: len(comparison)])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=20)
        if "loss" not in metric:
            ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / f"{metric}.png", dpi=180)
        plt.close(fig)

    shutil.copy2(EXPERIMENTS / "gnn_training_loss.png", FIGURES / "gnn_training_loss.png")
    observation_f1 = obs_f1[obs_f1["method"] == "Classifier-only"]
    ordered_f1 = observation_f1.set_index("model").reindex(comparison["model"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        ordered_f1["model"],
        ordered_f1["student_observation_macro_f1_trip_mean"],
        color=colors_by_model,
    )
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.set_title("Student-test observation-level macro F1 (mean across trips)")
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "student_observation_macro_f1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(comparison))
    width = 0.36
    ax.bar(x - width / 2, comparison["student_observation_top1_accuracy"], width, label="Classifier-only")
    ax.bar(x + width / 2, comparison["student_hmm_observation_accuracy"], width, label="HMM/Viterbi")
    ax.set_xticks(x, comparison["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Exact reference-arc accuracy")
    ax.set_title("Student-test observation accuracy before and after route decoding")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "student_classifier_vs_hmm_top1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = ["Train (80 trips)", "Student test (10 trips)"]
    missing = [int(inventories[key]["n_missing"].sum()) for key in ("train", "student_test")]
    absent_files = [
        sum(int((~inventories[key][f"has.{ext}"]).sum()) for ext in ("track", "nodes", "arcs", "route"))
        for key in ("train", "student_test")
    ]
    index = np.arange(2)
    ax.bar(index - 0.18, missing, 0.36, label="Missing track cells")
    ax.bar(index + 0.18, absent_files, 0.36, label="Missing required files")
    for offset, values in ((-0.18, missing), (0.18, absent_files)):
        for position, value in zip(index + offset, values):
            ax.annotate(str(value), (position, value), xytext=(0, 3), textcoords="offset points", ha="center")
    ax.set_xticks(index, labels)
    ax.set_ylabel("Count")
    ax.set_title("Missing data and required-file coverage")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "missing_data_analysis.png", dpi=180)
    plt.close(fig)

    _save_selected_trip_figures()

    # Keep data-validation observations partition-specific and directly measured.
    gap_counts = [
        int(checks["train"]["n_gaps_over_5s"].sum()),
        int(checks["student_test"]["n_gaps_over_5s"].sum()),
    ]
    jump_counts = [
        int(checks["train"]["n_jumps_over_100m_in_1s"].sum()),
        int(checks["student_test"]["n_jumps_over_100m_in_1s"].sum()),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(2)
    ax.bar(x - 0.18, gap_counts, 0.36, label="GPS intervals > 5 s")
    ax.bar(x + 0.18, jump_counts, 0.36, label="Jumps > 100 m in 1 s")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Count of observations")
    ax.set_title("GPS timing gaps and implausible jumps by partition")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "gps_gap_jump_counts.png", dpi=180)
    plt.close(fig)


def _save_selected_trip_figures() -> None:
    trip_id = "00000000"
    raw = load_trip(str(DATA_ROOT / "student_test"), trip_id)
    track = clean_track(raw["track"])
    nodes = raw["nodes"].to_numpy(dtype=float)
    arcs = raw["arcs"].to_numpy(dtype=np.int64)
    route = raw["route"]["arc_id"].to_numpy(dtype=np.int64)
    projector, _ = make_projector(float(track["lon"].mean()), float(track["lat"].mean()))
    track_xy = projector(track["lon"], track["lat"])
    node_xy = projector(nodes[:, 0], nodes[:, 1])
    predictions = pd.read_csv(PREDICTIONS / MODEL_FILES["Random Forest"])
    predictions = predictions[predictions["trip_id"].astype(str) == trip_id]
    decoded_ids = predictions.sort_values("observation_index")["viterbi_arc_id"].to_numpy(dtype=np.int64)

    starts = node_xy[arcs[:, 0]]
    ends = node_xy[arcs[:, 1]]
    midpoints = (starts + ends) / 2
    from scipy.spatial import cKDTree

    tree = cKDTree(midpoints)
    k = min(70, len(arcs))
    _, nearest = tree.query(track_xy, k=k)
    road_ids = np.unique(np.asarray(nearest).reshape(-1))
    if len(road_ids) > 2500:
        road_ids = road_ids[:2500]

    fig, ax = plt.subplots(figsize=(9, 7))
    for arc_id in road_ids:
        ax.plot(
            [starts[arc_id, 0], ends[arc_id, 0]],
            [starts[arc_id, 1], ends[arc_id, 1]],
            color="#888888",
            linewidth=0.35,
            alpha=0.42,
        )
    route_points = node_xy[arcs[route]]
    for segment in route_points:
        ax.plot(segment[:, 0], segment[:, 1], color="#2ca02c", linewidth=1.5, alpha=0.85)
    decoded_points = node_xy[arcs[decoded_ids]]
    for segment in decoded_points:
        ax.plot(segment[:, 0], segment[:, 1], color="#d62728", linewidth=1.05, alpha=0.78)
    ax.plot(track_xy[:, 0], track_xy[:, 1], color="#1f77b4", linewidth=0.7, alpha=0.5)
    ax.scatter(track_xy[:, 0], track_xy[:, 1], color="#1f77b4", s=5, label="GPS observations", zorder=4)
    ax.plot([], [], color="#888888", linewidth=2, label="Relevant road-network arcs")
    ax.plot([], [], color="#2ca02c", linewidth=2, label="Reference route (student test)")
    ax.plot([], [], color="#d62728", linewidth=2, label="Random Forest + Viterbi")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Local projected easting (m)")
    ax.set_ylabel("Local projected northing (m)")
    ax.set_title(f"Trip {trip_id}: GPS, road network, reference and predicted route")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "student_trip_map_export.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(track["t"] - track["t"].iloc[0], track["lon"], label="Longitude")
    ax.plot(track["t"] - track["t"].iloc[0], track["lat"], label="Latitude")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Coordinate (degrees)")
    ax.set_title(f"GPS trajectory coordinates over time, student-test trip {trip_id}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "student_trip_gps_trajectory.png", dpi=180)
    plt.close(fig)


def _add_docx_table(doc, headers, rows, font_size=8):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    for cell, header in zip(table.rows[0].cells, headers):
        cell.text = str(header)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(font_size)
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(font_size)
    return table


def _build_docx(comparison, summary, tuning, inventories, checks, obs_f1) -> None:
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(9)
    styles["Title"].font.name = "Aptos Display"
    styles["Title"].font.size = Pt(27)
    styles["Heading 1"].font.color.rgb = RGBColor(20, 72, 108)
    styles["Heading 2"].font.color.rgb = RGBColor(30, 96, 139)
    styles["Heading 3"].font.color.rgb = RGBColor(49, 112, 150)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.style = "Title"
    title.add_run("PathSnap Map-Matching\nTechnical Report")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Measured development experiment | 80 training trips, 10 student-test trips").italic = True
    doc.add_paragraph(
        "Prepared from the committed PathSnap source, quality-analysis outputs, grouped-CV results, "
        "student-test predictions and tuning logs. Evaluator data was not used."
    )
    doc.add_paragraph(
        "AI-use disclosure: AI tools assisted with code and report drafting. The team must review, "
        "understand, and revise the analysis and disclose assistance as required by the event policy."
    )
    doc.add_page_break()

    doc.add_heading("2.2 Dataset Understanding and Exploratory Data Analysis", level=1)
    doc.add_heading("Partitions and preprocessing", level=2)
    doc.add_paragraph(
        f"The experiment uses {summary['train_trip_count']} training trips and "
        f"{summary['student_test_trip_count']} student-testing trips. The evaluator partition "
        "was kept separate and was not loaded. Each trip contains .track (longitude, latitude, time), "
        ".nodes (road node coordinates), .arcs (directed node pairs), and .route (chronological arc IDs). "
        "GPS points are not paired with road rows by index."
    )
    doc.add_paragraph(
        "Track rows are coerced to numeric values; non-finite/out-of-range coordinates or timestamps "
        "are dropped, and the remaining points are stably sorted by timestamp. Road coordinates are "
        "validated. Each trip is projected to local UTM coordinates before metric distances are used. "
        "No interpolation/imputation was applied because the inventory found zero missing track cells."
    )
    doc.add_heading("Missing values and data quality", level=2)
    missing_rows = []
    for key, label in (("train", "Train"), ("student_test", "Student test")):
        inv = inventories[key]
        missing_rows.append([
            label, len(inv), int(inv["n_missing"].sum()),
            *(int((~inv[f"has.{ext}"]).sum()) for ext in ("track", "nodes", "arcs", "route")),
            int((checks[key]["n_gaps_over_5s"] > 0).sum()),
            int(checks[key]["n_jumps_over_100m_in_1s"].sum()),
        ])
    _add_docx_table(
        doc,
        ["Partition", "Trips", "Missing track cells", "Missing .track", "Missing .nodes",
         "Missing .arcs", "Missing .route", "Trips with >5s gaps", "Jumps >100m/1s"],
        missing_rows,
        7,
    )
    doc.add_picture(str(FIGURES / "missing_data_analysis.png"), width=Inches(6.7))
    doc.add_paragraph(
        "Quality notes found gaps over five seconds in 45/80 training trips and 2/10 student-test trips. "
        "Implausible >100 m/s GPS jumps occurred in training only. Very long gaps included parked cases "
        "with only a few metres of movement and moving cases with hundreds of metres travelled; the decoder "
        "therefore includes a stationary-gap rule and bounded directed-network path search."
    )
    doc.add_paragraph(
        "Training networks average about 24,169 nodes and 48,461 directed arcs; the mean number of "
        "disconnected components is 39 (maximum 421). Approximately 92% of arcs have a reverse arc on "
        "average, with a training minimum near 35%. Routes have valid IDs and remain in one component; "
        "repeated arcs occur, so route alignment preserves chronological order."
    )
    doc.add_paragraph(
        "GPS-to-reference-route distance analysis reported a median of 5.6 m in train and 5.2 m in "
        "student test; 95th percentiles were 140.3 m and 16.8 m. Within 50 m were 89.2% of training "
        "points and 99.86% of student-test points. Several training trips have persistently distant GPS "
        "points; their cause (route reference versus GPS offset) is not established."
    )
    doc.add_picture(str(FIGURES / "gps_gap_jump_counts.png"), width=Inches(6.7))
    doc.add_heading("Feature extraction and rationale", level=2)
    doc.add_paragraph(
        f"Candidate generation uses a {summary['candidate_radius_m']} m radius plus "
        f"{summary['candidate_nearest_count']} nearest directed arcs. The 16 edge features are "
        + ", ".join(summary["feature_columns"])
        + ". Geometry, heading, local degree and reverse-edge features describe road plausibility; "
        "GPS distance, progress, speed and heading alignment link each road arc to the trajectory. "
        "The GNN also uses node position, in/out degree, nearest-track distance and track progress."
    )
    doc.add_picture(str(FIGURES / "student_trip_gps_trajectory.png"), width=Inches(6.5))
    doc.add_picture(str(FIGURES / "student_trip_map_export.png"), width=Inches(6.5))
    doc.add_paragraph(
        "Both figures above are exported from actual student-test trip 00000000 data. The companion "
        "interactive Folium map is reports/PathSnap_student_test_map.html and contains GPS, road, "
        "reference, classifier-only and HMM/Viterbi layers with a layer control."
    )

    doc.add_page_break()
    doc.add_heading("3. Model Experimentation and Comparison", level=1)
    doc.add_paragraph(
        "Five distinct classifiers were fitted and compared. The experiment is bounded: two grouped "
        "CV folds, one Bayesian trial and one GNN epoch (seed 42). This report does not imply the "
        "full-default benchmark (four folds, eight trials, eight GNN epochs)."
    )
    doc.add_heading("3.1 Overall model comparison", level=2)
    obs_f1_map = obs_f1[obs_f1["method"] == "Classifier-only"].set_index("model")[
        "student_observation_macro_f1_trip_mean"
    ].to_dict()
    obs_f1_all = {
        (row["model"], row["method"]): row["student_observation_macro_f1_trip_mean"]
        for _, row in obs_f1.iterrows()
    }
    table_rows = []
    for _, row in comparison.iterrows():
        table_rows.append([
            row["model"],
            MODEL_FAMILY[row["model"]],
            row["configuration"],
            f"{row['cv_macro_f1']:.4f}",
            f"{row['cv_observation_top1_accuracy']:.4f}",
            f"{row['student_hmm_route_continuity']:.4f}",
            f"{row['student_arc_macro_f1']:.4f}",
            f"{row['student_observation_top1_accuracy']:.4f}",
            f"{obs_f1_map[row['model']]:.4f}",
            f"{row['student_candidate_miss_rate']:.4f}",
            "N/A (margin scores)" if pd.isna(row["student_log_loss"]) else f"{row['student_log_loss']:.4f}",
            f"{row['inference_seconds']:.3f}",
        ])
    _add_docx_table(
        doc,
        ["Model", "Family", "Measured configuration", "CV edge macro F1",
         "CV obs Top-1", "Student HMM route continuity", "Student edge macro F1",
         "Student obs Top-1", "Student obs macro F1*", "Candidate miss",
         "Student edge log loss", "Inference s"],
        table_rows,
        5.5,
    )
    doc.add_paragraph(
        "*Observation macro F1 is the unweighted mean of per-trip macro F1 over observed/reference "
        "directed arc IDs. Log loss in the experiment output is binary edge-membership log loss, not "
        "observation-level multiclass log loss. SVM scores are uncalibrated margins, so log loss is N/A. "
        "Route continuity is reported after HMM/Viterbi decoding. Inference time is model score generation "
        "for all ten student-test trips and excludes HMM decoding."
    )
    doc.add_paragraph(
        "Random Forest was selected by grouped-CV edge macro F1 (0.9137), narrowly ahead of its peers. "
        "The GNN had the highest student observation Top-1 (0.2780), illustrating that arc-membership "
        "selection and pointwise route accuracy are different objectives."
    )
    for filename, caption in [
        ("cv_macro_f1.png", "Grouped CV edge-membership macro F1"),
        ("student_observation_top1_accuracy.png", "Student-test observation Top-1 accuracy"),
        ("student_arc_macro_f1.png", "Student-test edge-membership macro F1"),
        ("student_observation_macro_f1.png", "Student-test observation-level macro F1 (trip mean)"),
        ("student_log_loss.png", "Student-test edge-membership log loss; SVM excluded"),
        ("student_classifier_vs_hmm_top1.png", "Classifier-only versus HMM/Viterbi Top-1"),
        ("gnn_training_loss.png", "Actual GNN training loss; one measured epoch"),
    ]:
        doc.add_picture(str(FIGURES / filename), width=Inches(6.6))
        doc.add_paragraph(caption, style="Caption")
    doc.add_paragraph(
        "The GNN plot is the actual one-epoch training-loss output. A single observed point cannot establish convergence; no artificial validation curve is shown."
    )

    doc.add_page_break()
    doc.add_heading("3.2 Model-specific analysis", level=2)
    for model in MODEL_FILES:
        row = comparison.loc[comparison["model"] == model].iloc[0]
        obs_score = obs_f1_map[model]
        hmm_obs_score = obs_f1_all[(model, "HMM/Viterbi")]
        details = MODEL_DISCUSSION[model]
        doc.add_heading(model, level=3)
        doc.add_heading("A. Model understanding", level=4)
        doc.add_paragraph("Suitability: " + details["why"])
        doc.add_paragraph("Observed strengths: " + details["strengths"])
        doc.add_paragraph("Observed limitations: " + details["limits"])
        doc.add_heading("B. Training and configuration", level=4)
        doc.add_paragraph("Measured configuration: " + str(row["configuration"]))
        doc.add_paragraph("Objective/loss: " + details["objective"])
        doc.add_paragraph(
            "The configurations shown are the actual baseline values used in the run; "
            "the selected Random Forest also has a separately tuned artifact and configuration."
        )
        doc.add_heading("C. Experimental results and fit", level=4)
        log_loss = "N/A (uncalibrated SVM margin)" if pd.isna(row["student_log_loss"]) else f"{row['student_log_loss']:.4f} edge log loss"
        doc.add_paragraph(
            f"Student-test observation Top-1: {row['student_observation_top1_accuracy']:.4f}; "
            f"observation macro F1 (mean across trips): {obs_score:.4f}; "
            f"student edge-membership macro F1: {row['student_arc_macro_f1']:.4f}; "
            f"{log_loss}. HMM/Viterbi observation macro F1: {hmm_obs_score:.4f}; "
            f"HMM/Viterbi observation Top-1: "
            f"{row['student_hmm_observation_accuracy']:.4f}; HMM route continuity: "
            f"{row['student_hmm_route_continuity']:.4f}."
        )
        delta = row["student_arc_macro_f1"] - row["cv_macro_f1"]
        doc.add_paragraph(
            f"Overfit/underfit evidence: grouped-CV edge macro F1 was {row['cv_macro_f1']:.4f} "
            f"versus student-test edge macro F1 {row['student_arc_macro_f1']:.4f} "
            f"(student minus CV {delta:+.4f}). These two partitions differ in trips and CV has only "
            "two folds, so the gap alone cannot establish overfitting. "
            + (
                "The GNN was trained for only one epoch, which is insufficient to judge convergence; "
                "the low edge precision and higher student score suggest unstable behavior, not a proven fit diagnosis."
                if model == "GNN"
                else (
                    "The SVM is weak on both CV and student observation accuracy, consistent with underfitting or a feature mismatch, but no learning curve was collected."
                    if model == "Graph-kernel SVM"
                    else "Scores are close across these partitions, so this run shows no strong overfit signal; a larger grouped-CV run is needed for confidence."
                )
            )
        )

    doc.add_page_break()
    doc.add_heading("4. Bayesian Hyperparameter Optimisation", level=1)
    doc.add_paragraph(
        "Random Forest was chosen for tuning because it had the highest grouped-CV edge macro F1. "
        "Optuna's seeded TPE sampler maximized the mean grouped-CV edge macro F1 using the two training-only folds."
    )
    doc.add_paragraph("Search space: " + json.dumps(tuning["search_space"], sort_keys=True))
    doc.add_paragraph(
        f"Trials performed: {tuning['tuning_trials']}. Best tried values: "
        + json.dumps(tuning["best_params"], sort_keys=True)
        + f". Best trial grouped-CV macro F1: {tuning['best_cv_macro_f1']:.4f}."
    )
    tuned = tuning["comparison"]
    tuned_obs_f1 = obs_f1_map["Random Forest (tuned)"]
    _add_docx_table(
        doc,
        ["Metric", "Untuned RF", "Tuned RF"],
        [
            ["Grouped-CV edge macro F1", f"{tuned['cv_macro_f1_untuned']:.4f}", f"{tuned['cv_macro_f1_tuned']:.4f}"],
            ["Student observation Top-1", f"{tuned['student_observation_top1_accuracy_untuned']:.4f}", f"{tuned['student_observation_top1_accuracy_tuned']:.4f}"],
            ["Student observation macro F1", f"{obs_f1_map['Random Forest']:.4f}", f"{tuned_obs_f1:.4f}"],
            ["Student edge macro F1", f"{tuned['student_arc_macro_f1_untuned']:.4f}", f"{tuned['student_arc_macro_f1_tuned']:.4f}"],
            ["Student edge log loss", f"{tuned['student_log_loss_untuned']:.4f}", f"{tuned['student_log_loss_tuned']:.4f}"],
            ["Student model inference time (s)", f"{tuned['inference_seconds_untuned']:.3f}", f"{tuned['inference_seconds_tuned']:.3f}"],
        ],
    )
    doc.add_paragraph(
        "The tuned model did not outperform the untuned model on the optimization metric: CV macro F1 "
        "fell from 0.9137 to 0.9091. Student observation Top-1 rose from 0.2055 to 0.2116, while "
        "student edge macro F1 fell from 0.9260 to 0.9215 and edge log loss worsened from 0.0139 to "
        "0.0164. Inference time increased from 0.832 s to 1.563 s. One trial is only a smoke test of "
        "the optimization workflow, not an adequate search for a reliable optimum."
    )

    doc.add_page_break()
    doc.add_heading("5. HMM and Viterbi Route Reconstruction", level=1)
    explanations = [
        ("1. Hidden states", "The hidden state at each GPS time is one directed road-arc candidate generated for that observation."),
        ("2. Emissions", "Classifier scores are converted to log scores (log-softmax for probabilities; margins retained for the SVM) and combined with point-to-segment distance using a 25 m Gaussian distance penalty. These are log potentials, not claimed calibrated posterior probabilities."),
        ("3. Transitions", "For each prior arc end node, bounded directed shortest paths estimate route distance to each next candidate's start; the next arc length is added. A negative absolute mismatch between route distance and observed GPS displacement is scaled by max(20 m, half the displacement + 15 m); routes over 50 m/s receive an extra penalty. Search radius is bounded by the observed displacement and 5 km."),
        ("4. Missing candidates and GPS gaps", "Candidate generation always adds nearest road arcs when possible and reports reference candidate misses. Stationary gaps over 60 s allow staying on the same arc without a transition penalty. GPS gaps are not removed during preprocessing."),
        ("5. Disconnected roads", "Unreachable transitions retain a finite -12 penalty rather than being made impossible, so the decoder can recover if all alternatives are disconnected. Zero-time positive-distance transitions are rejected."),
        ("6. Did decoding improve?", "On student-test data, HMM/Viterbi observation Top-1 improved for all five models. Examples: Random Forest 0.2055 to 0.5249, XGBoost 0.2529 to 0.5560, GNN 0.2780 to 0.4538. Route continuity also rose for every model, e.g. RF 0.9296 to 0.9539 and LightGBM 0.9342 to 0.9564. It does not solve all errors: misses in candidate generation, parallel roads, and imperfect sequence/path assumptions remain."),
    ]
    for heading, text in explanations:
        doc.add_heading(heading, level=2)
        doc.add_paragraph(text)
    doc.add_picture(str(FIGURES / "student_classifier_vs_hmm_top1.png"), width=Inches(6.5))

    doc.add_page_break()
    doc.add_heading("6. Geographical Visualisation Using Folium", level=1)
    doc.add_paragraph(
        "The committed interactive Folium HTML companion was generated by the project's actual "
        "training pipeline from student-test trip 00000000. It includes GPS observations, relevant "
        "candidate road-network geometry, the student-test reference route, classifier-only route, "
        "HMM/Viterbi route, distinct colors, and layer controls. The plotted figure below is an exported "
        "view of those same actual trip layers; OpenStreetMap tiles are not embedded into the report."
    )
    doc.add_picture(str(FIGURES / "student_trip_map_export.png"), width=Inches(7.0))
    doc.add_paragraph("Interactive companion: reports/PathSnap_student_test_map.html")

    doc.add_page_break()
    doc.add_heading("7. API, UI and Deployment", level=1)
    doc.add_paragraph(
        "The API provides GET /api/trips and POST /api/predict with {\"trip_id\":\"...\"}. It loads "
        "the selected saved model, predicts without reference labels in feature inputs, computes metrics "
        "server-side using its reference route, and returns predicted arcs, route coordinates, GPS points "
        "and metrics. Reference IDs are not returned to the browser. The UI lists trips and displays the "
        "predicted path and metrics; the server logs requests and inference duration."
    )
    doc.add_paragraph(
        "Laptop A (server): install dependencies, provide the train/student-test data, run training before "
        "adding evaluator data, then place the evaluator partition under data/evaluator_test and start "
        "`python -m uvicorn src.api:app --host 0.0.0.0 --port 8000`. Laptop B (presenter): open "
        "`http://<Laptop-A-private-IP>:8000` on the same trusted private network."
    )
    doc.add_heading("Required live-evaluation screenshots - pending", level=2)
    doc.add_paragraph(
        "The supplied development artifacts do not contain screenshots captured during the two-laptop "
        "evaluator session. Do not treat the placeholders below as evidence: before final submission, "
        "insert (1) a screenshot of the running UI with a selected evaluator trip and visible returned "
        "metrics/map from Laptop B and (2) a screenshot of the successful API response in the receiving "
        "laptop's CMD/PowerShell terminal. Evaluator data and live-machine screenshots were not available "
        "in this report-generation environment."
    )
    for label in     ("Screenshot A - Laptop B working UI (capture during live evaluator run)",
              "Screenshot B - receiving laptop terminal/API response (capture during live evaluator run)"):
        doc.add_heading(label, level=3)
        doc.add_paragraph("[LIVE EVALUATION SCREENSHOT REQUIRED - NOT CAPTURED]")

    doc.add_page_break()
    doc.add_heading("Reproducibility, limitations and assistance disclosure", level=1)
    doc.add_paragraph(
        "The report's metrics and tuning values are available in reports/model_comparison.csv, "
        "reports/run_summary.json, reports/tuning_summary.json, reports/tuning_comparison.csv and "
        "reports/bayesian_trials.csv. The source and dependencies are in the GitHub repository. The "
        "run used seed 42, two grouped folds, one tuning trial and one GNN epoch. Full-default results "
        "were not run, and the evaluator partition was not used. Runtime depends on hardware."
    )
    doc.add_paragraph(
        "AI assistance disclosure: AI tools were used to help implement parts of the pipeline and draft "
        "this report from the team's measured outputs. Team members must independently verify every "
        "claim, understand the code/results, edit the report into their own account of the work, and "
        "follow the event's disclosure rules. This draft must not replace the team's own technical review."
    )
    doc.save(REPORTS / "PathSnap_Technical_Report.docx")


def _build_pdf(comparison, summary, tuning, inventories, checks, obs_f1) -> None:
    pdf_path = REPORTS / "PathSnap_model_report.pdf"
    page_size = landscape(A4)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=page_size,
        leftMargin=13 * mm,
        rightMargin=13 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="PathSnap Map-Matching Technical Report",
        author="PathSnap team (AI-assisted draft; team review required)",
    )
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(name="CoverTitle", parent=base["Title"], fontSize=28, leading=34, alignment=TA_CENTER, textColor=colors.HexColor("#14486c")))
    base.add(ParagraphStyle(name="SectionTitle", parent=base["Heading1"], fontSize=18, leading=22, textColor=colors.HexColor("#14486c"), spaceBefore=8, spaceAfter=8))
    base.add(ParagraphStyle(name="SubTitle", parent=base["Heading2"], fontSize=12, leading=15, textColor=colors.HexColor("#1e608b"), spaceBefore=6, spaceAfter=4))
    base.add(ParagraphStyle(name="BodySmall", parent=base["BodyText"], fontSize=8.5, leading=11, alignment=TA_LEFT, spaceAfter=4))
    base.add(ParagraphStyle(name="CaptionSmall", parent=base["BodyText"], fontSize=7.5, leading=9, alignment=TA_CENTER, textColor=colors.HexColor("#555555")))
    base.add(ParagraphStyle(name="Tiny", parent=base["BodyText"], fontSize=6.5, leading=8))
    story = []

    def para(text, style="BodySmall"):
        story.append(Paragraph(text, base[style]))

    def heading(text, level=1):
        story.append(Paragraph(text, base["SectionTitle" if level == 1 else "SubTitle"]))

    def table(headers, rows, widths=None, font=6.8):
        data = [[Paragraph(str(v), base["Tiny"]) for v in headers]]
        data.extend([[Paragraph(str(v), base["Tiny"]) for v in row] for row in rows])
        obj = PdfTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        obj.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14486c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aab7c4")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4f8")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(obj)
        story.append(Spacer(1, 5))

    def image(name, width=245 * mm, caption=None):
        story.append(PdfImage(str(FIGURES / name), width=width, height=width * 0.53, kind="proportional"))
        if caption:
            story.append(Paragraph(caption, base["CaptionSmall"]))
        story.append(Spacer(1, 4))

    story.extend([Spacer(1, 30 * mm), Paragraph("PathSnap Map-Matching<br/>Technical Report", base["CoverTitle"]), Spacer(1, 10 * mm)])
    para(f"Measured development experiment | {summary['train_trip_count']} training trips | {summary['student_test_trip_count']} student-test trips", "SubTitle")
    para("Consolidated technical report following the supplied Deliverables.pdf structure. Evaluator data was not used.")
    para("AI-use disclosure: AI tools assisted with implementation and drafting from measured project outputs. Team review, independent understanding, and disclosure under event policy are required.")
    story.append(PageBreak())

    heading("2.2 Dataset Understanding and Exploratory Data Analysis")
    heading("Dataset partitions and preprocessing", 2)
    para("The development experiment used 80 training trips for fitting/grouped cross-validation and 10 student-testing trips for development comparisons. The separate 10-trip evaluator partition was not loaded. Raw files are .track (longitude, latitude, timestamp seconds), .nodes (longitude/latitude, row index=node ID), .arcs (directed from-node/to-node, row index=arc ID), and .route (chronological arc IDs). GPS samples are never aligned to road rows by index.")
    para("GPS values are coerced to numeric; non-finite or out-of-range coordinates/timestamps are removed, and remaining points are stably sorted by timestamp. Road node values and directed endpoint IDs are validated. Each trip is projected to a local UTM coordinate system before distances are calculated. No imputation was applied because inventory checks found no missing GPS cells.")
    heading("Missing-value analysis and partition coverage", 2)
    missing_rows = []
    for key, label in (("train", "Train (80)"), ("student_test", "Student test (10)")):
        inv = inventories[key]
        missing_rows.append([label, len(inv), int(inv["n_missing"].sum()), *(int((~inv[f"has.{ext}"]).sum()) for ext in ("track", "nodes", "arcs", "route"))])
    table(["Partition", "Trips", "Missing track cells", "Missing .track", "Missing .nodes", "Missing .arcs", "Missing .route"], missing_rows, [42*mm, 18*mm, 33*mm, 28*mm, 28*mm, 28*mm, 28*mm])
    image("missing_data_analysis.png", 170 * mm, "Figure 1. Inventory-based missing track values and required-file coverage.")
    para(    "The inventory reports zero missing track values and all four required files for all 80 train and 10 student-test trips. Numeric validation found valid road-node coordinates and directed arc endpoints for the processed partitions. Data-quality checks found >5 s gaps in 45/80 train trips and 2/10 student-test trips. GPS jumps exceeding 100 m in 1 s occur in train only. The longest gaps include parked cases (over 52,000 s with only a few metres movement) and moving cases (hundreds of metres over 25 to 184 s).")
    image("gps_gap_jump_counts.png", 170 * mm, "Figure 2. Observed time gaps and implausible GPS jumps by partition.")
    para("Train networks average approximately 24,169 nodes and 48,461 directed arcs, with mean 39 connected components (maximum 421). About 92% of arcs have a reverse arc on average; the training minimum is about 35%. Route IDs are valid and routes stay in one component, but repeated arcs occur. GPS-to-reference route distance has median 5.6 m train / 5.2 m student test; p95 140.3 m / 16.8 m; within 50 m 89.2% / 99.86%. Several difficult training trips remain unexplained (wrong/incomplete route versus GPS offset).",)
    heading("Trajectory patterns and engineered features", 2)
    para(f"Candidate sets use a {summary['candidate_radius_m']} m radius plus {summary['candidate_nearest_count']} nearest directed arcs. Edge features: " + ", ".join(summary["feature_columns"]) + ". These encode road geometry, direction/topology, GPS distance, progress, speed and heading. GNN node features encode relative position, degree, nearest-track distance and progress. Large gaps and implausible speeds are retained and represented, not silently deleted.")
    image("student_trip_gps_trajectory.png", 170 * mm, "Figure 3. GPS trajectory over time, student-test trip 00000000.")
    image("student_trip_map_export.png", 170 * mm, "Figure 4. Exported geographical view of actual student-test route layers; interactive Folium map is committed separately.")
    story.append(PageBreak())

    heading("3. Model Experimentation and Comparison")
    para(f"Five models were trained. This measured run used seed {summary['seed']}, {summary['group_cv_folds']} grouped folds, {summary['tuning_trials']} Bayesian trial, and {summary['gnn_epochs']} GNN epoch. It is a bounded validation run, not the README's full defaults (4 folds, 8 trials, 8 GNN epochs).")
    heading("3.1 Overall model comparison", 2)
    obsf1 = obs_f1[obs_f1["method"] == "Classifier-only"].set_index("model")["student_observation_macro_f1_trip_mean"].to_dict()
    hmmf1 = obs_f1[obs_f1["method"] == "HMM/Viterbi"].set_index("model")["student_observation_macro_f1_trip_mean"].to_dict()
    rows=[]
    for _,r in comparison.iterrows():
        rows.append([r["model"], MODEL_FAMILY[r["model"]], r["configuration"], f"{r['cv_macro_f1']:.4f}", f"{r['cv_observation_top1_accuracy']:.4f}", f"{r['student_hmm_route_continuity']:.4f}", f"{r['student_arc_macro_f1']:.4f}", f"{r['student_observation_top1_accuracy']:.4f}", f"{obsf1[r['model']]:.4f}", f"{r['student_candidate_miss_rate']:.4f}", "N/A" if pd.isna(r["student_log_loss"]) else f"{r['student_log_loss']:.4f}", f"{r['inference_seconds']:.3f}"])
    table(["Model / family / measured configuration", "CV macro F1", "CV Top-1", "Student HMM continuity", "Student edge macro F1", "Student Top-1", "Student obs macro F1*", "Candidate miss", "Student edge log loss", "Inference s"], [[r[0]+" / "+r[1]+" / "+r[2],*r[3:]] for r in rows], [76*mm,19*mm,18*mm,21*mm,21*mm,18*mm,22*mm,19*mm,20*mm,16*mm])
    para("*Observation macro F1 is the unweighted mean of trip-level macro F1 over local directed arc IDs. Log loss shown is binary edge-membership log loss from the existing experiment; it is not observation-level multiclass loss. SVM has N/A because it returns uncalibrated margins. Route continuity is HMM/Viterbi continuity. Inference time covers model scoring across ten student-test trips, excluding decoding.")
    para("Random Forest led grouped-CV edge macro F1 (0.9137), but GNN led student-test classifier-only observation Top-1 (0.2780). The selection criterion and pointwise sequence metric differ; the CV winner is not uniformly best on every metric.")
    for filename, caption in [("cv_macro_f1.png","Figure 5. Grouped CV macro F1."),("student_observation_top1_accuracy.png","Figure 6. Student-test observation Top-1."),("student_arc_macro_f1.png","Figure 7. Student-test edge macro F1."),("student_observation_macro_f1.png","Student-test observation-level macro F1, mean across trips."),("student_log_loss.png","Figure 8. Student-test edge log loss, SVM excluded."),("student_classifier_vs_hmm_top1.png","Figure 9. Student-test Top-1 before and after decoding."),("gnn_training_loss.png","Actual GNN training loss, one epoch only; no artificial curve.")]:
        image(filename, 170*mm, caption)
    story.append(PageBreak())

    heading("3.2 Model-specific experimentation (all five models)")
    for model in MODEL_FILES:
        row=comparison.loc[comparison["model"]==model].iloc[0]
        details=MODEL_DISCUSSION[model]
        score=obsf1[model]
        heading(model,2)
        para("<b>A. Suitability:</b> "+details["why"])
        para("<b>Strengths observed:</b> "+details["strengths"])
        para("<b>Limitations observed:</b> "+details["limits"])
        para("<b>B. Training/configuration:</b> "+str(row["configuration"]))
        para("<b>Loss/objective:</b> "+details["objective"])
        loss="N/A (SVM margin scores)" if pd.isna(row["student_log_loss"]) else f"edge log loss {row['student_log_loss']:.4f}"
        para(f"<b>C. Student-test results:</b> observation Top-1 {row['student_observation_top1_accuracy']:.4f}; observation macro F1 (trip mean) {score:.4f}; edge-membership macro F1 {row['student_arc_macro_f1']:.4f}; {loss}; HMM observation macro F1 {hmmf1[model]:.4f}; HMM Top-1 {row['student_hmm_observation_accuracy']:.4f}; HMM continuity {row['student_hmm_route_continuity']:.4f}.")
        delta=row["student_arc_macro_f1"]-row["cv_macro_f1"]
        evidence=("One epoch is too short to assess convergence; student scores above CV may reflect split differences, so overfitting is not established." if model=="GNN" else "The low performance on both partitions suggests underfitting or graph-feature mismatch, but no learning curve was collected." if model=="Graph-kernel SVM" else f"Student edge macro F1 minus CV edge macro F1 is {delta:+.4f}. With two folds and different trips, this alone does not establish overfit; results are broadly consistent across splits." )
        para(f"<b>Overfit/underfit assessment:</b> {evidence}")
    story.append(PageBreak())

    heading("4. Bayesian Hyperparameter Optimisation")
    para(f"The chosen tuning target was {tuning['selected_model']}, selected on grouped-CV edge macro F1. Optuna TPE maximized mean grouped-CV edge macro F1 over {tuning['group_cv_folds']} trip-grouped folds.")
    para("Search space: "+json.dumps(tuning["search_space"],sort_keys=True))
    para("Trials performed: "+str(tuning["tuning_trials"])+"; best tried parameters: "+json.dumps(tuning["best_params"],sort_keys=True)+f"; best CV macro F1 {tuning['best_cv_macro_f1']:.4f}.")
    observation_f1_lookup = {
        (row["model"], row["method"]): row["student_observation_macro_f1_trip_mean"]
        for _, row in obs_f1.iterrows()
    }
    t=tuning["comparison"]
    table(["Same-split metric","Untuned RF","Tuned RF"],[
        ["Grouped-CV edge macro F1",f"{t['cv_macro_f1_untuned']:.4f}",f"{t['cv_macro_f1_tuned']:.4f}"],
        ["Student observation Top-1",f"{t['student_observation_top1_accuracy_untuned']:.4f}",f"{t['student_observation_top1_accuracy_tuned']:.4f}"],
        ["Student observation macro F1",f"{observation_f1_lookup[('Random Forest','Classifier-only')]:.4f}",f"{observation_f1_lookup[('Random Forest (tuned)','Classifier-only')]:.4f}"],
        ["Student edge macro F1",f"{t['student_arc_macro_f1_untuned']:.4f}",f"{t['student_arc_macro_f1_tuned']:.4f}"],
        ["Student edge log loss",f"{t['student_log_loss_untuned']:.4f}",f"{t['student_log_loss_tuned']:.4f}"],
        ["Inference time (s)",f"{t['inference_seconds_untuned']:.3f}",f"{t['inference_seconds_tuned']:.3f}"],
    ],[90*mm,45*mm,45*mm])
    para("Tuning did not improve the objective: CV macro F1 decreased from 0.9137 to 0.9091. Student Top-1 improved slightly from 0.2055 to 0.2116, while edge macro F1 fell from 0.9260 to 0.9215 and log loss worsened from 0.0139 to 0.0164. Inference rose from 0.832 to 1.563 s. One trial only verifies the pipeline; it is not a robust search.")
    story.append(PageBreak())

    heading("5. HMM and Viterbi Route Reconstruction")
    for q,a in [
        ("1. Hidden states","Each time step's hidden states are the directed road arcs in its GPS candidate set."),
        ("2. Emissions","Classifier probability scores are log-normalized across local candidates; SVM margins are used directly as relative scores. A 25 m Gaussian distance penalty is subtracted using projected point-to-segment distance. These are log potentials, not calibrated posterior probabilities."),
        ("3. Transitions","Directed shortest paths connect the prior candidate arc's end node to the next candidate's start node. Route length, including next-arc length, is compared with GPS displacement. The mismatch penalty is scaled by max(20 m, half-displacement + 15 m), with an extra penalty above 50 m/s. Search is bounded (up to 5 km)."),
        ("4. Missing candidates / GPS gaps","Nearest roads are used to keep candidate sets populated; reference candidate misses are reported. Long stationary gaps (>60 s, <10 m movement) permit staying on the same arc. GPS gaps are preserved."),
        ("5. Disconnected suggestions","Unreachable transitions receive a finite -12 log-score penalty, not an impossible transition, allowing a fallback path when all candidates are disconnected. A positive route distance with no elapsed time is rejected."),
        ("6. Did decoding improve?","Student-test HMM Top-1 improved for all five models: RF 0.2055 to 0.5249, XGBoost 0.2529 to 0.5560, graph SVM 0.0549 to 0.2265, GNN 0.2780 to 0.4538, LightGBM 0.2547 to 0.5604. Route continuity also rose for each model (RF 0.9296 to 0.9539; LightGBM 0.9342 to 0.9564). Errors remain due to candidate misses, ambiguous roads and decoder assumptions."),
    ]:
        heading(q,2); para(a)
    story.append(PageBreak())

    heading("6. Geographical Visualisation Using Folium")
    para("The committed interactive Folium map was produced from the actual student-test trip 00000000 by the project implementation. It contains original GPS observations, relevant road-network arcs, reference route, classifier-only route, HMM/Viterbi route, distinct colors and layer controls. The following figure is a static exported view of the same actual geographic layers; the full interactive HTML remains available at reports/PathSnap_student_test_map.html.")
    image("student_trip_map_export.png", 230*mm, "Figure 10. Actual trip GPS/road/reference/predicted route export; not evaluator data.")
    para("Selected example is from student_test only. Map colors: GPS blue; local candidate roads gray; reference green; Random Forest + Viterbi prediction red. The map HTML is included alongside this report.")
    story.append(PageBreak())

    heading("7. API, UI and Deployment")
    para("The FastAPI service exposes GET /api/trips and POST /api/predict with {\"trip_id\":\"...\"}. It loads the selected saved model, forms features without reference labels, predicts the route, computes metrics internally against the server-held reference route and returns predicted arc IDs/coordinates, GPS points and metrics. It does not return reference arc IDs. Server logs include trip ID, point/arc counts and inference duration.")
    para("Laptop A holds the model and evaluator files and runs `python -m uvicorn src.api:app --host 0.0.0.0 --port 8000`. Laptop B opens the server on the same trusted private network, selects a trip and requests reconstruction. Evaluator data is not used to train or tune.")
    heading("Required live-demo evidence - not captured in this development environment",2)
    para("<b>Screenshot required 1:</b> working UI on Laptop B with one selected evaluator trip and returned route/metrics.")
    para("<b>Screenshot required 2:</b> successful API response in the receiving laptop's CMD/PowerShell window.")
    para("These cannot be truthfully supplied here: no live two-laptop evaluator session or evaluator archive was available. Insert actual screenshots before final submission; the report intentionally does not fabricate them.")
    story.append(PageBreak())

    heading("Reproducibility, Limitations and Academic Integrity")
    para("Supporting measured files: reports/model_comparison.csv, run_summary.json, tuning_summary.json, tuning_comparison.csv and bayesian_trials.csv. Full source and dependencies are in GitHub. Results are from seed 42, two grouped folds, one tuning trial and one GNN epoch; the evaluator partition was not used. The required full-default run and live evaluation screenshots remain outstanding.")
    para("AI assistance disclosure: AI tools assisted with implementation and drafting from measured project outputs. Team members must verify the analysis, understand and revise the report, and follow the event's AI disclosure rules. This draft is not a substitute for the team's own interpretation.")
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(13*mm, 8*mm, "PathSnap technical report | development partitions only")
        canvas.drawRightString(page_size[0]-13*mm, 8*mm, f"Page {document.page}")
        canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def main():
    REPORTS.mkdir(exist_ok=True)
    comparison, summary, tuning, notes, inventories, checks = _read_inputs()
    obs_f1 = _observation_macro_f1()
    obs_f1.to_csv(REPORTS / "observation_level_metrics.csv", index=False)
    _save_figures(comparison, inventories, checks, obs_f1)
    _build_docx(comparison, summary, tuning, inventories, checks, obs_f1)
    _build_pdf(comparison, summary, tuning, inventories, checks, obs_f1)
    print("Created", REPORTS / "PathSnap_Technical_Report.docx")
    print("Created", REPORTS / "PathSnap_model_report.pdf")
    print("Figures:", len(list(FIGURES.glob("*.png"))))


if __name__ == "__main__":
    main()
