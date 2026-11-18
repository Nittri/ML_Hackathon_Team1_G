# PathSnap map matching

This repository contains the PathSnap raw-data inspection tools and a reproducible
route-matching pipeline. It reads each trip's `.track`, `.nodes`, `.arcs`, and
`.route` files; the row number in `.nodes` and `.arcs` is the corresponding
zero-based ID. GPS observations are never aligned to network rows by index.

## Setup

Use Python 3.10 or newer. In PowerShell, extract the development partitions into
the repository's ignored `data` directory and install the dependencies:

```powershell
Expand-Archive -Path ".\Student folder.zip" -DestinationPath ".\data" -Force
python -m pip install -r requirements.txt
```

The archive should provide `data\Student folder\train` and
`data\Student folder\student_test`. Keep the separate evaluator archive out of
development runs.

## Train and compare models

```powershell
python -m src.train_models `
  --train "data\Student folder\train" `
  --student-test "data\Student folder\student_test" `
  --output "results\experiments" `
  --folds 4 `
  --trials 8
```

The experiment uses grouped cross-validation by trip on the 80 training trips,
then compares the final fits on the 10 student-testing trips. The student test
is not used for feature selection, threshold selection, or Bayesian tuning.
Optuna's seeded TPE sampler optimizes grouped-CV macro F1 for the model selected
by training-CV macro F1. The default candidate search uses a 50 m radius plus
the 20 nearest road arcs; its miss rate is reported rather than hidden.
The tuning summary records the objective, grouped fold count, search space,
trial log, and best parameters alongside the tuned-vs-untuned metrics.

The five implemented models are XGBoost with its binary logistic objective,
Random Forest, a linear SVM over an explicit Weisfeiler-Lehman graph-kernel
feature map, a directed message-passing GNN, and LightGBM. “lbgm” is interpreted
as LightGBM; “logistic regression base” is interpreted as XGBoost's
`binary:logistic` objective. The edge labels mark directed arcs appearing in the
reference route. Feature rows include projected geometry, local directed
degrees, reverse-arc availability, and GPS distance/progress/speed/heading
features. Invalid GPS rows are removed and remaining points are stably ordered
by timestamp; large gaps and implausible speeds are retained as observations
and represented by features rather than silently discarded.

The completed measured report is committed at
`reports\PathSnap_model_report.pdf`, with its interactive Folium companion at
`reports\PathSnap_student_test_map.html` and the underlying comparison/tuning
tables beside them. This report reflects the bounded run recorded in
`reports\run_summary.json` (2 CV folds, 1 tuning trial, 1 GNN epoch); it is not
a full-default benchmark. The command above uses the documented full defaults.

Each training run writes model comparison CSVs, tuning logs, plots, prediction
CSVs, and saved model artifacts beneath `results\experiments`. Those generated
run outputs are ignored by Git; rerun the command to reproduce them.

## Route decoding

`src.decoder` compares independent classifier choices with Viterbi decoding.
Hidden states are the directed road-arc candidates for each GPS observation.
Emission scores combine the model score and projected point-to-segment
distance. Transitions prefer a directed network path whose length agrees with
the observed displacement; unreachable transitions receive a finite penalty,
and large stationary gaps permit remaining on the same segment. Evaluation
reports observation Top-1 accuracy, candidate miss rate, arc-classification
metrics, and route continuity for both predictions.

## Evaluator API and UI

After development is complete, extract the evaluator archive into the same
ignored `data` directory; it should create `data\evaluator_test`. Do not run
training, tuning, or model selection on this partition. Start the server after
the experiment has produced its model artifact:

```powershell
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. The UI calls `GET /api/trips` and sends a selected
trip ID to `POST /api/predict` as `{"trip_id":"..."}`. The API loads the model,
predicts without reference labels, then uses the server-side route only to
calculate metrics. Reference arc IDs are not returned to the UI. API requests
and inference timing are logged by Uvicorn and the PathSnap API logger.

## Existing exploration utilities

Run the inventory, data checks, GPS-noise analysis, or map generator from the
repository root:

```powershell
python src\inventory.py "data\Student folder\train"
python src\checks.py "data\Student folder\train"
python src\noise.py "data\Student folder\train"
python src\map_trip.py "data\Student folder\train" 00000026 0 400
```