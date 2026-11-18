# error_analysis.py - what kind of mistakes does Logistic Regression make?
# usage (from the repo root):  python src/error_analysis.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from data_io import load_arcs

FEATURES = ["dist_m", "along_frac", "heading_diff", "speed", "arc_len", "dist_rank"]
BAD_TRIPS = ["00000054", "00000033", "00000002"]

# same data as train_logreg.py: only points whose correct arc is a candidate
t = pd.read_csv("data/candidates/train_candidates.csv", dtype={"trip_id": str})
has_correct = t.groupby(["trip_id", "point_idx"])["label"].transform("max") == 1
t = t[has_correct].reset_index(drop=True)

# 1. Out-of-fold probabilities: every row is predicted by a model that never saw its trip
t["prob"] = 0.0
for tr, va in GroupKFold(n_splits=5).split(t, groups=t["trip_id"]):
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=10, class_weight="balanced", max_iter=1000))
    model.fit(t.loc[tr, FEATURES], t.loc[tr, "label"])
    t.loc[va, "prob"] = model.predict_proba(t.loc[va, FEATURES])[:, 1]

# 2. For each point: the arc the model picked, and the correct arc
picked = t.loc[t.groupby(["trip_id", "point_idx"])["prob"].idxmax(),
               ["trip_id", "point_idx", "arc_id", "label", "dist_m"]]
truth = t.loc[t["label"] == 1, ["trip_id", "point_idx", "arc_id", "dist_m"]]
m = picked.merge(truth, on=["trip_id", "point_idx"], suffixes=("_pred", "_true"))
m["wrong"] = m["label"] == 0

# 3. For wrong picks: is it the same road in the opposite direction?
m["kind"] = "correct"
for trip_id, g in m[m["wrong"]].groupby("trip_id"):
    arcs = load_arcs(f"data/train/{trip_id}/{trip_id}.arcs")
    f, to = arcs["from_node"].values, arcs["to_node"].values
    p, q = g["arc_id_pred"].values, g["arc_id_true"].values
    reverse = (f[p] == to[q]) & (to[p] == f[q])        # same two nodes, opposite direction
    m.loc[g.index, "kind"] = np.where(reverse, "opposite direction", "different road")

# 4. Report
n, wrong = len(m), m["wrong"].sum()
print(f"points: {n}   accuracy: {1 - wrong / n:.4f}   wrong: {wrong}")
print(m["kind"].value_counts().to_string())
print("\nshare of mistakes:")
print((m[m["wrong"]]["kind"].value_counts(normalize=True) * 100).round(1).to_string())

ok = ~m["trip_id"].isin(BAD_TRIPS)
print(f"\naccuracy without {BAD_TRIPS}: {1 - m[ok]['wrong'].mean():.4f}")

per_trip = m.groupby("trip_id").agg(points=("wrong", "size"), error_rate=("wrong", "mean"))
per_trip = per_trip.sort_values("error_rate", ascending=False)
per_trip.to_csv("results/logreg_errors_by_trip.csv")
print("\nworst 10 trips by error rate:")
print(per_trip.head(10).round(3).to_string())