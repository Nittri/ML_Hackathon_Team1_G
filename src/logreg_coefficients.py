# logreg_coefficients.py - which features does the Logistic Regression rely on?
# usage (from the repo root):  python src/logreg_coefficients.py
# Partition used: train (all 80 trips). Nothing from student_test or evaluator data.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FEATURES = ["dist_m", "along_frac", "heading_diff", "speed", "arc_len", "dist_rank"]
C = 10    # the C picked by train_logreg.py

# Load the train candidates and keep only points whose correct arc is a candidate
# (same rule as train_logreg.py, so this is the same model)
t = pd.read_csv("data/candidates/train_candidates.csv", dtype={"trip_id": str})
has_correct = t.groupby(["trip_id", "point_idx"])["label"].transform("max") == 1
t = t[has_correct]

model = make_pipeline(StandardScaler(),
                      LogisticRegression(C=C, class_weight="balanced", max_iter=1000))
model.fit(t[FEATURES], t["label"])
lr = model.named_steps["logisticregression"]

# Features were standardised (mean 0, spread 1), so the weights are comparable:
#   positive weight = a bigger value makes the arc MORE likely to be correct
#   negative weight = a bigger value makes it LESS likely
table = pd.DataFrame({"feature": FEATURES, "weight": lr.coef_[0]})
table["abs_weight"] = table["weight"].abs()
table = table.sort_values("abs_weight", ascending=False)

print("rows:", len(t), "| trips:", t["trip_id"].nunique(), "| C =", C)
print(table.round(3).to_string(index=False))
print("intercept:", round(float(lr.intercept_[0]), 3))

table.to_csv("results/logreg_coefficients.csv", index=False)
print("saved results/logreg_coefficients.csv")
