# train_logreg.py - Logistic Regression on the candidate table
# usage (from the repo root):  python src/train_logreg.py
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, f1_score

FEATURES = ["dist_m", "along_frac", "heading_diff", "speed", "arc_len", "dist_rank"]
C_VALUES = [0.01, 0.1, 1, 10]     # regularisation strengths to try
SEED = 42


def load(partition):
    t = pd.read_csv(f"data/candidates/{partition}_candidates.csv", dtype={"trip_id": str})
    # keep only points where the correct arc is among the candidates
    has_correct = t.groupby(["trip_id", "point_idx"])["label"].transform("max") == 1
    return t[has_correct].reset_index(drop=True)


def make_model(C):
    # scaling matters: dist_m is in metres, along_frac is 0 to 1
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=1000))


def top1(df, prob):
    """Share of GPS points where the arc with the highest probability is correct."""
    d = df[["trip_id", "point_idx", "label"]].copy()
    d["prob"] = prob
    best = d.loc[d.groupby(["trip_id", "point_idx"])["prob"].idxmax()]
    return best["label"].mean()


def score(model, df):
    prob = model.predict_proba(df[FEATURES])[:, 1]
    return {"log_loss": log_loss(df["label"], prob),
            "macro_f1": f1_score(df["label"], prob >= 0.5, average="macro"),
            "top1": top1(df, prob)}


train = load("train")
print("train rows used:", len(train), "| trips:", train["trip_id"].nunique())

# 1. Grouped cross-validation: whole trips are held out in each fold
rows = []
gkf = GroupKFold(n_splits=5)
for C in C_VALUES:
    for fold, (tr, va) in enumerate(gkf.split(train, groups=train["trip_id"])):
        model = make_model(C).fit(train.loc[tr, FEATURES], train.loc[tr, "label"])
        s_tr, s_va = score(model, train.loc[tr]), score(model, train.loc[va])
        rows.append({"C": C, "fold": fold,
                     **{f"train_{k}": v for k, v in s_tr.items()},
                     **{f"val_{k}": v for k, v in s_va.items()}})
        print("C =", C, "fold", fold, "val top1 =", round(s_va["top1"], 4))

cv = pd.DataFrame(rows)
cv.to_csv("results/logreg_cv_folds.csv", index=False)
summary = cv.groupby("C").mean().drop(columns="fold")
summary.to_csv("results/logreg_cv_summary.csv")
print(summary.round(4).to_string())

# 2. Refit on all of train with the best C (chosen by CV macro-F1), test on student_test
best_C = summary["val_macro_f1"].idxmax()
t0 = time.time()
final = make_model(best_C).fit(train[FEATURES], train["label"])
fit_time = time.time() - t0

test = load("student_test")
t0 = time.time()
res = score(final, test)
infer_time = time.time() - t0
print("\nbest C:", best_C)
print("student_test:", {k: round(v, 4) for k, v in res.items()})
print(f"train time {fit_time:.1f} s, inference time {infer_time:.1f} s")