# Logistic Regression notes

Partitions: 5-fold grouped CV on train (80 trips, grouped by trip); comparison on student_test (10 trips, development only). Evaluator data not used.

Data: candidate table from src/build_candidates.py (50 m radius, at most the 30 nearest candidate arcs per point, label = closest arc of the reference route). Heading is computed from the GPS point 3 steps back to the point 3 steps ahead (smoothed). Features: dist_m, along_frac, heading_diff, speed, arc_len, dist_rank. Trained only on points where the correct arc is a candidate.

## 1. Why this model suits the problem
For each (GPS point, candidate arc) pair it outputs the probability that the arc is correct. The HMM needs exactly this kind of probability. It is fast, simple, and its weights can be read.

## 2. Strengths and limitations
- Strengths: very fast (about 0.5 s to train), almost no tuning needed, easy to explain, gives probabilities.
- Limitations: it can only draw a straight-line boundary, so it cannot capture interactions between features (for example, a far road may be fine when the car is slow). It looks at each candidate alone, with no knowledge of the previous or next point. Class weighting makes the probabilities less calibrated.

## 3. Hyperparameters tried (and why)
- C (inverse regularisation strength) in 0.01, 0.1, 1, 10, with 5-fold GroupKFold by trip. Features standardised, class_weight="balanced" (only about 11% of rows are correct arcs), default lbfgs solver, max_iter 1000. There is no random component (the solver and GroupKFold are deterministic), so results repeat exactly.
- Values of C gave near-identical results: validation top-1 0.8614 (C = 0.01) vs 0.8612 (C >= 0.1), log loss 0.2310 vs 0.2304. C >= 0.1 are tied at four decimals; the code picked C = 10 by highest mean macro-F1. With about 1.7M rows and 6 features, regularisation has little to fix.

## 4. Loss function
Log loss (cross-entropy) plus an L2 penalty. It punishes confident wrong probabilities heavily.

## 5. Results (v2: heading smoothed)
Partition named in every line.
- Nearest-arc baseline (no model), top-1: 0.588 (train), 0.583 (student_test)
- train, 5-fold grouped CV, C = 10, validation folds (mean, std across the 5 folds): top-1 0.861 (std 0.032), macro-F1 0.801 (std 0.014), log loss 0.230 (std 0.027)
- student_test (development comparison only): top-1 0.850, macro-F1 0.799, log loss 0.204
- Top-1 is measured only on points whose correct arc is a candidate: 89.15% of train points (185,809 of 208,417) and 99.86% of student_test points.
- Candidate miss rate (50 m radius, 30 nearest): 10.85% train, 0.14% student_test. About 94% of train misses come from trips 00000054, 00000033 and 00000002.
- Route continuity: pending the HMM.
- Training time 0.44 to 0.47 s. Inference time 0.03 s (one score() call on the student_test candidate rows, including metric calculation).
- Earlier version (v1, unsmoothed heading), for the record: train CV top-1 0.782, macro-F1 0.757, log loss 0.276; student_test top-1 0.806, macro-F1 0.759, log loss 0.240. Smoothing the heading (3 steps back to 3 steps ahead) lifted CV top-1 from 0.782 to 0.861.

Training without the 3 trips with miss rate above 10% (00000054, 00000033, 00000002), tested on student_test (development only):

| Training set | log loss | macro-F1 | top-1 |
|---|---|---|---|
| all 80 trips | 0.2042 | 0.7992 | 0.8499 |
| 77 trips (without the 3) | 0.1829 | 0.8163 | 0.8484 |

Top-1 is a tie (a difference of 0.0015); log loss and macro-F1 improve. CV scores of the two runs are not comparable because the validation trips differ.

## 6. Over/underfitting evidence
Train top-1 is 0.862 and validation top-1 is 0.861 (train partition, grouped CV, C = 10). Train log loss is 0.2255 and validation is 0.2304. The gap is small, so there is no sign of overfitting. Possible underfitting (a limit of the model or features) is only circumstantial: scores plateau near 0.86 and no value of C changes them, and the errors split about evenly between "opposite direction" (48.5%) and "different road" (51.5%). We have not tested this against a more flexible model; the tree and boosting results will show it. student_test top-1 (0.850) is slightly below CV (0.861) while its log loss (0.204) is lower; with 10 trips and fold top-1 ranging from 0.818 to 0.893, differences this size are noise-level.
