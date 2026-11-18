# Logistic Regression notes
Partitions: 5-fold grouped CV on train (80 trips, grouped by trip), comparison on student_test (10 trips). Evaluator data not used.
Data: candidate table from src/build_candidates.py (50 m radius, nearest 30 candidates per point, label = closest arc of the reference route). Trained only on points where the correct arc is a candidate.

## 1. Why this model suits the problem
For each (GPS point, candidate arc) pair it outputs the probability that the arc is correct. The HMM needs exactly this kind of probability. It is fast, simple, and its weights can be read.

## 2. Strengths and limitations
- Strengths: very fast (0.5 s to train), no tuning needed, easy to explain, gives probabilities.
- Limitations: it can only draw a straight-line boundary, so it cannot capture interactions between features (for example, a far road may be fine when the car is slow). It looks at each candidate alone, with no knowledge of the previous or next point. Class weighting makes the probabilities less calibrated.

## 3. Hyperparameters tried (and why)
- C (inverse regularisation strength) in 0.01, 0.1, 1, 10, with 5-fold GroupKFold by trip. Features standardised, class_weight="balanced" (only about 7% of rows are correct arcs), seed 42.
- All four values gave the same results to three decimals, because there are about 1.7M rows and only 6 features, so regularisation has nothing to fix. C = 10 had the highest mean macro-F1 by a tiny margin.

## 4. Loss function
Log loss (cross-entropy) plus an L2 penalty. It punishes confident wrong probabilities heavily.

## 5. Results
- Nearest-arc baseline (no model), top-1: 0.588 (train), 0.583 (student_test)
- CV macro-F1: 0.757 (std 0.008)
- CV top-1 accuracy: 0.782 (std 0.054)
- CV log loss: 0.276 (std 0.035)
- student_test: top-1 0.806, macro-F1 0.759, log loss 0.240
- Top-1 is measured on points whose correct arc is a candidate (all of train's used points; 99.86% of student_test points).
- Route continuity: pending the HMM.
- Candidate miss rate (50 m, 30 nearest): 10.85% train, 0.14% student_test. About 94% of train misses come from trips 00000054, 00000033 and 00000002.
- Training time 0.5 s; inference time: (measure predict_proba and add it)

## 6. Over/underfitting evidence (train vs CV vs student_test)
Train top-1 is 0.780 and validation top-1 is 0.782. Train log loss is 0.272 and validation is 0.276. The gap is tiny, so there is no overfitting. Both scores are modest, so the model slightly underfits: a linear model cannot capture the structure. Student_test is cleaner than train, so its scores are higher.
