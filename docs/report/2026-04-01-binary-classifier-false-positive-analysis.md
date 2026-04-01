# Binary Classifier False Positive Analysis

**Date:** 2026-04-01
**Model analyzed:** LR-v10 (LogisticRegression, L2-normalized, balanced class weights, SurfPerch 1280-d)
**Detection job:** Orcasound Lab 2021-11-01 00:00 UTC -- 2021-11-02 00:00 UTC

## Summary

The binary (whale/no whale) classifier achieves strong cross-validation metrics
(95.8% precision, 98.7% AUC) but produces a high false positive rate in
deployment. Analysis of a 24-hour Orcasound Lab detection job reveals ~90% of
detections are false positives, concentrated in the 0.90--0.99 confidence range
and dominated by ship noise.

The root cause is narrow negative training data -- the classifier hasn't seen
enough variety of ship sounds to reject them. The recommended fix is hard
negative mining from labeled detection jobs.

---

## 1. Model Configuration

| Parameter | Value |
|---|---|
| Classifier | LogisticRegression (sklearn) |
| Embedding model | SurfPerch TF2 SavedModel |
| Vector dimension | 1280 |
| L2 normalize | True |
| Class weight | balanced |
| C (regularization) | 1.0 (default) |
| Window size | 5.0 seconds |
| Sample rate | 32,000 Hz |

### Training Data Composition

**Positive samples (2,256 embeddings from 2,211 audio files):**

| Source | Count | % |
|---|---|---|
| Emily Vierling labeled vocalizations | 1,507 | 68% |
| SanctSound OC01 | 333 | 15% |
| Curated humpback positives | 371 | 17% |

Emily Vierling vocalization types represented: Whup (223), Grunt (213),
Ascending moan (192), Moan (168), Growl (131), Chirp (101), Upsweep (83),
Shriek (78), Buzz (62), Descending moan (49), Cry (38), Trumpet (34),
Bellow (33), Creak (30), Pop (29), Thwop (15), Whistle (11), Vibrate (4).

**Negative samples (3,072 embeddings from 2,186 audio files):**

| Category | Count | % |
|---|---|---|
| Background (ambient noise) | 1,424 | 65.1% |
| Ship noise | 762 | 34.9% |

Negative hydrophone sources: Orcasound Lab (1,059), North San Juan Channel (381),
SanctSound OC01 (314), NOAA Glacier Bay (240), Bush Point (103),
SanctSound OC (84), Port Townsend (5).

### Cross-Validation Metrics

| Metric | Value |
|---|---|
| Accuracy | 95.6% (+/- 0.3%) |
| Precision | 95.8% (+/- 0.6%) |
| Recall | 93.7% (+/- 0.4%) |
| F1 | 94.7% (+/- 0.3%) |
| ROC AUC | 98.7% (+/- 0.3%) |
| Score separation | 3.77 |
| Training confusion | TP=2197, FP=31, TN=3041, FN=59 |

---

## 2. Detection Job Analysis

**Job parameters:** confidence threshold 0.9, high threshold 0.9, low threshold
0.8, hop 1.0s, windowed detection mode.

**Overall statistics:**

| Metric | Value |
|---|---|
| Total windows analyzed | 80,597 |
| Windows above 0.9 | 9,336 (11.6%) |
| Final detections (after NMS) | 2,352 |
| Labeled humpback | 218 |
| Labeled ship | 21 |
| Labeled background | 2 |
| Unlabeled (likely FP) | ~2,111 |

### Window-Level Confidence Distribution

| Confidence range | Windows | % of total |
|---|---|---|
| 0.00 -- 0.10 | 15,609 | 19.4% |
| 0.10 -- 0.20 | 10,606 | 13.2% |
| 0.20 -- 0.30 | 7,936 | 9.8% |
| 0.30 -- 0.40 | 6,830 | 8.5% |
| 0.40 -- 0.50 | 5,924 | 7.4% |
| 0.50 -- 0.60 | 5,691 | 7.1% |
| 0.60 -- 0.70 | 5,814 | 7.2% |
| 0.70 -- 0.80 | 6,038 | 7.5% |
| 0.80 -- 0.90 | 6,813 | 8.5% |
| 0.90 -- 0.95 | 4,108 | 5.1% |
| 0.95 -- 0.99 | 3,659 | 4.5% |
| 0.99 -- 1.00 | 1,569 | 1.9% |

The confidence distribution is bimodal with a large mass below 0.5 (true
negatives) and a long tail above 0.9 (detections). The problematic region is
the 0.90--0.99 range where most false positives concentrate.

### Detection-Level Precision by Score Band

| Score band | Detections | Humpback | Ship/BG | Unlabeled | Approx. precision |
|---|---|---|---|---|---|
| 0.995 -- 1.000 | 207 | 162 | 23 | 22 | ~78% |
| 0.990 -- 0.995 | 89 | 9 | 0 | 80 | ~10% |
| 0.980 -- 0.990 | 214 | 10 | 0 | 204 | ~5% |
| 0.960 -- 0.980 | 398 | 8 | 0 | 390 | ~2% |
| 0.900 -- 0.960 | 1,444 | 13 | 0 | 1,431 | ~1% |

Precision drops sharply below 0.995. The vast majority of true humpback
detections (162/218 = 74%) are concentrated in the top score band (0.995+),
while 91% of all detections are likely false positives.

---

## 3. Model Evolution Trend

Analysis of 17 model versions from detector-v14 through LR-v11:

| Model | Positives | Negatives | CV Precision | Score Separation |
|---|---|---|---|---|
| ir-tflite-v1 | 1,676 | 1,412 | 0.989 | 4.93 |
| lr-v12 | 1,701 | 2,376 | 0.977 | 4.34 |
| LR-v1 | 2,033 | 2,411 | 0.983 | 4.52 |
| LR-v4 | 1,923 | 2,577 | 0.975 | 4.36 |
| LR-v9 | 2,087 | 2,976 | 0.960 | 3.97 |
| LR-v10 | 2,256 | 3,072 | 0.958 | 3.77 |
| LR-v11 | 2,880 | 3,191 | 0.961 | 3.72 |

**Key observation:** As training data grows, score separation steadily
decreases. Adding more diverse data makes the decision boundary harder, but the
linear classifier cannot increase its capacity to match. The model is near the
limits of what LogisticRegression can achieve on this data.

The MLP variant (mlp-v1, early in development) achieved a score separation of
15.1 -- roughly 3x better than any LR model -- suggesting a non-linear
boundary has significantly more capacity for this problem.

---

## 4. Root Cause Analysis

### 4.1 Narrow Negative Diversity

The negative training data contains only two categories: "background" (65%) and
"ship" (35%). Real-world hydrophone audio contains a much wider variety of
sounds. The 762 ship samples in training represent a limited set of vessel
passages from specific hydrophones and time periods. Ship noise varies widely
by vessel type, speed, distance, and sea state -- the training set does not
capture this diversity.

### 4.2 Domain Gap Between Training and Deployment

The classifier achieves 1% FPR on training data (31/3,072 negatives
misclassified) but ~90% of deployment detections are false positives. This
domain gap exists because curated negative samples are "easy" -- acoustically
distinct from whale calls. The false positives in deployment are "hard" --
ship sounds with harmonic structures that overlap with humpback call
frequencies in the SurfPerch embedding space.

### 4.3 Base Rate Problem

In 24 hours of audio, humpback vocalizations occur in ~1--2% of 5-second
windows. Even a 1% false positive rate on 80,000+ windows produces ~800 false
detections, swamping the ~200 true detections. High-volume detection requires
extremely low FPR, which in turn requires training data that covers the
distribution of sounds the classifier will encounter.

### 4.4 Linear Boundary Limitation

LogisticRegression draws a single hyperplane in 1280-dimensional space. If
confusing ship sounds are interspersed with whale calls in embedding space (not
linearly separable), no hyperplane can cleanly separate them. The declining
score separation across model versions (4.93 to 3.72) suggests the boundary is
increasingly constrained.

---

## 5. Recommendations

### 5.1 Hard Negative Mining from Detection Jobs (high impact, recommended)

Extract embeddings from false positive detections in labeled detection jobs and
add them as negative training samples.

**Rationale:** The detection_embeddings.parquet for the analyzed job contains
embeddings for all 2,352 detections. The ~2,100 unlabeled detections are
overwhelmingly false positives (only 218/2,352 are humpback). These are the
exact sounds the classifier misclassifies -- the hardest, most informative
negatives available.

**Implementation approach:**
- For detection jobs with labeled humpback positives, treat unlabeled detections
  as candidate hard negatives
- Use a conservative filter: exclude detections above 0.995 confidence (where
  78% are true positives) and only mine from lower score bands
- Extract embeddings from detection_embeddings.parquet (already computed)
- Add as negative embedding sets to the training pipeline
- Retrain and evaluate on the same detection job

**Expected impact:** Directly teaches the classifier what "confusing ship noise"
looks like. Even 200--500 hard negatives from a single detection job could
significantly improve separation in the 0.96--0.99 range.

### 5.2 MLP Classifier (medium impact, consider after 5.1)

If hard negative mining with LogisticRegression does not sufficiently reduce
false positives, switch to the MLP classifier. The existing mlp-v1 showed 3x
better score separation, suggesting a non-linear boundary has more capacity for
this problem.

**Note:** Should be evaluated after hard negative mining, not instead of it.
MLP with the current narrow negatives would likely show the same deployment
FP problem.

### 5.3 LR Hyperparameter Tuning (low impact, not recommended now)

Auto-tuning C (regularization strength) across a grid during retraining.
Analysis suggests this is low-value: the model already achieves 98.7% AUC, and
the bottleneck is training data composition, not regularization. After adding
hard negatives, manually testing C=0.1 and C=10 alongside C=1.0 would reveal
whether tuning matters for the new data distribution.

---

## 6. Evaluation Plan

After implementing hard negative mining and retraining:

1. Run the same detection job (Orcasound Lab 2021-11-01) with the retrained model
2. Compare score-banded precision against the baseline in Section 2
3. Target: reduce false positives in the 0.96--0.99 band by >50%
4. Check that true positive recall in the 0.995+ band is maintained
5. Validate on additional detection jobs from different hydrophones/dates
