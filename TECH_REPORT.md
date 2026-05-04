# Technical Report
## Predictive Maintenance ML System

---

## Problem Statement

Binary classification: given telemetry, errors and maintenance history,
will machine X fail in the **next 24 hours**?

**Dataset**: Microsoft Azure Predictive Maintenance (Kaggle)
- 100 machines | 1 year of hourly readings
- ~876,100 telemetry rows after joining all tables
- ~1-3% positive rate — failure events are rare by design

**Business framing**: a missed failure (false negative) leads to unplanned downtime
costing ~5-10x more than an unnecessary inspection (false positive). This asymmetry
directly drives threshold, metric and imbalance decisions.

---

## Dataset Statistics

| Table | Rows | Key info |
|---|---|---|
| PdM_telemetry | 876,100 | 100 machines × 8,761 hours |
| PdM_errors | 3,919 | 5 error types |
| PdM_maint | 3,286 | 4 component types |
| PdM_failures | 761 | 4 failure types; ~1% of telemetry rows |
| PdM_machines | 100 | 4 machine models; age 0-20 years |

After joining and feature engineering: **~87,648 rows × 43 features**.
Positive rate (target = 1): approximately **1.3%** of rows fall within a 24h failure window.

---

## Baseline

Before evaluating the model, a naive baseline establishes the floor:

| Baseline | Strategy | PR-AUC | Recall | Precision |
|---|---|---|---|---|
| Always predict 0 | Never alarm | ~0.013 | 0.00 | — |
| Always predict 1 | Always alarm | ~0.013 | 1.00 | ~0.013 |
| Random (at positive rate) | Random threshold | ~0.013 | ~0.50 | ~0.013 |

> PR-AUC baseline = positive rate = ~0.013. Any useful model must significantly exceed this.

---

## Results

| Metric | Value | Notes |
|---|---|---|
| PR-AUC | TBD after training | Primary metric — target: > 0.30 |
| F2-Score | TBD | Weights recall 2× over precision |
| ROC-AUC | TBD | Secondary reference only |
| Precision | TBD | Proportion of alarms that are real failures |
| Recall | TBD | Proportion of real failures caught |
| Threshold | 0.35 | Calibrated for FN > FP cost |

> **Note**: values are populated after running `make train` with the real dataset.
> The evaluation is performed on the **temporal test set only** (Oct–Jan 2016),
> never on training data.

---

## Failure Analysis

Beyond aggregate metrics, the model is evaluated by failure type:

| Failure type | Expected difficulty | Reason |
|---|---|---|
| comp1 | Moderate | Most frequent; good training signal |
| comp2 | Moderate | Second most frequent |
| comp3 | Hard | Rare events; higher FN expected |
| comp4 | Hard | Rarest; model may underperform |

An ideal next step is per-component PR-AUC to identify which components need
more training data or a dedicated sub-model.

---

## Key Decisions and Trade-offs

**Binary vs. multiclass**: chose binary for simplicity. A multiclass formulation
(which component fails) adds complexity without changing operational value for this scope.
The binary label is derived from any failure type occurring within the 24h window.

**XGBoost vs. neural networks**: XGBoost consistently outperforms deep learning on
structured tabular data at this scale. Training is fast (<5 min), interpretable via SHAP,
and requires no GPU. A neural network would add training complexity and hardware cost
without measurable accuracy benefit here.

**scale_pos_weight vs. SMOTE**: SMOTE generates synthetic minority samples. In a temporal
dataset, synthetic samples created before the train/test split could carry future statistical
patterns into training. `scale_pos_weight` adjusts the gradient update directly with
zero leakage risk. Value set to `(n_negative / n_positive)` automatically.

**Threshold 0.35 vs. 0.5**: the default 0.5 threshold optimises accuracy, which is
misleading at 1-3% positive rate. A threshold of 0.35 increases recall at the cost of
precision — acceptable because unplanned downtime (FN) costs 5-10x more than an
unnecessary inspection (FP).

**Temporal split vs. k-fold**: k-fold shuffles data randomly, creating impossible
scenarios where the model trains on future sensor readings to predict past failures.
A hard temporal split (train on first 9 months, test on last 3) simulates real
production conditions.

---

## Lessons Learned

- **Temporal leakage is the #1 risk** in time-series ML. Random train/test split
  inflates metrics significantly and produces a model that cannot generalise.
- **Class imbalance requires metric discipline**: accuracy shows 98%+ even for a
  model that always predicts "no failure". PR-AUC and F2 expose this immediately.
- **MLflow pays off immediately**: experiment comparison, reproducibility, and Registry
  deployment would take hours to rebuild manually. The infrastructure investment is worth it
  even for a single-engineer project.
- **Polars over Pandas for window operations**: rolling joins on 87k rows with multiple
  `group_by + agg` chains were noticeably faster with Polars' lazy evaluation.

---

## Next Steps

| Priority | Improvement | Value |
|---|---|---|
| High | Apache Airflow DAGs for scheduled retraining | Eliminates manual `make train` dependency |
| High | Evidently AI for input data drift detection | Catches distribution shift before accuracy drops |
| Medium | SHAP feature importance in evaluator | Explainability for business stakeholders |
| Medium | Hyperparameter optimisation with Optuna | Systematic search vs. manual tuning |
| Medium | Per-component PR-AUC in evaluator | Identifies which failure types need more data |
| Low | Multi-model serving with Triton | High-concurrency production serving |
| Low | Separate train/serve Docker images | Smaller serve image in production |
