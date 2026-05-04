# Technical Report
## Predictive Maintenance ML System

---

## Problem Statement

Binary classification: given telemetry, errors and maintenance history,
will machine X fail in the **next 24 hours**?

Dataset: Microsoft Azure Predictive Maintenance (Kaggle)
- 100 machines | 1 year of hourly readings | ~87,648 telemetry rows
- ~1-3% positive rate (failure events are rare)

---

## Results

| Metric | Value | Notes |
|---|---|---|
| PR-AUC | TBD after training | Primary metric |
| F2-Score | TBD | Weights recall 2x |
| ROC-AUC | TBD | Secondary reference |
| Precision | TBD | Proportion of alarms that are real |
| Recall | TBD | Proportion of real failures caught |

> Values populated after running `make train` with the real dataset.

---

## Key Decisions and Trade-offs

**Binary vs. multiclass**: chose binary for simplicity. A multiclass formulation
(which component fails) adds complexity without changing operational value for this scope.

**XGBoost vs. neural networks**: XGBoost consistently outperforms DL on structured tabular
data at this scale. Training is fast, interpretable via SHAP, and requires no GPU.

**scale_pos_weight vs. SMOTE**: SMOTE generates synthetic minority samples. In a temporal
dataset, synthetic samples could carry future information if applied before the split.
scale_pos_weight adjusts the loss function directly with zero leakage risk.

**Threshold 0.35**: in maintenance, a missed failure (FN) leads to unplanned downtime,
typically costing 5-10x more than a false alarm (FP). Lower threshold increases recall.

---

## Lessons Learned

- Temporal leakage is the #1 risk in time-series ML.
- Class imbalance requires metric discipline: accuracy shows 98%+ even for a naive model.
- MLflow pays off immediately: experiment comparison, reproducibility, and Registry deployment.

---

## Next Steps

| Priority | Improvement |
|---|---|
| High | Apache Airflow DAGs for scheduled retraining |
| High | Evidently AI for data drift monitoring |
| Medium | SHAP feature importance for explainability |
| Medium | Hyperparameter optimization with Optuna |
| Low | Multi-model serving with Triton Inference Server |
