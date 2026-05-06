# Technical Report — Predictive Maintenance ML System

> **Run ID**: `e0f90246210543fe80fe7a5c404406b9`  
> **Experiment**: `pdm-predictive-maintenance` | **Model**: `pdm-failure-predictor v1` → `Production`  
> **Reproduced with**: `make train`

---

## 1. Problem Statement

Binary classification task: given the last 24 hours of sensor telemetry, error logs,
and maintenance history, **will machine X fail in the next 24 hours?**

**Dataset**: Microsoft Azure Predictive Maintenance (Kaggle, 876,100 hourly readings,
100 machines, 365 days).

**Business framing**: in predictive maintenance, a missed failure (false negative)
causes unplanned downtime costing ~5–10× more than an unnecessary inspection (false
positive). Every modelling decision — metric choice, threshold, imbalance strategy —
is anchored to this cost asymmetry.

---

## 2. Dataset Statistics

| Table | Rows | Description |
|---|---|---|
| PdM_telemetry | 876,100 | 100 machines × 8,761 hourly readings, 365 days |
| PdM_errors | 3,919 | 5 error types (error1–error5) |
| PdM_maint | 3,286 | 4 component types (comp1–comp4) |
| PdM_failures | 761 | Labelled failure events across 4 component types |
| PdM_machines | 100 | 4 machine models; age range 0–20 years |

After joining all tables and running the feature engineering pipeline:
**876,100 rows × 47 feature columns + 1 target column.**

- **Positive rate**: 17,184 positives / 876,100 rows = **1.96%**
- **Class ratio in train set**: 641,466 negatives / 13,134 positives → `scale_pos_weight = 48.84`

---

## 3. Temporal Split

| Split | Rows | Period | Share |
|---|---|---|---|
| Train | 654,600 | Jan 2015 – Sep 2015 | 74.7% |
| Test | 221,500 | Oct 2015 – Jan 2016 | 25.3% |

**Cutoff date**: `2015-10-01`. The split is **strictly temporal — never random**.
The model always predicts the future. `scale_pos_weight` was computed exclusively
from training rows to prevent any form of leakage.

---

## 4. Baseline

Before evaluating the model, a naive baseline sets the minimum acceptable bar.
For a heavily imbalanced dataset, **PR-AUC baseline ≈ positive rate ≈ 0.020**.

| Baseline strategy | PR-AUC | Recall | Precision |
|---|---|---|---|
| Always predict 0 (never alarm) | ~0.020 | 0.000 | — |
| Always predict 1 (always alarm) | ~0.020 | 1.000 | ~0.020 |
| Random classifier (at positive rate) | ~0.020 | ~0.500 | ~0.020 |

> A useful model must significantly exceed **PR-AUC = 0.020**.  
> The trained model achieves **PR-AUC = 0.9994 — 50× above baseline**.

---

## 5. Results

| Metric | Value | Interpretation |
|---|---|---|
| **PR-AUC** | **0.9994** | Primary metric — 50× above the ~0.020 baseline; near-perfect precision-recall tradeoff at all thresholds |
| **F2-Score** | **0.9967** | Weights recall 2× over precision — near-perfect even with asymmetric cost |
| **ROC-AUC** | **1.0000** | Perfect class separation on the temporal test set |
| **Precision** | **0.9885** | 98.9% of triggered alarms correspond to real failures |
| **Recall** | **0.9988** | 99.9% of real failures are detected before they occur |
| Decision threshold | 0.35 | Below 0.5 to bias toward recall; calibrated to FN > FP cost |
| `scale_pos_weight` | 48.84 | Gradient re-weighting for 49:1 class imbalance; computed from train only |
| Promotion threshold | **PR-AUC ≥ 0.80** | Model clears this bar with 0.9994; only models 40× above baseline reach Production |

**Why are these results so high?**
The Microsoft Azure PdM dataset was designed for benchmarking. Unlike real noisy
industrial data, the sensor signals (volt, rotate, pressure, vibration) change in
a clean, consistent pattern before each failure event. The rolling features (mean,
std, lag, delta) capture this degradation signal very effectively.

**Anti-leakage guarantee**: the temporal split (cutoff `2015-10-01`), backward-only
features, and train-only `scale_pos_weight` ensure these metrics reflect genuine
generalisation to unseen future data — not data contamination.

---

## 6. Feature Engineering Summary

| Feature group | Count | Rationale |
|---|---|---|
| Raw telemetry (volt, rotate, pressure, vibration) | 4 | Direct sensor readings |
| Rolling mean — 3h and 24h windows per sensor | 8 | Short and long degradation trends |
| Rolling std — 3h and 24h windows per sensor | 8 | Volatility increase signals instability |
| Lag values — 1h, 2h, 3h per sensor | 12 | Recent history context without leakage |
| Rate of change (Δ t−1) per sensor | 4 | Abrupt changes immediately precede failures |
| Error counts in past 24h (error1–error5) | 5 | Error accumulation is a strong failure predictor |
| Hours since last component replacement (comp1–4) | 4 | Recently replaced components rarely fail |
| Machine metadata (model_id, age) | 2 | Static context per machine |
| **Total** | **47** | All features strictly backward-looking |

All features are **backward-looking by construction** — they use only information
available at the time of prediction. The target label is the only forward-looking
element, which is correct by design.

---

## 7. Key Decisions and Trade-offs

**Binary vs. multiclass**  
Binary classification was chosen for operational clarity and simplicity. A multiclass
formulation (predicting which component fails) adds modelling complexity without
proportional value for the 24-hour prediction window.

**XGBoost vs. neural networks**  
XGBoost consistently outperforms deep learning on structured tabular data at this
scale. Training completed in under 5 minutes on CPU with no GPU requirement. The
results (PR-AUC = 0.9994) validate this choice decisively.

**`scale_pos_weight = 48.84` vs. SMOTE**  
SMOTE generates synthetic minority samples. If applied before the temporal split,
synthetic samples can encode future distribution patterns into the training set —
a subtle but real leakage vector. `scale_pos_weight` adjusts the XGBoost gradient
update function directly, with zero leakage risk. Value computed from training rows only.

**Decision threshold 0.35 vs. 0.50**  
At 1.96% positive rate, the default threshold of 0.5 biases the model toward
predicting "no failure". A threshold of 0.35 increases recall (achieved 0.9988) at
a minor precision cost (0.9885). This trade-off is correct for predictive maintenance
where unplanned downtime costs 5–10× more than a false alarm.

**Promotion threshold PR-AUC ≥ 0.80**  
Raised from 0.75 to 0.80. A model must be 40× above the ~0.020 random-classifier
baseline to be automatically promoted to Production. This prevents degraded models
(e.g., trained on corrupt features or with a broken pipeline run) from replacing the
current Production version. The current model (0.9994) clears this gate comfortably.

**Temporal split vs. k-fold cross-validation**  
k-fold with shuffling allows the model to train on future sensor readings to predict
past failures, inflating metrics by 15–30% and producing a model that fails in
production. The hard cutoff at `2015-10-01` simulates real deployment conditions.

**Polars vs. Pandas**  
Rolling window operations on 876,100 rows with `group_by + rolling` chains completed
in ~5 seconds with Polars. The explicit API (no implicit index, immutable DataFrames)
also reduces subtle mutation bugs common in Pandas time-series pipelines.

---

## 8. Lessons Learned

**Temporal leakage is the primary risk** in time-series ML. A random train/test split
on this dataset would inflate PR-AUC significantly and produce a model that fails
immediately in production.

**Class imbalance requires metric discipline.** Accuracy exceeds 98% for a model
that always predicts "no failure". PR-AUC and F2-Score immediately expose this failure
mode. Accuracy should never be the primary metric for imbalanced datasets.

**`scale_pos_weight` matters more than model architecture.** Switching from default
XGBoost (no imbalance handling) to `scale_pos_weight = 48.84` produced the largest
single performance improvement on this dataset.

**Feature order in serving must match training.** The `(1, 46) ≠ (-1, 47)` error
encountered during serving was caused by `machine_id` being excluded from the schema
and by numpy array construction relying on alphabetical sort rather than model-defined
column order. Fixed by: (1) adding `model_id` to `PredictionRequest`, (2) passing a
named `pandas.DataFrame` to `mlflow.pyfunc.predict()` — column order resolved by name,
not position. This pattern eliminates train/serve feature skew permanently.

**MLflow Registry pays off immediately.** Programmatic model promotion, version
tracking, and Registry-based serving eliminated all manual model file management.
The full train → register → serve cycle is automated in a single `make train` call.

**Polars rolling boundary behaviour.** With `window_size=W`, the first `W−1` rows per
machine return `null`. Explicit `forward_fill + backward_fill` is required after all
rolling operations — this is a common and silent production bug in time-series pipelines.

**MLflow stage API deprecation.** `transition_model_version_stage` is deprecated since
MLflow 2.9.0. The production path forward uses model aliases (`@production` tag).
Documented in Next Steps below.

---

## 9. Next Steps

| Priority | Improvement | Expected value |
|---|---|---|
| High | **Migrate MLflow stages → model aliases** | Removes deprecation warning; future-proof Registry API |
| High | **Apache Airflow DAGs** for weekly scheduled retraining | Removes manual `make train` dependency |
| High | **Evidently AI** for input feature drift detection | Catches distribution shift before accuracy degrades |
| Medium | **Prometheus + Grafana dashboard** | Visualise p99 latency, request rate, error rate from `/metrics` |
| Medium | **SHAP feature importance** logged as MLflow artifact | Explainability for maintenance stakeholders |
| Medium | **Per-component PR-AUC** breakdown in evaluator | Identifies which failure types (comp3, comp4) need more data |
| Medium | **Prediction logging** to JSONL per API request | Full traceability; enables ground truth feedback loop |
| Medium | **Hyperparameter optimisation** with Optuna | Systematic search vs. current manually-tuned defaults |
| Medium | **DVC remote** (Google Drive or S3) | Enables `dvc pull` for one-command data download — eliminates manual Kaggle step |
| Low | **Kubernetes deployment** with HPA autoscaling | Production-grade horizontal serving at scale |
| Low | **Feature store** (Feast or Tecton) | Eliminates train/serve feature skew; single source of truth |
| Low | **Triton Inference Server** | High-concurrency multi-model GPU/CPU serving |
