# Technical Design Document
## Predictive Maintenance ML System

---

## Architecture Overview

```mermaid
flowchart LR
    subgraph Input
        A[PdM_telemetry.csv<br/>87k rows]
        B[PdM_errors.csv]
        C[PdM_maint.csv]
        D[PdM_failures.csv]
        E[PdM_Machines.csv]
    end
    subgraph Pipeline
        F[loader.py<br/>Schema validation]
        G[preprocessor.py<br/>Joins + target label]
        H[engineering.py<br/>Rolling + lags + deltas]
        I[trainer.py<br/>XGBoost + MLflow]
        J[evaluator.py<br/>PR-AUC, F2, CM]
    end
    subgraph Registry
        K[MLflow Experiment<br/>Tracking]
        L[Model Registry<br/>Production stage]
    end
    subgraph Serving
        M[FastAPI :8000<br/>/health /predict]
    end
    Input --> F --> G --> H --> I --> J
    I --> K --> L --> M
```

---

## Feature Engineering

| Group | Columns | Rationale |
|---|---|---|
| Rolling mean (3h, 24h) | 8 cols | Short and long degradation trends |
| Rolling std (3h, 24h) | 8 cols | Volatility signals instability |
| Lag values (1h, 2h, 3h) | 12 cols | Recent history per machine |
| Rate of change (delta) | 4 cols | Abrupt changes precede failures |
| Error counts (24h window) | 5 cols | Error frequency predicts failure |
| Hours since maintenance | 4 cols | Recently replaced components rarely fail |
| Machine metadata | 2 cols | Model type and age as static context |

**Total: ~41 features + 1 target**

---

## Anti-Leakage Strategy

| Rule | Implementation |
|---|---|
| All features backward-looking | Rolling/lag windows look left only |
| Target forward-looking | Label = failure in next N hours |
| Temporal split only | Strict cutoff date, never random shuffle |
| Error/maintenance counts | Only events before current timestamp counted |

---

## Technical Decisions

| Decision | Chosen | Alternatives | Reason |
|---|---|---|---|
| Formulation | Binary (fail / no fail) | Multiclass per component | Simpler, defensible, operationally useful |
| Model | XGBoost | LightGBM, Neural Net | Best for tabular; fast; interpretable |
| Imbalance | scale_pos_weight | SMOTE, class_weight | SMOTE risks temporal leakage |
| Metric | PR-AUC + F2 | Accuracy, ROC-AUC | Accuracy misleading at 1-3% positive rate |
| Threshold | 0.35 | 0.5 default | FN cost > FP cost in maintenance |
| Data layer | Polars | Pandas | Faster; modern API |
| Tracking | MLflow | W&B, Neptune | Self-hosted; in job description |
| Serving | FastAPI | Flask, Django | Async; Pydantic; auto OpenAPI docs |

---

## Monitoring Roadmap

- **Data drift**: Evidently AI to detect distribution shift in input features.
- **Performance**: weekly PR-AUC re-evaluation on labeled production data.
- **Alerts**: Prometheus/Grafana threshold alerts on recall drop.
- **Retraining trigger**: automatic pipeline re-run when drift exceeds threshold.
- **Rollback**: MLflow Registry stage transitions allow instant rollback.
