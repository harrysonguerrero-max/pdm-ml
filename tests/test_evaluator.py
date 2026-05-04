"""
Model evaluation focused on imbalanced binary classification.

Primary metric: PR-AUC (Precision-Recall Area Under Curve).

Why NOT accuracy?
    With ~98% negatives, a naive all-zeros model gets 98% accuracy.
    That model is useless — it never catches a failure.

Why F2-Score over F1?
    F2 weights recall 2x more than precision.
    In maintenance: missing a failure (FN) costs unplanned downtime.
    A false alarm (FP) costs an unnecessary inspection. FN >> FP.

Why PR-AUC over ROC-AUC?
    ROC-AUC is optimistic on imbalanced datasets because it includes
    true negatives in its calculation. PR-AUC focuses only on the
    minority class — exactly what matters here.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """
    Computes all evaluation metrics.

    Returns:
        dict with pr_auc, roc_auc, f2_score, precision, recall.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "pr_auc":    round(float(average_precision_score(y_test, y_prob)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_prob)), 4),
        "f2_score":  round(float(fbeta_score(y_test, y_pred, beta=2, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
    }
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def plot_confusion_matrix(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path = Path("data/processed"),
) -> Path:
    """
    Generates and saves the confusion matrix as a PNG.
    Logged as an MLflow artifact.

    Returns:
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "confusion_matrix.png"

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Failure", "Failure"],
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Temporal Test Split", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Confusion matrix saved: {out_path}")
    return out_path
