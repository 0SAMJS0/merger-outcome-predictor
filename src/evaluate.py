"""
Evaluation & reporting for the best model.

Produces:
  * confusion matrix (with explicit false-positive / false-negative framing),
  * ROC curve,
  * precision-recall curve,
  * a written markdown evaluation report.

A *false positive* here = predicting "will complete" for a deal that actually
fails. The report highlights this because acting on a false "will-complete"
signal is the costlier mistake for a decision-maker.

Run:  python -m src.evaluate
"""
from __future__ import annotations

import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src import config
from src.feature_engineering import MODEL_FEATURES


def _load_test_split():
    df = pd.read_csv(config.FEATURES_CSV)
    with open(config.METADATA_FILE) as f:
        meta = json.load(f)
    test_idx = meta["test_index"]
    test = df.loc[test_idx]
    X_test = test[MODEL_FEATURES]
    y_test = test[config.TARGET_COL].astype(int)
    return X_test, y_test, meta


def evaluate():
    model = joblib.load(config.MODEL_FILE)
    X_test, y_test, meta = _load_test_split()

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    auc = roc_auc_score(y_test, proba)

    # --- Confusion matrix figure --- #
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ConfusionMatrixDisplay(
        cm, display_labels=["Failed (0)", "Completed (1)"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {meta['best_model']}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    # --- ROC curve --- #
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {meta['best_model']}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "roc_curve.png", dpi=130)
    plt.close(fig)

    # --- Precision-Recall curve --- #
    prec, rec, _ = precision_recall_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(rec, prec)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {meta['best_model']}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "precision_recall.png", dpi=130)
    plt.close(fig)

    report_txt = classification_report(
        y_test, pred, target_names=["Failed (0)", "Completed (1)"], digits=3)

    comparison = pd.read_csv(config.METRICS_FILE)

    # Cost-sensitive framing: in this labeling a deal that fails but is
    # predicted to complete is a FALSE POSITIVE (predicting completion = 1).
    n = len(y_test)
    md = []
    md.append("# Evaluation Report\n")
    md.append("> Probability-based decision-support tool. **Not** a guarantee "
              "of any individual deal outcome.\n")
    md.append(f"**Best model:** {meta['best_model']}  ")
    md.append(f"**Test-set size:** {n}  ")
    md.append(f"**Class balance (completed):** {meta['class_balance_completed']:.1%}\n")

    md.append("## Model comparison (ranked by test ROC-AUC)\n")
    md.append(comparison.round(3).to_markdown(index=False))
    md.append("")

    md.append("## Best-model test metrics\n")
    md.append("```\n" + report_txt + "\n```")
    md.append(f"\n**ROC-AUC:** {auc:.3f}\n")

    md.append("## Confusion matrix & error analysis\n")
    md.append("|                       | Predicted Failed | Predicted Completed |")
    md.append("|-----------------------|------------------|---------------------|")
    md.append(f"| **Actually Failed**   | {tn} (TN)        | {fp} (FP)           |")
    md.append(f"| **Actually Completed**| {fn} (FN)        | {tp} (TP)           |")
    md.append("")
    fp_rate = fp / (fp + tn) if (fp + tn) else 0
    fn_rate = fn / (fn + tp) if (fn + tp) else 0
    md.append(f"- **False Positives (predict *complete*, deal actually fails): "
              f"{fp}** — rate {fp_rate:.1%} of truly-failed deals. "
              "This is the costliest error: a user could rely on a deal that "
              "ultimately collapses.")
    md.append(f"- **False Negatives (predict *fail*, deal actually completes): "
              f"{fn}** — rate {fn_rate:.1%} of truly-completed deals.")
    md.append("")
    md.append("Because false positives are the expensive error, the prediction "
              "interface lets you raise the completion-probability threshold "
              "above 0.5 to trade recall for precision on the 'will complete' "
              "call.\n")

    md.append("## Figures\n")
    md.append("- `figures/confusion_matrix.png`")
    md.append("- `figures/roc_curve.png`")
    md.append("- `figures/precision_recall.png`")
    md.append("- `figures/feature_importance.png` (from explainability stage)")
    md.append("- `figures/election_cancellation_rates.png`")
    md.append("- `figures/shap_summary.png` (if SHAP available)")

    config.EVAL_REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote evaluation report -> {config.EVAL_REPORT}")
    print(f"Test ROC-AUC={auc:.3f}  FP={fp}  FN={fn}")
    return auc, cm


def main():
    evaluate()


if __name__ == "__main__":
    main()
