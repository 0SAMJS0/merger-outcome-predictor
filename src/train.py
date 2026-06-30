"""
Model training & comparison.

Trains the full model zoo (Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting, XGBoost, KNN baseline) with a stratified train/test split
and 5-fold cross-validation, ranks them by ROC-AUC, and persists the best
fitted pipeline plus a comparison table.

Run:  python -m src.train
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src import config
from src.feature_engineering import MODEL_FEATURES
from src.model_pipeline import build_models, make_pipeline

warnings.filterwarnings("ignore")


def load_xy():
    df = pd.read_csv(config.FEATURES_CSV)
    X = df[MODEL_FEATURES].copy()
    y = df[config.TARGET_COL].astype(int)
    return X, y


def train_and_compare():
    X, y = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)

    rows = []
    fitted = {}
    for name, model in build_models().items():
        pipe = make_pipeline(model)

        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)

        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        rows.append({
            "model": name,
            "cv_roc_auc_mean": cv_auc.mean(),
            "cv_roc_auc_std": cv_auc.std(),
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_recall": recall_score(y_test, pred, zero_division=0),
            "test_f1": f1_score(y_test, pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, proba),
        })
        fitted[name] = pipe
        print(f"{name:22s} CV-AUC={cv_auc.mean():.3f}±{cv_auc.std():.3f}  "
              f"test-AUC={rows[-1]['test_roc_auc']:.3f}  "
              f"F1={rows[-1]['test_f1']:.3f}")

    results = pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False)
    results.to_csv(config.METRICS_FILE, index=False)

    best_name = results.iloc[0]["model"]
    best_pipe = fitted[best_name]
    joblib.dump(best_pipe, config.MODEL_FILE)

    # Persist the train/test split indices reference + metadata.
    meta = {
        "best_model": best_name,
        "features": MODEL_FEATURES,
        "target": config.TARGET_COL,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_balance_completed": float(y.mean()),
        "test_metrics": results[results["model"] == best_name]
        .iloc[0].to_dict(),
        "test_index": X_test.index.tolist(),
    }
    with open(config.METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2, default=float)

    print(f"\nBest model: {best_name}  ->  {config.MODEL_FILE}")
    print(f"Comparison -> {config.METRICS_FILE}")
    return results, best_name


def main():
    train_and_compare()


if __name__ == "__main__":
    main()
