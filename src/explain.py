"""
Explainability stage.

Generates:
  * permutation feature importance (model-agnostic, works for every model),
  * SHAP summary plot (if `shap` is installed and the model is supported),
  * a head-to-head comparison of *election-related* features vs.
    *business/financial* features (summed importance),
  * descriptive charts of cancellation rates in election years, transition
    years, new-president years, and ordinary years.

Run:  python -m src.explain
"""
from __future__ import annotations

import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from src import config
from src.feature_engineering import (
    ELECTION_FEATURE_NAMES,
    MODEL_FEATURES,
)


def _load_test_split():
    df = pd.read_csv(config.FEATURES_CSV)
    with open(config.METADATA_FILE) as f:
        meta = json.load(f)
    test = df.loc[meta["test_index"]]
    return df, test, meta


def permutation_importances(model, X_test, y_test) -> pd.DataFrame:
    result = permutation_importance(
        model, X_test, y_test, n_repeats=15,
        random_state=config.RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
    )
    imp = (
        pd.DataFrame({
            "feature": MODEL_FEATURES,
            "importance": result.importances_mean,
            "std": result.importances_std,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return imp


def plot_feature_importance(imp: pd.DataFrame):
    top = imp.head(15).iloc[::-1]
    colors = ["#d1495b" if f in ELECTION_FEATURE_NAMES else "#2e86ab"
              for f in top["feature"]]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color=colors)
    ax.set_xlabel("Permutation importance (drop in ROC-AUC)")
    ax.set_title("Top features  (red = election-related, blue = business/financial)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "feature_importance.png", dpi=130)
    plt.close(fig)


def election_vs_business(imp: pd.DataFrame) -> dict:
    imp = imp.copy()
    imp["importance"] = imp["importance"].clip(lower=0)
    is_election = imp["feature"].isin(ELECTION_FEATURE_NAMES)
    election_sum = float(imp.loc[is_election, "importance"].sum())
    business_sum = float(imp.loc[~is_election, "importance"].sum())
    total = election_sum + business_sum or 1.0

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Election-related", "Business/Financial"],
           [election_sum, business_sum], color=["#d1495b", "#2e86ab"])
    ax.set_ylabel("Summed permutation importance")
    ax.set_title("Election features vs. business/financial features")
    for i, v in enumerate([election_sum, business_sum]):
        ax.text(i, v, f"{v/total:.0%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "election_vs_business.png", dpi=130)
    plt.close(fig)

    return {
        "election_importance_share": election_sum / total,
        "business_importance_share": business_sum / total,
    }


def cancellation_rate_charts(df: pd.DataFrame) -> dict:
    """Descriptive cancellation rates by election-cycle context."""
    df = df.copy()
    df["cancelled"] = 1 - df[config.TARGET_COL]

    groups = {
        "Election year": df["election_year"] == 1,
        "Transition window": df["transition_year"] == 1,
        "New-president year": df["new_president_year"] == 1,
        "Ordinary year": (df["election_year"] == 0)
        & (df["transition_year"] == 0) & (df["new_president_year"] == 0),
    }
    rates = {k: float(df.loc[mask, "cancelled"].mean()) for k, mask in groups.items()}
    counts = {k: int(mask.sum()) for k, mask in groups.items()}

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(list(rates.keys()), [v * 100 for v in rates.values()],
                  color=["#d1495b", "#edae49", "#f4a261", "#2e86ab"])
    ax.set_ylabel("Cancellation rate (%)")
    ax.set_title("Deal cancellation rate by election-cycle context")
    for b, (k, v) in zip(bars, rates.items()):
        ax.text(b.get_x() + b.get_width() / 2, v * 100,
                f"{v:.1%}\n(n={counts[k]})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(v * 100 for v in rates.values()) * 1.25)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "election_cancellation_rates.png", dpi=130)
    plt.close(fig)

    # Cancellation rate by year (line) to visualize cycles.
    df["year"] = pd.to_datetime(df["announcement_date"]).dt.year
    by_year = df.groupby("year")["cancelled"].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(by_year.index, by_year.values * 100, marker="o", color="#2e86ab")
    for y in config.ELECTION_YEARS:
        if by_year.index.min() <= y <= by_year.index.max():
            ax.axvline(y, color="#d1495b", alpha=0.35, linestyle="--")
    ax.set_xlabel("Announcement year  (dashed red = U.S. presidential election)")
    ax.set_ylabel("Cancellation rate (%)")
    ax.set_title("Cancellation rate over time vs. election years")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "cancellation_rate_by_year.png", dpi=130)
    plt.close(fig)

    return {"cancellation_rates": rates, "counts": counts}


def shap_summary(model, X_test):
    """Best-effort SHAP summary; skips gracefully if unsupported."""
    try:
        import shap
    except Exception:
        print("SHAP not installed - skipping SHAP summary.")
        return False
    try:
        prep = model.named_steps["prep"]
        clf = model.named_steps["clf"]
        X_trans = prep.transform(X_test)
        feat_names = prep.get_feature_names_out()

        # Tree models -> fast TreeExplainer; otherwise sample + KernelExplainer.
        model_type = type(clf).__name__
        if model_type in {"RandomForestClassifier", "GradientBoostingClassifier",
                          "DecisionTreeClassifier", "XGBClassifier"}:
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_trans)
            if isinstance(sv, list):          # binary -> list of 2
                sv = sv[1]
            # newer shap returns (n, k, 2)
            if hasattr(sv, "ndim") and sv.ndim == 3:
                sv = sv[:, :, 1]
        else:
            sample = shap.sample(X_trans, 100, random_state=config.RANDOM_STATE)
            explainer = shap.KernelExplainer(
                lambda d: clf.predict_proba(d)[:, 1], sample)
            sv = explainer.shap_values(sample, nsamples=100)
            X_trans = sample

        shap.summary_plot(sv, X_trans, feature_names=feat_names, show=False,
                          max_display=15)
        plt.title("SHAP feature impact on completion probability")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "shap_summary.png", dpi=130,
                    bbox_inches="tight")
        plt.close()
        print("Wrote SHAP summary -> figures/shap_summary.png")
        return True
    except Exception as e:  # pragma: no cover
        print(f"SHAP summary failed ({e}); continuing.")
        return False


def main():
    df, test, meta = _load_test_split()
    model = joblib.load(config.MODEL_FILE)

    X_test = test[MODEL_FEATURES]
    y_test = test[config.TARGET_COL].astype(int)

    imp = permutation_importances(model, X_test, y_test)
    imp.to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(imp)

    shares = election_vs_business(imp)
    cancel = cancellation_rate_charts(df)
    shap_summary(model, X_test)

    summary = {
        "best_model": meta["best_model"],
        **shares,
        **cancel,
        "top_features": imp.head(10).to_dict(orient="records"),
    }
    with open(config.REPORTS_DIR / "explainability_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n--- Election vs. business/financial importance ---")
    print(f"  Election-related share : {shares['election_importance_share']:.1%}")
    print(f"  Business/financial share: {shares['business_importance_share']:.1%}")
    print("\n--- Cancellation rates ---")
    for k, v in cancel["cancellation_rates"].items():
        print(f"  {k:20s}: {v:.1%}  (n={cancel['counts'][k]})")
    print(f"\nExplainability summary -> "
          f"{config.REPORTS_DIR / 'explainability_summary.json'}")


if __name__ == "__main__":
    main()
