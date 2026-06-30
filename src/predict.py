"""
Prediction interface.

Score a *future / proposed* merger and get back:
  * predicted probability of completion,
  * predicted probability of cancellation,
  * the top factors pushing the prediction up or down (per-instance, via SHAP
    when available, else a model-importance fallback).

Use as a library:
    from src.predict import predict_deal
    predict_deal({...})

Or from the command line with a JSON file or inline flags:
    python -m src.predict --example
    python -m src.predict --json my_deal.json
    python -m src.predict --deal-value 25000 --industry Technology \\
        --sector Semiconductors --announcement-date 2028-03-01 \\
        --target-country US --acquirer-country China --regulatory-review extended

NOTE ON `time_to_close_or_cancel`: at announcement you don't yet know how long
a deal will take. The model includes this feature (per spec), but for a future
deal we default it to the dataset median via `expected_days_to_close`. Treat
the score as a probability-based decision aid, never a guarantee.
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src import config
from src.feature_engineering import (
    ELECTION_FEATURE_NAMES,
    MODEL_FEATURES,
    build_features,
)

# Sensible defaults so a user only has to supply what they know.
DEFAULTS = {
    "announcement_date": "2028-03-01",
    "deal_value_musd": 1000.0,
    "industry": "Technology",
    "sector": "Software",
    "target_country": "US",
    "acquirer_country": "US",
    "payment_method": "cash",
    "target_public": "public",
    "acquirer_public": "public",
    "regulatory_review": "standard",
    "rumor_or_leak": "no",
    "market_volatility_proxy": 20.0,
    "interest_rate_proxy": 4.0,
    "expected_days_to_close": 180,
}

EXAMPLE = {
    "announcement_date": "2028-03-15",   # election-year mega-deal
    "deal_value_musd": 42000.0,
    "industry": "Technology",
    "sector": "Semiconductors",
    "target_country": "US",
    "acquirer_country": "China",
    "payment_method": "stock",
    "target_public": "public",
    "acquirer_public": "public",
    "regulatory_review": "extended",
    "rumor_or_leak": "yes",
    "expected_days_to_close": 300,
}


def _to_feature_row(deal: dict) -> pd.DataFrame:
    d = {**DEFAULTS, **deal}
    ann = pd.to_datetime(d["announcement_date"])
    res = ann + pd.Timedelta(days=int(d["expected_days_to_close"]))
    raw = {
        "announcement_date": ann.isoformat(),
        "resolution_date": res.isoformat(),
        "deal_value_musd": float(d["deal_value_musd"]),
        "industry": d["industry"],
        "sector": d["sector"],
        "target_country": d["target_country"],
        "acquirer_country": d["acquirer_country"],
        "payment_method": d["payment_method"],
        "target_public": d["target_public"],
        "acquirer_public": d["acquirer_public"],
        "regulatory_review": d["regulatory_review"],
        "rumor_or_leak": d["rumor_or_leak"],
        "market_volatility_proxy": float(d["market_volatility_proxy"]),
        "interest_rate_proxy": float(d["interest_rate_proxy"]),
    }
    feats = build_features(pd.DataFrame([raw]))
    return feats[MODEL_FEATURES]


def _top_factors(model, X_row: pd.DataFrame, k: int = 6):
    """Per-instance contributions via SHAP; fall back to global importance."""
    try:
        import shap
        prep = model.named_steps["prep"]
        clf = model.named_steps["clf"]
        Xt = prep.transform(X_row)
        names = list(prep.get_feature_names_out())
        mt = type(clf).__name__
        if mt in {"RandomForestClassifier", "GradientBoostingClassifier",
                  "DecisionTreeClassifier", "XGBClassifier"}:
            expl = shap.TreeExplainer(clf)
            sv = expl.shap_values(Xt)
            if isinstance(sv, list):
                sv = sv[1]
            if hasattr(sv, "ndim") and sv.ndim == 3:
                sv = sv[:, :, 1]
            contrib = np.asarray(sv)[0]
            pairs = sorted(zip(names, contrib), key=lambda p: abs(p[1]),
                           reverse=True)[:k]
            return [
                {"feature": n,
                 "effect": "increases completion" if v > 0 else "increases cancellation",
                 "shap": float(v)}
                for n, v in pairs
            ]
        # Linear models: per-instance contribution = coef * transformed value.
        if hasattr(clf, "coef_"):
            contrib = clf.coef_[0] * np.asarray(Xt)[0]
            pairs = sorted(zip(names, contrib), key=lambda p: abs(p[1]),
                           reverse=True)[:k]
            return [
                {"feature": n,
                 "effect": "increases completion" if v > 0 else "increases cancellation",
                 "shap": float(v)}
                for n, v in pairs
            ]
    except Exception:
        pass

    # Fallback: report this deal's notable feature values ranked by global
    # permutation importance (if available).
    try:
        imp = pd.read_csv(config.REPORTS_DIR / "feature_importance.csv")
        top = imp.head(k)["feature"].tolist()
        return [{"feature": f, "value": _lookup(X_row, f)} for f in top]
    except Exception:
        return []


def _lookup(X_row, feat):
    return X_row.iloc[0][feat] if feat in X_row.columns else None


def predict_deal(deal: dict) -> dict:
    model = joblib.load(config.MODEL_FILE)
    X_row = _to_feature_row(deal)
    p_complete = float(model.predict_proba(X_row)[0, 1])
    result = {
        "probability_completion": round(p_complete, 4),
        "probability_cancellation": round(1 - p_complete, 4),
        "predicted_outcome": "likely COMPLETE" if p_complete >= 0.5
        else "likely CANCEL/FAIL",
        "top_factors": _top_factors(model, X_row),
        "disclaimer": "Probability-based decision support only. Not a guarantee "
                      "of any individual deal's outcome.",
    }
    return result


def _print(deal, res):
    print("\n=== Merger Outcome Prediction ===")
    print(f"Probability of COMPLETION : {res['probability_completion']:.1%}")
    print(f"Probability of CANCEL/FAIL: {res['probability_cancellation']:.1%}")
    print(f"Predicted outcome         : {res['predicted_outcome']}")
    print("\nTop influencing factors:")
    for f in res["top_factors"]:
        if "shap" in f:
            arrow = "-> completion" if f["shap"] > 0 else "-> cancellation"
            print(f"  - {f['feature']:35s} {arrow:16s} (contrib={f['shap']:+.3f})")
        else:
            print(f"  - {f['feature']:35s} value={f.get('value')}")
    print(f"\n{res['disclaimer']}")


def main():
    ap = argparse.ArgumentParser(description="Predict a merger outcome")
    ap.add_argument("--json", help="path to a JSON file describing the deal")
    ap.add_argument("--example", action="store_true",
                    help="score the built-in election-year mega-deal example")
    for key in DEFAULTS:
        ap.add_argument(f"--{key.replace('_', '-')}")
    args = ap.parse_args()

    if args.example:
        deal = EXAMPLE
    elif args.json:
        with open(args.json) as f:
            deal = json.load(f)
    else:
        deal = {k: getattr(args, k) for k in DEFAULTS
                if getattr(args, k) is not None}
        if not deal:
            print("No deal provided; using built-in example. "
                  "Use --json or flags, or --example.")
            deal = EXAMPLE

    res = predict_deal(deal)
    _print(deal, res)


if __name__ == "__main__":
    main()
