"""
Feature engineering stage.

Turns the cleaned deal records into the modeling table, building the
election-cycle, deal-economics, and structural features described in the
project spec. Categorical encoding and numeric scaling are intentionally left
to the sklearn ColumnTransformer in `model_pipeline.py` so they are fit on the
training split only (no leakage).

Run:  python -m src.feature_engineering
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src import config


# Columns the model will actually consume (the ColumnTransformer references these).
NUMERIC_FEATURES = [
    "deal_value_log",
    "months_to_election",
    "months_after_election",
    "time_to_close_or_cancel",
    "market_volatility_proxy",
    "interest_rate_proxy",
    "antitrust_risk_proxy",
]
CATEGORICAL_FEATURES = [
    "industry",
    "sector",
    "payment_method",
    "regulatory_review",
]
BINARY_FEATURES = [
    "election_year",
    "transition_year",
    "new_president_year",
    "cross_border_deal",
    "public_target",
    "public_acquirer",
    "rumor_or_leak_indicator",
]
ELECTION_FEATURE_NAMES = [
    "election_year", "transition_year", "new_president_year",
    "months_to_election", "months_after_election",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def _signed_election_months(d: pd.Timestamp) -> float:
    """Signed months to the nearest election (negative = before election)."""
    best = None
    for y in config.ELECTION_YEARS:
        eday = date(y, config.ELECTION_MONTH, config.ELECTION_DAY)
        delta = (d.year - eday.year) * 12 + (d.month - eday.month)
        if best is None or abs(delta) < abs(best):
            best = delta
    return float(best)


def _antitrust_risk(row: pd.Series) -> float:
    """Heuristic 0-1 antitrust/regulatory-risk proxy from structural fields."""
    score = 0.0
    score += {"extended": 0.5, "standard": 0.25, "none": 0.0}.get(
        row.get("regulatory_review", "none"), 0.0)
    if row.get("sector") in {
        "Semiconductors", "Aerospace & Defense", "Banks", "Telecom",
        "Pharmaceuticals", "Oil & Gas", "Health Services",
    }:
        score += 0.25
    if row.get("cross_border_deal", 0) == 1:
        score += 0.15
    dv = row.get("deal_value_musd", 0) or 0
    if dv > 10_000:
        score += 0.20
    elif dv > 2_000:
        score += 0.10
    return float(min(score, 1.0))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["announcement_date"] = pd.to_datetime(df["announcement_date"])
    df["resolution_date"] = pd.to_datetime(df["resolution_date"])

    ann = df["announcement_date"]
    ann_year = ann.dt.year

    # --- Election-cycle features ------------------------------------------- #
    df["election_year"] = ann_year.isin(config.ELECTION_YEARS).astype(int)
    df["new_president_year"] = ann_year.isin(config.INAUGURATION_YEARS).astype(int)

    signed = ann.apply(_signed_election_months)
    # months_to_election: only counts the run-up (0 if past the nearest election)
    df["months_to_election"] = signed.apply(lambda m: float(-m) if m < 0 else 0.0)
    # months_after_election: only counts the aftermath
    df["months_after_election"] = signed.apply(lambda m: float(m) if m >= 0 else 0.0)
    # transition_year: within the +/- window of an election day
    df["transition_year"] = (signed.abs() <= config.TRANSITION_WINDOW_MONTHS).astype(int)

    # --- Deal-economics features ------------------------------------------- #
    dv = pd.to_numeric(df["deal_value_musd"], errors="coerce")
    # Impute missing deal value with the median before logging (structural,
    # not statistical-leaky: median taken over the full history is acceptable
    # here, but we also expose a missingness flag).
    df["deal_value_missing"] = dv.isna().astype(int)
    dv = dv.fillna(dv.median())
    df["deal_value_musd"] = dv
    df["deal_value_log"] = np.log1p(dv)

    df["time_to_close_or_cancel"] = (
        (df["resolution_date"] - df["announcement_date"]).dt.days.clip(lower=0)
    )

    # --- Structural / binary features -------------------------------------- #
    df["cross_border_deal"] = (
        df["target_country"].astype(str) != df["acquirer_country"].astype(str)
    ).astype(int)
    df["public_target"] = (df["target_public"].astype(str) == "public").astype(int)
    df["public_acquirer"] = (df["acquirer_public"].astype(str) == "public").astype(int)
    df["rumor_or_leak_indicator"] = (
        df["rumor_or_leak"].astype(str).str.lower() == "yes"
    ).astype(int)

    # --- Proxies ----------------------------------------------------------- #
    # Macro proxies: present in synthetic export; if a real export lacks them,
    # fill with sensible neutral defaults so the column always exists.
    if "market_volatility_proxy" not in df:
        df["market_volatility_proxy"] = 20.0
    if "interest_rate_proxy" not in df:
        df["interest_rate_proxy"] = 4.0
    df["market_volatility_proxy"] = pd.to_numeric(
        df["market_volatility_proxy"], errors="coerce").fillna(20.0)
    df["interest_rate_proxy"] = pd.to_numeric(
        df["interest_rate_proxy"], errors="coerce").fillna(4.0)

    df["antitrust_risk_proxy"] = df.apply(_antitrust_risk, axis=1)

    return df


def main() -> None:
    import pandas as pd
    clean_df = pd.read_csv(config.CLEAN_CSV)
    feats = build_features(clean_df)
    feats.to_csv(config.FEATURES_CSV, index=False)
    print(f"Built {len(MODEL_FEATURES)} model features for {len(feats):,} deals.")
    print(f"Wrote -> {config.FEATURES_CSV}")


if __name__ == "__main__":
    main()
