"""
Data cleaning stage.

Reads the raw M&A export, then:
  * removes exact duplicate deals,
  * standardizes the date columns,
  * normalizes the messy `deal_status` text and derives the binary target,
  * drops deals that are still pending (no resolved outcome) or have
    impossible dates,
  * imputes obvious missing categorical flags,
  * leaves numeric imputation/scaling to the modeling pipeline so we never
    leak test-set statistics into training.

Run:  python -m src.data_cleaning
"""
from __future__ import annotations

import pandas as pd

from src import config


def _normalize_status(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.lower()
        .replace({"canceled": "cancelled"})
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) De-duplicate on deal_id (keep first) then on full-row duplicates.
    before = len(df)
    df = df.drop_duplicates(subset="deal_id", keep="first")
    df = df.drop_duplicates(keep="first")
    removed = before - len(df)

    # 2) Standardize dates.
    for col in ("announcement_date", "resolution_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # 3) Normalize status text and map to the binary target.
    status = _normalize_status(df["deal_status"])
    df["deal_status"] = status

    completed = status.isin(config.COMPLETED_STATUSES)
    failed = status.isin(config.FAILED_STATUSES)

    # Keep only resolved deals (drop pending/unknown).
    resolved = completed | failed
    df = df.loc[resolved].copy()
    df[config.TARGET_COL] = completed[resolved].astype(int).values

    # 4) Date sanity: resolution must be on/after announcement.
    bad_dates = df["resolution_date"] < df["announcement_date"]
    df = df.loc[~bad_dates].copy()
    df = df.dropna(subset=["announcement_date", "resolution_date"])

    # 5) Impute obvious categorical gaps.
    df["payment_method"] = df["payment_method"].fillna("unknown")
    df["rumor_or_leak"] = df["rumor_or_leak"].fillna("unknown")
    df["regulatory_review"] = df.get("regulatory_review", "none").fillna("none")
    for col in ("target_public", "acquirer_public"):
        df[col] = df[col].fillna("unknown")

    print(f"Cleaning: removed {removed} duplicates, "
          f"{(~resolved).sum()} unresolved, {int(bad_dates.sum())} bad-date rows.")
    print(f"Clean dataset: {len(df):,} rows | "
          f"completion rate {df[config.TARGET_COL].mean():.1%}")
    return df


def main() -> None:
    raw = pd.read_csv(config.RAW_CSV)
    clean_df = clean(raw)
    clean_df.to_csv(config.CLEAN_CSV, index=False)
    print(f"Wrote -> {config.CLEAN_CSV}")


if __name__ == "__main__":
    main()
