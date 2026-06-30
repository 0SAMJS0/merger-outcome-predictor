"""
Synthetic M&A dataset generator.

No proprietary M&A database (Refinitiv/SDC, Bloomberg, Capital IQ) ships with
this project, so we generate a realistic *synthetic* dataset that mirrors the
schema and statistical structure of real deal data. The generator embeds known
ground-truth relationships so the rest of the pipeline has signal to learn:

  * Business/financial factors dominate outcomes (deal size, payment method,
    antitrust/regulatory risk, cross-border friction, hostile/leaked deals).
  * Macro factors (market volatility, interest rates) add moderate signal.
  * Election-cycle factors add a *modest but real* effect, so the explainability
    stage can honestly compare "do elections matter vs. business fundamentals?"

This file also writes an empty template CSV and the data dictionary so a user
can drop in their own real data using the same column names.

Run:  python -m src.generate_data --n 4000
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src import config


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
INDUSTRIES = {
    "Technology": ["Software", "Semiconductors", "IT Services", "Hardware"],
    "Healthcare": ["Pharmaceuticals", "Biotech", "Medical Devices", "Health Services"],
    "Financials": ["Banks", "Insurance", "Asset Management", "Fintech"],
    "Energy": ["Oil & Gas", "Renewables", "Utilities"],
    "Industrials": ["Aerospace & Defense", "Machinery", "Transportation"],
    "Consumer": ["Retail", "Food & Beverage", "Media", "Telecom"],
    "Materials": ["Chemicals", "Metals & Mining", "Construction Materials"],
}
# Sectors that draw heavier antitrust / national-security scrutiny.
HIGH_SCRUTINY_SECTORS = {
    "Semiconductors", "Aerospace & Defense", "Banks", "Telecom",
    "Pharmaceuticals", "Oil & Gas", "Health Services",
}

COUNTRIES = ["US", "US", "US", "US", "UK", "Canada", "Germany",
             "France", "China", "Japan", "Australia", "India"]

PAYMENT_METHODS = ["cash", "stock", "mixed", "unknown"]


def _election_distance_months(d: date) -> float:
    """Signed months from the nearest election day (negative = before)."""
    best = None
    for y in config.ELECTION_YEARS:
        eday = date(y, config.ELECTION_MONTH, config.ELECTION_DAY)
        delta_months = (d.year - eday.year) * 12 + (d.month - eday.month)
        if best is None or abs(delta_months) < abs(best):
            best = delta_months
    return best


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
def generate(n: int = 4000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = date(1996, 1, 1)
    span_days = (date(2024, 12, 31) - start).days

    rows = []
    for i in range(n):
        ann = start + timedelta(days=int(rng.integers(0, span_days)))

        industry = rng.choice(list(INDUSTRIES.keys()))
        sector = rng.choice(INDUSTRIES[industry])

        # Deal value: heavy right-skew (log-normal), in $M.
        deal_value = float(np.round(rng.lognormal(mean=5.4, sigma=1.5), 1))  # ~$220M median
        deal_value = min(deal_value, 500_000.0)

        target_country = rng.choice(COUNTRIES)
        acquirer_country = rng.choice(COUNTRIES)
        cross_border = int(target_country != acquirer_country)

        payment = rng.choice(PAYMENT_METHODS, p=[0.45, 0.25, 0.22, 0.08])
        target_public = int(rng.random() < 0.62)
        acquirer_public = int(rng.random() < 0.78)
        rumor_leak = int(rng.random() < 0.16)  # leaked/rumored before announcement

        high_scrutiny = sector in HIGH_SCRUTINY_SECTORS
        big_deal = deal_value > 10_000  # > $10B mega-deals draw scrutiny

        # Regulatory review intensity grows with size + scrutiny + cross-border.
        reg_score = (
            0.6 * big_deal
            + 0.5 * high_scrutiny
            + 0.4 * cross_border
            + 0.3 * (deal_value > 2_000)
            + rng.normal(0, 0.3)
        )
        if reg_score > 1.4:
            reg_review = "extended"
        elif reg_score > 0.7:
            reg_review = "standard"
        elif reg_score > 0.2:
            reg_review = "standard"
        else:
            reg_review = "none"

        # Macro proxies driven by the announcement year.
        # Volatility spikes around 2000-2002, 2008-2009, 2020.
        yr = ann.year + ann.month / 12.0
        vol = (
            18
            + 12 * np.exp(-((yr - 2008.7) ** 2) / 1.2)   # GFC
            + 10 * np.exp(-((yr - 2020.2) ** 2) / 0.6)   # COVID
            + 8 * np.exp(-((yr - 2001.5) ** 2) / 1.5)    # dot-com
            + rng.normal(0, 2.5)
        )
        # Interest-rate proxy (10y-ish), secular decline then 2022+ rise.
        rate = (
            6.0
            - 0.13 * (ann.year - 1996)
            + 2.5 * max(0, ann.year - 2021)
            + rng.normal(0, 0.4)
        )
        rate = float(np.clip(rate, 0.3, 9.0))

        # Election-cycle context.
        dist = _election_distance_months(ann)
        is_election_year = int(ann.year in config.ELECTION_YEARS)
        # Transition period: within +/- window months of an election day.
        in_transition = int(abs(dist) <= config.TRANSITION_WINDOW_MONTHS)
        # Heightened-uncertainty bump, strongest right around the election and
        # for party-change cycles affecting large/cross-border/scrutiny deals.
        nearest_year = min(config.ELECTION_YEARS, key=lambda y: abs(ann.year - y))
        party_change = nearest_year in config.PARTY_CHANGE_ELECTIONS

        # --------------------------------------------------------------- #
        # Latent log-odds of FAILURE (higher => more likely to cancel).
        # Business/financial terms dominate; election terms are modest.
        # --------------------------------------------------------------- #
        fail_logodds = (
            -2.10                                   # base: most deals complete
            + 0.95 * (reg_review == "extended")     # regulatory friction (strong)
            + 0.35 * (reg_review == "standard")
            + 0.55 * high_scrutiny                   # antitrust-prone sectors
            + 0.45 * big_deal
            + 0.25 * np.log1p(deal_value) / 3.0      # size effect (smooth)
            + 0.40 * cross_border                    # deal friction
            + 0.50 * rumor_leak                      # leaked deals more fragile
            + 0.35 * (payment == "stock")            # stock deals riskier to close
            + 0.20 * (payment == "mixed")
            - 0.25 * acquirer_public                 # public acquirers more reliable
            + 0.030 * (vol - 20)                     # market stress (moderate)
            + 0.10 * (rate - 4)                      # financing cost (moderate)
            # --- election-cycle terms (deliberately modest) ---
            + 0.30 * in_transition
            + 0.18 * is_election_year
            + 0.22 * (in_transition and party_change and (big_deal or high_scrutiny))
            + rng.normal(0, 0.45)                    # irreducible noise
        )
        p_fail = float(_sigmoid(np.array(fail_logodds)))
        failed = rng.random() < p_fail

        # Time to resolution: failures resolve faster on average; extended
        # regulatory review and large deals take longer.
        base_days = rng.normal(160, 50)
        base_days += 90 * (reg_review == "extended") + 40 * (reg_review == "standard")
        base_days += 60 * big_deal + 30 * cross_border
        if failed:
            base_days *= rng.uniform(0.45, 0.9)
        days_to_resolve = int(np.clip(base_days, 12, 1000))
        resolution = ann + timedelta(days=days_to_resolve)

        if failed:
            status = rng.choice(["withdrawn", "cancelled", "failed", "terminated"],
                                 p=[0.5, 0.25, 0.15, 0.10])
        else:
            status = "completed"

        rows.append({
            "deal_id": f"DEAL{i:06d}",
            "announcement_date": ann.isoformat(),
            "resolution_date": resolution.isoformat(),
            "target_name": f"Target_{i:06d}",
            "acquirer_name": f"Acquirer_{rng.integers(0, n):06d}",
            "deal_value_musd": deal_value,
            "industry": industry,
            "sector": sector,
            "target_country": target_country,
            "acquirer_country": acquirer_country,
            "deal_status": status,
            "payment_method": payment,
            "target_public": "public" if target_public else "private",
            "acquirer_public": "public" if acquirer_public else "private",
            "regulatory_review": reg_review,
            "rumor_or_leak": "yes" if rumor_leak else "no",
            "source_notes": "synthetic",
            # macro proxies are part of the export in this synthetic set;
            # for real data they may be merged in during feature engineering.
            "market_volatility_proxy": float(np.round(vol, 2)),
            "interest_rate_proxy": rate,
        })

    df = pd.DataFrame(rows)

    # --- Inject realistic messiness so the cleaning stage has work to do --- #
    # 1) duplicate a handful of rows
    dupes = df.sample(frac=0.01, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    # 2) missing deal values
    miss_idx = rng.choice(df.index, size=int(0.04 * len(df)), replace=False)
    df.loc[miss_idx, "deal_value_musd"] = np.nan
    # 3) missing payment method / rumor flags
    for col, frac in [("payment_method", 0.05), ("rumor_or_leak", 0.07)]:
        idx = rng.choice(df.index, size=int(frac * len(df)), replace=False)
        df.loc[idx, col] = np.nan
    # 4) inconsistent status casing/spelling
    case_idx = rng.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[case_idx, "deal_status"] = df.loc[case_idx, "deal_status"].str.upper()

    return df


def write_template() -> None:
    """An empty CSV with the expected columns for users bringing real data."""
    cols = config.RAW_COLUMNS + ["market_volatility_proxy", "interest_rate_proxy"]
    pd.DataFrame(columns=cols).to_csv(config.TEMPLATE_CSV, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic M&A dataset")
    ap.add_argument("--n", type=int, default=4000, help="number of base deals")
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = ap.parse_args()

    df = generate(args.n, args.seed)
    df.to_csv(config.RAW_CSV, index=False)
    write_template()

    rate = 1 - df["deal_status"].str.lower().isin(config.COMPLETED_STATUSES).mean()
    print(f"Wrote {len(df):,} rows -> {config.RAW_CSV}")
    print(f"Overall failure/cancellation rate: {rate:.1%}")
    print(f"Template -> {config.TEMPLATE_CSV}")


if __name__ == "__main__":
    main()
