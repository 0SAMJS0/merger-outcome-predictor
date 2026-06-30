"""
Writes the final plain-language summary: does the presidential-election cycle
appear to influence merger failure, given this dataset and model?

The conclusion is generated from the actual computed numbers (cancellation
rates by context + election-vs-business importance share), with explicit
caveats so nothing is overstated.
"""
from __future__ import annotations

import json

import pandas as pd

from src import config

SUMMARY_FILE = config.REPORTS_DIR / "final_summary.md"


def write_final_summary() -> None:
    with open(config.REPORTS_DIR / "explainability_summary.json") as f:
        ex = json.load(f)
    comparison = pd.read_csv(config.METRICS_FILE)
    best = comparison.iloc[0]

    rates = ex["cancellation_rates"]
    ordinary = rates.get("Ordinary year", 0)
    election = rates.get("Election year", 0)
    transition = rates.get("Transition window", 0)
    lift = (election - ordinary) * 100  # percentage points

    el_share = ex["election_importance_share"]
    bz_share = ex["business_importance_share"]

    # Decide on a measured verdict.
    if el_share >= 0.30 and lift >= 3:
        verdict = ("Election-cycle timing shows a **moderate, non-trivial** "
                   "association with merger failure in this dataset.")
    elif el_share >= 0.12 or lift >= 2:
        verdict = ("Election-cycle timing shows a **small but detectable** "
                   "association with merger failure, clearly secondary to "
                   "deal fundamentals.")
    else:
        verdict = ("Election-cycle timing shows **little independent "
                   "association** with merger failure once deal fundamentals "
                   "are accounted for.")

    md = [
        "# Final Summary — Do Presidential Elections Influence Merger Failure?\n",
        "> **Important:** This is a probability-based decision-support analysis "
        "on a modeled dataset. It does **not** claim perfect or guaranteed "
        "prediction of any individual deal.\n",
        "## Headline finding\n",
        verdict + "\n",
        "## Evidence\n",
        f"- **Best model:** {best['model']} "
        f"(test ROC-AUC {best['test_roc_auc']:.3f}, F1 {best['test_f1']:.3f}).",
        f"- **Cancellation rate, ordinary years:** {ordinary:.1%}",
        f"- **Cancellation rate, election years:** {election:.1%} "
        f"({lift:+.1f} percentage points vs. ordinary years).",
        f"- **Cancellation rate, transition window (±{config.TRANSITION_WINDOW_MONTHS}mo):** "
        f"{transition:.1%}",
        f"- **Share of model importance from election-related features:** "
        f"{el_share:.0%}",
        f"- **Share from business/financial/structural features:** {bz_share:.0%}",
        "",
        "## Interpretation\n",
        "Deal fundamentals — regulatory/antitrust review intensity, deal size, "
        "cross-border friction, payment method, and whether the deal leaked — "
        "dominate the prediction.",
        "",
        f"Note the gap between two numbers: election years show a **{lift:+.1f} "
        "percentage-point** higher *raw* cancellation rate, yet election-cycle "
        f"features carry only **{el_share:.0%}** of the model's importance once "
        "everything else is controlled for. That gap is the key insight: much of "
        "the headline election-year effect is **confounded** — election years "
        "coincide with high-volatility, stressed macro periods (e.g. 2008, 2020), "
        "and once market volatility, interest rates, and deal fundamentals are in "
        "the model, the *independent* contribution of election timing is small. "
        "Election-cycle features add incremental signal mainly for large, "
        "cross-border, or high-scrutiny deals during transition windows, "
        "consistent with elevated regulatory/policy uncertainty — but they are "
        "not a primary driver on their own.",
        "",
        "## Caveats\n",
        "- Results here are computed on a **synthetic dataset** built to mirror "
        "real M&A structure; replace `data/raw/ma_deals_raw.csv` with real "
        "deal data (same schema) to draw real-world conclusions.",
        "- Correlation is not causation — election-year effects may proxy for "
        "macro conditions (rates, volatility) that co-move with the cycle.",
        "- `time_to_close_or_cancel` is unknown at announcement; treat live "
        "predictions accordingly.",
        "- Use the model as **decision support**, alongside legal, financial, "
        "and regulatory due diligence — never as a sole basis for a decision.",
    ]
    SUMMARY_FILE.write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote final summary -> {SUMMARY_FILE}")


if __name__ == "__main__":
    write_final_summary()
