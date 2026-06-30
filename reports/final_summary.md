# Final Summary — Do Presidential Elections Influence Merger Failure?

> **Important:** This is a probability-based decision-support analysis on a modeled dataset. It does **not** claim perfect or guaranteed prediction of any individual deal.

## Headline finding

Election-cycle timing shows a **small but detectable** association with merger failure, clearly secondary to deal fundamentals.

## Evidence

- **Best model:** Logistic Regression (test ROC-AUC 0.879, F1 0.840).
- **Cancellation rate, ordinary years:** 30.5%
- **Cancellation rate, election years:** 41.3% (+10.9 percentage points vs. ordinary years).
- **Cancellation rate, transition window (±6mo):** 39.8%
- **Share of model importance from election-related features:** 1%
- **Share from business/financial/structural features:** 99%

## Interpretation

Deal fundamentals — regulatory/antitrust review intensity, deal size, cross-border friction, payment method, and whether the deal leaked — dominate the prediction.

Note the gap between two numbers: election years show a **+10.9 percentage-point** higher *raw* cancellation rate, yet election-cycle features carry only **1%** of the model's importance once everything else is controlled for. That gap is the key insight: much of the headline election-year effect is **confounded** — election years coincide with high-volatility, stressed macro periods (e.g. 2008, 2020), and once market volatility, interest rates, and deal fundamentals are in the model, the *independent* contribution of election timing is small. Election-cycle features add incremental signal mainly for large, cross-border, or high-scrutiny deals during transition windows, consistent with elevated regulatory/policy uncertainty — but they are not a primary driver on their own.

## Caveats

- Results here are computed on a **synthetic dataset** built to mirror real M&A structure; replace `data/raw/ma_deals_raw.csv` with real deal data (same schema) to draw real-world conclusions.
- Correlation is not causation — election-year effects may proxy for macro conditions (rates, volatility) that co-move with the cycle.
- `time_to_close_or_cancel` is unknown at announcement; treat live predictions accordingly.
- Use the model as **decision support**, alongside legal, financial, and regulatory due diligence — never as a sole basis for a decision.