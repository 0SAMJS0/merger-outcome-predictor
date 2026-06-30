# Machine Learning Analysis of Factors Influencing Merger Cancellations

A supervised machine-learning project that predicts whether a proposed merger or
acquisition will **complete** or **be cancelled/withdrawn/failed**, and measures
how much (if at all) the **U.S. presidential-election cycle** influences merger
failure relative to business and financial fundamentals.

> ⚠️ **This is a probability-based decision-support tool, not a crystal ball.**
> It estimates the *probability* of an outcome. It does **not** claim perfect or
> guaranteed prediction of any individual deal, and should be used alongside
> legal, financial, and regulatory due diligence — never as a sole basis for a
> decision.

---

## What it does

- Builds a clean, documented M&A modeling dataset (bring your own data, or use
  the included realistic **synthetic** generator).
- Engineers election-cycle features (election year, transition window,
  new-president year, months-to/after election) alongside deal-economics and
  structural features.
- Trains and compares six models: Logistic Regression, Decision Tree, Random
  Forest, Gradient Boosting, XGBoost, and a KNN baseline.
- Evaluates with train/test split, 5-fold cross-validation, accuracy,
  precision, recall, F1, ROC-AUC, and a confusion matrix — with explicit
  attention to **false positives** (predicting a doomed deal will complete).
- Explains predictions with permutation importance and SHAP, and directly
  compares **election-related vs. business/financial** feature importance.
- Ships a **prediction interface** to score a future deal and get the top
  factors behind the score.
- Writes a plain-language **final summary** answering: *do elections appear to
  influence merger failure?*

---

## Project structure

```
ML MERGER/
├── data/
│   ├── raw/          ma_deals_raw.csv  +  ma_deals_template.csv (bring-your-own)
│   └── processed/    ma_deals_clean.csv, ma_deals_features.csv, data_dictionary.md
├── notebooks/        ma_merger_analysis.ipynb  (end-to-end walkthrough)
├── src/
│   ├── config.py             paths, election reference data, schema
│   ├── generate_data.py      synthetic dataset generator + template
│   ├── data_cleaning.py      dedupe, date standardization, target creation
│   ├── feature_engineering.py election + deal + structural features
│   ├── model_pipeline.py     preprocessing ColumnTransformer + model zoo
│   ├── train.py              train, cross-validate, rank, persist best model
│   ├── evaluate.py           metrics, confusion matrix, ROC/PR curves, report
│   ├── explain.py            permutation importance, SHAP, election-vs-business
│   ├── predict.py            prediction interface (library + CLI)
│   ├── make_data_dictionary.py
│   ├── summary.py            final plain-language verdict
│   └── run_pipeline.py       one command to run everything
├── models/           best_model.joblib, model_metadata.json
├── reports/
│   ├── model_comparison.csv
│   ├── evaluation_report.md
│   ├── feature_importance.csv
│   ├── explainability_summary.json
│   ├── final_summary.md
│   └── figures/      confusion_matrix, roc_curve, precision_recall,
│                     feature_importance, election_vs_business,
│                     election_cancellation_rates, cancellation_rate_by_year,
│                     shap_summary
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the entire pipeline (generate → clean → features → train → evaluate → explain → summary)
python -m src.run_pipeline --n 5000

# 3. Score a proposed future merger
python -m src.predict --example
#   ...or with your own inputs:
python -m src.predict --deal-value 25000 --industry Technology \
    --sector Semiconductors --announcement-date 2028-03-01 \
    --target-country US --acquirer-country China --regulatory-review extended
```

Or open `notebooks/ma_merger_analysis.ipynb` and run top-to-bottom.

---

## Using your own data

The synthetic generator exists only so the project runs out of the box. To draw
real-world conclusions, replace the raw file with real deal data:

1. Export deals (e.g. from Refinitiv/SDC, Bloomberg, Capital IQ) into
   `data/raw/ma_deals_raw.csv` using the column names in
   `data/raw/ma_deals_template.csv` (see also `data/processed/data_dictionary.md`).
2. Run `python -m src.run_pipeline --skip-generate`.

Minimum useful columns: `announcement_date`, `resolution_date`,
`deal_value_musd`, `industry`, `sector`, `target_country`, `acquirer_country`,
`deal_status`, `payment_method`, `target_public`, `acquirer_public`. Optional
columns (`regulatory_review`, `rumor_or_leak`, `market_volatility_proxy`,
`interest_rate_proxy`) improve the model when present and degrade gracefully when
absent.

---

## Target variable

| `deal_completed` | Meaning |
|---|---|
| `1` | completed / closed / successful |
| `0` | withdrawn / cancelled / failed / terminated |

Pending/unresolved deals are dropped during cleaning.

## Key engineered features

- **Election-cycle:** `election_year`, `transition_year`, `new_president_year`,
  `months_to_election`, `months_after_election`
- **Deal economics:** `deal_value_log`, `payment_method`, `time_to_close_or_cancel`,
  `market_volatility_proxy`, `interest_rate_proxy`
- **Structure / risk:** `cross_border_deal`, `public_target`, `public_acquirer`,
  `regulatory_review`, `antitrust_risk_proxy`, `rumor_or_leak_indicator`

Full definitions live in `data/processed/data_dictionary.md`.

---

## Why false positives get special attention

With the labeling `completed = 1`, predicting **completion for a deal that
actually fails is a false positive** — the costlier error, because a user could
rely on a deal that ultimately collapses. The evaluation report breaks out FP vs.
FN counts and rates, and the prediction threshold can be raised above 0.5 to
trade recall for precision on the "will complete" call.

---

## Example result (synthetic run, n=5000)

| Model | Test ROC-AUC | F1 |
|---|---|---|
| Logistic Regression | ~0.88 | ~0.84 |
| Gradient Boosting | ~0.88 | ~0.86 |
| Random Forest / XGBoost | ~0.87 | ~0.86 |
| KNN (baseline) | ~0.83 | ~0.85 |

**Election vs. fundamentals:** election years showed a markedly higher *raw*
cancellation rate (~41% vs. ~31% in ordinary years), **but** election-cycle
features carried only ~1% of model importance once deal fundamentals and macro
conditions were controlled for — i.e. most of the headline election-year effect
is **confounded** with stressed-market periods. See `reports/final_summary.md`
for the full, caveated verdict (numbers vary slightly per random seed / dataset).

---

## Caveats & responsible use

- The bundled results use **synthetic** data; swap in real data before citing any
  finding.
- Correlation ≠ causation — election-year signals may proxy for macro conditions.
- `time_to_close_or_cancel` is unknown at announcement; the prediction interface
  defaults it to an expected timeline, so treat live scores accordingly.
- Probabilities are estimates. Use as **decision support**, not as a guarantee.
