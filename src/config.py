"""
Central configuration for the M&A merger-outcome prediction project.

Holds shared paths, U.S. presidential-election reference data, and column
definitions so every stage of the pipeline (data generation, cleaning,
feature engineering, training, evaluation, explainability, prediction) reads
from one source of truth.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV = DATA_RAW / "ma_deals_raw.csv"
TEMPLATE_CSV = DATA_RAW / "ma_deals_template.csv"
CLEAN_CSV = DATA_PROCESSED / "ma_deals_clean.csv"
FEATURES_CSV = DATA_PROCESSED / "ma_deals_features.csv"
DATA_DICTIONARY = DATA_PROCESSED / "data_dictionary.md"

MODEL_FILE = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.json"
METRICS_FILE = REPORTS_DIR / "model_comparison.csv"
EVAL_REPORT = REPORTS_DIR / "evaluation_report.md"

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --------------------------------------------------------------------------- #
# U.S. presidential-election reference data
# --------------------------------------------------------------------------- #
# Presidential elections happen every 4 years on the first Tuesday after the
# first Monday of November. We approximate election day as Nov 5 for distance
# calculations - good enough for a months-to/from-election feature.
ELECTION_YEARS = [1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024, 2028]

# Years that mark the first full calendar year of a new administration
# (i.e. the year after an election that produced a change of party, OR any
# first year of a presidential term). Inauguration is Jan 20 of the year
# following the election.
INAUGURATION_YEARS = [y + 1 for y in ELECTION_YEARS]  # e.g. 2001, 2005, ...

# Elections that produced a change of the party controlling the White House.
# Used to flag genuine administration "transitions" vs. an incumbent re-election.
PARTY_CHANGE_ELECTIONS = [2000, 2008, 2016, 2020, 2024]

ELECTION_MONTH = 11
ELECTION_DAY = 5

# Window (in months) on either side of election day considered the
# "transition period" (heightened policy/regulatory uncertainty).
TRANSITION_WINDOW_MONTHS = 6

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
# Raw columns expected from a real M&A data export (e.g. SDC Platinum,
# Refinitiv, Bloomberg, Capital IQ). The synthetic generator mirrors this.
RAW_COLUMNS = [
    "deal_id",
    "announcement_date",
    "resolution_date",          # completion OR cancellation date
    "target_name",
    "acquirer_name",
    "deal_value_musd",          # deal value in millions USD
    "industry",
    "sector",
    "target_country",
    "acquirer_country",
    "deal_status",              # completed / withdrawn / cancelled / failed / pending
    "payment_method",           # cash / stock / mixed / unknown
    "target_public",            # public / private
    "acquirer_public",          # public / private
    "regulatory_review",        # none / standard / extended / blocked
    "rumor_or_leak",            # yes / no / unknown
    "source_notes",
]

# Status values that map to each side of the binary target.
COMPLETED_STATUSES = {"completed", "closed", "successful"}
FAILED_STATUSES = {"withdrawn", "cancelled", "canceled", "failed", "terminated"}

TARGET_COL = "deal_completed"  # 1 = completed, 0 = failed/cancelled/withdrawn
