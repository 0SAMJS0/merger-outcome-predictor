"""
Writes the data dictionary describing every raw column and engineered feature.
"""
from __future__ import annotations

from src import config

RAW_FIELDS = [
    ("deal_id", "string", "Unique deal identifier."),
    ("announcement_date", "date", "Date the deal was publicly announced."),
    ("resolution_date", "date", "Date the deal completed OR was cancelled/withdrawn."),
    ("target_name", "string", "Target (acquired) company name."),
    ("acquirer_name", "string", "Acquiring company name."),
    ("deal_value_musd", "float", "Deal value in millions of USD."),
    ("industry", "category", "High-level industry of the target."),
    ("sector", "category", "Finer-grained sector of the target."),
    ("target_country", "category", "Country of the target company."),
    ("acquirer_country", "category", "Country of the acquirer."),
    ("deal_status", "category",
     "completed / withdrawn / cancelled / failed / terminated / pending."),
    ("payment_method", "category", "cash / stock / mixed / unknown."),
    ("target_public", "category", "public / private (target listing status)."),
    ("acquirer_public", "category", "public / private (acquirer listing status)."),
    ("regulatory_review", "category", "none / standard / extended / blocked."),
    ("rumor_or_leak", "category", "yes / no / unknown — deal leaked/rumored pre-announcement."),
    ("source_notes", "string", "Free-text source / provenance note."),
    ("market_volatility_proxy", "float", "Equity-market volatility proxy at announcement (VIX-like)."),
    ("interest_rate_proxy", "float", "Benchmark interest-rate proxy at announcement (%)."),
]

TARGET = [
    (config.TARGET_COL, "binary",
     "1 = completed; 0 = failed/cancelled/withdrawn/terminated. Modeling target."),
]

ENGINEERED = [
    ("election_year", "binary", "1 if announcement year is a U.S. presidential election year."),
    ("transition_year", "binary",
     f"1 if announced within ±{config.TRANSITION_WINDOW_MONTHS} months of an election day."),
    ("new_president_year", "binary",
     "1 if announcement year is the first calendar year of a new administration (inauguration year)."),
    ("months_to_election", "float", "Months remaining until the next election (0 once past nearest)."),
    ("months_after_election", "float", "Months elapsed since the most recent election."),
    ("deal_value_log", "float", "log1p of deal_value_musd (tames right-skew)."),
    ("deal_value_missing", "binary", "1 if original deal value was missing (imputed)."),
    ("time_to_close_or_cancel", "int",
     "Days between announcement and resolution. NOTE: unknown at announcement for "
     "future deals; the prediction interface uses an expected-timeline default."),
    ("cross_border_deal", "binary", "1 if target and acquirer countries differ."),
    ("public_target", "binary", "1 if the target is publicly listed."),
    ("public_acquirer", "binary", "1 if the acquirer is publicly listed."),
    ("rumor_or_leak_indicator", "binary", "1 if the deal was rumored/leaked pre-announcement."),
    ("antitrust_risk_proxy", "float",
     "0-1 heuristic blending regulatory review level, sensitive sector, cross-border, and size."),
    ("industry", "category (one-hot)", "Target industry (encoded for modeling)."),
    ("sector", "category (one-hot)", "Target sector (encoded for modeling)."),
    ("payment_method", "category (one-hot)", "Consideration type (encoded)."),
    ("regulatory_review", "category (one-hot)", "Regulatory review intensity (encoded)."),
    ("market_volatility_proxy", "float", "Market volatility proxy (scaled in modeling)."),
    ("interest_rate_proxy", "float", "Interest-rate proxy (scaled in modeling)."),
]


def _table(rows):
    out = ["| Field | Type | Description |", "|---|---|---|"]
    out += [f"| `{n}` | {t} | {d} |" for n, t, d in rows]
    return "\n".join(out)


def write_dictionary() -> None:
    md = [
        "# Data Dictionary\n",
        "Schema for the M&A merger-outcome dataset. Bring your own data by "
        "matching the **raw** column names (see `data/raw/ma_deals_template.csv`).\n",
        "## Raw input columns\n", _table(RAW_FIELDS), "",
        "## Target variable\n", _table(TARGET), "",
        "## Engineered model features\n", _table(ENGINEERED), "",
        "### Election-cycle reference\n",
        f"- Election years: {config.ELECTION_YEARS}",
        f"- Inauguration (new-president) years: {config.INAUGURATION_YEARS}",
        f"- Party-change elections: {config.PARTY_CHANGE_ELECTIONS}",
        f"- Transition window: ±{config.TRANSITION_WINDOW_MONTHS} months around election day.",
    ]
    config.DATA_DICTIONARY.write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote data dictionary -> {config.DATA_DICTIONARY}")


if __name__ == "__main__":
    write_dictionary()
