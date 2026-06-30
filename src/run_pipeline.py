"""
One-command, end-to-end pipeline runner.

    python -m src.run_pipeline               # full run with defaults
    python -m src.run_pipeline --n 6000      # larger synthetic dataset
    python -m src.run_pipeline --skip-generate   # reuse existing raw data

Stages: generate -> clean -> features -> data dictionary -> train ->
evaluate -> explain -> final summary.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src import data_cleaning, evaluate, explain, feature_engineering, generate_data
from src.make_data_dictionary import write_dictionary
from src.summary import write_final_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("STAGE 1/8  Generate raw data")
    print("=" * 70)
    if not args.skip_generate:
        df = generate_data.generate(args.n)
        df.to_csv(config.RAW_CSV, index=False)
        generate_data.write_template()
        print(f"  wrote {len(df):,} rows")
    else:
        print("  skipped (reusing existing raw CSV)")

    print("\n" + "=" * 70)
    print("STAGE 2/8  Clean data")
    print("=" * 70)
    clean_df = data_cleaning.clean(pd.read_csv(config.RAW_CSV))
    clean_df.to_csv(config.CLEAN_CSV, index=False)

    print("\n" + "=" * 70)
    print("STAGE 3/8  Feature engineering")
    print("=" * 70)
    feats = feature_engineering.build_features(clean_df)
    feats.to_csv(config.FEATURES_CSV, index=False)
    print(f"  {len(feature_engineering.MODEL_FEATURES)} features, "
          f"{len(feats):,} deals")

    print("\n" + "=" * 70)
    print("STAGE 4/8  Data dictionary")
    print("=" * 70)
    write_dictionary()

    print("\n" + "=" * 70)
    print("STAGE 5/8  Train & compare models")
    print("=" * 70)
    from src import train
    train.train_and_compare()

    print("\n" + "=" * 70)
    print("STAGE 6/8  Evaluate best model")
    print("=" * 70)
    evaluate.evaluate()

    print("\n" + "=" * 70)
    print("STAGE 7/8  Explainability")
    print("=" * 70)
    explain.main()

    print("\n" + "=" * 70)
    print("STAGE 8/8  Final summary")
    print("=" * 70)
    write_final_summary()

    print("\nDONE. See the reports/ folder for outputs.")


if __name__ == "__main__":
    main()
