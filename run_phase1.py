"""
run_phase1.py
=============
Entry point for Milestone 1 (Phase 1) of the TED Talks Recommendation System.

Executes:
  1. Module 1 – Data Collection
  2. Module 2 – EDA + Preprocessing

Usage
-----
    python run_phase1.py                 # full pipeline (synthetic fallback)
    python run_phase1.py --synthetic     # force synthetic dataset
    python run_phase1.py --no-plots      # skip plot generation
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_collection import load_or_collect_data
from preprocessing import run_preprocessing, generate_summary
from eda import run_eda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 – TED Talks Recommendation System"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Force the synthetic dataset (skip Kaggle / local CSV)."
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip EDA plot generation (report still generated)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    print("=" * 60)
    print("  TED Talks Recommendation System – Phase 1 (Milestone 1)")
    print("=" * 60)

    # ── Module 1: Data Collection ────────────────────────────────────────────
    print("\n>>> MODULE 1: DATA COLLECTION\n")
    raw_df = load_or_collect_data(force_synthetic=args.synthetic)
    print(f"    Dataset shape : {raw_df.shape}")
    print(f"    Columns       : {list(raw_df.columns)}")

    # ── Module 2: Preprocessing ──────────────────────────────────────────────
    print("\n>>> MODULE 2: PREPROCESSING\n")
    processed_df = run_preprocessing(raw_df, save=True)

    print("\n── Data Quality Summary (Processed) ──")
    summary = generate_summary(processed_df)
    print(summary.to_string())

    # ── Module 2: EDA ────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n>>> MODULE 2: EDA & VISUALISATIONS\n")
        run_eda(processed_df, raw_df=raw_df)
    else:
        print("\n[INFO] Plot generation skipped (--no-plots flag set).")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Phase 1 complete in {elapsed:.1f}s")
    print(f"  Processed data  → data/ted_talks_processed.csv")
    print(f"  Plots           → outputs/plots/")
    print(f"  EDA report      → outputs/reports/eda_report.txt")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
