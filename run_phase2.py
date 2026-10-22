"""
run_phase2.py
=============
Entry point for Milestone 2 (Phase 2) of the TED Talks Recommendation System.

Executes Module 3: Feature Engineering and Vectorization.

Usage
-----
    python run_phase2.py                    # full pipeline (TF-IDF + Word2Vec)
    python run_phase2.py --no-word2vec      # skip Word2Vec (faster)
    python run_phase2.py --n-users 200      # set number of simulated users
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from feature_engineering import run_feature_engineering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 – Feature Engineering & Vectorization"
    )
    parser.add_argument(
        "--no-word2vec", action="store_true",
        help="Skip Word2Vec training (TF-IDF only)."
    )
    parser.add_argument(
        "--n-users", type=int, default=100,
        help="Number of simulated users for the user-item matrix."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    print("=" * 60)
    print("  TED Talks Recommendation System – Phase 2 (Milestone 2)")
    print("  Module 3: Feature Engineering & Vectorization")
    print("=" * 60)

    run_feature_engineering(
        df=None,
        use_word2vec=not args.no_word2vec,
        n_users=args.n_users,
    )

    elapsed = time.time() - t0
    print(f"\nPhase 2 complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
