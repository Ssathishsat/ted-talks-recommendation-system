"""
Module 2 – Part A: Data Preprocessing
---------------------------------------
Cleans text data, handles missing values, encodes categorical features,
and prepares the dataset for feature engineering in Module 3.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── NLTK lazy-download ──────────────────────────────────────────────────────
import nltk

for _pkg in ("stopwords", "punkt", "wordnet", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg
                       else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_PATH = DATA_DIR / "ted_talks_processed.csv"


# ────────────────────────────────────────────────────────────────────────────
# Text helpers
# ────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lowercase, remove special characters, stopwords, and lemmatize.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Cleaned text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def parse_tags(raw: object) -> list[str]:
    """
    Normalise a tags value into a clean Python list.

    Handles:
    - Comma-separated strings: "ai, health, future"
    - Python-list-like strings: "['ai', 'health']"
    - Actual Python lists (pass-through)
    - NaN / None → empty list

    Parameters
    ----------
    raw : object

    Returns
    -------
    list[str]
    """
    if isinstance(raw, list):
        return [str(t).strip().lower() for t in raw if str(t).strip()]

    if not isinstance(raw, str) or not raw.strip():
        return []

    # Strip Python-list brackets and quotes
    cleaned = re.sub(r"[\[\]'\"]", "", raw)
    tags = [t.strip().lower() for t in cleaned.split(",") if t.strip()]
    return tags


# ────────────────────────────────────────────────────────────────────────────
# Core preprocessing pipeline
# ────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Steps
    -----
    1.  Column normalisation  – lower-case, strip column names.
    2.  Duplicate removal.
    3.  Type coercions        – numeric columns, date parsing.
    4.  Missing-value handling.
    5.  Text cleaning         – title, description, transcript.
    6.  Tag normalisation.
    7.  Derived features      – combined_text, year, tag_count.
    8.  Categorical encoding  – topic label-encoded.

    Parameters
    ----------
    df : pd.DataFrame
        Raw TED Talks DataFrame.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    df = df.copy()

    # ── 1. Normalise column names ────────────────────────────────────────────
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print(f"[PREPROCESS] Columns: {list(df.columns)}")

    # ── 2. Remove duplicates ─────────────────────────────────────────────────
    before = len(df)
    title_col = _find_col(df, ["title", "name"])
    if title_col:
        df.drop_duplicates(subset=[title_col], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[PREPROCESS] Duplicates removed: {before - len(df)} rows. "
          f"Remaining: {len(df)}")

    # ── 3. Type coercions ────────────────────────────────────────────────────
    for col in ("views", "num_ratings", "duration_minutes", "talk_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "average_rating" in df.columns:
        df["average_rating"] = pd.to_numeric(df["average_rating"],
                                              errors="coerce")

    date_col = _find_col(df, ["film_date", "date", "published_date"])
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["year"] = df[date_col].dt.year
    elif "year" not in df.columns:
        df["year"] = np.nan

    # ── 4. Handle missing values ─────────────────────────────────────────────
    # Numeric imputation with median
    for col in ("views", "num_ratings", "duration_minutes", "average_rating"):
        if col in df.columns:
            median_val = df[col].median()
            missing = df[col].isna().sum()
            df[col].fillna(median_val, inplace=True)
            if missing:
                print(f"[PREPROCESS] '{col}': filled {missing} NaN → "
                      f"median={median_val:.2f}")

    # Text columns: fill NaN with empty string before cleaning
    for col in ("description", "transcript", "tags"):
        if col in df.columns:
            missing = df[col].isna().sum()
            df[col].fillna("", inplace=True)
            if missing:
                print(f"[PREPROCESS] '{col}': filled {missing} missing values.")

    # Speaker
    speaker_col = _find_col(df, ["speaker_name", "speaker", "main_speaker"])
    if speaker_col:
        df[speaker_col].fillna("Unknown", inplace=True)

    # ── 5. Clean text fields ─────────────────────────────────────────────────
    print("[PREPROCESS] Cleaning text fields …")

    if title_col:
        df["title_clean"] = df[title_col].apply(clean_text)

    desc_col = _find_col(df, ["description", "headline", "about"])
    if desc_col:
        df["description_clean"] = df[desc_col].apply(clean_text)

    if "transcript" in df.columns:
        df["transcript_clean"] = df["transcript"].apply(clean_text)
        print("[PREPROCESS] Transcripts cleaned.")

    # ── 6. Normalise tags ────────────────────────────────────────────────────
    if "tags" in df.columns:
        df["tags_list"] = df["tags"].apply(parse_tags)
        df["tags_clean"] = df["tags_list"].apply(lambda lst: " ".join(lst))
        df["tag_count"] = df["tags_list"].apply(len)
        print(f"[PREPROCESS] Average tag count: "
              f"{df['tag_count'].mean():.1f}")

    # ── 7. Combined text for vectorisation ───────────────────────────────────
    parts = []
    for col in ("title_clean", "description_clean",
                "tags_clean", "transcript_clean"):
        if col in df.columns:
            parts.append(df[col].fillna(""))
    if parts:
        df["combined_text"] = parts[0]
        for p in parts[1:]:
            df["combined_text"] = df["combined_text"] + " " + p
        df["combined_text"] = df["combined_text"].str.strip()

    # Word count of transcript
    if "transcript_clean" in df.columns:
        df["transcript_word_count"] = (
            df["transcript_clean"].apply(lambda t: len(t.split()) if t else 0)
        )

    # ── 8. Encode topic ──────────────────────────────────────────────────────
    topic_col = _find_col(df, ["topic", "category"])
    if topic_col:
        df["topic_encoded"] = df[topic_col].astype("category").cat.codes
        print(f"[PREPROCESS] '{topic_col}' encoded: "
              f"{df[topic_col].nunique()} unique categories.")

    print(f"[PREPROCESS] ✓ Done. Final shape: {df.shape}")
    return df


# ────────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ────────────────────────────────────────────────────────────────────────────

def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with per-column quality statistics.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Summary with columns: dtype, non_null, missing, missing_pct,
        n_unique, mean, std, min, max.
    """
    records = []
    for col in df.columns:
        s = df[col]
        try:
            n_unique = s.nunique()
        except TypeError:
            n_unique = "N/A"

        rec = {
            "column": col,
            "dtype": str(s.dtype),
            "non_null": s.notna().sum(),
            "missing": s.isna().sum(),
            "missing_pct": round(s.isna().mean() * 100, 2),
            "n_unique": n_unique,
        }

        if pd.api.types.is_numeric_dtype(s):
            rec.update({
                "mean": round(s.mean(), 3),
                "std": round(s.std(), 3),
                "min": s.min(),
                "max": s.max(),
            })
        records.append(rec)

    return pd.DataFrame(records).set_index("column")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def run_preprocessing(df: pd.DataFrame,
                      save: bool = True) -> pd.DataFrame:
    """
    Preprocess the raw DataFrame and optionally persist the result.

    Parameters
    ----------
    df : pd.DataFrame
    save : bool

    Returns
    -------
    pd.DataFrame
    """
    processed = preprocess(df)

    if save:
        processed.to_csv(PROCESSED_PATH, index=False)
        print(f"[PREPROCESS] Processed dataset saved → {PROCESSED_PATH}")

    return processed


if __name__ == "__main__":
    from data_collection import load_or_collect_data

    raw = load_or_collect_data()
    processed = run_preprocessing(raw)

    summary = generate_summary(processed)
    print("\n── Data Quality Summary ──")
    print(summary.to_string())
