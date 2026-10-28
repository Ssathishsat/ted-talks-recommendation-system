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

    # ── 9. Encode speaker (for collaborative filtering) ─────────────────────
    speaker_col = _find_col(df, ["speaker_name", "speaker", "main_speaker"])
    if speaker_col:
        df["speaker_encoded"] = df[speaker_col].astype("category").cat.codes
        print(f"[PREPROCESS] Speaker encoded: "
              f"{df[speaker_col].nunique()} unique speakers.")

    # ── 10. Parse ratings dictionary ────────────────────────────────────────
    ratings_col = _find_col(df, ["ratings"])
    if ratings_col:
        try:
            import ast
            df["_ratings_parsed"] = df[ratings_col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else {}
            )
            # Extract top rating category and its count
            def _top_rating(ratings_dict):
                if not ratings_dict or not isinstance(ratings_dict, dict):
                    return ("unknown", 0)
                top = max(ratings_dict.items(), key=lambda x: x[1])
                return top
            rating_results = df["_ratings_parsed"].apply(_top_rating)
            df["top_rating_category"] = rating_results.apply(lambda x: x[0])
            df["top_rating_count"] = rating_results.apply(lambda x: x[1])
            df.drop(columns=["_ratings_parsed"], inplace=True)
            print(f"[PREPROCESS] Ratings parsed: "
                  f"{df['top_rating_category'].nunique()} unique categories.")
        except Exception:
            print("[PREPROCESS] Could not parse ratings column — skipping.")

    # ── 11. Encode event ────────────────────────────────────────────────────
    event_col = _find_col(df, ["event"])
    if event_col:
        df["event_encoded"] = df[event_col].astype("category").cat.codes
        print(f"[PREPROCESS] Event encoded: "
              f"{df[event_col].nunique()} unique events.")

    # ── 12. Language encoding ──────────────────────────────────────────────
    lang_col = _find_col(df, ["language"])
    if lang_col:
        df["language_encoded"] = df[lang_col].astype("category").cat.codes
        print(f"[PREPROCESS] Language encoded: "
              f"{df[lang_col].nunique()} unique languages.")

    # ── 13. Popularity score (weighted combination) ─────────────────────────
    if "views" in df.columns and "average_rating" in df.columns:
        # Normalize views to 0-1
        views_norm = (df["views"] - df["views"].min()) / (df["views"].max() - df["views"].min() + 1)
        rating_norm = (df["average_rating"] - df["average_rating"].min()) / (df["average_rating"].max() - df["average_rating"].min() + 1)
        df["popularity_score"] = (0.6 * views_norm + 0.4 * rating_norm).round(4)
        print(f"[PREPROCESS] Popularity score created (60% views + 40% rating).")

    # ── 14. Engagement ratio ───────────────────────────────────────────────
    if "comments" in df.columns and "views" in df.columns:
        df["engagement_ratio"] = (df["comments"] / (df["views"] + 1)).round(6)
        print(f"[PREPROCESS] Engagement ratio created (comments / views).")

    # ── 15. Description sentiment (simple polarity) ────────────────────────
    if "description_clean" in df.columns:
        positive_words = {"remarkable", "amazing", "inspiring", "powerful",
                         "transformative", "innovative", "fascinating",
                         "breakthrough", "revolutionary", "incredible",
                         "extraordinary", "brilliant", "pioneering"}
        negative_words = {"challenge", "problem", "crisis", "concerning",
                         "difficult", "struggling", "failing", "broken"}
        def _simple_sentiment(text):
            if not text:
                return 0.0
            tokens = set(text.split())
            pos = len(tokens & positive_words)
            neg = len(tokens & negative_words)
            return round((pos - neg) / max(len(tokens), 1), 4)
        df["description_sentiment"] = df["description_clean"].apply(_simple_sentiment)
        print(f"[PREPROCESS] Description sentiment created.")

    # ── 16. Transcript diversity (unique words / total words) ──────────────
    if "transcript_clean" in df.columns:
        def _lexical_diversity(text):
            if not text:
                return 0.0
            tokens = text.split()
            return round(len(set(tokens)) / max(len(tokens), 1), 4)
        df["lexical_diversity"] = df["transcript_clean"].apply(_lexical_diversity)
        print(f"[PREPROCESS] Lexical diversity created (unique/total word ratio).")

    # ── 17. Content richness (combined text length) ───────────────────────
    if "combined_text" in df.columns:
        df["content_length"] = df["combined_text"].apply(
            lambda t: len(t.split()) if t else 0
        )
        print(f"[PREPROCESS] Content length created (total words in combined text).")

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
