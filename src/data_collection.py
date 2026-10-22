"""
Module 1: Data Collection
--------------------------
Handles downloading/loading the TED Talks dataset from Kaggle or a local CSV.
Falls back to generating a realistic synthetic dataset if neither is available,
so the pipeline can always run end-to-end for demonstration purposes.
"""

import os
import zipfile
import warnings
import random
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_TALKS_PATH = DATA_DIR / "ted_talks_raw.csv"


# ─────────────────────────────────────────────
# 1. Try Kaggle download
# ─────────────────────────────────────────────
def download_from_kaggle(dataset: str = "rounakbanik/ted-talks") -> bool:
    """
    Attempt to pull the dataset via the Kaggle API.

    Parameters
    ----------
    dataset : str
        Kaggle dataset slug (<owner>/<name>).

    Returns
    -------
    bool
        True if the download succeeded, False otherwise.
    """
    try:
        import kaggle  # noqa: F401 — triggers credential check
        from kaggle.api.kaggle_api_extended import KaggleApiExtended

        api = KaggleApiExtended()
        api.authenticate()
        print(f"[INFO] Downloading Kaggle dataset '{dataset}' …")
        api.dataset_download_files(dataset, path=str(DATA_DIR), unzip=True)

        # Locate the main CSV
        candidates = list(DATA_DIR.glob("*.csv"))
        if candidates:
            primary = candidates[0]
            if primary != RAW_TALKS_PATH:
                primary.rename(RAW_TALKS_PATH)
            print(f"[INFO] Dataset saved to {RAW_TALKS_PATH}")
            return True

    except Exception as exc:
        print(f"[WARNING] Kaggle download failed: {exc}")

    return False


# ─────────────────────────────────────────────
# 2. Synthetic dataset (fallback)
# ─────────────────────────────────────────────
TOPICS = [
    "Technology", "Science", "Health", "Education", "Business",
    "Environment", "Psychology", "Arts", "Politics", "Society",
    "Innovation", "Design", "Philosophy", "Economics", "Space",
    "Biology", "Neuroscience", "Artificial Intelligence", "Climate",
    "Culture",
]

TAGS_POOL = [
    "ai", "machine learning", "deep learning", "neuroscience", "brain",
    "climate change", "sustainability", "creativity", "leadership",
    "innovation", "education", "future", "health", "medicine",
    "psychology", "behavior", "society", "culture", "science",
    "technology", "design", "business", "economics", "politics",
    "space", "biology", "evolution", "language", "communication",
    "environment", "energy", "data", "privacy", "ethics",
]

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Elena", "Frank", "Grace",
    "Henry", "Iris", "James", "Karen", "Leo", "Maria", "Neil",
    "Olivia", "Paul", "Quinn", "Rachel", "Samuel", "Tina",
    "Uma", "Victor", "Wendy", "Xavier", "Yara", "Zack",
    "Aisha", "Ben", "Chloe", "Daniel",
]

LAST_NAMES = [
    "Anderson", "Brown", "Chen", "Davis", "Evans", "Ford", "Garcia",
    "Harris", "Ibrahim", "Johnson", "Kim", "Lee", "Martinez",
    "Nelson", "Okafor", "Patel", "Quinn", "Roberts", "Smith",
    "Taylor", "Ueda", "Voss", "Williams", "Xu", "Young", "Zhang",
]

TRANSCRIPT_TEMPLATES = [
    (
        "Thank you. I want to talk about {topic}. "
        "For the past {years} years, I have been studying {topic} and its impact "
        "on our world. What we found is remarkable. "
        "{topic} is fundamentally changing the way we think about {related}. "
        "Let me share three key insights. First, {insight1}. "
        "Second, {insight2}. Third, {insight3}. "
        "If we embrace these ideas, we can build a better future."
    ),
    (
        "Imagine a world where {topic} is accessible to everyone. "
        "I have spent {years} years working on {topic} and I believe we are "
        "at a tipping point. The data shows that {insight1}. "
        "But more importantly, {insight2}. "
        "Our research in {related} confirms that {insight3}. "
        "The question is not whether this will happen, but when."
    ),
    (
        "I grew up fascinated by {topic}. "
        "As a researcher in {related}, I have seen first-hand how {insight1}. "
        "Over {years} years of work, my team discovered that {insight2}. "
        "This has profound implications. {insight3}. "
        "We must act now to shape the future of {topic}."
    ),
]

INSIGHTS = [
    "human collaboration amplifies individual effort tenfold",
    "small behavioural changes lead to systemic transformation",
    "data-driven decisions consistently outperform intuition",
    "diversity in teams produces more creative solutions",
    "early education interventions yield lifelong returns",
    "open-source models accelerate scientific discovery",
    "decentralised networks are more resilient than centralised ones",
    "empathy is a trainable skill with measurable outcomes",
    "renewable energy is now cheaper than fossil fuels in many regions",
    "micro-habits compound into macro-level change over time",
]

RELATEDS = [
    "cognitive science", "climate policy", "urban planning",
    "public health", "economic theory", "social psychology",
    "computational biology", "human-computer interaction",
    "behavioural economics", "evolutionary biology",
]


def _make_transcript(topic: str, rng: random.Random) -> str:
    template = rng.choice(TRANSCRIPT_TEMPLATES)
    return template.format(
        topic=topic,
        years=rng.randint(3, 20),
        related=rng.choice(RELATEDS),
        insight1=rng.choice(INSIGHTS),
        insight2=rng.choice(INSIGHTS),
        insight3=rng.choice(INSIGHTS),
    )


def _make_title(topic: str, rng: random.Random) -> str:
    prefixes = [
        f"The Future of {topic}",
        f"How {topic} is Changing Everything",
        f"Why {topic} Matters More Than Ever",
        f"Rethinking {topic}",
        f"The Power of {topic}",
        f"What {topic} Taught Me",
        f"A New Way to Think About {topic}",
        f"The Hidden Side of {topic}",
        f"{topic}: A New Perspective",
        f"Inside {topic}",
    ]
    return rng.choice(prefixes)


def generate_synthetic_dataset(n_talks: int = 500, n_users: int = 200,
                                seed: int = 42) -> pd.DataFrame:
    """
    Build a realistic synthetic TED Talks dataset.

    Parameters
    ----------
    n_talks : int
        Number of talks to generate.
    n_users : int
        Number of simulated users (generates a watch-history column).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching the real Kaggle TED dataset schema.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    rows = []
    for i in range(1, n_talks + 1):
        topic = rng.choice(TOPICS)
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        speaker = f"{first} {last}"
        year = rng.randint(2006, 2023)
        month = rng.randint(1, 12)
        duration = rng.randint(4, 22)          # minutes
        views = int(np.random.lognormal(13, 1.5))
        rating = round(rng.uniform(3.5, 5.0), 2)
        num_ratings = int(np.random.lognormal(7, 1.2))
        tags = list({topic.lower().replace(" ", "_")}
                    | set(rng.sample(TAGS_POOL, k=rng.randint(3, 7))))

        transcript = _make_transcript(topic, rng)
        # Occasionally drop transcript to simulate missing data
        if rng.random() < 0.05:
            transcript = np.nan

        # Occasionally drop tags
        tag_str = ", ".join(tags) if rng.random() > 0.03 else np.nan

        rows.append({
            "talk_id": i,
            "title": _make_title(topic, rng),
            "speaker_name": speaker,
            "description": (
                f"A talk by {speaker} exploring the frontiers of {topic.lower()}. "
                f"Filmed at TED {year}."
            ),
            "tags": tag_str,
            "topic": topic,
            "film_date": f"{year}-{month:02d}-01",
            "duration_minutes": duration,
            "views": views,
            "average_rating": rating,
            "num_ratings": num_ratings,
            "transcript": transcript,
            "url": f"https://www.ted.com/talks/talk_{i}",
        })

    df = pd.DataFrame(rows)
    print(f"[INFO] Synthetic dataset generated: {len(df)} talks, "
          f"{df.shape[1]} columns.")
    return df


# ─────────────────────────────────────────────
# 3. Public entry point
# ─────────────────────────────────────────────
def load_or_collect_data(force_synthetic: bool = False) -> pd.DataFrame:
    """
    Return the TED Talks DataFrame, using the following priority:

    1. Pre-existing raw CSV in data/
    2. Kaggle API download
    3. Synthetic fallback

    Parameters
    ----------
    force_synthetic : bool
        If True, skip steps 1–2 and always generate the synthetic dataset.

    Returns
    -------
    pd.DataFrame
    """
    if not force_synthetic and RAW_TALKS_PATH.exists():
        print(f"[INFO] Loading existing dataset from {RAW_TALKS_PATH} …")
        df = pd.read_csv(RAW_TALKS_PATH)
        print(f"[INFO] Loaded {len(df)} rows, {df.shape[1]} columns.")
        return df

    if not force_synthetic and download_from_kaggle():
        df = pd.read_csv(RAW_TALKS_PATH)
        print(f"[INFO] Loaded Kaggle dataset: {len(df)} rows.")
        return df

    print("[INFO] Using synthetic dataset for demonstration …")
    df = generate_synthetic_dataset()
    df.to_csv(RAW_TALKS_PATH, index=False)
    print(f"[INFO] Synthetic dataset saved to {RAW_TALKS_PATH}")
    return df


if __name__ == "__main__":
    data = load_or_collect_data()
    print(data.head())
    print(f"\nShape: {data.shape}")
    print(f"\nColumns: {list(data.columns)}")
