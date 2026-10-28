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
    "Culture", "Music", "History", "Mathematics", "Law",
    "Sustainability", "Mental Health", "Robotics", "Data Science",
    "Architecture", "Literature",
]

EVENTS = [
    "TED2006", "TED2007", "TED2008", "TED2009", "TED2010",
    "TED2011", "TED2012", "TED2013", "TED2014", "TED2015",
    "TED2016", "TED2017", "TED2018", "TED2019", "TED2020",
    "TED2021", "TED2022", "TED2023", "TEDGlobal", "TEDx",
    "TED Salon", "TED Residency", "TED Countdown", "TED Women",
]

LANGUAGES = ["English", "English", "English", "English", "English",
             "Spanish", "French", "Portuguese", "Arabic", "Chinese"]

TAGS_POOL = [
    "ai", "machine learning", "deep learning", "neuroscience", "brain",
    "climate change", "sustainability", "creativity", "leadership",
    "innovation", "education", "future", "health", "medicine",
    "psychology", "behavior", "society", "culture", "science",
    "technology", "design", "business", "economics", "politics",
    "space", "biology", "evolution", "language", "communication",
    "environment", "energy", "data", "privacy", "ethics",
    "robotics", "quantum computing", "blockchain", "virtual reality",
    "mental health", "wellness", "mindfulness", "storytelling",
    "social media", "globalization", "human rights", "gender",
    "race", "inequality", "poverty", "nutrition", "fitness",
    "aging", "genetics", "crispr", "biotech", "nanotechnology",
    "renewable energy", "solar", "battery", "carbon capture",
    "ocean", "biodiversity", "agriculture", "food security",
    "urban planning", "transportation", "automation", "future of work",
    "education reform", "online learning", "critical thinking",
    "philosophy", "ethics", "morality", "consciousness",
    "music", "art", "poetry", "writing", "film",
    "history", "archaeology", "anthropology", "religion",
    "mathematics", "statistics", "probability", "algorithms",
    "law", "justice", "democracy", "governance",
    "collaboration", "teamwork", "productivity", "motivation",
    "happiness", "resilience", "grit", "compassion",
    "tedx", "ted talk", "inspiring", "informative", "perspective",
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
        "Let me give you a concrete example. In our lab, we tested this idea "
        "with {example_group}. The results were striking. "
        "{example_detail} "
        "This tells us something profound about {topic}. "
        "When we look at the broader picture, we see that {insight4}. "
        "And this is not just theory. Real world applications are already emerging. "
        "Companies and institutions are starting to adopt these principles. "
        "The implications for {related} are enormous. "
        "If we embrace these ideas, we can build a better future. "
        "But it requires courage, collaboration, and a willingness to change. "
        "So I ask you today: what role will you play in shaping the future of {topic}? "
        "Thank you."
    ),
    (
        "Imagine a world where {topic} is accessible to everyone. "
        "I have spent {years} years working on {topic} and I believe we are "
        "at a tipping point. The data shows that {insight1}. "
        "But more importantly, {insight2}. "
        "Our research in {related} confirms that {insight3}. "
        "Consider this: when we surveyed {example_group}, we discovered that "
        "{example_detail} "
        "This is not an isolated finding. Across multiple studies, the pattern holds. "
        "{insight4} "
        "Now, you might be wondering: how do we get there? "
        "The answer is both simple and complex. It starts with {insight5}. "
        "Then we need investment, policy changes, and public engagement. "
        "The technology is ready. The question is not whether this will happen, but when. "
        "And the when depends on all of us in this room today."
    ),
    (
        "I grew up fascinated by {topic}. "
        "As a researcher in {related}, I have seen first-hand how {insight1}. "
        "Over {years} years of work, my team discovered that {insight2}. "
        "This has profound implications. {insight3}. "
        "Let me walk you through what we found. "
        "We worked with {example_group} over a period of several months. "
        "{example_detail} "
        "What surprised us most was {insight4}. "
        "It challenges everything we thought we knew about {topic}. "
        "We published our findings and the response was overwhelming. "
        "Researchers from {n_countries} countries reached out to collaborate. "
        "Together, we are now building a global network focused on {topic}. "
        "The future is not something we wait for. It is something we create. "
        "We must act now to shape the future of {topic}. "
        "Thank you for listening."
    ),
    (
        "Let me start with a question. What if everything we know about {topic} is wrong? "
        "That is the question I asked myself {years} years ago. "
        "And it led me down a path I never expected. "
        "Working in {related}, I began to notice patterns that did not fit "
        "the conventional wisdom. {insight1}. "
        "So my team and I decided to dig deeper. "
        "We collected data from {example_group}. "
        "{example_detail} "
        "What we found changed my perspective entirely. {insight2}. "
        "And it gets more interesting. {insight3}. "
        "When we shared our results, some people were skeptical. That is fair. "
        "But the evidence is compelling. {insight4}. "
        "I believe we are standing at the beginning of a revolution in {topic}. "
        "Not a violent one, but a revolution of ideas. "
        "The kind that reshapes how we see the world and our place in it. "
        "So I leave you with this thought: the future of {topic} is not written. "
        "It is being written right now, by people like you. "
        "Thank you."
    ),
]

EXAMPLE_GROUPS = [
    "a group of 500 volunteers", "students across 12 universities",
    "professionals from 30 companies", "a community of 1000 participants",
    "patients in a clinical trial", "citizens from 5 different cities",
    "a team of 200 researchers", "children in 15 schools",
    "workers across 3 industries", "a diverse group of 800 individuals",
]

EXAMPLE_DETAILS = [
    "Those who adopted the new approach showed a 40% improvement in outcomes.",
    "The data revealed clear patterns that contradicted existing assumptions.",
    "Participants reported significantly higher satisfaction and engagement.",
    "The intervention reduced error rates by more than half within six months.",
    "We observed measurable behavioural changes that persisted over time.",
    "The results were consistent across age groups, genders, and backgrounds.",
    "Productivity increased by 35% without any additional resources required.",
    "The approach proved scalable and cost-effective in real-world settings.",
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
    "the brain has remarkable capacity to rewire itself at any age",
    "storytelling activates multiple brain regions simultaneously",
    "most decisions are made emotionally and justified rationally afterwards",
    "sleep deprivation impairs judgment as much as alcohol intoxication",
    "exposure to nature reduces stress hormones within minutes",
    "social connections are the strongest predictor of long-term happiness",
    "practice alone does not make perfect — deliberate practice does",
    "the most innovative ideas often come from outside the field",
]

RELATEDS = [
    "cognitive science", "climate policy", "urban planning",
    "public health", "economic theory", "social psychology",
    "computational biology", "human-computer interaction",
    "behavioural economics", "evolutionary biology",
    "neural networks", "sustainable development", "public policy",
    "organizational behavior", "educational technology",
    "behavioral neuroscience", "environmental science",
    "digital transformation", "social innovation", "systems thinking",
]

RATING_CATEGORIES = {
    "inspiring": (0.15, 0.35),
    "informative": (0.20, 0.40),
    "funny": (0.05, 0.20),
    "persuasive": (0.15, 0.30),
    "fascinating": (0.10, 0.30),
    "beautiful": (0.05, 0.15),
    "courageous": (0.05, 0.15),
    "ingenious": (0.05, 0.20),
    "ok": (0.01, 0.05),
    "confusing": (0.01, 0.05),
    "jaw-dropping": (0.05, 0.15),
    "longwinded": (0.01, 0.05),
    "obnoxious": (0.01, 0.03),
}


def _make_transcript(topic: str, rng: random.Random) -> str:
    template = rng.choice(TRANSCRIPT_TEMPLATES)
    return template.format(
        topic=topic,
        years=rng.randint(3, 20),
        related=rng.choice(RELATEDS),
        insight1=rng.choice(INSIGHTS),
        insight2=rng.choice(INSIGHTS),
        insight3=rng.choice(INSIGHTS),
        insight4=rng.choice(INSIGHTS),
        insight5=rng.choice(INSIGHTS),
        example_group=rng.choice(EXAMPLE_GROUPS),
        example_detail=rng.choice(EXAMPLE_DETAILS),
        n_countries=rng.randint(5, 30),
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
        f"The Surprising Truth About {topic}",
        f"What I Learned From {topic}",
        f"The Art of {topic}",
        f"{topic} and the Future of Humanity",
        f"Breaking the Rules of {topic}",
        f"The Science Behind {topic}",
        f"Why {topic} is the Most Important Conversation Today",
    ]
    return rng.choice(prefixes)


def _make_ratings(rng: random.Random) -> dict:
    """Generate realistic rating counts for each rating category."""
    ratings = {}
    for cat, (lo, hi) in RATING_CATEGORIES.items():
        # Higher total for popular talks
        base = int(np.random.lognormal(6, 1.5))
        count = int(base * rng.uniform(lo, hi))
        ratings[cat] = max(count, 0)
    return ratings


def generate_synthetic_dataset(n_talks: int = 1000, n_users: int = 200,
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
        event = rng.choice(EVENTS)
        language = rng.choice(LANGUAGES)
        year = rng.randint(2006, 2023)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        duration = rng.randint(4, 25)          # minutes
        views = int(np.random.lognormal(13, 1.5))
        rating = round(rng.uniform(3.0, 5.0), 2)
        num_ratings = int(np.random.lognormal(7, 1.2))
        comments = int(np.random.lognormal(5, 1.3))

        tags = list({topic.lower().replace(" ", "_")}
                    | set(rng.sample(TAGS_POOL, k=rng.randint(3, 10))))

        transcript = _make_transcript(topic, rng)
        # Occasionally drop transcript to simulate missing data
        if rng.random() < 0.04:
            transcript = np.nan

        # Occasionally drop tags
        tag_str = ", ".join(tags) if rng.random() > 0.02 else np.nan

        # Generate rating breakdown
        ratings = _make_ratings(rng)

        # Speaker occupation
        occupations = [
            f"Professor of {topic}", f"{topic} Researcher", f"{topic} Expert",
            f"Author and {topic} Specialist", f"Director of {topic} Institute",
            f"Founder, {topic} Initiative", f"Chief {topic} Officer",
            f"Senior {topic} Analyst", f"{topic} Consultant",
            f"Entrepreneur in {topic}", f"Journalist covering {topic}",
            f"Activist for {topic}", f"Pioneer in {topic}",
        ]
        speaker_occupation = rng.choice(occupations)

        rows.append({
            "talk_id": i,
            "title": _make_title(topic, rng),
            "speaker_name": speaker,
            "speaker_occupation": speaker_occupation,
            "description": (
                f"A talk by {speaker} exploring the frontiers of {topic.lower()}. "
                f"Filmed at {event}."
            ),
            "tags": tag_str,
            "topic": topic,
            "event": event,
            "language": language,
            "film_date": f"{year}-{month:02d}-{day:02d}",
            "published_date": f"{year}-{month:02d}-{day:02d}",
            "duration_minutes": duration,
            "views": views,
            "average_rating": rating,
            "num_ratings": num_ratings,
            "comments": comments,
            "ratings": str(ratings),
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
