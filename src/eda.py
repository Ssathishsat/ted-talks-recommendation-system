"""
Module 2 – Part B: Exploratory Data Analysis (EDA)
----------------------------------------------------
Generates all visualisations and summary statistics required by Milestone 1:

  • Talk-length distribution
  • Views distribution (log-scale)
  • Top topics / categories
  • Tag frequency (bar + word-cloud)
  • Ratings distribution
  • Talks per year (time-series)
  • Missing-value heat-map
  • Correlation heat-map (numeric features)
  • Speaker appearances (top-N)
  • Transcript word-count distribution

All plots are saved to outputs/plots/ and a text report is written to
outputs/reports/eda_report.txt.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Optional: WordCloud ──────────────────────────────────────────────────────
try:
    from wordcloud import WordCloud
    _HAS_WORDCLOUD = True
except ImportError:
    _HAS_WORDCLOUD = False
    print("[EDA] wordcloud not installed – word-cloud plot will be skipped.")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR    = PROJECT_ROOT / "outputs" / "plots"
REPORTS_DIR  = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = sns.color_palette("Set2")
FIGSIZE_WIDE  = (12, 5)
FIGSIZE_MED   = (10, 6)
FIGSIZE_SMALL = (8, 5)


# ────────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved → {path}")


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ────────────────────────────────────────────────────────────────────────────
# Individual plots
# ────────────────────────────────────────────────────────────────────────────

def plot_duration_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of talk durations (minutes)."""
    col = _find_col(df, ["duration_minutes", "duration"])
    if col is None:
        print("[EDA] Duration column not found – skipping.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_MED)
    vals = df[col].dropna()
    ax.hist(vals, bins=30, color=PALETTE[0], edgecolor="white",
            alpha=0.85, density=True, label="Histogram")

    # KDE overlay
    from scipy.stats import gaussian_kde   # local import – optional dep
    try:
        kde = gaussian_kde(vals)
        xs  = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(xs, kde(xs), color=PALETTE[2], lw=2, label="KDE")
    except Exception:
        pass

    ax.axvline(vals.mean(), color="crimson", ls="--", lw=1.5,
               label=f"Mean = {vals.mean():.1f} min")
    ax.axvline(vals.median(), color="steelblue", ls=":", lw=1.5,
               label=f"Median = {vals.median():.1f} min")

    ax.set_title("Talk Duration Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Density")
    ax.legend()
    _save(fig, "01_duration_distribution.png")


def plot_views_distribution(df: pd.DataFrame) -> None:
    """Log-scale histogram of view counts."""
    col = _find_col(df, ["views", "view_count"])
    if col is None:
        print("[EDA] Views column not found – skipping.")
        return

    vals = df[col].dropna()
    vals = vals[vals > 0]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Raw
    axes[0].hist(vals, bins=40, color=PALETTE[1], edgecolor="white",
                 alpha=0.85)
    axes[0].set_title("Views (Raw)", fontsize=13)
    axes[0].set_xlabel("Views")
    axes[0].set_ylabel("Count")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))

    # Log
    axes[1].hist(np.log10(vals + 1), bins=40, color=PALETTE[2],
                 edgecolor="white", alpha=0.85)
    axes[1].set_title("Views (Log₁₀ Scale)", fontsize=13)
    axes[1].set_xlabel("log₁₀(Views)")
    axes[1].set_ylabel("Count")

    fig.suptitle("View Count Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "02_views_distribution.png")


def plot_top_topics(df: pd.DataFrame, top_n: int = 15) -> None:
    """Horizontal bar chart of top N topics/categories."""
    col = _find_col(df, ["topic", "category", "main_tag"])
    if col is None:
        print("[EDA] Topic column not found – skipping.")
        return

    counts = df[col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=FIGSIZE_MED)
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=sns.color_palette("Spectral", len(counts)),
                   edgecolor="white")

    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=9)

    ax.set_title(f"Top {top_n} Talk Topics", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Talks")
    ax.set_ylabel("Topic")
    plt.tight_layout()
    _save(fig, "03_top_topics.png")


def plot_tag_frequency(df: pd.DataFrame, top_n: int = 25) -> None:
    """Bar chart of most common individual tags."""
    if "tags_list" not in df.columns:
        print("[EDA] 'tags_list' column not found – skipping tag frequency.")
        return

    from collections import Counter
    all_tags: list[str] = []
    for lst in df["tags_list"].dropna():
        all_tags.extend(lst)

    counter = Counter(all_tags)
    top = pd.DataFrame(counter.most_common(top_n),
                       columns=["tag", "count"])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(top["tag"], top["count"],
           color=sns.color_palette("tab20", len(top)), edgecolor="white")
    ax.set_xticklabels(top["tag"], rotation=45, ha="right", fontsize=9)
    ax.set_title(f"Top {top_n} Tag Frequencies", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("Tag")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    _save(fig, "04_tag_frequency.png")


def plot_tag_wordcloud(df: pd.DataFrame) -> None:
    """Word cloud of all tags."""
    if not _HAS_WORDCLOUD:
        return
    if "tags_list" not in df.columns:
        print("[EDA] 'tags_list' not found – skipping word cloud.")
        return

    text = " ".join(
        tag for lst in df["tags_list"].dropna()
        for tag in lst if isinstance(tag, str)
    )
    if not text.strip():
        return

    wc = WordCloud(width=900, height=450, background_color="white",
                   colormap="Set2", max_words=120,
                   collocations=False).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("TED Talks – Tag Word Cloud", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "05_tag_wordcloud.png")


def plot_ratings_distribution(df: pd.DataFrame) -> None:
    """Distribution of average ratings."""
    col = _find_col(df, ["average_rating", "rating"])
    if col is None:
        print("[EDA] Ratings column not found – skipping.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    vals = df[col].dropna()
    ax.hist(vals, bins=25, color=PALETTE[3], edgecolor="white", alpha=0.85)
    ax.axvline(vals.mean(), color="crimson", ls="--", lw=1.5,
               label=f"Mean = {vals.mean():.2f}")
    ax.set_title("Average Rating Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rating (out of 5)")
    ax.set_ylabel("Number of Talks")
    ax.legend()
    plt.tight_layout()
    _save(fig, "06_ratings_distribution.png")


def plot_talks_per_year(df: pd.DataFrame) -> None:
    """Line chart of talks published per year."""
    year_col = _find_col(df, ["year", "film_year"])
    if year_col is None and "film_date" in df.columns:
        df = df.copy()
        df["year"] = pd.to_datetime(df["film_date"], errors="coerce").dt.year
        year_col = "year"
    if year_col is None:
        print("[EDA] Year column not found – skipping talks-per-year plot.")
        return

    counts = (df[year_col]
              .dropna()
              .astype(int)
              .value_counts()
              .sort_index())

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(counts.index, counts.values, marker="o", color=PALETTE[0],
            linewidth=2, markersize=5)
    ax.fill_between(counts.index, counts.values, alpha=0.15,
                    color=PALETTE[0])
    ax.set_title("Number of TED Talks per Year", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Talks")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    _save(fig, "07_talks_per_year.png")


def plot_missing_values(df: pd.DataFrame) -> None:
    """Heat-map of missing values per column."""
    missing_pct = df.isnull().mean().sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        print("[EDA] No missing values detected – skipping missing-value plot.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(missing_pct) * 0.4)))
    bars = ax.barh(missing_pct.index[::-1], missing_pct.values[::-1] * 100,
                   color=sns.color_palette("flare", len(missing_pct)),
                   edgecolor="white")

    for bar, val in zip(bars, missing_pct.values[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=9)

    ax.set_title("Missing Value Percentage by Column", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("Missing (%)")
    ax.set_xlim(0, max(missing_pct.values) * 110)
    plt.tight_layout()
    _save(fig, "08_missing_values.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heat-map for numeric features."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude  = [c for c in num_cols if "encoded" in c or "id" in c.lower()]
    num_cols = [c for c in num_cols if c not in exclude]

    if len(num_cols) < 2:
        print("[EDA] Not enough numeric columns for correlation heat-map.")
        return

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(max(8, len(num_cols)), max(6, len(num_cols) - 1)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                annot_kws={"size": 9}, ax=ax)
    ax.set_title("Numeric Feature Correlation Matrix", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, "09_correlation_heatmap.png")


def plot_top_speakers(df: pd.DataFrame, top_n: int = 15) -> None:
    """Bar chart of speakers with the most talks."""
    col = _find_col(df, ["speaker_name", "speaker", "main_speaker"])
    if col is None:
        print("[EDA] Speaker column not found – skipping.")
        return

    counts = df[col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=FIGSIZE_MED)
    ax.bar(counts.index, counts.values,
           color=sns.color_palette("husl", len(counts)), edgecolor="white")
    ax.set_xticklabels(counts.index, rotation=40, ha="right", fontsize=9)
    ax.set_title(f"Top {top_n} Speakers by Number of Talks", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("Speaker")
    ax.set_ylabel("Number of Talks")
    plt.tight_layout()
    _save(fig, "10_top_speakers.png")


def plot_transcript_word_count(df: pd.DataFrame) -> None:
    """Histogram of cleaned transcript word counts."""
    col = _find_col(df, ["transcript_word_count"])
    if col is None:
        print("[EDA] 'transcript_word_count' not found – skipping.")
        return

    vals = df[col].dropna()
    vals = vals[vals > 0]

    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    ax.hist(vals, bins=35, color=PALETTE[4], edgecolor="white", alpha=0.85)
    ax.axvline(vals.mean(), color="crimson", ls="--", lw=1.5,
               label=f"Mean = {vals.mean():.0f} words")
    ax.set_title("Transcript Word Count Distribution", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("Word Count (cleaned tokens)")
    ax.set_ylabel("Number of Talks")
    ax.legend()
    plt.tight_layout()
    _save(fig, "11_transcript_word_count.png")


def plot_views_vs_rating(df: pd.DataFrame) -> None:
    """Scatter plot: views vs average rating, coloured by topic."""
    views_col  = _find_col(df, ["views", "view_count"])
    rating_col = _find_col(df, ["average_rating", "rating"])
    topic_col  = _find_col(df, ["topic", "category"])

    if views_col is None or rating_col is None:
        print("[EDA] views/rating columns missing – skipping scatter.")
        return

    plot_df = df[[views_col, rating_col]].copy()
    plot_df = plot_df[(plot_df[views_col] > 0) &
                      plot_df[rating_col].notna()]

    fig, ax = plt.subplots(figsize=FIGSIZE_MED)

    if topic_col and df[topic_col].nunique() <= 25:
        categories = df[topic_col].fillna("Unknown").astype(str)
        unique_cats = categories.unique()
        palette = dict(zip(unique_cats,
                           sns.color_palette("tab20", len(unique_cats))))
        colors = [palette[c] for c in categories.loc[plot_df.index]]
        sc = ax.scatter(np.log10(plot_df[views_col] + 1),
                        plot_df[rating_col], c=colors,
                        alpha=0.5, s=20, edgecolors="none")
    else:
        ax.scatter(np.log10(plot_df[views_col] + 1),
                   plot_df[rating_col], alpha=0.5,
                   s=20, color=PALETTE[0], edgecolors="none")

    ax.set_title("Views vs Average Rating", fontsize=14, fontweight="bold")
    ax.set_xlabel("log₁₀(Views)")
    ax.set_ylabel("Average Rating")
    plt.tight_layout()
    _save(fig, "12_views_vs_rating.png")


# ────────────────────────────────────────────────────────────────────────────
# Text report
# ────────────────────────────────────────────────────────────────────────────

def generate_eda_report(df: pd.DataFrame,
                        raw_df: pd.DataFrame | None = None) -> str:
    """
    Build a plaintext EDA summary report.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.
    raw_df : pd.DataFrame or None
        Original raw DataFrame (for before/after comparison).

    Returns
    -------
    str
        Report text, also written to outputs/reports/eda_report.txt.
    """
    lines = [
        "=" * 65,
        "  TED Talks Recommendation System – EDA Report (Phase 1)",
        "=" * 65,
        "",
    ]

    def _section(title: str) -> None:
        lines.extend(["", f"── {title} " + "─" * max(1, 60 - len(title)), ""])

    # ── Dataset dimensions ───────────────────────────────────────────────────
    _section("Dataset Dimensions")
    if raw_df is not None:
        lines.append(f"  Raw dataset     : {raw_df.shape[0]:,} rows × "
                     f"{raw_df.shape[1]} columns")
    lines.append(f"  Processed dataset: {df.shape[0]:,} rows × "
                 f"{df.shape[1]} columns")

    # ── Missing values ───────────────────────────────────────────────────────
    _section("Missing Value Summary")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        lines.append("  No missing values in the processed dataset.")
    else:
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            lines.append(f"  {col:<35} {cnt:>5} missing  ({pct:.1f}%)")

    # ── Numeric summary ──────────────────────────────────────────────────────
    _section("Numeric Feature Summary")
    num_cols = [c for c in ["duration_minutes", "views", "average_rating",
                            "num_ratings", "transcript_word_count",
                            "tag_count"]
                if c in df.columns]
    if num_cols:
        stats = df[num_cols].describe().T[
            ["count", "mean", "std", "min", "50%", "max"]
        ].rename(columns={"50%": "median"})
        lines.append(stats.to_string())

    # ── Topic distribution ───────────────────────────────────────────────────
    topic_col = next((c for c in ["topic", "category"]
                      if c in df.columns), None)
    if topic_col:
        _section(f"Top 10 Topics  ('{topic_col}')")
        for topic, cnt in df[topic_col].value_counts().head(10).items():
            lines.append(f"  {topic:<30} {cnt:>4} talks")

    # ── Year range ───────────────────────────────────────────────────────────
    if "year" in df.columns:
        yr = df["year"].dropna()
        _section("Year Range")
        lines.append(f"  Earliest: {int(yr.min())}  –  Latest: {int(yr.max())}")
        lines.append(f"  Mode year: {int(yr.mode().iloc[0])}")

    # ── Tags ─────────────────────────────────────────────────────────────────
    if "tag_count" in df.columns:
        _section("Tag Statistics")
        tc = df["tag_count"]
        lines.append(f"  Mean tags per talk  : {tc.mean():.1f}")
        lines.append(f"  Median tags per talk: {tc.median():.0f}")
        lines.append(f"  Max tags on one talk: {int(tc.max())}")
        lines.append(f"  Talks with 0 tags   : {(tc == 0).sum()}")

    # ── Data quality verdict ─────────────────────────────────────────────────
    _section("Data Quality Verdict")
    missing_pct = df.isnull().mean().max() * 100
    if missing_pct < 5:
        verdict = "GOOD – Missing values < 5 % across all columns."
    elif missing_pct < 15:
        verdict = "ACCEPTABLE – Some columns have 5–15 % missing values."
    else:
        verdict = "NEEDS ATTENTION – Some columns have > 15 % missing values."
    lines.append(f"  {verdict}")

    lines += ["", "=" * 65, "  End of Report", "=" * 65]
    report = "\n".join(lines)

    # Persist
    report_path = REPORTS_DIR / "eda_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"[EDA] Report saved → {report_path}")
    return report


# ────────────────────────────────────────────────────────────────────────────
# Master runner
# ────────────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame,
            raw_df: pd.DataFrame | None = None) -> None:
    """
    Execute the complete EDA pipeline: generate all plots and the report.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed TED Talks DataFrame.
    raw_df : pd.DataFrame or None
        Original raw DataFrame passed through for report comparison.
    """
    print("\n[EDA] Starting exploratory data analysis …\n")

    plot_duration_distribution(df)
    plot_views_distribution(df)
    plot_top_topics(df)
    plot_tag_frequency(df)
    plot_tag_wordcloud(df)
    plot_ratings_distribution(df)
    plot_talks_per_year(df)
    plot_missing_values(df)
    plot_correlation_heatmap(df)
    plot_top_speakers(df)
    plot_transcript_word_count(df)
    plot_views_vs_rating(df)

    report = generate_eda_report(df, raw_df=raw_df)
    print("\n" + report)
    print(f"\n[EDA] ✓ All plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    from pathlib import Path as _Path
    import sys
    sys.path.insert(0, str(_Path(__file__).resolve().parent))

    from data_collection import load_or_collect_data
    from preprocessing import run_preprocessing

    raw  = load_or_collect_data()
    proc = run_preprocessing(raw, save=True)
    run_eda(proc, raw_df=raw)
