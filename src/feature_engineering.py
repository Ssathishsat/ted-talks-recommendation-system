"""
Module 3 – Feature Engineering and Vectorization
-------------------------------------------------
Converts text data (transcripts, tags, combined text) into numerical
representations using:

  1. TF-IDF vectorization  (primary method)
  2. Word2Vec embeddings    (secondary method)
  3. Cosine similarity matrices for content-based filtering
  4. Simulated user–item interaction matrix for collaborative filtering

All models and matrices are saved to data/models/ for reuse in deployment.
"""

import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = DATA_DIR / "models"
PLOTS_DIR    = PROJECT_ROOT / "outputs" / "plots"
REPORTS_DIR  = PROJECT_ROOT / "outputs" / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_PATH = DATA_DIR / "ted_talks_processed.csv"


# ────────────────────────────────────────────────────────────────────────────
# 1. TF-IDF Vectorization
# ────────────────────────────────────────────────────────────────────────────

def build_tfidf_vectorizer(texts: pd.Series,
                           max_features: int = 5000,
                           ngram_range: tuple = (1, 2),
                           min_df: int = 1,
                           max_df: float = 0.85,
                           label: str = "combined") -> tuple:
    """
    Fit a TF-IDF vectorizer on the given text series.

    Parameters
    ----------
    texts : pd.Series
        Clean text to vectorize.
    max_features : int
        Maximum number of features (vocabulary size).
    ngram_range : tuple
        N-gram range (e.g., (1, 2) for unigrams + bigrams).
    min_df : int
        Minimum document frequency — ignores terms in fewer than min_df docs.
    max_df : float
        Maximum document fraction — ignores terms in more than max_df fraction.
    label : str
        Name for logging and file naming.

    Returns
    -------
    tuple
        (vectorizer, tfidf_matrix)
    """
    print(f"\n[TF-IDF] Building vectorizer for '{label}' …")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        sublinear_tf=True,          # 1 + log(tf)
        norm="l2",
    )

    tfidf_matrix = vectorizer.fit_transform(texts.fillna(""))

    vocab_size = len(vectorizer.vocabulary_)
    print(f"[TF-IDF] '{label}' matrix shape : {tfidf_matrix.shape}")
    print(f"[TF-IDF] '{label}' vocabulary   : {vocab_size} terms")
    print(f"[TF-IDF] '{label}' ngram_range   : {ngram_range}")
    print(f"[TF-IDF] '{label}' sparsity      : "
          f"{1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4f}")

    return vectorizer, tfidf_matrix


def save_tfidf_model(vectorizer: TfidfVectorizer,
                     matrix: sparse.csr_matrix,
                     label: str) -> None:
    """Persist the TF-IDF vectorizer and its matrix."""
    vec_path  = MODELS_DIR / f"tfidf_{label}_vectorizer.pkl"
    mat_path  = MODELS_DIR / f"tfidf_{label}_matrix.npz"

    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    sparse.save_npz(str(mat_path), matrix)

    print(f"[TF-IDF] Saved vectorizer → {vec_path}")
    print(f"[TF-IDF] Saved matrix     → {mat_path}")


# ────────────────────────────────────────────────────────────────────────────
# 2. Word2Vec Embeddings
# ────────────────────────────────────────────────────────────────────────────

def build_word2vec_embeddings(texts: pd.Series,
                              vector_size: int = 100,
                              window: int = 5,
                              min_count: int = 1,
                              workers: int = 4,
                              seed: int = 42,
                              label: str = "combined") -> tuple:
    """
    Train a Word2Vec model on tokenized text and produce per-document
    averaged embeddings.

    Parameters
    ----------
    texts : pd.Series
        Clean text to embed.
    vector_size : int
        Dimensionality of the word vectors.
    window : int
        Context window size.
    min_count : int
        Minimum word frequency for inclusion.
    workers : int
        Number of training threads.
    seed : int
        Random seed for reproducibility.
    label : str
        Name for logging and file naming.

    Returns
    -------
    tuple
        (model, embeddings_matrix)
    """
    from gensim.models import Word2Vec

    print(f"\n[Word2Vec] Training model for '{label}' …")

    # Tokenize
    tokenized = texts.fillna("").apply(lambda t: t.split() if t else [])
    total_tokens = sum(len(t) for t in tokenized)
    print(f"[Word2Vec] Total tokens   : {total_tokens}")
    print(f"[Word2Vec] Avg tokens/doc : {total_tokens / len(tokenized):.1f}")

    # Train
    model = Word2Vec(
        sentences=tokenized.tolist(),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
        sg=1,               # Skip-gram
        epochs=20,
    )

    print(f"[Word2Vec] Vocabulary size : {len(model.wv)}")
    print(f"[Word2Vec] Vector size     : {vector_size}")

    # Average word vectors per document
    embeddings = np.zeros((len(texts), vector_size), dtype=np.float32)
    for i, tokens in enumerate(tokenized):
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if vecs:
            embeddings[i] = np.mean(vecs, axis=0)

    print(f"[Word2Vec] Embeddings shape : {embeddings.shape}")
    return model, embeddings


def save_word2vec_model(model, embeddings: np.ndarray, label: str) -> None:
    """Persist the Word2Vec model and its embedding matrix."""
    model_path = MODELS_DIR / f"word2vec_{label}.model"
    emb_path   = MODELS_DIR / f"word2vec_{label}_embeddings.npy"

    model.save(str(model_path))
    np.save(str(emb_path), embeddings)

    print(f"[Word2Vec] Saved model      → {model_path}")
    print(f"[Word2Vec] Saved embeddings → {emb_path}")


# ────────────────────────────────────────────────────────────────────────────
# 3. Cosine Similarity Matrices
# ────────────────────────────────────────────────────────────────────────────

def build_similarity_matrix(tfidf_matrix: sparse.csr_matrix,
                            label: str = "combined") -> np.ndarray:
    """
    Compute cosine similarity matrix from a TF-IDF matrix.

    For TF-IDF, cosine similarity == linear_kernel (vectors are L2-normalized),
    which is faster than cosine_similarity.

    Parameters
    ----------
    tfidf_matrix : sparse.csr_matrix
        TF-IDF matrix from build_tfidf_vectorizer().
    label : str
        Name for logging.

    Returns
    -------
    np.ndarray
        N×N similarity matrix.
    """
    print(f"\n[SIM] Building cosine similarity matrix for '{label}' …")

    # linear_kernel is faster for L2-normalized vectors
    sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    print(f"[SIM] Similarity matrix shape : {sim.shape}")
    print(f"[SIM] Mean similarity          : {np.mean(sim):.4f}")
    print(f"[SIM] Max similarity (excl. diag): "
          f"{np.max(sim - np.eye(sim.shape[0])):.4f}")

    return sim


def save_similarity_matrix(sim_matrix: np.ndarray, label: str) -> None:
    """Persist a similarity matrix."""
    path = MODELS_DIR / f"similarity_{label}.npy"
    np.save(str(path), sim_matrix)
    print(f"[SIM] Saved → {path}")


def get_top_similar_talks(sim_matrix: np.ndarray,
                          df: pd.DataFrame,
                          talk_idx: int,
                          top_n: int = 5) -> pd.DataFrame:
    """
    Return the top-N most similar talks for a given talk index.

    Parameters
    ----------
    sim_matrix : np.ndarray
    df : pd.DataFrame
    talk_idx : int
        Index of the query talk.
    top_n : int

    Returns
    -------
    pd.DataFrame
        Columns: title, speaker_name, topic, similarity_score
    """
    scores = sim_matrix[talk_idx]
    # Exclude the talk itself
    scores[talk_idx] = -1
    top_indices = np.argsort(scores)[::-1][:top_n]

    results = pd.DataFrame({
        "title": df.iloc[top_indices]["title"].values,
        "speaker_name": df.iloc[top_indices]["speaker_name"].values,
        "topic": df.iloc[top_indices]["topic"].values,
        "similarity_score": scores[top_indices],
    })
    return results


# ────────────────────────────────────────────────────────────────────────────
# 4. User–Item Interaction Matrix (Simulated)
# ────────────────────────────────────────────────────────────────────────────

def build_user_item_matrix(df: pd.DataFrame,
                           n_users: int = 100,
                           seed: int = 42) -> sparse.csr_matrix:
    """
    Simulate a user–item interaction matrix for collaborative filtering.

    Each user is randomly assigned a set of preferred topics. Users then
    "interact" (watch) talks matching their preferred topics with a probability
    weighted by the talk's view count.

    Parameters
    ----------
    df : pd.DataFrame
        Processed TED Talks DataFrame.
    n_users : int
        Number of simulated users.
    seed : int
        Random seed.

    Returns
    -------
    sparse.csr_matrix
        Shape: (n_users, n_talks). Values: 1 (watched) or 0 (not watched).
    """
    print(f"\n[USER-ITEM] Simulating {n_users} users × {len(df)} talks …")

    rng = np.random.RandomState(seed)
    n_talks = len(df)

    topics = df["topic"].unique() if "topic" in df.columns else []
    n_topics = len(topics)

    # Assign each user 2–4 preferred topics
    user_prefs = []
    for _ in range(n_users):
        k = rng.randint(2, min(5, n_topics + 1))
        prefs = set(rng.choice(topics, size=k, replace=False))
        user_prefs.append(prefs)

    # Build interaction matrix
    rows, cols = [], []

    # Normalize views for probability weighting
    views = df["views"].fillna(1).values.astype(float)
    view_probs = views / views.sum()

    for user_id in range(n_users):
        prefs = user_prefs[user_id]

        # Candidate talks: those matching user's preferred topics
        if "topic" in df.columns:
            mask = df["topic"].isin(prefs).values
        else:
            mask = np.ones(n_talks, dtype=bool)

        candidate_indices = np.where(mask)[0]

        if len(candidate_indices) == 0:
            continue

        # Each user watches 5–20 talks
        n_watches = rng.randint(5, min(21, len(candidate_indices) + 1))

        # Weight selection by views
        candidate_view_probs = view_probs[candidate_indices]
        candidate_view_probs = candidate_view_probs / candidate_view_probs.sum()

        watched = rng.choice(
            candidate_indices,
            size=n_watches,
            replace=False,
            p=candidate_view_probs,
        )

        rows.extend([user_id] * len(watched))
        cols.extend(watched.tolist())

    data = np.ones(len(rows), dtype=np.float32)
    user_item = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_users, n_talks),
        dtype=np.float32,
    )

    sparsity = 1 - user_item.nnz / (n_users * n_talks)
    print(f"[USER-ITEM] Matrix shape     : {user_item.shape}")
    print(f"[USER-ITEM] Total interactions: {user_item.nnz}")
    print(f"[USER-ITEM] Avg watches/user  : {user_item.nnz / n_users:.1f}")
    print(f"[USER-ITEM] Sparsity          : {sparsity:.4f}")

    return user_item


def save_user_item_matrix(user_item: sparse.csr_matrix) -> None:
    """Persist the user–item interaction matrix."""
    path = MODELS_DIR / "user_item_matrix.npz"
    sparse.save_npz(str(path), user_item)
    print(f"[USER-ITEM] Saved → {path}")


# ────────────────────────────────────────────────────────────────────────────
# 5. Visualization — Feature Importance & Similarity
# ────────────────────────────────────────────────────────────────────────────

def plot_tfidf_top_terms(vectorizer: TfidfVectorizer,
                          tfidf_matrix: sparse.csr_matrix,
                          label: str,
                          top_n: int = 20) -> None:
    """
    Bar chart of top-N terms by average TF-IDF score across all documents.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"\n[VIZ] Plotting top TF-IDF terms for '{label}' …")

    # Mean TF-IDF score per term
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()

    top_indices = np.argsort(mean_scores)[::-1][:top_n]
    top_terms = [feature_names[i] for i in top_indices]
    top_scores = mean_scores[top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_terms[::-1], top_scores[::-1],
                   color=sns.color_palette("Spectral", top_n),
                   edgecolor="white")
    ax.set_title(f"Top {top_n} TF-IDF Terms — '{label}'",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean TF-IDF Score")
    ax.set_ylabel("Term")
    plt.tight_layout()

    path = PLOTS_DIR / f"13_tfidf_top_terms_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved → {path}")


def plot_similarity_heatmap_sample(sim_matrix: np.ndarray,
                                   df: pd.DataFrame,
                                   label: str,
                                   n_sample: int = 20,
                                   seed: int = 42) -> None:
    """
    Heatmap of pairwise cosine similarity for a random sample of talks.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"\n[VIZ] Plotting similarity heatmap for '{label}' …")

    rng = np.random.RandomState(seed)
    n = min(n_sample, len(df))
    sample_idx = rng.choice(len(df), size=n, replace=False)
    sample_sim = sim_matrix[np.ix_(sample_idx, sample_idx)]

    # Short labels for display
    title_col = "title" if "title" in df.columns else df.columns[1]
    labels = [df.iloc[i][title_col][:25] for i in sample_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(sample_sim, dtype=bool), k=1)
    sns.heatmap(sample_sim, mask=mask, xticklabels=labels, yticklabels=labels,
                cmap="YlOrRd", annot=False, fmt=".2f",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Cosine Similarity"})
    ax.set_title(f"Content Similarity Matrix (sample of {n} talks) — '{label}'",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    path = PLOTS_DIR / f"14_similarity_heatmap_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved → {path}")


def plot_similarity_distribution(sim_matrix: np.ndarray,
                                 label: str) -> None:
    """
    Histogram of pairwise similarity scores (excluding diagonal).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n[VIZ] Plotting similarity distribution for '{label}' …")

    # Extract upper triangle (excluding diagonal)
    n = sim_matrix.shape[0]
    upper = sim_matrix[np.triu_indices(n, k=1)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(upper, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(upper), color="crimson", ls="--", lw=1.5,
               label=f"Mean = {np.mean(upper):.4f}")
    ax.set_title(f"Pairwise Similarity Score Distribution — '{label}'",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Pair Count")
    ax.legend()
    plt.tight_layout()

    path = PLOTS_DIR / f"15_similarity_distribution_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved → {path}")


def plot_user_item_sparsity(user_item: sparse.csr_matrix) -> None:
    """
    Visualize the sparsity pattern of the user–item interaction matrix.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n[VIZ] Plotting user-item matrix sparsity …")

    # Sample for visualization
    n_show_users = min(50, user_item.shape[0])
    n_show_items = min(50, user_item.shape[1])
    sample = user_item[:n_show_users, :n_show_items].toarray()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.spy(sample, markersize=3, color="#4C72B0", aspect="auto")
    ax.set_title("User–Item Interaction Matrix (sample)\n"
                 "(dots = watched talks)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Talk Index")
    ax.set_ylabel("User Index")
    plt.tight_layout()

    path = PLOTS_DIR / "16_user_item_sparsity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved → {path}")


# ────────────────────────────────────────────────────────────────────────────
# 6. Documentation Report
# ────────────────────────────────────────────────────────────────────────────

def generate_feature_report(df: pd.DataFrame,
                            tfidf_results: dict,
                            w2v_results: dict | None,
                            sim_results: dict,
                            user_item: sparse.csr_matrix) -> str:
    """
    Generate a text report documenting vectorization choices and results.
    """
    lines = [
        "=" * 65,
        "  TED Talks Recommendation System – Feature Engineering Report",
        "  Module 3 (Milestone 2, Phase 2)",
        "=" * 65,
        "",
        f"  Processed dataset : {df.shape[0]} talks × {df.shape[1]} features",
        "",
    ]

    def _section(title: str) -> None:
        lines.extend(["", f"── {title} " + "─" * max(1, 60 - len(title)), ""])

    # ── TF-IDF ─────────────────────────────────────────────────────────────────
    _section("1. TF-IDF Vectorization")
    lines.append("  Method : Term Frequency–Inverse Document Frequency")
    lines.append("  Library: scikit-learn TfidfVectorizer")
    lines.append("  Parameters:")
    lines.append("    • max_features = 5000")
    lines.append("    • ngram_range  = (1, 2)  [unigrams + bigrams]")
    lines.append("    • min_df = 1  (keep terms in at least 1 doc)")
    lines.append("    • max_df = 0.85  (drop terms in > 85% of docs)")
    lines.append("    • sublinear_tf = True  (1 + log(tf))")
    lines.append("    • norm = 'l2'")
    lines.append("")
    for label, (vec, mat) in tfidf_results.items():
        lines.append(f"  '{label}':")
        lines.append(f"    Matrix shape  : {mat.shape}")
        lines.append(f"    Vocabulary    : {len(vec.vocabulary_)} terms")
        sparsity = 1 - mat.nnz / (mat.shape[0] * mat.shape[1])
        lines.append(f"    Sparsity      : {sparsity:.4f}")

    # ── Word2Vec ───────────────────────────────────────────────────────────────
    if w2v_results:
        _section("2. Word2Vec Embeddings")
        lines.append("  Method : Skip-gram Word2Vec with averaged document vectors")
        lines.append("  Library: gensim.models.Word2Vec")
        lines.append("  Parameters:")
        lines.append("    • vector_size = 100")
        lines.append("    • window     = 5")
        lines.append("    • min_count  = 1")
        lines.append("    • sg = 1  (Skip-gram)")
        lines.append("    • epochs = 20")
        lines.append("")
        for label, (model, emb) in w2v_results.items():
            lines.append(f"  '{label}':")
            lines.append(f"    Embedding shape : {emb.shape}")
            lines.append(f"    Vocabulary size  : {len(model.wv)}")

    # ── Similarity ────────────────────────────────────────────────────────────
    _section("3. Cosine Similarity Matrices")
    lines.append("  Method : linear_kernel (equivalent to cosine for L2-normalized)")
    for label, sim in sim_results.items():
        lines.append(f"  '{label}':")
        lines.append(f"    Shape            : {sim.shape}")
        upper = sim[np.triu_indices(sim.shape[0], k=1)]
        lines.append(f"    Mean similarity  : {np.mean(upper):.4f}")
        lines.append(f"    Std similarity   : {np.std(upper):.4f}")
        lines.append(f"    Min / Max (excl diag): {np.min(upper):.4f} / {np.max(upper):.4f}")

    # ── User-Item ──────────────────────────────────────────────────────────────
    _section("4. User–Item Interaction Matrix")
    lines.append(f"  Shape             : {user_item.shape}")
    lines.append(f"  Total interactions: {user_item.nnz}")
    lines.append(f"  Avg watches/user  : {user_item.nnz / user_item.shape[0]:.1f}")
    sparsity = 1 - user_item.nnz / (user_item.shape[0] * user_item.shape[1])
    lines.append(f"  Sparsity          : {sparsity:.4f}")
    lines.append("  Method            : Simulated — each user assigned 2-4")
    lines.append("                      preferred topics, watches 5-20 talks")
    lines.append("                      weighted by view count")

    # ── Files saved ───────────────────────────────────────────────────────────
    _section("5. Saved Artifacts")
    lines.append("  data/models/")
    for f in sorted(MODELS_DIR.glob("*")):
        lines.append(f"    • {f.name}")

    lines += ["", "=" * 65, "  End of Report", "=" * 65]
    report = "\n".join(lines)

    report_path = REPORTS_DIR / "feature_engineering_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[REPORT] Saved → {report_path}")
    return report


# ────────────────────────────────────────────────────────────────────────────
# 7. Master Pipeline
# ────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(df: pd.DataFrame | None = None,
                            use_word2vec: bool = True,
                            n_users: int = 100) -> dict:
    """
    Execute the complete feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame or None
        Processed TED Talks DataFrame. If None, loads from disk.
    use_word2vec : bool
        Whether to train Word2Vec embeddings (requires gensim).
    n_users : int
        Number of simulated users for the interaction matrix.

    Returns
    -------
    dict
        Dictionary of all produced artifacts.
    """
    print("=" * 60)
    print("  MODULE 3: FEATURE ENGINEERING & VECTORIZATION")
    print("=" * 60)

    # Load data
    if df is None:
        print(f"\n[INFO] Loading processed dataset from {PROCESSED_PATH} …")
        df = pd.read_csv(PROCESSED_PATH)
    print(f"[INFO] Dataset: {df.shape[0]} talks, {df.shape[1]} columns")

    artifacts = {}

    # ── 1. TF-IDF on combined_text (primary) ──────────────────────────────────
    combined_col = _find_col(df, ["combined_text", "transcript_clean"])
    if combined_col:
        vec_combined, mat_combined = build_tfidf_vectorizer(
            df[combined_col], label="combined"
        )
        save_tfidf_model(vec_combined, mat_combined, "combined")
        artifacts["tfidf_combined"] = (vec_combined, mat_combined)

        plot_tfidf_top_terms(vec_combined, mat_combined, "combined")

    # ── 2. TF-IDF on tags_clean ───────────────────────────────────────────────
    tags_col = _find_col(df, ["tags_clean", "tags"])
    if tags_col:
        vec_tags, mat_tags = build_tfidf_vectorizer(
            df[tags_col],
            max_features=2000,
            ngram_range=(1, 1),
            label="tags"
        )
        save_tfidf_model(vec_tags, mat_tags, "tags")
        artifacts["tfidf_tags"] = (vec_tags, mat_tags)

        plot_tfidf_top_terms(vec_tags, mat_tags, "tags")

    # ── 3. TF-IDF on transcript_clean ─────────────────────────────────────────
    transcript_col = _find_col(df, ["transcript_clean"])
    if transcript_col:
        vec_transcript, mat_transcript = build_tfidf_vectorizer(
            df[transcript_col],
            max_features=5000,
            ngram_range=(1, 2),
            label="transcript"
        )
        save_tfidf_model(vec_transcript, mat_transcript, "transcript")
        artifacts["tfidf_transcript"] = (vec_transcript, mat_transcript)

        plot_tfidf_top_terms(vec_transcript, mat_transcript, "transcript")

    # ── 4. Word2Vec embeddings ────────────────────────────────────────────────
    w2v_results = {}
    if use_word2vec:
        try:
            if combined_col:
                w2v_model, w2v_emb = build_word2vec_embeddings(
                    df[combined_col], label="combined"
                )
                save_word2vec_model(w2v_model, w2v_emb, "combined")
                w2v_results["combined"] = (w2v_model, w2v_emb)
        except ImportError:
            print("\n[Word2Vec] gensim not installed — skipping Word2Vec.")
            print("[Word2Vec] Install with: pip install gensim")

    artifacts["word2vec"] = w2v_results

    # ── 5. Cosine similarity matrices ─────────────────────────────────────────
    sim_results = {}
    if "tfidf_combined" in artifacts:
        sim_combined = build_similarity_matrix(mat_combined, "combined")
        save_similarity_matrix(sim_combined, "combined")
        sim_results["combined"] = sim_combined
        plot_similarity_heatmap_sample(sim_combined, df, "combined")
        plot_similarity_distribution(sim_combined, "combined")

    if "tfidf_tags" in artifacts:
        sim_tags = build_similarity_matrix(mat_tags, "tags")
        save_similarity_matrix(sim_tags, "tags")
        sim_results["tags"] = sim_tags

    if "tfidf_transcript" in artifacts:
        sim_transcript = build_similarity_matrix(mat_transcript, "transcript")
        save_similarity_matrix(sim_transcript, "transcript")
        sim_results["transcript"] = sim_transcript

    artifacts["similarity"] = sim_results

    # ── 6. User–item interaction matrix ───────────────────────────────────────
    user_item = build_user_item_matrix(df, n_users=n_users, seed=42)
    save_user_item_matrix(user_item)
    plot_user_item_sparsity(user_item)
    artifacts["user_item"] = user_item

    # ── 7. Demo: show sample recommendations ──────────────────────────────────
    if "combined" in sim_results:
        print("\n" + "=" * 60)
        print("  DEMO: Sample Recommendations (Content-Based)")
        print("=" * 60)

        for sample_idx in [0, 5, 10]:
            if sample_idx < len(df):
                title_col = "title" if "title" in df.columns else df.columns[1]
                print(f"\n  If you liked: '{df.iloc[sample_idx][title_col]}'")
                recs = get_top_similar_talks(
                    sim_results["combined"], df, sample_idx, top_n=3
                )
                for _, row in recs.iterrows():
                    print(f"    → {row['title']} "
                          f"({row['topic']}, sim={row['similarity_score']:.3f})")

    # ── 8. Report ─────────────────────────────────────────────────────────────
    report = generate_feature_report(
        df, {k: v for k, v in artifacts.items() if k.startswith("tfidf")},
        w2v_results if w2v_results else None,
        sim_results,
        user_item,
    )
    print("\n" + report)

    print(f"\n{'=' * 60}")
    print("  Module 3 complete. All artifacts saved to data/models/")
    print(f"{'=' * 60}\n")

    return artifacts


# ────────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_feature_engineering()
