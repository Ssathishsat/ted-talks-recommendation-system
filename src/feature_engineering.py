"""
Module 3 – Feature Engineering and Vectorization (Enhanced)
-----------------------------------------------------------
Converts text data (transcripts, tags, descriptions, titles) into numerical
representations using:

  1. TF-IDF vectorization        (primary — combined, tags, transcript, title, description)
  2. CountVectorizer / BoW       (secondary — for comparison)
  3. SVD / LSA dimensionality     (reduced representations for visualization)
  4. Cosine similarity matrices   (content-based filtering)
  5. Hybrid similarity matrix     (weighted combination of all signals)
  6. User–item interaction matrix (with ratings + user demographics for CF)

All models and matrices are saved to data/models/ for reuse in deployment.
"""

import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.decomposition import TruncatedSVD

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
    """
    print(f"\n[TF-IDF] Building vectorizer for '{label}' …")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        sublinear_tf=True,
        norm="l2",
    )

    tfidf_matrix = vectorizer.fit_transform(texts.fillna(""))

    vocab_size = len(vectorizer.vocabulary_)
    print(f"[TF-IDF] '{label}' matrix shape : {tfidf_matrix.shape}")
    print(f"[TF-IDF] '{label}' vocabulary   : {vocab_size} terms")
    print(f"[TF-IDF] '{label}' ngram_range   : {ngram_range}")
    sparsity = 1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
    print(f"[TF-IDF] '{label}' sparsity      : {sparsity:.4f}")

    return vectorizer, tfidf_matrix


def save_tfidf_model(vectorizer: TfidfVectorizer,
                     matrix: sparse.csr_matrix,
                     label: str) -> None:
    """Persist the TF-IDF vectorizer and its matrix."""
    vec_path = MODELS_DIR / f"tfidf_{label}_vectorizer.pkl"
    mat_path = MODELS_DIR / f"tfidf_{label}_matrix.npz"

    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    sparse.save_npz(str(mat_path), matrix)

    print(f"[TF-IDF] Saved vectorizer → {vec_path}")
    print(f"[TF-IDF] Saved matrix     → {mat_path}")


# ────────────────────────────────────────────────────────────────────────────
# 2. CountVectorizer / Bag of Words
# ────────────────────────────────────────────────────────────────────────────

def build_count_vectorizer(texts: pd.Series,
                           max_features: int = 5000,
                           ngram_range: tuple = (1, 2),
                           label: str = "combined") -> tuple:
    """
    Fit a CountVectorizer (Bag of Words) for comparison with TF-IDF.
    """
    print(f"\n[BoW] Building CountVectorizer for '{label}' …")

    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=1,
        max_df=0.85,
        stop_words="english",
    )

    bow_matrix = vectorizer.fit_transform(texts.fillna(""))

    vocab_size = len(vectorizer.vocabulary_)
    print(f"[BoW] '{label}' matrix shape : {bow_matrix.shape}")
    print(f"[BoW] '{label}' vocabulary   : {vocab_size} terms")

    return vectorizer, bow_matrix


def save_bow_model(vectorizer: CountVectorizer,
                   matrix: sparse.csr_matrix,
                   label: str) -> None:
    """Persist the BoW vectorizer and its matrix."""
    vec_path = MODELS_DIR / f"bow_{label}_vectorizer.pkl"
    mat_path = MODELS_DIR / f"bow_{label}_matrix.npz"

    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    sparse.save_npz(str(mat_path), matrix)

    print(f"[BoW] Saved vectorizer → {vec_path}")
    print(f"[BoW] Saved matrix     → {mat_path}")


# ────────────────────────────────────────────────────────────────────────────
# 3. SVD / LSA Dimensionality Reduction
# ────────────────────────────────────────────────────────────────────────────

def build_svd_reduction(tfidf_matrix: sparse.csr_matrix,
                        n_components: int = 100,
                        label: str = "combined") -> tuple:
    """
    Apply Truncated SVD (Latent Semantic Analysis) to reduce dimensionality.

    This captures latent topics — groups of words that frequently appear
    together across talks.

    Parameters
    ----------
    tfidf_matrix : sparse.csr_matrix
        TF-IDF matrix to reduce.
    n_components : int
        Number of latent topics to extract.
    label : str

    Returns
    -------
    tuple
        (svd_model, reduced_matrix)
    """
    n_components = min(n_components, tfidf_matrix.shape[1] - 1,
                       tfidf_matrix.shape[0] - 1)

    print(f"\n[SVD] Reducing '{label}' TF-IDF to {n_components} latent topics …")

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(tfidf_matrix)

    explained = svd.explained_variance_ratio_.sum()
    print(f"[SVD] '{label}' reduced shape       : {reduced.shape}")
    print(f"[SVD] '{label}' variance explained   : {explained:.4f} ({explained*100:.1f}%)")

    return svd, reduced


def save_svd_model(svd_model: TruncatedSVD,
                   reduced_matrix: np.ndarray,
                   label: str) -> None:
    """Persist the SVD model and its reduced matrix."""
    svd_path = MODELS_DIR / f"svd_{label}_model.pkl"
    red_path = MODELS_DIR / f"svd_{label}_reduced.npy"

    with open(svd_path, "wb") as f:
        pickle.dump(svd_model, f)
    np.save(str(red_path), reduced_matrix)

    print(f"[SVD] Saved model    → {svd_path}")
    print(f"[SVD] Saved reduced  → {red_path}")


# ────────────────────────────────────────────────────────────────────────────
# 4. Word2Vec Embeddings
# ────────────────────────────────────────────────────────────────────────────

def build_word2vec_embeddings(texts: pd.Series,
                              vector_size: int = 100,
                              window: int = 5,
                              min_count: int = 1,
                              workers: int = 4,
                              seed: int = 42,
                              label: str = "combined") -> tuple:
    """
    Train a Word2Vec model and produce per-document averaged embeddings.
    """
    from gensim.models import Word2Vec

    print(f"\n[Word2Vec] Training model for '{label}' …")

    tokenized = texts.fillna("").apply(lambda t: t.split() if t else [])
    total_tokens = sum(len(t) for t in tokenized)
    print(f"[Word2Vec] Total tokens   : {total_tokens}")
    print(f"[Word2Vec] Avg tokens/doc : {total_tokens / len(tokenized):.1f}")

    model = Word2Vec(
        sentences=tokenized.tolist(),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
        sg=1,
        epochs=20,
    )

    print(f"[Word2Vec] Vocabulary size : {len(model.wv)}")
    print(f"[Word2Vec] Vector size     : {vector_size}")

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
# 5. Cosine Similarity Matrices
# ────────────────────────────────────────────────────────────────────────────

def build_similarity_matrix(matrix, label: str = "combined") -> np.ndarray:
    """
    Compute cosine similarity matrix from a TF-IDF or SVD matrix.
    Uses linear_kernel for sparse matrices (faster for L2-normalized).
    """
    print(f"\n[SIM] Building cosine similarity matrix for '{label}' …")

    if sparse.issparse(matrix):
        sim = linear_kernel(matrix, matrix)
    else:
        sim = cosine_similarity(matrix)

    print(f"[SIM] '{label}' shape          : {sim.shape}")
    print(f"[SIM] '{label}' mean similarity : {np.mean(sim):.4f}")
    n = sim.shape[0]
    upper = sim[np.triu_indices(n, k=1)]
    print(f"[SIM] '{label}' max (excl diag)  : {np.max(upper):.4f}")
    print(f"[SIM] '{label}' std              : {np.std(upper):.4f}")

    return sim


def save_similarity_matrix(sim_matrix: np.ndarray, label: str) -> None:
    """Persist a similarity matrix."""
    path = MODELS_DIR / f"similarity_{label}.npy"
    np.save(str(path), sim_matrix)
    print(f"[SIM] Saved → {path}")


# ────────────────────────────────────────────────────────────────────────────
# 6. Hybrid Similarity Matrix
# ────────────────────────────────────────────────────────────────────────────

def build_hybrid_similarity(sim_combined: np.ndarray,
                            sim_tags: np.ndarray,
                            sim_transcript: np.ndarray,
                            w_combined: float = 0.5,
                            w_tags: float = 0.2,
                            w_transcript: float = 0.3) -> np.ndarray:
    """
    Build a hybrid similarity matrix by combining multiple similarity signals.

    Parameters
    ----------
    sim_combined : np.ndarray
        Similarity based on combined text (title + desc + tags + transcript).
    sim_tags : np.ndarray
        Similarity based on tags only.
    sim_transcript : np.ndarray
        Similarity based on transcripts only.
    w_combined : float
        Weight for combined similarity.
    w_tags : float
        Weight for tag similarity.
    w_transcript : float
        Weight for transcript similarity.

    Returns
    -------
    np.ndarray
        Weighted hybrid similarity matrix.
    """
    print(f"\n[HYBRID] Building hybrid similarity matrix …")
    print(f"[HYBRID] Weights: combined={w_combined}, tags={w_tags}, "
          f"transcript={w_transcript}")

    hybrid = (w_combined * sim_combined
              + w_tags * sim_tags
              + w_transcript * sim_transcript)

    n = hybrid.shape[0]
    upper = hybrid[np.triu_indices(n, k=1)]
    print(f"[HYBRID] Shape            : {hybrid.shape}")
    print(f"[HYBRID] Mean similarity   : {np.mean(upper):.4f}")
    print(f"[HYBRID] Max (excl diag)   : {np.max(upper):.4f}")
    print(f"[HYBRID] Std               : {np.std(upper):.4f}")

    return hybrid


# ────────────────────────────────────────────────────────────────────────────
# 7. User–Item Interaction Matrix (Enhanced with ratings + demographics)
# ────────────────────────────────────────────────────────────────────────────

def build_user_item_matrix(df: pd.DataFrame,
                           n_users: int = 200,
                           seed: int = 42) -> tuple:
    """
    Simulate a user–item interaction matrix for collaborative filtering.

    Each user gets:
    - 2-4 preferred topics
    - An age group and country (demographics)
    - 5-20 watched talks (weighted by views and rating)
    - A 1-5 star rating per watched talk

    Parameters
    ----------
    df : pd.DataFrame
    n_users : int
    seed : int

    Returns
    -------
    tuple
        (user_item_matrix, user_profiles_df)
    """
    print(f"\n[USER-ITEM] Simulating {n_users} users × {len(df)} talks …")

    rng = np.random.RandomState(seed)
    n_talks = len(df)

    topics = df["topic"].unique() if "topic" in df.columns else []
    n_topics = len(topics)

    # ── User demographics ─────────────────────────────────────────────────────
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    countries = ["USA", "UK", "India", "Germany", "Brazil",
                 "Japan", "Canada", "Australia", "France", "Nigeria"]
    genders = ["M", "F", "M", "F", "M", "F"]  # roughly balanced

    user_profiles = []
    for uid in range(n_users):
        k = rng.randint(2, min(5, n_topics + 1))
        prefs = list(rng.choice(topics, size=k, replace=False))
        user_profiles.append({
            "user_id": uid,
            "age_group": rng.choice(age_groups),
            "country": rng.choice(countries),
            "gender": rng.choice(genders),
            "preferred_topics": prefs,
        })

    # ── Build interaction matrix with ratings ─────────────────────────────────
    rows, cols, ratings_data = [], [], []

    views = df["views"].fillna(1).values.astype(float)
    view_probs = views / views.sum()

    talk_ratings = df["average_rating"].fillna(4.0).values

    for user_id in range(n_users):
        prefs = set(user_profiles[user_id]["preferred_topics"])

        if "topic" in df.columns:
            mask = df["topic"].isin(prefs).values
        else:
            mask = np.ones(n_talks, dtype=bool)

        candidate_indices = np.where(mask)[0]
        if len(candidate_indices) == 0:
            continue

        n_watches = rng.randint(5, min(21, len(candidate_indices) + 1))

        candidate_view_probs = view_probs[candidate_indices]
        candidate_view_probs = candidate_view_probs / candidate_view_probs.sum()

        watched = rng.choice(
            candidate_indices,
            size=n_watches,
            replace=False,
            p=candidate_view_probs,
        )

        for talk_idx in watched:
            rows.append(user_id)
            cols.append(int(talk_idx))
            # Rating: base on talk's average rating + user noise
            base_rating = talk_ratings[talk_idx]
            user_noise = rng.normal(0, 0.3)
            user_rating = round(np.clip(base_rating + user_noise, 1.0, 5.0), 2)
            ratings_data.append(user_rating)

    user_item = sparse.csr_matrix(
        (ratings_data, (rows, cols)),
        shape=(n_users, n_talks),
        dtype=np.float32,
    )

    # Binary version for some algorithms
    binary_ui = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_talks),
        dtype=np.float32,
    )

    sparsity = 1 - user_item.nnz / (n_users * n_talks)
    print(f"[USER-ITEM] Matrix shape       : {user_item.shape}")
    print(f"[USER-ITEM] Total interactions : {user_item.nnz}")
    print(f"[USER-ITEM] Avg watches/user   : {user_item.nnz / n_users:.1f}")
    print(f"[USER-ITEM] Sparsity           : {sparsity:.4f}")
    print(f"[USER-ITEM] Avg rating         : {np.mean(ratings_data):.2f}")
    print(f"[USER-ITEM] Rating range       : {min(ratings_data):.1f} - {max(ratings_data):.1f}")

    # Save user profiles
    user_profiles_df = pd.DataFrame([
        {k: v if k != "preferred_topics" else ", ".join(v)
         for k, v in p.items()}
        for p in user_profiles
    ])

    return user_item, binary_ui, user_profiles_df


def save_user_item_matrix(user_item: sparse.csr_matrix,
                          binary_ui: sparse.csr_matrix,
                          user_profiles: pd.DataFrame) -> None:
    """Persist the user–item matrices and profiles."""
    ui_path = MODELS_DIR / "user_item_matrix.npz"
    binary_path = MODELS_DIR / "user_item_binary.npz"
    profiles_path = MODELS_DIR / "user_profiles.csv"

    sparse.save_npz(str(ui_path), user_item)
    sparse.save_npz(str(binary_path), binary_ui)
    user_profiles.to_csv(str(profiles_path), index=False)

    print(f"[USER-ITEM] Saved rated matrix  → {ui_path}")
    print(f"[USER-ITEM] Saved binary matrix → {binary_path}")
    print(f"[USER-ITEM] Saved user profiles → {profiles_path}")


# ────────────────────────────────────────────────────────────────────────────
# 8. Visualization — Feature Importance & Similarity
# ────────────────────────────────────────────────────────────────────────────

def plot_tfidf_top_terms(vectorizer: TfidfVectorizer,
                          tfidf_matrix: sparse.csr_matrix,
                          label: str,
                          top_n: int = 20) -> None:
    """Bar chart of top-N terms by average TF-IDF score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"\n[VIZ] Plotting top TF-IDF terms for '{label}' …")

    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()

    top_indices = np.argsort(mean_scores)[::-1][:top_n]
    top_terms = [feature_names[i] for i in top_indices]
    top_scores = mean_scores[top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_terms[::-1], top_scores[::-1],
            color=sns.color_palette("Spectral", top_n), edgecolor="white")
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
    """Heatmap of pairwise cosine similarity for a random sample of talks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"\n[VIZ] Plotting similarity heatmap for '{label}' …")

    rng = np.random.RandomState(seed)
    n = min(n_sample, len(df))
    sample_idx = rng.choice(len(df), size=n, replace=False)
    sample_sim = sim_matrix[np.ix_(sample_idx, sample_idx)]

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


def plot_similarity_distribution(sim_matrix: np.ndarray, label: str) -> None:
    """Histogram of pairwise similarity scores (excluding diagonal)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n[VIZ] Plotting similarity distribution for '{label}' …")

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
    """Visualize the sparsity pattern of the user–item interaction matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n[VIZ] Plotting user-item matrix sparsity …")

    n_show_users = min(50, user_item.shape[0])
    n_show_items = min(50, user_item.shape[1])
    sample = user_item[:n_show_users, :n_show_items].toarray()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.spy(sample, markersize=3, color="#4C72B0", aspect="auto")
    ax.set_title("User–Item Interaction Matrix (sample)\n(dots = watched talks)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Talk Index")
    ax.set_ylabel("User Index")
    plt.tight_layout()

    path = PLOTS_DIR / "16_user_item_sparsity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved → {path}")


def plot_svd_scatter(reduced_matrix: np.ndarray,
                     df: pd.DataFrame,
                     label: str = "combined") -> None:
    """Scatter plot of talks in SVD latent space (first 2 components)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"\n[VIZ] Plotting SVD scatter for '{label}' …")

    fig, ax = plt.subplots(figsize=(10, 8))

    topic_col = "topic" if "topic" in df.columns else None
    if topic_col:
        topics = df[topic_col]
        palette = dict(zip(topics.unique(),
                           sns.color_palette("tab20", topics.nunique())))
        colors = [palette[t] for t in topics]
    else:
        colors = "#4C72B0"

    ax.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1],
               c=colors, alpha=0.5, s=15, edgecolors="none")

    ax.set_title(f"SVD Latent Space — '{label}' (first 2 components)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()

    path = PLOTS_DIR / f"17_svd_scatter_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved → {path}")


def plot_rating_distribution(user_item: sparse.csr_matrix) -> None:
    """Histogram of user ratings in the interaction matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n[VIZ] Plotting user rating distribution …")

    ratings = user_item.data
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratings, bins=20, color="#DD8452", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(ratings), color="crimson", ls="--", lw=1.5,
               label=f"Mean = {np.mean(ratings):.2f}")
    ax.set_title("User Rating Distribution in Interaction Matrix",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Rating (1-5 stars)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()

    path = PLOTS_DIR / "18_user_rating_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved → {path}")


def plot_tfidf_vs_bow_comparison(tfidf_matrix: sparse.csr_matrix,
                                  bow_matrix: sparse.csr_matrix,
                                  label: str = "combined") -> None:
    """Compare TF-IDF vs BoW sparsity and density."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"\n[VIZ] Plotting TF-IDF vs BoW comparison for '{label}' …")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sparsity comparison
    methods = ["TF-IDF", "BoW"]
    matrices = [tfidf_matrix, bow_matrix]
    sparsities = [1 - m.nnz / (m.shape[0] * m.shape[1]) for m in matrices]
    nnz_counts = [m.nnz for m in matrices]

    axes[0].bar(methods, sparsities, color=["#4C72B0", "#DD8452"],
                edgecolor="white")
    axes[0].set_title("Sparsity Comparison", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Sparsity (higher = more sparse)")
    for i, v in enumerate(sparsities):
        axes[0].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=10)

    axes[1].bar(methods, nnz_counts, color=["#4C72B0", "#DD8452"],
                edgecolor="white")
    axes[1].set_title("Non-Zero Elements", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(nnz_counts):
        axes[1].text(i, v + 50, f"{v:,}", ha="center", fontsize=10)

    fig.suptitle(f"TF-IDF vs Bag of Words — '{label}'",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = PLOTS_DIR / f"19_tfidf_vs_bow_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved → {path}")


def plot_hybrid_vs_individual_similarity(sim_combined: np.ndarray,
                                          sim_hybrid: np.ndarray) -> None:
    """Compare hybrid similarity distribution vs individual."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n[VIZ] Plotting hybrid vs individual similarity comparison …")

    n = sim_combined.shape[0]
    combined_upper = sim_combined[np.triu_indices(n, k=1)]
    hybrid_upper = sim_hybrid[np.triu_indices(n, k=1)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(combined_upper, bins=50, alpha=0.6, color="#4C72B0",
            label=f"Combined only (mean={np.mean(combined_upper):.4f})")
    ax.hist(hybrid_upper, bins=50, alpha=0.6, color="#DD8452",
            label=f"Hybrid (mean={np.mean(hybrid_upper):.4f})")
    ax.set_title("Hybrid vs Combined Similarity Distribution",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Pair Count")
    ax.legend()
    plt.tight_layout()

    path = PLOTS_DIR / "20_hybrid_vs_combined_similarity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved → {path}")


# ────────────────────────────────────────────────────────────────────────────
# 9. Documentation Report
# ────────────────────────────────────────────────────────────────────────────

def generate_feature_report(df: pd.DataFrame,
                            tfidf_results: dict,
                            bow_results: dict,
                            svd_results: dict,
                            w2v_results: dict | None,
                            sim_results: dict,
                            user_item: sparse.csr_matrix) -> str:
    """Generate a text report documenting all vectorization choices and results."""
    lines = [
        "=" * 65,
        "  TED Talks Recommendation System – Feature Engineering Report",
        "  Module 3 (Milestone 2, Phase 2) — Enhanced",
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
    lines.append("    • min_df = 1, max_df = 0.85")
    lines.append("    • sublinear_tf = True, norm = 'l2'")
    lines.append("")
    for label, (vec, mat) in tfidf_results.items():
        lines.append(f"  '{label}':")
        lines.append(f"    Matrix shape  : {mat.shape}")
        lines.append(f"    Vocabulary    : {len(vec.vocabulary_)} terms")
        sparsity = 1 - mat.nnz / (mat.shape[0] * mat.shape[1])
        lines.append(f"    Sparsity      : {sparsity:.4f}")

    # ── BoW ───────────────────────────────────────────────────────────────────
    _section("2. Bag of Words (CountVectorizer)")
    lines.append("  Method : Raw term counts (for comparison with TF-IDF)")
    lines.append("  Library: scikit-learn CountVectorizer")
    lines.append("")
    for label, (vec, mat) in bow_results.items():
        lines.append(f"  '{label}':")
        lines.append(f"    Matrix shape  : {mat.shape}")
        lines.append(f"    Vocabulary    : {len(vec.vocabulary_)} terms")

    # ── SVD ───────────────────────────────────────────────────────────────────
    _section("3. SVD / LSA Dimensionality Reduction")
    lines.append("  Method : Truncated SVD (Latent Semantic Analysis)")
    lines.append("  Library: scikit-learn TruncatedSVD")
    lines.append("")
    for label, (svd, reduced) in svd_results.items():
        explained = svd.explained_variance_ratio_.sum()
        lines.append(f"  '{label}':")
        lines.append(f"    Reduced shape : {reduced.shape}")
        lines.append(f"    Components     : {svd.n_components}")
        lines.append(f"    Var explained  : {explained:.4f} ({explained*100:.1f}%)")

    # ── Word2Vec ───────────────────────────────────────────────────────────────
    if w2v_results:
        _section("4. Word2Vec Embeddings")
        lines.append("  Method : Skip-gram Word2Vec with averaged document vectors")
        lines.append("  Library: gensim.models.Word2Vec")
        lines.append("  Parameters: vector_size=100, window=5, sg=1, epochs=20")
        lines.append("")
        for label, (model, emb) in w2v_results.items():
            lines.append(f"  '{label}':")
            lines.append(f"    Embedding shape : {emb.shape}")
            lines.append(f"    Vocabulary size  : {len(model.wv)}")

    # ── Similarity ────────────────────────────────────────────────────────────
    _section("5. Cosine Similarity Matrices")
    lines.append("  Method : linear_kernel (cosine for L2-normalized vectors)")
    lines.append("")
    for label, sim in sim_results.items():
        lines.append(f"  '{label}':")
        lines.append(f"    Shape            : {sim.shape}")
        upper = sim[np.triu_indices(sim.shape[0], k=1)]
        lines.append(f"    Mean similarity  : {np.mean(upper):.4f}")
        lines.append(f"    Std similarity   : {np.std(upper):.4f}")
        lines.append(f"    Min / Max (excl diag): {np.min(upper):.4f} / {np.max(upper):.4f}")

    # ── User-Item ──────────────────────────────────────────────────────────────
    _section("6. User–Item Interaction Matrix")
    lines.append(f"  Shape             : {user_item.shape}")
    lines.append(f"  Total interactions: {user_item.nnz}")
    lines.append(f"  Avg watches/user  : {user_item.nnz / user_item.shape[0]:.1f}")
    sparsity = 1 - user_item.nnz / (user_item.shape[0] * user_item.shape[1])
    lines.append(f"  Sparsity          : {sparsity:.4f}")
    if user_item.nnz > 0:
        lines.append(f"  Avg rating        : {np.mean(user_item.data):.2f}")
        lines.append(f"  Rating range      : {np.min(user_item.data):.1f} - {np.max(user_item.data):.1f}")
    lines.append("  Method            : Simulated with demographics —")
    lines.append("                      200 users with age, country, gender,")
    lines.append("                      preferred topics. Watches 5-20 talks")
    lines.append("                      weighted by views. Ratings based on")
    lines.append("                      talk's average rating + user noise.")

    # ── Files saved ───────────────────────────────────────────────────────────
    _section("7. Saved Artifacts")
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
# 10. Helper
# ────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_top_similar_talks(sim_matrix: np.ndarray,
                          df: pd.DataFrame,
                          talk_idx: int,
                          top_n: int = 5) -> pd.DataFrame:
    """Return the top-N most similar talks for a given talk index."""
    scores = sim_matrix[talk_idx].copy()
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
# 11. Master Pipeline
# ────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(df: pd.DataFrame | None = None,
                            use_word2vec: bool = True,
                            n_users: int = 200) -> dict:
    """
    Execute the complete enhanced feature engineering pipeline.
    """
    print("=" * 60)
    print("  MODULE 3: FEATURE ENGINEERING & VECTORIZATION (ENHANCED)")
    print("=" * 60)

    # Load data
    if df is None:
        print(f"\n[INFO] Loading processed dataset from {PROCESSED_PATH} …")
        df = pd.read_csv(PROCESSED_PATH)
    print(f"[INFO] Dataset: {df.shape[0]} talks, {df.shape[1]} columns")

    artifacts = {}

    # ── 1. TF-IDF on combined_text ────────────────────────────────────────────
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
            df[tags_col], max_features=2000, ngram_range=(1, 1), label="tags"
        )
        save_tfidf_model(vec_tags, mat_tags, "tags")
        artifacts["tfidf_tags"] = (vec_tags, mat_tags)
        plot_tfidf_top_terms(vec_tags, mat_tags, "tags")

    # ── 3. TF-IDF on transcript_clean ─────────────────────────────────────────
    transcript_col = _find_col(df, ["transcript_clean"])
    if transcript_col:
        vec_transcript, mat_transcript = build_tfidf_vectorizer(
            df[transcript_col], max_features=5000, ngram_range=(1, 2),
            label="transcript"
        )
        save_tfidf_model(vec_transcript, mat_transcript, "transcript")
        artifacts["tfidf_transcript"] = (vec_transcript, mat_transcript)
        plot_tfidf_top_terms(vec_transcript, mat_transcript, "transcript")

    # ── 4. TF-IDF on title_clean ──────────────────────────────────────────────
    title_col = _find_col(df, ["title_clean"])
    if title_col:
        vec_title, mat_title = build_tfidf_vectorizer(
            df[title_col], max_features=1000, ngram_range=(1, 1), label="title"
        )
        save_tfidf_model(vec_title, mat_title, "title")
        artifacts["tfidf_title"] = (vec_title, mat_title)
        plot_tfidf_top_terms(vec_title, mat_title, "title")

    # ── 5. TF-IDF on description_clean ────────────────────────────────────────
    desc_col = _find_col(df, ["description_clean"])
    if desc_col:
        vec_desc, mat_desc = build_tfidf_vectorizer(
            df[desc_col], max_features=2000, ngram_range=(1, 1), label="description"
        )
        save_tfidf_model(vec_desc, mat_desc, "description")
        artifacts["tfidf_description"] = (vec_desc, mat_desc)

    # ── 6. BoW on combined_text (for comparison) ──────────────────────────────
    if combined_col:
        bow_vec, bow_mat = build_count_vectorizer(
            df[combined_col], label="combined"
        )
        save_bow_model(bow_vec, bow_mat, "combined")
        artifacts["bow_combined"] = (bow_vec, bow_mat)
        plot_tfidf_vs_bow_comparison(mat_combined, bow_mat, "combined")

    # ── 7. SVD / LSA on combined TF-IDF ───────────────────────────────────────
    if "tfidf_combined" in artifacts:
        svd_model, svd_reduced = build_svd_reduction(
            mat_combined, n_components=100, label="combined"
        )
        save_svd_model(svd_model, svd_reduced, "combined")
        artifacts["svd_combined"] = (svd_model, svd_reduced)
        plot_svd_scatter(svd_reduced, df, "combined")

    # ── 8. SVD on transcript TF-IDF ────────────────────────────────────────────
    if "tfidf_transcript" in artifacts:
        svd_t, svd_r_t = build_svd_reduction(
            mat_transcript, n_components=50, label="transcript"
        )
        save_svd_model(svd_t, svd_r_t, "transcript")
        artifacts["svd_transcript"] = (svd_t, svd_r_t)

    # ── 9. Word2Vec embeddings ────────────────────────────────────────────────
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
            print("\n[Word2Vec] gensim not installed — skipping.")
            print("[Word2Vec] Install: pip install gensim")

    artifacts["word2vec"] = w2v_results

    # ── 10. Cosine similarity matrices ────────────────────────────────────────
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

    # ── 11. SVD-based similarity ──────────────────────────────────────────────
    if "svd_combined" in artifacts:
        sim_svd = build_similarity_matrix(svd_reduced, "svd_combined")
        save_similarity_matrix(sim_svd, "svd_combined")
        sim_results["svd_combined"] = sim_svd

    # ── 12. Hybrid similarity matrix ──────────────────────────────────────────
    if sim_results.get("combined") is not None and \
       sim_results.get("tags") is not None and \
       sim_results.get("transcript") is not None:
        sim_hybrid = build_hybrid_similarity(
            sim_combined, sim_tags, sim_transcript,
            w_combined=0.5, w_tags=0.2, w_transcript=0.3
        )
        save_similarity_matrix(sim_hybrid, "hybrid")
        sim_results["hybrid"] = sim_hybrid
        plot_hybrid_vs_individual_similarity(sim_combined, sim_hybrid)

    artifacts["similarity"] = sim_results

    # ── 13. User–item interaction matrix ──────────────────────────────────────
    user_item, binary_ui, user_profiles = build_user_item_matrix(
        df, n_users=n_users, seed=42
    )
    save_user_item_matrix(user_item, binary_ui, user_profiles)
    plot_user_item_sparsity(user_item)
    plot_rating_distribution(user_item)
    artifacts["user_item"] = user_item

    # ── 14. Demo: show sample recommendations ─────────────────────────────────
    if sim_results.get("combined") is not None:
        print("\n" + "=" * 60)
        print("  DEMO: Sample Recommendations (Content-Based)")
        print("=" * 60)

        for sample_idx in [0, 50, 100, 200]:
            if sample_idx < len(df):
                title_col_name = "title" if "title" in df.columns else df.columns[1]
                print(f"\n  If you liked: '{df.iloc[sample_idx][title_col_name]}'")
                recs = get_top_similar_talks(
                    sim_results["combined"], df, sample_idx, top_n=3
                )
                for _, row in recs.iterrows():
                    print(f"    → {row['title']} "
                          f"({row['topic']}, sim={row['similarity_score']:.3f})")

    # ── 15. Demo: hybrid recommendations ──────────────────────────────────────
    if sim_results.get("hybrid") is not None:
        print("\n" + "-" * 60)
        print("  DEMO: Sample Recommendations (Hybrid)")
        print("-" * 60)

        for sample_idx in [0, 50, 100]:
            if sample_idx < len(df):
                title_col_name = "title" if "title" in df.columns else df.columns[1]
                print(f"\n  If you liked: '{df.iloc[sample_idx][title_col_name]}'")
                recs = get_top_similar_talks(
                    sim_results["hybrid"], df, sample_idx, top_n=3
                )
                for _, row in recs.iterrows():
                    print(f"    → {row['title']} "
                          f"({row['topic']}, sim={row['similarity_score']:.3f})")

    # ── 16. Report ────────────────────────────────────────────────────────────
    tfidf_only = {k: v for k, v in artifacts.items() if k.startswith("tfidf")}
    bow_only = {k: v for k, v in artifacts.items() if k.startswith("bow")}
    svd_only = {k: v for k, v in artifacts.items() if k.startswith("svd")}

    report = generate_feature_report(
        df, tfidf_only, bow_only, svd_only,
        w2v_results if w2v_results else None,
        sim_results, user_item
    )
    print("\n" + report)

    print(f"\n{'=' * 60}")
    print("  Module 3 (Enhanced) complete.")
    print(f"  {len(list(MODELS_DIR.glob('*')))} artifacts saved to data/models/")
    print(f"{'=' * 60}\n")

    return artifacts


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_feature_engineering()
