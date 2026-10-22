# Personalized TED Talks Recommendation System

A machine learning–based recommender system that suggests relevant TED Talks
by analysing content (transcripts, tags) and user preferences.

## Project Structure

```
├── data/                        # Raw and processed datasets
├── notebooks/
│   └── Phase1_EDA_Preprocessing.ipynb   # Interactive EDA notebook
├── outputs/
│   ├── plots/                   # Generated visualisations
│   └── reports/                 # EDA text report
├── src/
│   ├── data_collection.py       # Module 1 – Dataset loading / generation
│   ├── preprocessing.py         # Module 2 – Text cleaning & preprocessing
│   └── eda.py                   # Module 2 – EDA visualisations & report
├── run_phase1.py                # Phase 1 entry point
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Phase 1 (Data Collection + EDA + Preprocessing)
```bash
python run_phase1.py
```

Flags:
- `--synthetic`  Force the built-in synthetic dataset (no Kaggle credentials needed)
- `--no-plots`   Skip plot generation

### 3. Use the real Kaggle dataset (optional)
1. Get a Kaggle API key from https://www.kaggle.com/settings
2. Place `kaggle.json` in `~/.kaggle/`
3. Run `python run_phase1.py` (omit `--synthetic`)

The script will download the **rounakbanik/ted-talks** dataset automatically.

### 4. Jupyter Notebook
```bash
jupyter notebook notebooks/Phase1_EDA_Preprocessing.ipynb
```

## Phase 1 Outputs

| Output | Location |
|--------|----------|
| Processed dataset | `data/ted_talks_processed.csv` |
| Duration distribution | `outputs/plots/01_duration_distribution.png` |
| Views distribution | `outputs/plots/02_views_distribution.png` |
| Top topics | `outputs/plots/03_top_topics.png` |
| Tag frequency | `outputs/plots/04_tag_frequency.png` |
| Tag word cloud | `outputs/plots/05_tag_wordcloud.png` |
| Ratings distribution | `outputs/plots/06_ratings_distribution.png` |
| Talks per year | `outputs/plots/07_talks_per_year.png` |
| Missing values | `outputs/plots/08_missing_values.png` |
| Correlation heatmap | `outputs/plots/09_correlation_heatmap.png` |
| Top speakers | `outputs/plots/10_top_speakers.png` |
| Transcript word count | `outputs/plots/11_transcript_word_count.png` |
| Views vs Rating | `outputs/plots/12_views_vs_rating.png` |
| EDA text report | `outputs/reports/eda_report.txt` |

## Milestones

| Milestone | Weeks | Status |
|-----------|-------|--------|
| **Phase 1** – Data Collection, EDA, Preprocessing | 1–2 | ✅ Complete |
| Phase 2 – Feature Engineering & Vectorisation | 3–4 | 🔜 |
| Phase 3 – Recommendation Engine & Evaluation | 5–6 | 🔜 |
| Phase 4 – Streamlit App & Documentation | 7–8 | 🔜 |
