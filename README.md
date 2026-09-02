# LyRemo — Multilingual Lyrical Emotion Model

> **Repo `LSA-model` · model published as `LyRemo`.**

An open-source multilingual emotion classification model trained on ~3M song lyrics.
Given lyrics in any language, LyRemo outputs a probability distribution across
**28 emotions** based on the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) taxonomy.

To be published on HuggingFace Hub for public use. Designed to power downstream
music recommenders (such as [MusicMatch](https://github.com/brinez-juan/MusicMatch))
that match songs by emotional similarity.

**This is a standalone ML project** — it produces a trained, evaluated, published
model. It contains no recommender, API, or frontend; those consume the model once
it's published.

---

## How it works

LyRemo is trained by **knowledge distillation**: a strong English emotion model
pseudo-labels a huge multilingual lyric corpus, and a multilingual student model
learns to reproduce those labels across all languages.

```
data/raw/genius_raw.csv  (~3M songs)
        │  preprocess.py   ── clean lyrics, split into verses
        ▼
genius_verses.parquet     (one row per verse)
        │  label.py        ── pseudo-label each verse with the GoEmotions teacher
        ▼
genius_labeled.parquet    (+ 28-dim emotion vector)
        │  split.py        ── train / val / test partition
        ▼
train / val / test.parquet
        │  train.py        ── fine-tune XLM-RoBERTa (Kaggle T4 GPU)
        ▼
model  →  evaluate.py (per-emotion F1)  →  publish to HuggingFace Hub
```

| Layer                | Tool                                |
|----------------------|-------------------------------------|
| Base (student) model | `xlm-roberta-base`                  |
| Teacher (labeler)    | `SamLowe/roberta-base-go_emotions`  |
| Framework            | HuggingFace `transformers`, `datasets` |
| Training compute     | Kaggle Notebooks (free T4 GPU)      |
| Tracking             | Weights & Biases                    |

---

## Emotion Taxonomy (28 labels)

Canonical order — alphabetical within each group; groups in the order below. This
ordering is fixed across the whole codebase (see `src/lyremo/constants.py`).

| Category       | Labels |
|----------------|--------|
| Positive (12)  | admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief |
| Negative (11)  | anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness |
| Ambiguous (4)  | confusion, curiosity, realization, surprise |
| Neutral (1)    | neutral |

---

## Project Status

| Phase | Status         | Description                                 |
|-------|----------------|---------------------------------------------|
| 0     | ✅ Done        | Project scaffold                            |
| 1     | ✅ Done        | Data cleaning & verse splitting (26.7M verses from ~3M songs; 21 tests passing) |
| 2     | ⏳ Pending     | Pseudo-labeling with GoEmotions teacher     |
| 2b    | ⏳ Pending     | Train / val / test split                    |
| 3     | ⏳ Pending     | Fine-tune XLM-RoBERTa                       |
| 4     | ⏳ Pending     | Evaluation (F1, confusion matrix)           |
| 5     | ⏳ Pending     | HuggingFace Hub publish + model card        |

---

## Setup

```bash
git clone https://github.com/brinez-juan/LSA-model.git
cd LSA-model
pip install -r requirements.txt
```

For development (testing and linting tools):

```bash
pip install -r requirements-dev.txt
```

Run the test suite:

```bash
make test        # or: pytest tests/ -v
```

---

## Data

The raw Genius lyrics dataset is **not included** in this repository (it's ~9 GB).
Source it separately and place it at:

```
data/raw/genius_raw.csv   # or .parquet
```

### Raw schema (columns used)

`id`, `title`, `artist`, `tag` (genre), `year`, `lyrics`, `language` (the
agreed language when both detectors concur). Other columns (`views`, `features`,
`language_cld3`, `language_ft`) are dropped.

### Filtering policy (lenient)

- Drop explicit non-music tags (`book`, `poem`, `prose`, `novel`, `story`); **keep** `misc`.
- Drop rows with empty lyrics or an undetermined language.
- Deduplicate on `(artist, title)`, keeping the first occurrence.

### Verse splitting

Lyrics are split into verse-level rows (the model's training unit):

1. Use `[Verse]` / `[Chorus]` / `[Bridge]` bracket tags as boundaries, then strip them.
2. Fall back to blank-line (double-newline) boundaries when there are no tags.
3. Split any chunk over ~400 words on sentence boundaries (XLM-RoBERTa has a
   512-token limit).
4. Drop chunks shorter than 8 words (noise like "Yeah", "Uh").

Stopwords, punctuation, slang, and contractions are **kept** — transformers need
them. URLs, `@handles`, and bracket annotations are stripped.

### Output schema (verse-level parquet)

`song_id`, `verse_index`, `verse_id` (`{song_id}_{verse_index}`), `title`,
`artist`, `tag`, `year`, `language`, `lyrics_clean`, `token_count`.

Later stages add: `emotions` (28-dim float array) and `dominant_emotion`
(`label.py`), and `split` (`split.py`).

---

## Data Pipeline

Run each step via Make:

```bash
make preprocess-sample   # test on a 10k-song sample first
make preprocess          # full run (~3M songs)
make label               # pseudo-label with GoEmotions teacher
make split               # create train/val/test splits
make help                # list all commands
```

**Phase 1 sanity checks** (verified on the 10k sample): verse count is 4–8× the
song count, median `token_count` is 20–80 words, no NaNs in the key columns, and
`verse_id` is unique across the output.

---

## Training

Training runs on a **Kaggle Notebook** (free T4 GPU) — see
`notebooks/kaggle_training.ipynb`.

---

## License

MIT — see `LICENSE`.
