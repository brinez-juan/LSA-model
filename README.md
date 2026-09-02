# LyRemo — Multilingual Lyrical Emotion Model

An open-source multilingual emotion classification model trained on ~3M song lyrics.
Given lyrics in any language, LyRemo outputs a probability distribution across
**28 emotions** based on the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) taxonomy.

Published on HuggingFace Hub for public use. Designed to power downstream music
recommenders that match songs by emotional similarity.

---

## Emotion Taxonomy

| Category       | Labels |
|----------------|--------|
| Positive (12)  | admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief |
| Negative (11)  | anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness |
| Ambiguous (4)  | confusion, curiosity, realization, surprise |
| Neutral (1)    | neutral |

---

## Project Status

| Phase | Status      | Description                                 |
|-------|-------------|---------------------------------------------|
| 0     | ✅ Done     | Project scaffold                            |
| 1     | 🔄 In progress | Data cleaning & verse splitting          |
| 2     | ⏳ Pending  | Pseudo-labeling with GoEmotions teacher     |
| 2b    | ⏳ Pending  | Train / val / test split                    |
| 3     | ⏳ Pending  | Fine-tune XLM-RoBERTa                       |
| 4     | ⏳ Pending  | Evaluation (F1, confusion matrix)           |
| 5     | ⏳ Pending  | HuggingFace Hub publish + model card        |

---

## Setup

```bash
git clone https://github.com/yourusername/lyremo.git
cd lyremo
pip install -r requirements.txt
```

For development (includes testing and linting tools):
```bash
pip install -r requirements-dev.txt
```

---

## Data

The raw Genius lyrics dataset is **not included** in this repository. You must
source it separately. Once you have it, place it at:

```
data/raw/genius_raw.csv   # or .parquet
```

The dataset schema expected by the pipeline is documented in `CLAUDE.md`.

---

## Data Pipeline

Run each step via Make:

```bash
# Test on a 10k sample first
make preprocess-sample

# Full preprocessing run (~3M songs)
make preprocess

# Pseudo-label with GoEmotions teacher model
make label

# Create train/val/test splits
make split
```

See `Makefile` for all available commands:
```bash
make help
```

---

## Training

Training is designed to run on a **Kaggle Notebook** (free T4 GPU).
See `notebooks/kaggle_training.ipynb` for the self-contained training notebook.

---

## License

MIT
