"""
preprocess.py
=============
Phase 1: Clean Genius lyrics and split into verse-level rows.

Input:  Raw Genius CSV/parquet (~3M songs)
Output: Verse-level parquet, one row per verse, ready for pseudo-labeling

Usage (via Make):
    make preprocess           # full run
    make preprocess-sample    # 10k song sample for testing

Usage (direct):
    python -m lyremo.data.preprocess \
        --input data/raw/genius_raw.csv \
        --output data/processed/genius_verses.parquet \
        [--sample 10000]
"""

import re
import time
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Default rows per CSV/parquet streaming chunk. ~100k rows of Genius lyrics
# is roughly 150–300 MB of string data in pandas — safely below typical RAM
# limits while amortizing per-chunk overhead.
DEFAULT_CHUNKSIZE = 100_000


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Tags that are explicitly non-music — drop these entries
NON_MUSIC_TAGS = {"book", "poem", "prose", "novel", "story"}

# Columns to load from the raw dataset (others are dropped)
KEEP_COLUMNS = ["id", "title", "artist", "tag", "year", "language", "lyrics"]

# Minimum words for a verse to be kept (filters out noise like "Yeah" or "Uh")
MIN_WORDS_PER_VERSE = 8

# Maximum words per chunk before falling back to sentence-level splitting.
# ~400 words stays safely under XLM-RoBERTa's 512-token limit after subword expansion.
MAX_WORDS_PER_CHUNK = 400

# Patterns to strip from lyrics (applied after using brackets as boundaries)
STRIP_PATTERNS = [
    r"\[.*?\]",           # Any remaining [bracket] annotations
    r"http\S+",           # URLs
    r"@\w+",              # Social handles
    r"\(adsbygoogle.*",   # Ad injection artifacts from scraping
]


# ─────────────────────────────────────────────
# FILTERING
# ─────────────────────────────────────────────

def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply lenient filtering policy:
    - Drop explicit non-music tags (books, poems, etc.)
    - Keep 'misc' — many non-English songs are miscategorized there
    - Drop rows with no lyrics
    - Drop rows where language is NaN (both detectors disagreed)
    - Deduplicate on (artist, title), keep first occurrence
    """
    initial = len(df)

    if "tag" in df.columns:
        df = df[~df["tag"].str.lower().isin(NON_MUSIC_TAGS)]

    df = df[df["lyrics"].notna() & (df["lyrics"].str.strip() != "")]
    df = df[df["language"].notna()]
    df = df.drop_duplicates(subset=["artist", "title"], keep="first")

    print(f"[filter] {initial:,} → {len(df):,} rows after filtering")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────

def extract_verse_boundaries(text: str) -> list[str]:
    """
    Use [Verse], [Chorus], [Bridge] etc. as explicit verse boundary markers
    before stripping them. This is the primary splitting strategy for annotated songs.

    Returns a list of raw verse strings (not yet cleaned of other noise).
    Returns an empty list if no bracket annotations are found.
    """
    # Find all positions of [bracket] tags
    bracket_positions = [m.start() for m in re.finditer(r"\[.*?\]", text)]

    if not bracket_positions:
        return []

    verses = []
    for i, pos in enumerate(bracket_positions):
        start = pos
        end = bracket_positions[i + 1] if i + 1 < len(bracket_positions) else len(text)
        verse = text[start:end].strip()
        if verse:
            verses.append(verse)

    return verses


def clean_text(text: str) -> str:
    """
    Strip annotations and noise from a verse or lyric block.
    Keeps stopwords, punctuation, slang, and contractions — transformers need them.
    """
    for pattern in STRIP_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Normalize whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3+ consecutive newlines to 2 (verse boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ─────────────────────────────────────────────
# VERSE SPLITTING
# ─────────────────────────────────────────────

def split_long_chunk(text: str) -> list[str]:
    """
    Fallback: split a chunk that exceeds MAX_WORDS_PER_CHUNK on sentence
    boundaries (.!?) to stay within XLM-RoBERTa's token limit.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current: list[str] = []
    current_count = 0

    for sentence in sentences:
        s_count = len(sentence.split())
        if current_count + s_count > MAX_WORDS_PER_CHUNK and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_count = s_count
        else:
            current.append(sentence)
            current_count += s_count

    if current:
        chunks.append(" ".join(current))

    return chunks


def split_into_verses(raw_lyrics: str) -> list[str]:
    """
    Split raw lyrics into cleaned verse chunks using this hierarchy:

    1. Use [bracket] tags as explicit verse boundaries (annotated songs)
    2. Fall back to double newlines for unannotated songs
    3. If any chunk exceeds MAX_WORDS_PER_CHUNK, split on sentence boundaries
    4. Drop chunks below MIN_WORDS_PER_VERSE

    Returns a list of cleaned verse strings.
    """
    # Strategy 1: bracket-annotated verses
    raw_verses = extract_verse_boundaries(raw_lyrics)

    # Strategy 2: fallback to double newlines
    if not raw_verses:
        raw_verses = re.split(r"\n{2,}", raw_lyrics)

    # Clean each verse and handle oversized chunks
    final_verses = []
    for verse in raw_verses:
        cleaned = clean_text(verse)
        if not cleaned:
            continue

        word_count = len(cleaned.split())

        if word_count > MAX_WORDS_PER_CHUNK:
            # Strategy 3: sentence-level fallback for very long verses
            final_verses.extend(split_long_chunk(cleaned))
        else:
            final_verses.append(cleaned)

    # Strategy 4: drop noise chunks
    final_verses = [v for v in final_verses if len(v.split()) >= MIN_WORDS_PER_VERSE]

    return final_verses


# ─────────────────────────────────────────────
# EXPLODE TO VERSE-LEVEL ROWS
# ─────────────────────────────────────────────

def explode_to_verses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform from one-row-per-song to one-row-per-verse.
    Each verse row inherits all metadata from its parent song.
    """
    records = []

    for _, row in df.iterrows():
        verses = split_into_verses(row["lyrics"])

        for i, verse_text in enumerate(verses):
            records.append({
                "song_id":      row["id"],
                "verse_index":  i,
                "verse_id":     f"{row['id']}_{i}",
                "title":        row["title"],
                "artist":       row["artist"],
                "tag":          row.get("tag"),
                "year":         row.get("year"),
                "language":     row["language"],
                "lyrics_clean": verse_text,
                "token_count":  len(verse_text.split()),
            })

    result = pd.DataFrame(records)
    print(f"[explode] {len(df):,} songs → {len(result):,} verses")
    return result


# ─────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────

def print_diagnostics(df: pd.DataFrame) -> None:
    print("\n── Dataset Diagnostics ──────────────────────────")
    print(f"  Total verses:             {len(df):,}")
    print(f"  Unique songs:             {df['song_id'].nunique():,}")
    print(f"  Unique artists:           {df['artist'].nunique():,}")
    print(f"  Languages detected:       {df['language'].nunique()}")
    print(f"\n  Top 10 languages:")
    print(df["language"].value_counts().head(10).to_string())
    print(f"\n  Verses per song (median): {df.groupby('song_id').size().median():.0f}")
    print(f"\n  Token count distribution:")
    print(df["token_count"].describe().to_string())
    print("─────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# PROGRESS TRACKING
# ─────────────────────────────────────────────

def print_progress(
    chunk_idx: int,
    rows_in_chunk: int,
    verses_in_chunk: int,
    total_rows: int,
    total_verses: int,
    start_time: float,
    expected_rows: int | None = None,
) -> None:
    """Print a single-line progress update for one streamed chunk.

    Shows per-chunk and cumulative row/verse counts, elapsed wall time, and
    throughput. If `expected_rows` is known (parquet input exposes row count
    cheaply; CSV does not), also prints percent complete and an ETA.
    """
    elapsed = time.time() - start_time
    rate = total_rows / elapsed if elapsed > 0 else 0.0
    msg = (
        f"[chunk {chunk_idx:>4}] "
        f"rows={rows_in_chunk:>7,}  verses+={verses_in_chunk:>7,}  |  "
        f"total rows={total_rows:>11,}  verses={total_verses:>11,}  |  "
        f"{elapsed:6.1f}s  {rate:>7,.0f} rows/s"
    )
    if expected_rows:
        pct = 100.0 * total_rows / expected_rows
        remaining = max(expected_rows - total_rows, 0)
        eta_min = (remaining / rate / 60.0) if rate > 0 else 0.0
        msg += f"  |  {pct:5.1f}%  ETA {eta_min:5.1f}m"
    print(msg, flush=True)


# ─────────────────────────────────────────────
# STREAMING I/O HELPERS
# ─────────────────────────────────────────────

def _normalize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce verse DataFrame columns to nullable, consistent dtypes so
    PyArrow schemas align across streamed chunks. Safe to call repeatedly.
    """
    df = df.copy()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    text_cols = ("song_id", "verse_id", "title", "artist", "tag", "language", "lyrics_clean")
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
    df["verse_index"] = df["verse_index"].astype("int32")
    df["token_count"] = df["token_count"].astype("int32")
    return df


def _drop_seen_songs(df: pd.DataFrame, seen_keys: set) -> tuple[pd.DataFrame, int]:
    """Drop rows whose (artist, title) was already seen in a prior chunk.

    `filter_rows` only dedups within its input, so in streaming mode we need
    this second pass to enforce global deduplication across chunks. Mirrors
    the same key semantics as `drop_duplicates(subset=["artist", "title"])`.

    Updates `seen_keys` in place with new survivors. Returns (df, dropped).
    """
    if df.empty:
        return df, 0
    keys = list(zip(df["artist"].astype(str), df["title"].astype(str)))
    mask = [k not in seen_keys for k in keys]
    survivors = df[mask]
    seen_keys.update(k for k, keep in zip(keys, mask) if keep)
    return survivors.reset_index(drop=True), len(df) - len(survivors)


def _iter_input_chunks(input_path: Path, chunksize: int):
    """Yield (chunk_df, expected_rows_or_None) tuples for CSV or parquet input.
    For parquet we can read `num_rows` from metadata; CSV row count is not
    computed (would require scanning the file twice), so ETA is skipped.
    """
    if input_path.suffix == ".parquet":
        pf = pq.ParquetFile(input_path)
        expected_rows = pf.metadata.num_rows
        schema_cols = [c for c in KEEP_COLUMNS if c in pf.schema.names]
        for batch in pf.iter_batches(batch_size=chunksize, columns=schema_cols):
            yield batch.to_pandas(), expected_rows
    else:
        reader = pd.read_csv(input_path, chunksize=chunksize, low_memory=False)
        for chunk in reader:
            available = [c for c in KEEP_COLUMNS if c in chunk.columns]
            yield chunk[available], None


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def _run_sample(input_path: Path, output_path: Path, sample_n: int) -> None:
    """Non-streaming path for small samples. 10k rows fits easily in memory,
    so we keep the original load-filter-explode-save flow for reproducibility.
    """
    print(f"[load] Reading {input_path} (sample mode, n={sample_n:,})...")
    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" \
        else pd.read_csv(input_path, low_memory=False)

    available = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[available]
    df = df.sample(n=min(sample_n, len(df)), random_state=42)
    print(f"[sample] Using {len(df):,} songs")

    df = filter_rows(df)
    df = explode_to_verses(df)

    print_diagnostics(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[save] Saved to {output_path}")


def _run_streaming(input_path: Path, output_path: Path, chunksize: int) -> None:
    """Streaming path: read the raw dataset in fixed-size chunks, pipe each
    chunk through filter_rows + explode_to_verses, and append it to the
    output parquet via a single ParquetWriter kept open for the whole run.

    `filter_rows` only dedups within its input; `_drop_seen_songs` runs a
    second pass against a persistent `(artist, title)` set so deduplication
    is enforced globally across chunks.
    """
    print(f"[stream] Reading {input_path} in chunks of {chunksize:,}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    start_time = time.time()

    total_rows_read = 0
    total_verses_written = 0
    cross_chunk_duplicates = 0
    seen_song_keys: set = set()
    unique_songs: set = set()
    language_counts: Counter = Counter()

    try:
        for chunk_idx, (chunk_df, expected_rows) in enumerate(
            _iter_input_chunks(input_path, chunksize), start=1
        ):
            rows_in_chunk = len(chunk_df)
            total_rows_read += rows_in_chunk

            # ── Core logic (untouched) ─────────────────────────
            chunk_df = filter_rows(chunk_df)
            chunk_df, dropped_xchunk = _drop_seen_songs(chunk_df, seen_song_keys)
            cross_chunk_duplicates += dropped_xchunk
            verses_df = explode_to_verses(chunk_df)
            # ───────────────────────────────────────────────────

            verses_in_chunk = len(verses_df)

            if verses_in_chunk > 0:
                # Running stats for the final summary
                unique_songs.update(verses_df["song_id"].astype(str).unique().tolist())
                language_counts.update(
                    verses_df["language"].astype(str).value_counts().to_dict()
                )

                # Append this chunk to the output parquet
                verses_df = _normalize_for_parquet(verses_df)
                table = pa.Table.from_pandas(verses_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path, table.schema, compression="snappy"
                    )
                else:
                    # Align schema in case pandas inferred a different nullability
                    table = table.cast(writer.schema, safe=False)
                writer.write_table(table)
                total_verses_written += verses_in_chunk

            print_progress(
                chunk_idx=chunk_idx,
                rows_in_chunk=rows_in_chunk,
                verses_in_chunk=verses_in_chunk,
                total_rows=total_rows_read,
                total_verses=total_verses_written,
                start_time=start_time,
                expected_rows=expected_rows,
            )
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - start_time
    print(f"\n[stream] Done in {elapsed:.1f}s. "
          f"{total_rows_read:,} rows → {total_verses_written:,} verses.")
    print(f"[dedup] Cross-chunk duplicates dropped: {cross_chunk_duplicates:,}")
    print(f"[save] Saved to {output_path}")

    # Final full diagnostics — read back only the small metadata columns from
    # the saved parquet (skipping `lyrics_clean`) so memory stays low even
    # for 10M+ verse outputs.
    if total_verses_written > 0:
        print("[diag] Reading metadata columns from output for diagnostics...")
        diag_df = pd.read_parquet(
            output_path, columns=["song_id", "artist", "language", "token_count"]
        )
        print_diagnostics(diag_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Genius lyrics dataset")
    parser.add_argument("--input",  required=True, help="Path to raw CSV or parquet")
    parser.add_argument("--output", required=True, help="Path to save verse-level parquet")
    parser.add_argument("--sample", type=int, default=None,
                        help="Optional: process only N songs (loads fully, non-streaming)")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE,
                        help=f"Rows per streaming chunk (default: {DEFAULT_CHUNKSIZE:,})")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if args.sample is not None:
        _run_sample(input_path, output_path, args.sample)
    else:
        _run_streaming(input_path, output_path, args.chunksize)


if __name__ == "__main__":
    main()
