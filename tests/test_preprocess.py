"""Tests for preprocess.py — cleaning and verse splitting logic."""
import pytest
from lyremo.data.preprocess import (
    clean_text,
    split_into_verses,
    extract_verse_boundaries,
)


# ─────────────────────────────────────────────
# clean_text
# ─────────────────────────────────────────────

def test_bracket_annotations_stripped():
    text = "[Verse 1]\nI walk alone at night"
    result = clean_text(text)
    assert "[Verse 1]" not in result
    assert "I walk alone at night" in result


def test_url_stripped():
    text = "Follow me at https://genius.com tonight"
    result = clean_text(text)
    assert "https://" not in result
    assert "Follow me at" in result
    assert "tonight" in result


def test_handle_stripped():
    text = "Shoutout @drake for this one"
    result = clean_text(text)
    assert "@drake" not in result
    assert "Shoutout" in result
    assert "for this one" in result


def test_stopwords_preserved():
    text = "That is the beat and it is sick"
    result = clean_text(text)
    assert "is" in result
    assert "the" in result
    assert "and" in result


def test_punctuation_preserved():
    text = "Wait, are you serious? Yes! Absolutely."
    result = clean_text(text)
    assert "," in result
    assert "?" in result
    assert "!" in result
    assert "." in result


def test_surrounding_text_preserved_after_bracket_strip():
    text = "[Bridge]\nStay with me now, don't go"
    result = clean_text(text)
    assert "Stay with me now, don't go" in result


def test_ambiguous_bracket_stripped():
    text = "[?] Something unclear here"
    result = clean_text(text)
    assert "[?]" not in result
    assert "Something unclear here" in result


# ─────────────────────────────────────────────
# extract_verse_boundaries
# ─────────────────────────────────────────────

def test_bracket_annotations_produce_splits():
    text = "[Verse 1]\nLine one\nLine two\n\n[Chorus]\nLine three"
    verses = extract_verse_boundaries(text)
    assert len(verses) == 2


def test_no_brackets_returns_empty():
    text = "Line one\nLine two\n\nLine three"
    verses = extract_verse_boundaries(text)
    assert verses == []


# ─────────────────────────────────────────────
# split_into_verses
# ─────────────────────────────────────────────

def test_annotated_song_splits_by_bracket():
    text = (
        "[Verse 1]\n"
        "I'm walking down the street alone tonight\n"
        "The city lights are blurring out my sight\n\n"
        "[Chorus]\n"
        "And I can't find my way back home again\n"
        "Lost in the rain without a friend\n"
    )
    verses = split_into_verses(text)
    assert len(verses) == 2


def test_unannotated_song_splits_by_double_newline():
    text = (
        "I'm walking down the street alone tonight\n"
        "The city lights are blurring out my sight\n\n"
        "And I can't find my way back home again\n"
        "Lost in the rain without a friend\n"
    )
    verses = split_into_verses(text)
    assert len(verses) == 2


def test_short_chunks_dropped():
    text = (
        "[Verse 1]\nYeah\n\n"
        "[Chorus]\nUh huh\n\n"
        "[Verse 2]\n"
        "I walked through the fire and came out the other side stronger than before\n"
        "The scars on my skin tell the story of a hundred battles and a thousand more\n"
    )
    verses = split_into_verses(text)
    # Only the last verse has enough words
    assert len(verses) == 1
    assert "walked through the fire" in verses[0]


def test_long_verse_triggers_sentence_fallback():
    # Generate a verse > 400 words using repeated sentences
    sentence = "The rain keeps falling down on me tonight and I don't know what to do. "
    long_verse = sentence * 30  # ~300+ words, well over limit
    text = f"[Verse 1]\n{long_verse}"
    verses = split_into_verses(text)
    # Should have been split into multiple chunks
    assert len(verses) > 1
    for v in verses:
        assert len(v.split()) <= 400


def test_all_short_returns_empty():
    text = "Yeah\n\nUh\n\nOkay\n\nMmm"
    verses = split_into_verses(text)
    assert verses == []


def test_slang_preserved():
    text = (
        "[Verse 1]\n"
        "That beat is sick yo, the flow is straight fire\n"
        "We out here getting lit and never getting tired\n"
        "She said I'm lowkey wildin but I keep it a buck\n"
        "No cap I'm built different and I don't give a cluck\n"
    )
    verses = split_into_verses(text)
    assert len(verses) == 1
    assert "sick" in verses[0]
    assert "fire" in verses[0]
    assert "lowkey" in verses[0]
