"""Tests for constants.py — label list integrity and map consistency."""
from lyremo.constants import EMOTIONS, NUM_LABELS, LABEL2ID, ID2LABEL


def test_emotions_has_28_entries():
    assert len(EMOTIONS) == 28


def test_no_duplicate_labels():
    assert len(set(EMOTIONS)) == len(EMOTIONS)


def test_label2id_and_id2label_are_inverses():
    for label, idx in LABEL2ID.items():
        assert ID2LABEL[idx] == label
    for idx, label in ID2LABEL.items():
        assert LABEL2ID[label] == idx


def test_num_labels_matches_emotions_length():
    assert NUM_LABELS == 28
    assert NUM_LABELS == len(EMOTIONS)


def test_label2id_covers_all_emotions():
    assert set(LABEL2ID.keys()) == set(EMOTIONS)


def test_id2label_indices_are_contiguous():
    assert sorted(ID2LABEL.keys()) == list(range(NUM_LABELS))
