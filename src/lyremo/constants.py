"""
constants.py
============
Single source of truth for emotion labels and mappings.
All other modules must import from here — never hardcode label strings elsewhere.
"""

EMOTIONS: list[str] = [
    # Positive (12)
    "admiration",
    "amusement",
    "approval",
    "caring",
    "desire",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
    # Negative (11)
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
    # Ambiguous (4)
    "confusion",
    "curiosity",
    "realization",
    "surprise",
    # Neutral (1)
    "neutral",
]

NUM_LABELS: int = len(EMOTIONS)  # 28

LABEL2ID: dict[str, int] = {emotion: i for i, emotion in enumerate(EMOTIONS)}
ID2LABEL: dict[int, str] = {i: emotion for i, emotion in enumerate(EMOTIONS)}

# Grouped for reference and analysis
POSITIVE_EMOTIONS: list[str] = EMOTIONS[:12]
NEGATIVE_EMOTIONS: list[str] = EMOTIONS[12:23]
AMBIGUOUS_EMOTIONS: list[str] = EMOTIONS[23:27]
NEUTRAL_EMOTIONS: list[str] = EMOTIONS[27:]
