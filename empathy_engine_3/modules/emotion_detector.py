"""
emotion_detector.py
---------------------
Handles emotion detection from input text using a pretrained HuggingFace model.
Model: j-hartmann/emotion-english-distilroberta-base
Detects: joy, anger, sadness, neutral, surprise, fear, disgust
Also extracts intensity score (confidence) for each prediction.
"""

from transformers import pipeline
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class EmotionResult:
    """Structured output from emotion detection."""
    emotion: str          # Primary detected emotion label
    intensity: float      # Confidence score [0.0 – 1.0]
    all_scores: dict      # Full distribution over all emotion classes
    punctuation_cues: dict  # Detected punctuation signals


# Lazy-load the model so it's only downloaded once
_classifier = None

def _get_classifier():
    global _classifier
    if _classifier is None:
        print("[EmotionDetector] Loading HuggingFace model...")
        _classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,           # Return scores for ALL labels
            truncation=True,
            max_length=512,
        )
        print("[EmotionDetector] Model loaded.")
    return _classifier


def detect_punctuation_cues(text: str) -> dict:
    """
    Scan text for punctuation marks that imply prosodic signals.

    Rules:
      "!"  → boost energy + pitch  (exclamation_count)
      "?"  → rising pitch cue      (question_count)
      "…" or "..."  → slow down + add pause  (ellipsis_count)
      ALL_CAPS words → emphasis boost  (caps_ratio)
    """
    exclamation_count = text.count("!")
    question_count    = text.count("?")
    ellipsis_count    = text.count("…") + text.count("...")

    words = text.split()
    caps_words = [w for w in words if w.isupper() and len(w) > 1]
    caps_ratio = len(caps_words) / max(len(words), 1)

    return {
        "exclamation_count": exclamation_count,
        "question_count":    question_count,
        "ellipsis_count":    ellipsis_count,
        "caps_ratio":        caps_ratio,
    }


def detect_emotion(text: str) -> EmotionResult:
    """
    Run the HuggingFace classifier on `text` and return an EmotionResult.

    The primary emotion is the label with the highest score.
    Intensity = that top score (a proxy for how strongly the model
    commits to the predicted class).
    """
    clf = _get_classifier()
    raw = clf(text)[0]  # List[{label, score}]

    # Build a {label: score} dict
    all_scores = {item["label"]: round(item["score"], 4) for item in raw}

    # Top prediction
    top = max(raw, key=lambda x: x["score"])
    emotion   = top["label"].lower()
    intensity = round(top["score"], 4)

    punctuation_cues = detect_punctuation_cues(text)

    return EmotionResult(
        emotion=emotion,
        intensity=intensity,
        all_scores=all_scores,
        punctuation_cues=punctuation_cues,
    )


# ── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "I just got promoted — this is the BEST day of my life!!!",
        "I can't believe you did that. I am absolutely furious.",
        "She passed away quietly last night. I'm going to miss her so much...",
        "The package will arrive on Tuesday.",
        "Oh wow, I had no idea that was even possible!",
        "I'm scared. What if everything goes wrong?",
    ]
    for s in samples:
        result = detect_emotion(s)
        print(f"\nText     : {s}")
        print(f"Emotion  : {result.emotion}  (intensity={result.intensity})")
        print(f"Punct    : {result.punctuation_cues}")
