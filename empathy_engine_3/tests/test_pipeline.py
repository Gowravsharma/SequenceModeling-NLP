"""
test_pipeline.py
-----------------
Unit tests for the Empathy Engine pipeline modules.
Run with:  python -m pytest tests/test_pipeline.py -v

Tests cover:
  - EmotionResult structure
  - Punctuation cue detection
  - Prosody parameter math / clamping
  - SSML generation structure
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from modules.emotion_detector import detect_punctuation_cues, EmotionResult
from modules.prosody_mapper   import map_to_prosody, ProsodyParams, _intensity_scale
from modules.ssml_builder     import build_ssml, _split_sentences


# ─── Punctuation Cue Tests ────────────────────────────────────────────────────

def test_exclamation_count():
    cues = detect_punctuation_cues("Wow! This is amazing! Really!!")
    assert cues["exclamation_count"] == 4

def test_question_count():
    cues = detect_punctuation_cues("Are you sure? Is this real?")
    assert cues["question_count"] == 2

def test_ellipsis_detected():
    cues = detect_punctuation_cues("I don't know... it's just so sad...")
    assert cues["ellipsis_count"] >= 2

def test_caps_ratio():
    cues = detect_punctuation_cues("This is GREAT and AWESOME")
    assert cues["caps_ratio"] > 0.0

def test_no_cues():
    cues = detect_punctuation_cues("The meeting is at three o'clock.")
    assert cues["exclamation_count"] == 0
    assert cues["question_count"] == 0
    assert cues["ellipsis_count"] == 0


# ─── Intensity Scale Tests ─────────────────────────────────────────────────────

def test_intensity_scale_low():
    scale = _intensity_scale(0.0)
    assert scale == pytest.approx(0.15)  # MIN_SCALE

def test_intensity_scale_high():
    scale = _intensity_scale(1.0)
    assert scale == pytest.approx(1.0)   # MAX_SCALE

def test_intensity_scale_mid():
    scale = _intensity_scale(0.5)
    assert 0.15 < scale < 1.0


# ─── Prosody Mapping Tests ─────────────────────────────────────────────────────

EMPTY_CUES = {"exclamation_count": 0, "question_count": 0,
              "ellipsis_count": 0, "caps_ratio": 0.0}

def test_joy_faster_and_higher():
    p = map_to_prosody("joy", 0.9, EMPTY_CUES)
    assert p.rate  > 1.0,  "Joy should speed up speech"
    assert p.pitch > 0.0,  "Joy should raise pitch"

def test_sadness_slower_and_lower():
    p = map_to_prosody("sadness", 0.9, EMPTY_CUES)
    assert p.rate  < 1.0,  "Sadness should slow speech"
    assert p.pitch < 0.0,  "Sadness should lower pitch"

def test_anger_louder():
    p = map_to_prosody("anger", 0.9, EMPTY_CUES)
    assert p.volume_db > 0.0, "Anger should be louder"
    assert p.emphasis in ("moderate", "strong")

def test_neutral_near_baseline():
    p = map_to_prosody("neutral", 0.6, EMPTY_CUES)
    assert abs(p.rate - 1.0)       < 0.1
    assert abs(p.pitch)            < 1.0
    assert abs(p.volume_db)        < 1.0

def test_low_intensity_small_change():
    p_low  = map_to_prosody("joy", 0.2, EMPTY_CUES)
    p_high = map_to_prosody("joy", 0.95, EMPTY_CUES)
    assert p_low.pitch < p_high.pitch, "Higher intensity should give larger pitch boost"

def test_clamping():
    # Artificial extreme cues should still stay in valid range
    extreme_cues = {"exclamation_count": 100, "question_count": 100,
                    "ellipsis_count": 100, "caps_ratio": 1.0}
    p = map_to_prosody("anger", 1.0, extreme_cues)
    assert 0.5 <= p.rate     <= 2.0
    assert -20  <= p.pitch    <= 20
    assert -6   <= p.volume_db<= 6
    assert 0.5 <= p.pause_scale <= 3.0

def test_exclamation_boosts_pitch():
    cues_exc = {"exclamation_count": 2, "question_count": 0,
                "ellipsis_count": 0, "caps_ratio": 0.0}
    p_plain = map_to_prosody("joy", 0.7, EMPTY_CUES)
    p_exc   = map_to_prosody("joy", 0.7, cues_exc)
    assert p_exc.pitch > p_plain.pitch

def test_question_boosts_pitch():
    cues_q = {"exclamation_count": 0, "question_count": 1,
              "ellipsis_count": 0, "caps_ratio": 0.0}
    p_plain = map_to_prosody("neutral", 0.6, EMPTY_CUES)
    p_q     = map_to_prosody("neutral", 0.6, cues_q)
    assert p_q.pitch > p_plain.pitch

def test_ellipsis_slows_and_pauses():
    cues_ell = {"exclamation_count": 0, "question_count": 0,
                "ellipsis_count": 2, "caps_ratio": 0.0}
    p_plain = map_to_prosody("neutral", 0.6, EMPTY_CUES)
    p_ell   = map_to_prosody("neutral", 0.6, cues_ell)
    assert p_ell.pause_scale > p_plain.pause_scale


# ─── SSML Tests ───────────────────────────────────────────────────────────────

def test_ssml_has_speak_root():
    p = ProsodyParams(rate=1.1, pitch=2.0, volume_db=1.0,
                      pause_scale=1.0, emphasis="none")
    ssml = build_ssml("Hello world.", p)
    assert ssml.startswith("<speak>")
    assert ssml.endswith("</speak>")

def test_ssml_has_prosody_tag():
    p = ProsodyParams(rate=1.2, pitch=3.0, volume_db=0.0,
                      pause_scale=1.0, emphasis="moderate")
    ssml = build_ssml("This is great.", p)
    assert "<prosody" in ssml
    assert "rate=" in ssml
    assert "pitch=" in ssml

def test_ssml_break_between_sentences():
    p = ProsodyParams(rate=0.8, pitch=-3.0, volume_db=-1.0,
                      pause_scale=2.0, emphasis="none")
    ssml = build_ssml("I am sad. Life is hard. Everything hurts.", p)
    assert "<break" in ssml

def test_ssml_ellipsis_becomes_break():
    p = ProsodyParams(rate=0.9, pitch=-2.0, volume_db=0.0,
                      pause_scale=1.5, emphasis="none")
    ssml = build_ssml("I waited... and waited...", p)
    assert "<break" in ssml

def test_sentence_splitter():
    sentences = _split_sentences("Hello! How are you? I am fine.")
    assert len(sentences) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
