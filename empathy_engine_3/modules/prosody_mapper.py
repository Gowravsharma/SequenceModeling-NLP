"""
prosody_mapper.py
------------------
Maps detected emotion + intensity → concrete TTS voice parameters.

┌─────────────────────────────────────────────────────────────────┐
│                    PARAMETER SPACE                              │
│  rate    : speaking speed  [0.5 – 2.0]  (1.0 = normal)        │
│  pitch   : tonal height    [-20 – +20 semitones from baseline] │
│  volume  : amplitude dB    [-6 – +6 dB relative to baseline]   │
│  pause_scale : multiplier on inter-sentence pauses [0.5 – 3.0] │
│  emphasis: word stress flag [none | moderate | strong]         │
└─────────────────────────────────────────────────────────────────┘

Formula (applied per parameter):

    param_final = base + emotion_boost * intensity_scale

Where:
    intensity_scale = lerp(MIN_SCALE, MAX_SCALE, intensity)
                    = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * intensity

This means:
  • A weakly detected emotion (intensity ≈ 0.3) causes subtle change
  • A strongly detected emotion (intensity ≈ 0.95) causes full change

Punctuation cues are then layered on top:
    "!"  → +pitch_delta, +volume_delta  per exclamation
    "?"  → +pitch_delta (rising)
    "…"  → +pause_delta, -rate_delta
    CAPS → emphasis escalated
"""

from dataclasses import dataclass
from typing import Literal
import math


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class ProsodyParams:
    """Final prosody parameters to hand to the TTS engine."""
    rate:         float   # 0.5–2.0
    pitch:        float   # semitones, −20 to +20
    volume_db:    float   # dB offset, −6 to +6
    pause_scale:  float   # multiplier on natural pause durations
    emphasis:     Literal["none", "moderate", "strong"]

    def clamp(self):
        """Ensure all values stay within valid ranges."""
        self.rate        = max(0.5,  min(2.0,  self.rate))
        self.pitch       = max(-20,  min(20,   self.pitch))
        self.volume_db   = max(-6,   min(6,    self.volume_db))
        self.pause_scale = max(0.5,  min(3.0,  self.pause_scale))
        return self

    def to_ssml_attrs(self) -> dict:
        """
        Convert to SSML-friendly attributes for <prosody> tags.
        pitch in SSML uses semitones as "+Nst" / "-Nst"
        rate  in SSML uses a percentage string like "120%"
        volume in SSML uses dB like "+3dB"
        """
        pitch_st  = f"{self.pitch:+.1f}st"
        rate_pct  = f"{self.rate * 100:.0f}%"
        vol_db    = f"{self.volume_db:+.1f}dB"
        return {"pitch": pitch_st, "rate": rate_pct, "volume": vol_db}


# ── Emotion Baselines & Boosts ───────────────────────────────────────────────

# Baseline (neutral voice)
BASE = {
    "rate":        1.0,
    "pitch":       0.0,
    "volume_db":   0.0,
    "pause_scale": 1.0,
}

# Maximum boost per parameter for each emotion (at full intensity=1.0)
# Negative values slow/lower the parameter; positive values raise it.
EMOTION_PROFILES = {
    #          rate   pitch  vol_db  pause_scale  emphasis
    "joy":     ( 0.25,  4.0,   2.0,   -0.3,  "moderate"),
    "anger":   ( 0.15,  2.0,   5.0,   -0.2,  "strong"),
    "sadness": (-0.30, -4.0,  -2.0,    1.5,  "none"),
    "fear":    (-0.15,  3.0,   1.0,    0.8,  "moderate"),
    "surprise":( 0.20,  6.0,   3.0,   -0.3,  "moderate"),
    "disgust": (-0.10, -2.0,   2.0,    0.5,  "moderate"),
    "neutral": ( 0.0,   0.0,   0.0,    0.0,  "none"),
}

# Intensity scaling bounds:
# At intensity=0  → scale=MIN_SCALE  (very small change even if emotion detected)
# At intensity=1  → scale=MAX_SCALE  (full boost applied)
MIN_SCALE = 0.15
MAX_SCALE = 1.0


def _intensity_scale(intensity: float) -> float:
    """Linear interpolation between MIN_SCALE and MAX_SCALE."""
    return MIN_SCALE + (MAX_SCALE - MIN_SCALE) * intensity


def _emphasis_escalate(current: str, delta: float) -> str:
    """Escalate emphasis level by delta (0–1)."""
    levels = ["none", "moderate", "strong"]
    idx = levels.index(current)
    idx = min(len(levels) - 1, idx + round(delta * 2))
    return levels[idx]


# ── Punctuation Deltas ───────────────────────────────────────────────────────

EXCLAMATION_PITCH_BOOST   =  1.0   # semitones per "!"
EXCLAMATION_VOLUME_BOOST  =  0.5   # dB per "!"
QUESTION_PITCH_BOOST      =  2.0   # semitones per "?" (rising)
ELLIPSIS_RATE_REDUCE      = -0.08  # rate delta per "…"
ELLIPSIS_PAUSE_BOOST      =  0.4   # pause_scale delta per "…"
CAPS_EMPHASIS_THRESHOLD   =  0.15  # caps_ratio above which emphasis steps up


# ── Main Mapping Function ────────────────────────────────────────────────────

def map_to_prosody(emotion: str, intensity: float, punctuation_cues: dict) -> ProsodyParams:
    """
    Compute final ProsodyParams for a given emotion + intensity + punctuation context.

    Algorithm
    ---------
    1. Look up the emotion profile for boost vectors.
    2. Compute intensity_scale via linear interpolation.
    3. Apply:  param = base + boost * intensity_scale
    4. Layer punctuation adjustments on top.
    5. Clamp to valid ranges.
    6. Return ProsodyParams.
    """
    # Fallback to neutral if emotion not in profiles
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])
    rate_boost, pitch_boost, vol_boost, pause_boost, base_emphasis = profile

    scale = _intensity_scale(intensity)

    rate        = BASE["rate"]        + rate_boost   * scale
    pitch       = BASE["pitch"]       + pitch_boost  * scale
    volume_db   = BASE["volume_db"]   + vol_boost    * scale
    pause_scale = BASE["pause_scale"] + pause_boost  * scale
    emphasis    = base_emphasis

    # ── Punctuation adjustments ──────────────────────────────────────────────
    exc = punctuation_cues.get("exclamation_count", 0)
    if exc:
        pitch     += EXCLAMATION_PITCH_BOOST  * min(exc, 3)   # cap at 3
        volume_db += EXCLAMATION_VOLUME_BOOST * min(exc, 3)
        emphasis   = _emphasis_escalate(emphasis, 0.5)

    qst = punctuation_cues.get("question_count", 0)
    if qst:
        pitch += QUESTION_PITCH_BOOST * min(qst, 2)

    ell = punctuation_cues.get("ellipsis_count", 0)
    if ell:
        rate        += ELLIPSIS_RATE_REDUCE * min(ell, 3)
        pause_scale += ELLIPSIS_PAUSE_BOOST * min(ell, 3)

    if punctuation_cues.get("caps_ratio", 0) > CAPS_EMPHASIS_THRESHOLD:
        emphasis = _emphasis_escalate(emphasis, 0.6)

    params = ProsodyParams(
        rate=rate,
        pitch=pitch,
        volume_db=volume_db,
        pause_scale=pause_scale,
        emphasis=emphasis,
    )
    return params.clamp()


def explain_mapping(emotion: str, intensity: float, params: ProsodyParams) -> str:
    """Return a human-readable explanation of the prosody decision."""
    scale = _intensity_scale(intensity)
    lines = [
        f"Emotion   : {emotion}  (intensity={intensity:.2f}, scale={scale:.2f})",
        f"Rate      : {params.rate:.2f}  (1.0 = normal; >1 = faster)",
        f"Pitch     : {params.pitch:+.1f} st  (semitones from baseline)",
        f"Volume    : {params.volume_db:+.1f} dB",
        f"Pause×    : {params.pause_scale:.2f}x  (>1 = longer pauses)",
        f"Emphasis  : {params.emphasis}",
    ]
    return "\n".join(lines)


# ── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("joy",     0.92, {"exclamation_count": 2, "question_count": 0, "ellipsis_count": 0, "caps_ratio": 0.0}),
        ("sadness", 0.75, {"exclamation_count": 0, "question_count": 0, "ellipsis_count": 2, "caps_ratio": 0.0}),
        ("anger",   0.88, {"exclamation_count": 1, "question_count": 0, "ellipsis_count": 0, "caps_ratio": 0.3}),
        ("neutral", 0.60, {"exclamation_count": 0, "question_count": 1, "ellipsis_count": 0, "caps_ratio": 0.0}),
    ]
    for emotion, intensity, cues in test_cases:
        params = map_to_prosody(emotion, intensity, cues)
        print(explain_mapping(emotion, intensity, params))
        print("SSML attrs:", params.to_ssml_attrs())
        print()
