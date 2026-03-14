"""
ssml_builder.py
----------------
Converts plain text + ProsodyParams into an SSML document
compatible with Google Cloud TTS (and most W3C-compliant engines).

SSML Reference:
  <speak>            : root element
  <prosody>          : rate, pitch, volume control
  <emphasis>         : word-level stress
  <break>            : explicit pause injection
  <say-as>           : pronunciation hints (not used here but importable)

Design notes
------------
• We split the text into sentences and wrap each in its own
  <prosody> block so per-sentence pitch/rate can be adjusted.
• Ellipsis ("…" / "...") → <break time="700ms"/> injected at occurrence.
• Exclamation words before "!" get <emphasis level="strong">.
• Question sentences get an extra pitch bump on the last 2 words.
• The pause_scale from ProsodyParams adjusts the <break> between sentences.
"""

import re
import math
from modules.prosody_mapper import ProsodyParams


# ── Sentence splitter ────────────────────────────────────────────────────────

_SENTENCE_RE = re.compile(r'(?<=[.!?…])\s+')

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping the delimiter attached."""
    sentences = _SENTENCE_RE.split(text.strip())
    return [s for s in sentences if s]


# ── Pause duration helper ─────────────────────────────────────────────────────

def _inter_sentence_break_ms(pause_scale: float) -> int:
    """
    Natural inter-sentence pause is ~500 ms.
    pause_scale ∈ [0.5, 3.0]  → break in [250 ms, 1500 ms]
    """
    base_ms = 500
    return int(base_ms * pause_scale)


# ── SSML builder ─────────────────────────────────────────────────────────────

def build_ssml(text: str, params: ProsodyParams) -> str:
    """
    Build a complete SSML string from plain text and ProsodyParams.

    Parameters
    ----------
    text   : The input sentence(s).
    params : ProsodyParams from prosody_mapper.

    Returns
    -------
    ssml   : A <speak>…</speak> string ready for a TTS API.
    """
    attrs = params.to_ssml_attrs()
    rate_str   = attrs["rate"]
    pitch_str  = attrs["pitch"]
    volume_str = attrs["volume"]

    sentences = _split_sentences(text)
    break_ms  = _inter_sentence_break_ms(params.pause_scale)

    parts = []

    for i, sentence in enumerate(sentences):
        processed = _process_sentence(sentence, params)

        # Wrap in prosody tag
        ssml_sentence = (
            f'<prosody rate="{rate_str}" pitch="{pitch_str}" volume="{volume_str}">'
            f'{processed}'
            f'</prosody>'
        )
        parts.append(ssml_sentence)

        # Add inter-sentence break (not after the last sentence)
        if i < len(sentences) - 1:
            parts.append(f'<break time="{break_ms}ms"/>')

    body = "\n  ".join(parts)
    ssml = f'<speak>\n  {body}\n</speak>'
    return ssml


def _process_sentence(sentence: str, params: ProsodyParams) -> str:
    """
    Apply intra-sentence SSML markup:
      - Replace "…" / "..." with a <break>
      - Wrap exclamation-adjacent words with <emphasis>
      - Add rising pitch on last 2 words of questions
    """
    # Ellipsis → break
    sentence = re.sub(r'\.{3}|…', '<break time="700ms"/>', sentence)

    # Exclamation: wrap the last word before "!" in <emphasis>
    def emphasize_exclaim(m):
        word = m.group(1)
        return f'<emphasis level="{params.emphasis if params.emphasis != "none" else "moderate"}">{word}</emphasis>!'
    sentence = re.sub(r'(\w+)!', emphasize_exclaim, sentence)

    # Question: raise pitch on the last 2 words
    if sentence.rstrip().endswith("?"):
        sentence = _raise_question_tail(sentence, params)

    return sentence


def _raise_question_tail(sentence: str, params: ProsodyParams) -> str:
    """
    For question sentences, wrap the final 2 words in a slightly
    higher-pitch <prosody> to simulate natural rising intonation.
    """
    # Extra pitch on tail: +2 semitones beyond the already-set pitch
    current_pitch_st = params.pitch
    tail_pitch_st    = current_pitch_st + 2.0
    tail_pitch_str   = f"{tail_pitch_st:+.1f}st"

    words = sentence.rstrip("?").split()
    if len(words) < 2:
        return sentence

    tail   = " ".join(words[-2:])
    body_  = " ".join(words[:-2])

    tail_ssml = f'<prosody pitch="{tail_pitch_str}">{tail}</prosody>?'
    return f"{body_} {tail_ssml}" if body_ else tail_ssml


# ── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from modules.prosody_mapper import ProsodyParams

    params = ProsodyParams(rate=1.2, pitch=4.0, volume_db=2.0,
                           pause_scale=0.8, emphasis="moderate")
    text = "This is amazing! I can't believe it worked. Are you sure this is real?"
    ssml = build_ssml(text, params)
    print(ssml)
