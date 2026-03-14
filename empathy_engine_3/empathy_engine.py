"""
empathy_engine.py
------------------
Main orchestrator for the Empathy Engine pipeline.

Pipeline:
  Text Input
    → Emotion Detection       (emotion_detector.py)
    → Prosody Mapping         (prosody_mapper.py)
    → SSML Construction       (ssml_builder.py)
    → Speech Synthesis        (tts_engine.py)
    → Audio Output (.mp3)

Usage (CLI):
  python empathy_engine.py --text "I can't believe this happened!" --backend gtts

Usage (Python API):
  from empathy_engine import EmpathyEngine
  engine = EmpathyEngine(backend="gtts")
  result = engine.process("Hello, how are you today?")
  print(result)
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from modules.emotion_detector import detect_emotion, EmotionResult
from modules.prosody_mapper   import map_to_prosody, explain_mapping, ProsodyParams
from modules.ssml_builder     import build_ssml
from modules.tts_engine       import synthesize


# ── Pipeline Result ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    text:            str
    emotion:         str
    intensity:       float
    all_scores:      dict
    punctuation_cues: dict
    prosody_params:  dict        # ProsodyParams as dict
    ssml:            str
    audio_path:      str
    backend:         str
    timestamp:       str


# ── Engine ───────────────────────────────────────────────────────────────────

class EmpathyEngine:
    """
    High-level interface to the full Empathy Engine pipeline.

    Parameters
    ----------
    backend : "gtts" | "google" | "elevenlabs"
    output_dir : directory where audio files are saved
    """

    def __init__(self, backend: str = "gtts", output_dir: str = "output"):
        self.backend    = backend
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def process(self, text: str, filename: str | None = None) -> PipelineResult:
        """
        Run the full pipeline on a single piece of text.

        Steps
        -----
        1. Detect emotion + intensity from text
        2. Map emotion → voice parameters (ProsodyParams)
        3. Build SSML document
        4. Synthesize audio via selected backend
        5. Return a PipelineResult with all metadata
        """
        print(f"\n{'='*60}")
        print(f"[Empathy Engine] Processing: {text[:80]}...")

        # ── Step 1: Emotion Detection ────────────────────────────────────────
        print("[1/4] Detecting emotion...")
        emotion_result: EmotionResult = detect_emotion(text)
        print(f"      → {emotion_result.emotion}  (intensity={emotion_result.intensity:.2f})")

        # ── Step 2: Prosody Mapping ──────────────────────────────────────────
        print("[2/4] Mapping to prosody parameters...")
        params: ProsodyParams = map_to_prosody(
            emotion=emotion_result.emotion,
            intensity=emotion_result.intensity,
            punctuation_cues=emotion_result.punctuation_cues,
        )
        print(explain_mapping(emotion_result.emotion, emotion_result.intensity, params))

        # ── Step 3: SSML Construction ────────────────────────────────────────
        print("[3/4] Building SSML...")
        ssml = build_ssml(text, params)

        # ── Step 4: Speech Synthesis ─────────────────────────────────────────
        print(f"[4/4] Synthesizing speech (backend={self.backend})...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename is None:
            safe_emotion = emotion_result.emotion.replace(" ", "_")
            filename = f"{safe_emotion}_{timestamp}.mp3"
        audio_path = os.path.join(self.output_dir, filename)

        # gTTS gets plain text; Google/ElevenLabs can receive SSML via the module
        synthesize(
            text=text,
            params=params,
            output_path=audio_path,
            backend=self.backend,
            emotion=emotion_result.emotion,
        )

        result = PipelineResult(
            text=text,
            emotion=emotion_result.emotion,
            intensity=emotion_result.intensity,
            all_scores=emotion_result.all_scores,
            punctuation_cues=emotion_result.punctuation_cues,
            prosody_params=asdict(params),
            ssml=ssml,
            audio_path=audio_path,
            backend=self.backend,
            timestamp=timestamp,
        )

        print(f"\n✅  Done!  Audio → {audio_path}")
        return result

    def process_batch(self, texts: list[str]) -> list[PipelineResult]:
        """Process multiple texts and return a list of results."""
        return [self.process(t) for t in texts]


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Empathy Engine: Emotionally expressive text-to-speech"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Text to synthesize. If omitted, runs built-in demo sentences.",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["gtts", "google", "elevenlabs"],
        default="gtts",
        help="TTS backend to use (default: gtts)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory for audio output files (default: output/)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full pipeline result as JSON",
    )
    args = parser.parse_args()

    engine = EmpathyEngine(backend=args.backend, output_dir=args.output_dir)

    if args.text:
        result = engine.process(args.text)
        if args.json:
            print(json.dumps(asdict(result) if hasattr(result, '__dataclass_fields__') else result.__dict__, indent=2))
    else:
        # Built-in demo sentences covering all emotion categories
        demo_texts = [
            "I just got promoted — this is the BEST day of my life!!!",
            "I can't believe you lied to me. I am absolutely furious.",
            "She passed away quietly last night. I'm going to miss her so much...",
            "The meeting is scheduled for 3 PM on Tuesday.",
            "Oh wow, I had no idea that was even possible! Are you serious?",
            "I'm scared. What if everything goes wrong?",
            "Ugh, that smell is absolutely revolting.",
        ]
        print(f"\nRunning demo with {len(demo_texts)} test sentences...\n")
        results = engine.process_batch(demo_texts)
        print(f"\n{'='*60}")
        print(f"SUMMARY — {len(results)} audio files generated:")
        for r in results:
            print(f"  [{r.emotion:10s}  {r.intensity:.2f}]  {r.audio_path}")


if __name__ == "__main__":
    main()
