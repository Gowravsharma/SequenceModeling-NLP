"""
tts_engine.py — TTS synthesis with prosody control.
Backends: edge (default), gtts, google, elevenlabs
"""

import os
from pathlib import Path

from modules.ssml_builder import build_ssml
from modules.prosody_mapper import ProsodyParams

EDGE_STYLE_MAP = {
    "joy":      "cheerful",
    "anger":    "angry",
    "sadness":  "sad",
    "fear":     "terrified",
    "surprise": "excited",
    "disgust":  "disgruntled",
    "neutral":  "chat",
}

def _synthesize_edge_emotional(text, params, output_path, emotion="neutral"):
    try:
        import asyncio
        import edge_tts
    except ImportError:
        raise ImportError("Run: pip install edge-tts")

    abs_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    rate_pct   = f"{int((params.rate - 1.0) * 100):+d}%"
    pitch_st   = f"{params.pitch:+.1f}st"
    volume_pct = f"{int(params.volume_db * 8):+d}%"

    print(f"[EdgeTTS] emotion={emotion} rate={rate_pct} pitch={pitch_st} volume={volume_pct}")

    # Build kwargs — only pass pitch/volume if non-zero
    # Edge TTS rejects "+0.0st" and "+0%" as invalid
    kwargs = {
        "text":  text,
        "voice": "en-US-AriaNeural",
        "rate":  rate_pct,
    }
    if abs(params.pitch) >= 0.5:
        kwargs["pitch"] = pitch_st
    if abs(params.volume_db) >= 0.5:
        kwargs["volume"] = volume_pct

    async def _run():
        communicate = edge_tts.Communicate(**kwargs)
        await communicate.save(abs_path)

    asyncio.run(_run())

    if not os.path.exists(abs_path) or os.path.getsize(abs_path) == 0:
        raise RuntimeError(f"Edge TTS produced no audio at {abs_path}")

    print(f"[EdgeTTS] Done -> {abs_path} ({os.path.getsize(abs_path)} bytes)")
    return output_path

def _synthesize_gtts(text, params, output_path):
    try:
        from gtts import gTTS
    except ImportError:
        raise ImportError("Run: pip install gtts")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    slow     = params.rate < 0.85
    abs_path = os.path.abspath(output_path)
    gTTS(text=text, lang="en", slow=slow).save(abs_path)
    print(f"[gTTS] slow={slow} -> {abs_path} ({os.path.getsize(abs_path)} bytes)")
    return output_path


def _synthesize_google(text, params, output_path):
    try:
        from google.cloud import texttospeech
    except ImportError:
        raise ImportError("Run: pip install google-cloud-texttospeech")

    ssml     = build_ssml(text, params)
    client   = texttospeech.TextToSpeechClient()
    voice    = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Neural2-F")
    acfg     = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=texttospeech.SynthesisInput(ssml=ssml), voice=voice, audio_config=acfg)
    with open(output_path, "wb") as f:
        f.write(response.audio_content)
    return output_path


def _synthesize_elevenlabs(text, params, output_path):
    import requests
    api_key  = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ELEVENLABS_API_KEY environment variable.")
    voice_id  = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    style_map = {"none": 0.2, "moderate": 0.5, "strong": 0.8}
    payload   = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5, "similarity_boost": 0.75,
            "style": style_map.get(params.emphasis, 0.4),
            "use_speaker_boost": True,
            "speed": max(0.7, min(1.2, params.rate)),
        },
    }
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        json=payload,
        headers={"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
    )
    r.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(r.content)
    return output_path


BACKENDS = {
    "gtts":       _synthesize_gtts,
    "google":     _synthesize_google,
    "elevenlabs": _synthesize_elevenlabs,
}


def synthesize(text, params, output_path="output/speech.mp3", backend="edge", emotion="neutral"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if backend == "edge":
        return _synthesize_edge_emotional(text, params, output_path, emotion)
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: edge, {', '.join(BACKENDS)}")
    result = BACKENDS[backend](text, params, output_path)
    print(f"[TTS] Audio saved -> {result}")
    return result
