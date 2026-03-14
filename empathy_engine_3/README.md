# 🎙 Empathy Engine
> Emotionally expressive text-to-speech — giving AI a human voice.

The Empathy Engine detects the emotion in a piece of text and automatically
modulates the synthesized speech to match: joyful text sounds upbeat and fast,
sad text sounds slow and subdued, angry text sounds sharp and loud.

---

## Architecture

```
Text Input
    │
    ▼
┌─────────────────────────────────────┐
│  1. Emotion Detection               │
│     HuggingFace DistilRoBERTa       │
│     → emotion label + intensity     │
│     → punctuation cue signals       │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  2. Prosody Mapping                 │
│     emotion × intensity → params   │
│     rate / pitch / volume / pause  │
│     + punctuation adjustments      │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  3. SSML Construction               │
│     <prosody rate pitch volume>     │
│     <emphasis> / <break> injection  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  4. Speech Synthesis                │
│     Edge TTS (default, free)        │
│     gTTS / Google Cloud / ElevenLabs│
└──────────────────┬──────────────────┘
                   │
                   ▼
              Audio File (.mp3)
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourname/empathy-engine
cd empathy-engine
pip install -r requirements.txt
pip install edge-tts
```

> For CPU-only PyTorch (faster install, no GPU needed):
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 2. Run the Web UI

```bash
python app.py
# → open http://localhost:5000
```

Paste any text, select **Edge TTS (free, neural)** from the dropdown, and click **Generate Speech ▶**.

### 3. Run the CLI

```bash
# Single sentence
python empathy_engine.py --text "I just got promoted — this is the BEST day of my life!!!"

# Run all built-in demo sentences
python empathy_engine.py

# JSON output with full pipeline metadata
python empathy_engine.py --text "I'm scared." --json
```

Audio files are saved to the `output/` folder.

---

## TTS Backends

| Backend | Quality | Cost | Requires | Status |
|---|---|---|---|---|
| `edge` | ★★★★☆ | **Free** | `pip install edge-tts` | ✅ Default — recommended |
| `gtts` | ★★☆☆☆ | **Free** | `pip install gtts` | ⚠️ No pitch control, speed only |
| `google` | ★★★★☆ | **Paid** | GCP account + credentials | 🔧 Optional upgrade |
| `elevenlabs` | ★★★★★ | **Paid** | API key (no free tier) | 🔧 Optional upgrade |

### Edge TTS ← used in this project
Microsoft's neural TTS engine, completely free with no API key required.
Supports real pitch, rate, and volume control via direct parameters.

```bash
pip install edge-tts
python app.py   # select "Edge TTS" in the dropdown
```

Voices used: `en-US-AriaNeural` — a high-quality expressive neural voice that
genuinely responds to pitch and rate changes, making emotional differences clearly audible.

### gTTS (basic fallback)
Free but severely limited — it only supports normal/slow speed and has no pitch
control whatsoever. Use this only if Edge TTS is unavailable.

```bash
pip install gtts
python empathy_engine.py --text "Hello!" --backend gtts
```

---

## Optional Paid Backends

These backends are **not used in the default setup** due to cost, but the code
is fully implemented and ready to use if you have credentials.

### Google Cloud TTS (optional)

Supports full SSML including `<prosody>`, `<emphasis>`, and `<break>` tags.
Requires a Google Cloud account and billing enabled.

**Setup:**
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project and enable the **Cloud Text-to-Speech API**
3. Go to IAM & Admin → Service Accounts → create one → download the JSON key
4. Install the library:
```bash
pip install google-cloud-texttospeech
```
5. Set credentials and run:
```bash
# Windows
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service_account.json

# Mac / Linux
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json

python empathy_engine.py --text "Hello!" --backend google
```

**Pricing:** ~$4 per 1 million characters for Neural2 voices.
Free tier: 1 million characters/month for WaveNet voices.
See [cloud.google.com/text-to-speech/pricing](https://cloud.google.com/text-to-speech/pricing).

---

### ElevenLabs (optional)

The highest quality TTS available. Voices are extremely expressive and
human-like. However, **ElevenLabs does not offer a permanently free tier** —
the starter plan requires a paid subscription.

**Setup:**
1. Create an account at [elevenlabs.io](https://elevenlabs.io)
2. Go to Profile → API Keys → copy your key
3. Install requests:
```bash
pip install requests
```
4. Set the key and run:
```bash
# Windows
set ELEVENLABS_API_KEY=your_key_here

# Mac / Linux
export ELEVENLABS_API_KEY=your_key_here

python empathy_engine.py --text "Hello!" --backend elevenlabs
```

**To change the voice**, set the voice ID from your ElevenLabs dashboard:
```bash
set ELEVENLABS_VOICE_ID=your_voice_id_here
```

**Pricing:** Starts at $5/month for 30,000 characters.
See [elevenlabs.io/pricing](https://elevenlabs.io/pricing).

---

## Emotion → Prosody Mapping

### Formula

```
intensity_scale = 0.15 + (1.0 - 0.15) × intensity

rate        = 1.0 + rate_boost        × intensity_scale
pitch       = 0.0 + pitch_boost       × intensity_scale   (semitones)
volume_db   = 0.0 + volume_boost      × intensity_scale   (dB)
pause_scale = 1.0 + pause_boost       × intensity_scale   (multiplier)
```

The `intensity_scale` ensures a weakly-detected emotion (intensity ≈ 0.2) causes
only a subtle change, while a strongly-detected one (intensity ≈ 0.95) causes
the full boost. The floor of 0.15 means even low-confidence detections produce
a noticeable effect rather than sounding completely flat.

### Emotion Profiles

| Emotion   | Rate   | Pitch  | Volume | Pause× | Emphasis |
|-----------|--------|--------|--------|--------|----------|
| joy       | +0.25  | +4 st  | +2 dB  | −0.3×  | moderate |
| anger     | +0.15  | +2 st  | +5 dB  | −0.2×  | strong   |
| sadness   | −0.30  | −4 st  | −2 dB  | +1.5×  | none     |
| fear      | −0.15  | +3 st  | +1 dB  | +0.8×  | moderate |
| surprise  | +0.20  | +6 st  | +3 dB  | −0.3×  | moderate |
| disgust   | −0.10  | −2 st  | +2 dB  | +0.5×  | moderate |
| neutral   |  0.0   |  0 st  |  0 dB  |  1.0×  | none     |

### Punctuation Cues

On top of the emotion base, punctuation shifts the parameters further:

| Symbol | Effect |
|--------|--------|
| `!`    | +1 st pitch, +0.5 dB volume (per `!`, capped at 3) |
| `?`    | +2 st pitch (rising intonation, per `?`, capped at 2) |
| `…`    | −0.08× rate, +0.4× pause_scale (per ellipsis, capped at 3) |
| ALL CAPS | Escalates emphasis level by one step |

---

## Module Reference

```
empathy_engine/
├── empathy_engine.py        ← Main orchestrator & CLI
├── app.py                   ← Flask web UI
├── requirements.txt         ← pip dependencies
├── modules/
│   ├── emotion_detector.py  ← HuggingFace DistilRoBERTa wrapper
│   ├── prosody_mapper.py    ← Emotion → voice parameter math
│   ├── ssml_builder.py      ← SSML document construction
│   └── tts_engine.py        ← Multi-backend TTS synthesis
└── tests/
    └── test_pipeline.py     ← pytest unit tests
```

---

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

Tests cover: punctuation parsing, intensity scaling math, prosody clamping,
SSML structure, and emotion-to-parameter direction (joy → faster, sadness → slower, etc.).

---

## Example Outputs

| Text | Detected | Rate | Pitch | Volume |
|------|----------|------|-------|--------|
| "I just got promoted — this is the BEST day of my life!!!" | joy 0.94 | 1.23× | +3.8 st | +1.9 dB |
| "I can't believe you lied to me." | anger 0.87 | 1.12× | +1.7 st | +4.3 dB |
| "She passed away last night..." | sadness 0.91 | 0.73× | −3.6 st | −1.8 dB |
| "The meeting is at 3 PM." | neutral 0.72 | 1.0× | 0.0 st | 0.0 dB |

---

## Design Decisions

**Why Edge TTS as default?**
Edge TTS (Microsoft) is completely free, requires no API key, and uses genuine
neural voices that actually respond to pitch and rate parameters. Unlike gTTS
which ignores all prosody, Edge TTS makes the emotional differences clearly
audible — a sad sentence genuinely sounds slower and lower, a joyful one faster
and higher. It runs on Windows, Mac, and Linux with a single `pip install edge-tts`.

**Why not ElevenLabs or Google Cloud by default?**
Both produce higher quality audio, but require paid accounts. ElevenLabs has no
free tier; Google Cloud charges after the free monthly quota. For a demo or
prototype, Edge TTS delivers clearly audible emotional variation at zero cost.
The code for both is fully implemented in `tts_engine.py` and can be switched
in with a single environment variable — see the Optional Paid Backends section above.

**Why DistilRoBERTa for emotion detection?**
`j-hartmann/emotion-english-distilroberta-base` gives 7-class emotion output
(joy, anger, sadness, fear, surprise, disgust, neutral) with per-class confidence
scores. The confidence score of the top prediction doubles as our intensity signal
with no extra computation — a 0.95 confidence joy detection sounds more
enthusiastic than a 0.45 one.

**Why SSML?**
SSML is the W3C standard for speech markup, supported by Google Cloud TTS,
Amazon Polly, Azure TTS, and ElevenLabs. The SSML builder in `ssml_builder.py`
produces standard-compliant output that works with any of these backends.
Edge TTS on the other hand accepts rate/pitch as direct constructor parameters,
which is more reliable across versions than injecting SSML strings.

**Intensity scaling floor (0.15)**
A floor of 0.15 instead of 0.0 ensures that even a weakly detected emotion
(intensity = 0.2) still produces a subtle but noticeable prosody change rather
than sounding completely flat.
