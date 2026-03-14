"""
app.py
-------
Flask web interface for the Empathy Engine.

Run:
  python app.py

Then open: http://localhost:5000
"""

import os
import json
from dataclasses import asdict
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string

from empathy_engine import EmpathyEngine

app = Flask(__name__)
engine = EmpathyEngine(backend=os.environ.get("TTS_BACKEND", "gtts"), output_dir="output")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Empathy Engine</title>
<style>
  :root {
    --bg: #0a0a12;
    --surface: #13131f;
    --card: #1a1a2e;
    --accent: #7c6af7;
    --accent2: #f06292;
    --text: #e8e8f0;
    --muted: #6e6e8a;
    --joy: #fbbf24; --anger: #ef4444; --sadness: #60a5fa;
    --fear: #a78bfa; --surprise: #34d399; --disgust: #fb923c; --neutral: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif;
         min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem; }
  h1 { font-size: 2.4rem; background: linear-gradient(135deg, var(--accent), var(--accent2));
       -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: .25rem; }
  .subtitle { color: var(--muted); margin-bottom: 2rem; font-size: .95rem; }
  .card { background: var(--card); border-radius: 16px; padding: 1.5rem; width: 100%; max-width: 700px;
          border: 1px solid #2a2a3e; margin-bottom: 1.5rem; }
  textarea { width: 100%; background: var(--surface); border: 1px solid #2a2a3e; border-radius: 10px;
             color: var(--text); padding: 1rem; font-size: 1rem; resize: vertical; min-height: 120px;
             outline: none; transition: border-color .2s; }
  textarea:focus { border-color: var(--accent); }
  .controls { display: flex; gap: 1rem; margin-top: 1rem; align-items: center; flex-wrap: wrap; }
  select { background: var(--surface); border: 1px solid #2a2a3e; color: var(--text);
           padding: .6rem 1rem; border-radius: 8px; font-size: .9rem; cursor: pointer; }
  button { background: linear-gradient(135deg, var(--accent), var(--accent2));
           border: none; color: white; padding: .7rem 2rem; border-radius: 8px;
           font-size: 1rem; font-weight: 600; cursor: pointer; transition: opacity .2s; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  #result { display: none; }
  .emotion-badge { display: inline-flex; align-items: center; gap: .5rem; padding: .4rem 1rem;
                   border-radius: 20px; font-weight: 700; font-size: 1.1rem; margin-bottom: 1rem; }
  .scores-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: .75rem; margin: 1rem 0; }
  .score-item { background: var(--surface); border-radius: 10px; padding: .75rem; }
  .score-label { font-size: .8rem; color: var(--muted); text-transform: capitalize; }
  .score-bar-wrap { background: #2a2a3e; border-radius: 4px; height: 6px; margin: .4rem 0; }
  .score-bar { height: 6px; border-radius: 4px; transition: width .5s ease; }
  .score-val { font-size: .85rem; font-weight: 600; }
  .params-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: .75rem; }
  .param-item { background: var(--surface); border-radius: 10px; padding: .75rem; }
  .param-label { font-size: .75rem; color: var(--muted); }
  .param-val { font-size: 1.2rem; font-weight: 700; margin-top: .2rem; }
  audio { width: 100%; margin-top: 1rem; }
  .ssml-box { background: var(--surface); border-radius: 10px; padding: 1rem; font-family: monospace;
              font-size: .8rem; color: #7ec8e3; overflow-x: auto; white-space: pre-wrap; margin-top: .5rem;
              max-height: 200px; overflow-y: auto; }
  .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(255,255,255,.2);
             border-top-color: white; border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .section-title { font-size: .85rem; color: var(--muted); text-transform: uppercase;
                   letter-spacing: .1em; margin-bottom: .75rem; font-weight: 600; }
  details summary { cursor: pointer; color: var(--muted); font-size: .85rem; }
</style>
</head>
<body>

<h1>🎙 Empathy Engine</h1>
<p class="subtitle">Emotionally expressive text-to-speech powered by AI</p>

<div class="card">
  <div class="section-title">Input</div>
  <textarea id="inputText" placeholder="Type something emotional...&#10;&#10;e.g. I just got promoted — this is the BEST day of my life!!!"></textarea>
  <div class="controls">
    <select id="backend">
      <option value="edge">Edge TTS (free, neural &#x2605;&#x2605;&#x2605;)</option>
      <option value="gtts">gTTS (basic, no pitch)</option>
      <option value="google">Google Cloud TTS</option>
      <option value="elevenlabs">ElevenLabs</option>
    </select>
    <button id="generateBtn" onclick="generate()">Generate Speech ▶</button>
    <span id="spinner" style="display:none"><span class="spinner"></span></span>
  </div>
</div>

<div class="card" id="result">
  <div class="section-title">Detected Emotion</div>
  <div id="emotionBadge" class="emotion-badge"></div>

  <div class="section-title">Emotion Scores</div>
  <div class="scores-grid" id="scoresGrid"></div>

  <div class="section-title" style="margin-top:1rem">Prosody Parameters</div>
  <div class="params-grid" id="paramsGrid"></div>

  <div class="section-title" style="margin-top:1rem">Audio Output</div>
  <audio id="audioPlayer" controls></audio>

  <details style="margin-top:1rem">
    <summary>View generated SSML</summary>
    <div class="ssml-box" id="ssmlBox"></div>
  </details>
</div>

<script>
const EMOTION_COLORS = {
  joy:"#fbbf24",anger:"#ef4444",sadness:"#60a5fa",fear:"#a78bfa",
  surprise:"#34d399",disgust:"#fb923c",neutral:"#94a3b8"
};
const EMOTION_EMOJI = {
  joy:"😄",anger:"😡",sadness:"😢",fear:"😨",surprise:"😲",disgust:"🤢",neutral:"😐"
};

async function generate() {
  const text = document.getElementById("inputText").value.trim();
  if (!text) return;
  const backend = document.getElementById("backend").value;

  document.getElementById("generateBtn").disabled = true;
  document.getElementById("spinner").style.display = "inline";
  document.getElementById("result").style.display = "none";

  try {
    const res = await fetch("/synthesize", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({text, backend})
    });
    const data = await res.json();
    if (!res.ok) { alert(data.error || "Error"); return; }
    renderResult(data);
  } catch(e) {
    alert("Request failed: " + e.message);
  } finally {
    document.getElementById("generateBtn").disabled = false;
    document.getElementById("spinner").style.display = "none";
  }
}

function renderResult(d) {
  const color = EMOTION_COLORS[d.emotion] || "#94a3b8";
  const emoji = EMOTION_EMOJI[d.emotion] || "🎤";

  // Badge
  document.getElementById("emotionBadge").innerHTML =
    `<span style="font-size:1.5rem">${emoji}</span>
     <span style="color:${color}">${d.emotion.toUpperCase()}</span>
     <span style="color:var(--muted);font-size:.85rem;font-weight:400">intensity ${(d.intensity*100).toFixed(0)}%</span>`;

  // Scores
  const sg = document.getElementById("scoresGrid");
  sg.innerHTML = "";
  Object.entries(d.all_scores).sort((a,b)=>b[1]-a[1]).forEach(([label,score])=>{
    const c = EMOTION_COLORS[label] || "#94a3b8";
    sg.innerHTML += `
      <div class="score-item">
        <div class="score-label">${EMOTION_EMOJI[label]||""} ${label}</div>
        <div class="score-bar-wrap"><div class="score-bar" style="width:${score*100}%;background:${c}"></div></div>
        <div class="score-val" style="color:${c}">${(score*100).toFixed(1)}%</div>
      </div>`;
  });

  // Params
  const p = d.prosody_params;
  const pg = document.getElementById("paramsGrid");
  pg.innerHTML = `
    <div class="param-item"><div class="param-label">Rate</div><div class="param-val">${p.rate.toFixed(2)}×</div></div>
    <div class="param-item"><div class="param-label">Pitch</div><div class="param-val">${p.pitch>0?"+":""}${p.pitch.toFixed(1)} st</div></div>
    <div class="param-item"><div class="param-label">Volume</div><div class="param-val">${p.volume_db>0?"+":""}${p.volume_db.toFixed(1)} dB</div></div>
    <div class="param-item"><div class="param-label">Pause ×</div><div class="param-val">${p.pause_scale.toFixed(2)}</div></div>
    <div class="param-item"><div class="param-label">Emphasis</div><div class="param-val">${p.emphasis}</div></div>
  `;

  // Audio
  document.getElementById("audioPlayer").src = d.audio_url + "?t=" + Date.now();

  // SSML
  document.getElementById("ssmlBox").textContent = d.ssml;

  document.getElementById("result").style.display = "block";
  document.getElementById("result").scrollIntoView({behavior:"smooth"});
}

document.getElementById("inputText").addEventListener("keydown", e => {
  if (e.ctrlKey && e.key === "Enter") generate();
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/synthesize", methods=["POST"])
def api_synthesize():
    data = request.get_json()
    text    = data.get("text", "").strip()
    backend = data.get("backend", "gtts")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Re-create engine with the requested backend
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        eng = EmpathyEngine(backend=backend, output_dir=output_dir)
        result = eng.process(text)
        audio_filename = Path(result.audio_path).name

        return jsonify({
            "emotion":       result.emotion,
            "intensity":     result.intensity,
            "all_scores":    result.all_scores,
            "punctuation_cues": result.punctuation_cues,
            "prosody_params":result.prosody_params,
            "ssml":          result.ssml,
            "audio_url":     f"/audio/{audio_filename}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/audio/<filename>")
def serve_audio(filename):
    import os
    # Get absolute path — fixes Windows backslash issues
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    path = os.path.join(output_dir, filename)
    
    print(f"[serve_audio] Requested: {filename}")
    print(f"[serve_audio] Full path: {path}")
    print(f"[serve_audio] Exists: {os.path.exists(path)}")
    
    if not os.path.exists(path):
        return f"File not found: {path}", 404
    
    return send_file(
        path,
        mimetype="audio/mpeg",
        as_attachment=False,
        conditional=False,      # disable caching — forces fresh load every time
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
