import sys
import struct
from pathlib import Path

from flask import Flask, Response, request

# -----------------------------
# Local repo package import
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "orpheus_tts_pypi"))

from orpheus_tts import OrpheusModel


# -----------------------------
# App / model config
# -----------------------------
app = Flask(__name__)

engine = None

MODEL_NAME = "canopylabs/orpheus-tts-0.1-finetune-prod"
DEFAULT_VOICE = "tara"

SAMPLE_RATE = 24000

MAX_MODEL_LEN = 1024
GPU_MEMORY_UTILIZATION = 0.45

PORT = 6021

# Realtime demo guardrails.
# Keep this conservative because generate_speech() wrapper can kill EngineCore
# for unsupported, long, or repetitive prompts.
MAX_PROMPT_CHARS = 250
MAX_WORDS = 45
MAX_REPEAT_PHRASE_COUNT = 2


# -----------------------------
# Safety / input validation
# -----------------------------
def is_ascii_safe_text(text: str) -> bool:
    """
    Current tested Orpheus finetuned path is English/preset-voice focused.

    Bengali/non-ASCII input caused vLLM EngineDeadError in realtime streaming.
    Rejecting non-ASCII text before generation prevents unsupported input from
    killing the shared vLLM engine.
    """
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def is_valid_voice(voice: str) -> bool:
    return voice in {
        "tara",
        "leah",
        "jess",
        "leo",
        "dan",
        "mia",
        "zac",
        "zoe",
    }


def is_too_repetitive(text: str) -> bool:
    """
    Very repetitive prompts caused instability in the high-level generate_speech()
    realtime wrapper. This simple guard catches repeated sentences/phrases.
    """
    normalized = " ".join(text.lower().split())

    # Check repeated exact sentences.
    sentence_parts = [
        part.strip()
        for part in normalized.replace("?", ".").replace("!", ".").split(".")
        if part.strip()
    ]

    seen = {}
    for part in sentence_parts:
        seen[part] = seen.get(part, 0) + 1
        if seen[part] > MAX_REPEAT_PHRASE_COUNT:
            return True

    # Check repeated 5-word windows.
    words = normalized.split()
    if len(words) >= 10:
        windows = {}
        for i in range(0, len(words) - 4):
            window = " ".join(words[i : i + 5])
            windows[window] = windows.get(window, 0) + 1
            if windows[window] > MAX_REPEAT_PHRASE_COUNT:
                return True

    return False


def validate_prompt(prompt: str):
    if not prompt:
        return "Prompt is empty."

    if len(prompt) > MAX_PROMPT_CHARS:
        return (
            f"Prompt too long for this realtime demo. "
            f"Max {MAX_PROMPT_CHARS} characters allowed."
        )

    words = prompt.split()
    if len(words) > MAX_WORDS:
        return (
            f"Prompt too long for this realtime demo. "
            f"Max {MAX_WORDS} words allowed."
        )

    if not is_ascii_safe_text(prompt):
        return (
            "Only English/ASCII text is supported in this Orpheus realtime demo. "
            "Bengali/non-ASCII input is blocked because it caused vLLM EngineDeadError "
            "with the tested finetuned preset-voice model path."
        )

    if is_too_repetitive(prompt):
        return (
            "Prompt looks too repetitive for this realtime demo. "
            "Please use a shorter non-repeated English sentence."
        )

    return None


# -----------------------------
# Model lifecycle
# -----------------------------
def get_engine():
    global engine

    if engine is None:
        print("Loading Orpheus realtime streaming engine...")
        print(f"Model: {MODEL_NAME}")
        print(f"max_model_len={MAX_MODEL_LEN}")
        print(f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION}")

        engine = OrpheusModel(
            model_name=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )

        print("Engine loaded.")

    return engine


def reset_engine():
    """
    If vLLM EngineCore dies, the current engine object is no longer healthy.
    Reset it so the next valid request can reload the engine.
    """
    global engine
    print("Resetting Orpheus engine after failure.")
    engine = None


# -----------------------------
# WAV streaming helper
# -----------------------------
def create_wav_header(sample_rate=SAMPLE_RATE, bits_per_sample=16, channels=1):
    """
    Create a minimal WAV header for streaming PCM 16-bit mono audio.

    data_size is unknown at stream start, so it is set to 0.
    Many browsers can still play the streamed WAV response.
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )

    return header


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return f"""
<!doctype html>
<html>
<head>
  <title>Orpheus Realtime Streaming</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      max-width: 900px;
      margin: 40px auto;
      line-height: 1.5;
    }}
    textarea {{
      width: 100%;
      font-size: 15px;
    }}
    select, button {{
      font-size: 15px;
      padding: 8px;
      margin-top: 10px;
    }}
    audio {{
      width: 100%;
      margin-top: 20px;
    }}
    .note {{
      background: #fff3cd;
      padding: 12px;
      border-radius: 8px;
      margin-bottom: 16px;
      border: 1px solid #ffeeba;
    }}
    .error {{
      color: #b00020;
      margin-top: 12px;
      white-space: pre-wrap;
    }}
    .meta {{
      color: #444;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <h2>Orpheus-TTS Realtime Streaming</h2>

  <div class="note">
    This demo is configured for the tested English Orpheus preset-voice path.
    Non-ASCII text such as Bengali is blocked. Long or highly repetitive prompts
    are also blocked because they caused vLLM EngineDeadError through the high-level
    realtime wrapper.
  </div>

  <div class="meta">
    Limits: max {MAX_PROMPT_CHARS} characters, max {MAX_WORDS} words.
  </div>

  <br>

  <label for="voice">Voice:</label>
  <br>
  <select id="voice">
    <option value="tara" selected>tara</option>
    <option value="leah">leah</option>
    <option value="jess">jess</option>
    <option value="leo">leo</option>
    <option value="dan">dan</option>
    <option value="mia">mia</option>
    <option value="zac">zac</option>
    <option value="zoe">zoe</option>
  </select>

  <br><br>

  <label for="prompt">Prompt:</label>
  <textarea id="prompt" rows="6">Hello, this is a realtime streaming test using Orpheus text to speech.</textarea>

  <br>
  <button onclick="playAudio()">Play Audio</button>

  <div id="error" class="error"></div>

  <audio id="audio" controls></audio>

  <script>
    async function playAudio() {{
      const prompt = document.getElementById("prompt").value;
      const voice = document.getElementById("voice").value;
      const audio = document.getElementById("audio");
      const errorBox = document.getElementById("error");

      errorBox.textContent = "";

      const url = "/tts?prompt=" + encodeURIComponent(prompt) +
                  "&voice=" + encodeURIComponent(voice);

      try {{
        const res = await fetch(url);

        if (!res.ok) {{
          const text = await res.text();
          errorBox.textContent = text;
          audio.removeAttribute("src");
          return;
        }}

        const blob = await res.blob();
        const objectUrl = URL.createObjectURL(blob);

        audio.src = objectUrl;
        await audio.play();
      }} catch (err) {{
        errorBox.textContent = "Request failed: " + err;
      }}
    }}
  </script>
</body>
</html>
"""


@app.route("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "engine_loaded": engine is not None,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "sample_rate": SAMPLE_RATE,
        "max_prompt_chars": MAX_PROMPT_CHARS,
        "max_words": MAX_WORDS,
    }


@app.route("/tts", methods=["GET"])
def tts():
    prompt = request.args.get(
        "prompt",
        "Hey there, looks like you forgot to provide a prompt!",
    ).strip()

    voice = request.args.get("voice", DEFAULT_VOICE).strip()

    if not is_valid_voice(voice):
        return Response(
            f"Unsupported voice: {voice}. Supported voices: "
            "tara, leah, jess, leo, dan, mia, zac, zoe.",
            status=400,
            mimetype="text/plain",
        )

    prompt_error = validate_prompt(prompt)
    if prompt_error:
        return Response(
            prompt_error,
            status=400,
            mimetype="text/plain",
        )

    def generate_audio_stream():
        try:
            yield create_wav_header()

            model = get_engine()

            print("=" * 60)
            print("Realtime TTS request")
            print(f"Voice: {voice}")
            print(f"Prompt: {prompt}")

            syn_tokens = model.generate_speech(
                prompt=prompt,
                voice=voice,
                repetition_penalty=1.3,
                stop_token_ids=[128258],
                max_tokens=900,
                temperature=0.6,
                top_p=0.8,
            )

            chunk_count = 0

            for chunk in syn_tokens:
                chunk_count += 1
                yield chunk

            print(f"Streaming complete. chunks={chunk_count}")
            print("=" * 60)

        except Exception as e:
            print("TTS generation failed:", repr(e))
            reset_engine()
            return

    return Response(
        generate_audio_stream(),
        mimetype="audio/wav",
    )


def main():
    # Load once at startup so the first browser request is not blocked by model init.
    get_engine()

    app.run(
        host="0.0.0.0",
        port=PORT,
        threaded=False,
    )


if __name__ == "__main__":
    main()
