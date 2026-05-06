import os
import time
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch

from cli.SparkTTS import SparkTTS


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "pretrained_models" / "Spark-TTS-0.5B"
OUTPUT_DIR = BASE_DIR / "outputs"
GRADIO_TMP_DIR = BASE_DIR / "gradio_tmp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRADIO_TMP_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TMPDIR"] = str(GRADIO_TMP_DIR)
os.environ["GRADIO_TEMP_DIR"] = str(GRADIO_TMP_DIR)


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_TEXT = "Today is a beautiful day. I am testing Spark TTS voice cloning."

DEFAULT_PROMPT_TEXT = (
    "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。"
    "豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"
)

DEFAULT_PROMPT_AUDIO = str(BASE_DIR / "example" / "prompt_audio.wav")


# -----------------------------
# Load model once
# -----------------------------
DEVICE_ID = 0

print(f"Loading Spark-TTS from: {MODEL_DIR}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = SparkTTS(
    model_dir=str(MODEL_DIR),
    device=DEVICE_ID,
)

print("Spark-TTS model loaded.")


def save_wav(wav, sample_rate: int = 16000):
    timestamp_ms = int(time.time() * 1000)
    out_path = OUTPUT_DIR / f"spark_tts_{timestamp_ms}.wav"

    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().float().numpy()

    wav = wav.squeeze()

    sf.write(str(out_path), wav, sample_rate)
    return str(out_path)


def generate_voice_clone(text, prompt_text, prompt_audio):
    text = (text or "").strip()
    prompt_text = (prompt_text or "").strip()

    if not text:
        raise gr.Error("Please enter text.")

    if not prompt_audio:
        raise gr.Error("Please provide prompt/reference audio.")

    if not prompt_text:
        raise gr.Error("Please provide the exact transcript of the prompt/reference audio.")

    start = time.perf_counter()

    wav = model.inference(
        text=text,
        prompt_text=prompt_text,
        prompt_speech_path=prompt_audio,
    )

    elapsed = time.perf_counter() - start

    # Spark-TTS outputs 16 kHz audio in the default path.
    sample_rate = 16000
    out_path = save_wav(wav, sample_rate)

    audio_duration = len(sf.read(out_path)[0]) / sample_rate
    rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

    stats = (
        f"Generation time: {elapsed:.2f} sec\n"
        f"Audio duration: {audio_duration:.2f} sec\n"
        f"RTF: {rtf:.3f}\n"
        f"Sample rate: {sample_rate}\n"
        f"Device: cuda:{DEVICE_ID}\n"
        f"Output file: {out_path}"
    )

    return out_path, stats


def generate_created_voice(text, gender, pitch, speed):
    text = (text or "").strip()

    if not text:
        raise gr.Error("Please enter text.")

    start = time.perf_counter()

    wav = model.inference(
        text=text,
        gender=gender,
        pitch=pitch,
        speed=speed,
    )

    elapsed = time.perf_counter() - start

    sample_rate = 16000
    out_path = save_wav(wav, sample_rate)

    audio_duration = len(sf.read(out_path)[0]) / sample_rate
    rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

    stats = (
        f"Generation time: {elapsed:.2f} sec\n"
        f"Audio duration: {audio_duration:.2f} sec\n"
        f"RTF: {rtf:.3f}\n"
        f"Sample rate: {sample_rate}\n"
        f"Device: cuda:{DEVICE_ID}\n"
        f"Output file: {out_path}"
    )

    return out_path, stats


with gr.Blocks(title="Spark-TTS Custom UI") as demo:
    gr.Markdown("# Spark-TTS Custom UI")
    gr.Markdown("Custom Gradio UI for Spark-TTS single inference testing.")

    with gr.Tab("Voice Cloning"):
        with gr.Row():
            with gr.Column():
                clone_text = gr.Textbox(
                    label="Target Text",
                    value=DEFAULT_TEXT,
                    lines=4,
                )

                clone_prompt_text = gr.Textbox(
                    label="Prompt Audio Transcript",
                    value=DEFAULT_PROMPT_TEXT,
                    lines=4,
                )

                clone_prompt_audio = gr.Audio(
                    label="Prompt / Reference Audio",
                    type="filepath",
                    value=DEFAULT_PROMPT_AUDIO,
                )

                clone_button = gr.Button("Generate Voice Clone")

            with gr.Column():
                clone_audio = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                )
                clone_stats = gr.Textbox(
                    label="Stats",
                    lines=7,
                )

        clone_button.click(
            fn=generate_voice_clone,
            inputs=[clone_text, clone_prompt_text, clone_prompt_audio],
            outputs=[clone_audio, clone_stats],
        )

    with gr.Tab("Voice Creation"):
        with gr.Row():
            with gr.Column():
                create_text = gr.Textbox(
                    label="Target Text",
                    value="Hello, this is Spark TTS speaking with a generated male voice.",
                    lines=4,
                )

                gender = gr.Dropdown(
                    label="Gender",
                    choices=["male", "female"],
                    value="male",
                )

                pitch = gr.Dropdown(
                    label="Pitch",
                    choices=["very_low", "low", "moderate", "high", "very_high"],
                    value="moderate",
                )

                speed = gr.Dropdown(
                    label="Speed",
                    choices=["very_low", "low", "moderate", "high", "very_high"],
                    value="moderate",
                )

                create_button = gr.Button("Generate Created Voice")

            with gr.Column():
                create_audio = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                )
                create_stats = gr.Textbox(
                    label="Stats",
                    lines=7,
                )

        create_button.click(
            fn=generate_created_voice,
            inputs=[create_text, gender, pitch, speed],
            outputs=[create_audio, create_stats],
        )


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=6012,
        share=False,
    )
