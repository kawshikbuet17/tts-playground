import os
import time
from pathlib import Path

import torch
import gradio as gr
import soundfile as sf

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


# -----------------------------
# Paths / folders
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
GRADIO_TMP_DIR = BASE_DIR / "gradio_tmp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRADIO_TMP_DIR.mkdir(parents=True, exist_ok=True)

# Force Gradio/temp files into project-local folder instead of /tmp/gradio
os.environ["TMPDIR"] = str(GRADIO_TMP_DIR)
os.environ["GRADIO_TEMP_DIR"] = str(GRADIO_TMP_DIR)


# -----------------------------
# Model config
# -----------------------------
MODEL_ID = "ai4bharat/indic-parler-tts"

DEFAULT_TEXT = "আজকের আবহাওয়া খুব সুন্দর। আমি বাংলা টেক্সট টু স্পিচ পরীক্ষা করছি।"

DEFAULT_DESCRIPTION = (
    "Aditi speaks in a natural Bengali voice, at a moderate pace, "
    "with clear pronunciation and very clear audio."
)


# -----------------------------
# Load model
# -----------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading model on {device}...")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
).to(device)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Indic Parler-TTS uses separate tokenizer for voice/style description.
description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

print("Model loaded.")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Gradio temp directory: {GRADIO_TMP_DIR}")


# -----------------------------
# Inference function
# -----------------------------
def generate_tts(text: str, description: str):
    text = (text or "").strip()
    description = (description or "").strip()

    if not text:
        raise gr.Error("Please enter Bengali text.")

    if not description:
        description = DEFAULT_DESCRIPTION

    start = time.perf_counter()

    description_inputs = description_tokenizer(
        description,
        return_tensors="pt",
    ).to(device)

    prompt_inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        generation = model.generate(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )

    elapsed = time.perf_counter() - start

    # soundfile cannot write float16, so convert to float32.
    audio_arr = generation.detach().cpu().float().numpy().squeeze()
    sampling_rate = model.config.sampling_rate

    timestamp_ms = int(time.time() * 1000)
    out_path = OUTPUT_DIR / f"indic_parler_bn_{timestamp_ms}.wav"

    sf.write(str(out_path), audio_arr, sampling_rate)

    audio_duration = len(audio_arr) / sampling_rate
    rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

    stats = (
        f"Generation time: {elapsed:.2f} sec\n"
        f"Audio duration: {audio_duration:.2f} sec\n"
        f"RTF: {rtf:.3f}\n"
        f"Sampling rate: {sampling_rate}\n"
        f"Device: {device}\n"
        f"Output file: {out_path}"
    )

    return str(out_path), stats


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="Indic Parler-TTS Bengali Test UI") as demo:
    gr.Markdown("# Indic Parler-TTS Bengali Test UI")
    gr.Markdown("Local Bengali TTS test using `ai4bharat/indic-parler-tts`.")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Bengali Text",
                value=DEFAULT_TEXT,
                lines=5,
            )

            description_input = gr.Textbox(
                label="Voice Description",
                value=DEFAULT_DESCRIPTION,
                lines=4,
            )

            generate_button = gr.Button("Generate TTS")

        with gr.Column():
            audio_output = gr.Audio(
                label="Generated Audio",
                type="filepath",
            )

            stats_output = gr.Textbox(
                label="Stats",
                lines=7,
            )

    gr.Examples(
        examples=[
            [
                "আমি আজকে একটি বাংলা টেক্সট টু স্পিচ মডেল পরীক্ষা করছি।",
                "Aditi speaks in a natural Bengali voice, at a moderate pace, with clear pronunciation and very clear audio.",
            ],
            [
                "আপনার অর্ডারটি সফলভাবে সম্পন্ন হয়েছে। ধন্যবাদ।",
                "Aditi speaks in a calm customer-service tone, with clear Bengali pronunciation and very clear audio.",
            ],
            [
                "ওয়াও! এটা সত্যিই অসাধারণ লাগছে। তুমি কি আবার এটা শুনতে চাও?",
                "Aditi speaks in an excited and expressive Bengali voice, with moderate pace and very clear audio.",
            ],
            [
                "আজ ঢাকায় আবহাওয়া আংশিক মেঘলা থাকতে পারে। দিনের তাপমাত্রা সামান্য বাড়তে পারে।",
                "Aditi speaks like a professional Bengali news presenter, with a calm tone, steady pace, and very clear audio.",
            ],
        ],
        inputs=[text_input, description_input],
    )

    generate_button.click(
        fn=generate_tts,
        inputs=[text_input, description_input],
        outputs=[audio_output, stats_output],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=6010,
        share=False,
    )
