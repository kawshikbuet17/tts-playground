import os
import time
from pathlib import Path

import torch
import gradio as gr
import soundfile as sf

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "benchmark_outputs"
GRADIO_TMP_DIR = BASE_DIR / "gradio_tmp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRADIO_TMP_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TMPDIR"] = str(GRADIO_TMP_DIR)
os.environ["GRADIO_TEMP_DIR"] = str(GRADIO_TMP_DIR)


# -----------------------------
# Model config
# -----------------------------
MODEL_ID = "ai4bharat/indic-parler-tts"

DEFAULT_DESCRIPTION = (
    "Aditi speaks in a natural Bengali voice, at a moderate pace, "
    "with clear pronunciation and very clear audio."
)

DEFAULT_TEXTS = """আজকের আবহাওয়া খুব সুন্দর। আমি বাংলা টেক্সট টু স্পিচ পরীক্ষা করছি।
আপনার অর্ডারটি সফলভাবে সম্পন্ন হয়েছে। ধন্যবাদ।
বাংলাদেশের নদীগুলো আমাদের জীবনের সঙ্গে গভীরভাবে জড়িয়ে আছে।
ওয়াও! এটা সত্যিই অসাধারণ লাগছে। তুমি কি আবার এটা শুনতে চাও?"""

MAX_UI_AUDIOS = 8


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
description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

print("Model loaded.")
print(f"Output directory: {OUTPUT_DIR}")


# -----------------------------
# Helpers
# -----------------------------
def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_texts(texts_blob: str):
    texts = []
    for line in (texts_blob or "").splitlines():
        line = line.strip()
        if line:
            texts.append(line)
    return texts


def save_audio(audio_arr, sampling_rate, path: Path):
    audio_arr = audio_arr.astype("float32")
    sf.write(str(path), audio_arr, sampling_rate)


def generate_one(text: str, description: str):
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

    return generation.detach().cpu().float().numpy().squeeze()


def run_warmup(description: str):
    _ = generate_one(
        text="আমি বাংলা টেক্সট টু স্পিচ পরীক্ষা করছি।",
        description=description,
    )
    sync_cuda()


def pad_items(items, max_items=MAX_UI_AUDIOS, fill_value=None):
    items = list(items[:max_items])
    while len(items) < max_items:
        items.append(fill_value)
    return items


# -----------------------------
# Benchmark function
# -----------------------------
def benchmark(texts_blob: str, description: str, run_warmup_first: bool):
    texts = parse_texts(texts_blob)
    description = (description or "").strip() or DEFAULT_DESCRIPTION

    if not texts:
        raise gr.Error("Please enter at least one text line.")

    if len(texts) < 2:
        raise gr.Error("Please enter at least 2 text lines to compare sequential vs batch.")

    if len(texts) > MAX_UI_AUDIOS:
        raise gr.Error(f"Please use at most {MAX_UI_AUDIOS} texts for UI playback.")

    sampling_rate = model.config.sampling_rate
    timestamp = int(time.time() * 1000)
    run_dir = OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logs = []
    sequential_audio_files = []
    batch_audio_files = []
    seq_times = []
    batch_durations = []
    seq_durations = []

    logs.append(f"Device: {device}")
    if torch.cuda.is_available():
        logs.append(f"GPU: {torch.cuda.get_device_name(0)}")
    logs.append(f"Number of texts: {len(texts)}")
    logs.append(f"Output directory: {run_dir}")

    if run_warmup_first:
        logs.append("\nRunning warmup...")
        run_warmup(description)
        logs.append("Warmup done.")

    # -----------------------------
    # Sequential benchmark
    # -----------------------------
    logs.append("\n==============================")
    logs.append("Sequential inference benchmark")
    logs.append("==============================")

    seq_total_audio_duration = 0.0

    sync_cuda()
    seq_total_start = time.perf_counter()

    for idx, text in enumerate(texts, start=1):
        logs.append(f"\nSequential item {idx}/{len(texts)}")
        logs.append(f"Text: {text}")

        sync_cuda()
        start = time.perf_counter()

        audio_arr = generate_one(text=text, description=description)

        sync_cuda()
        elapsed = time.perf_counter() - start

        out_path = run_dir / f"seq_{idx}.wav"
        save_audio(audio_arr, sampling_rate, out_path)
        sequential_audio_files.append(str(out_path))

        audio_duration = len(audio_arr) / sampling_rate
        seq_total_audio_duration += audio_duration
        seq_times.append(elapsed)
        seq_durations.append(audio_duration)

        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

        logs.append(f"Saved: {out_path.name}")
        logs.append(f"Generation time: {elapsed:.2f} sec")
        logs.append(f"Audio duration: {audio_duration:.2f} sec")
        logs.append(f"RTF: {rtf:.3f}")

    sync_cuda()
    seq_wall_time = time.perf_counter() - seq_total_start

    seq_throughput = (
        seq_total_audio_duration / seq_wall_time if seq_wall_time > 0 else 0.0
    )
    seq_rtf = (
        seq_wall_time / seq_total_audio_duration if seq_total_audio_duration > 0 else 0.0
    )
    seq_avg_item_time = sum(seq_times) / len(seq_times)

    logs.append("\n--- Sequential Summary ---")
    logs.append(f"Total wall time: {seq_wall_time:.2f} sec")
    logs.append(f"Average item generation time: {seq_avg_item_time:.2f} sec")
    logs.append(f"Total audio duration: {seq_total_audio_duration:.2f} sec")
    logs.append(f"Audio seconds / wall second: {seq_throughput:.2f}")
    logs.append(f"Overall RTF: {seq_rtf:.3f}")

    # -----------------------------
    # Model-level batch benchmark
    # -----------------------------
    logs.append("\n==============================")
    logs.append("Model-level batch benchmark")
    logs.append("==============================")

    descriptions = [description] * len(texts)

    description_inputs = description_tokenizer(
        descriptions,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    prompt_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    logs.append(f"description input_ids shape: {tuple(description_inputs.input_ids.shape)}")
    logs.append(f"prompt input_ids shape: {tuple(prompt_inputs.input_ids.shape)}")

    sync_cuda()
    batch_start = time.perf_counter()

    with torch.inference_mode():
        generation = model.generate(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )

    sync_cuda()
    batch_wall_time = time.perf_counter() - batch_start

    logs.append(f"Generation tensor shape: {tuple(generation.shape)}")

    audio_batch = generation.detach().cpu().float().numpy()

    if audio_batch.ndim == 1:
        audio_batch = audio_batch[None, :]

    batch_total_audio_duration = 0.0

    for idx, audio_arr in enumerate(audio_batch, start=1):
        audio_arr = audio_arr.squeeze()

        out_path = run_dir / f"batch_{idx}.wav"
        save_audio(audio_arr, sampling_rate, out_path)
        batch_audio_files.append(str(out_path))

        audio_duration = len(audio_arr) / sampling_rate
        batch_total_audio_duration += audio_duration
        batch_durations.append(audio_duration)

        logs.append(f"Saved: {out_path.name}")
        logs.append(f"Batch audio duration {idx}: {audio_duration:.2f} sec")

    batch_throughput = (
        batch_total_audio_duration / batch_wall_time if batch_wall_time > 0 else 0.0
    )
    batch_rtf = (
        batch_wall_time / batch_total_audio_duration if batch_total_audio_duration > 0 else 0.0
    )

    logs.append("\n--- Batch Summary ---")
    logs.append(f"Batch size: {len(texts)}")
    logs.append(f"Batch wall time: {batch_wall_time:.2f} sec")
    logs.append(f"Total audio duration: {batch_total_audio_duration:.2f} sec")
    logs.append(f"Audio seconds / wall second: {batch_throughput:.2f}")
    logs.append(f"Overall RTF: {batch_rtf:.3f}")

    # -----------------------------
    # Final comparison
    # -----------------------------
    speedup = seq_wall_time / batch_wall_time if batch_wall_time > 0 else 0.0

    logs.append("\n==============================")
    logs.append("Final Comparison")
    logs.append("==============================")
    logs.append(f"Sequential wall time: {seq_wall_time:.2f} sec")
    logs.append(f"Batch wall time:      {batch_wall_time:.2f} sec")
    logs.append(f"Speedup:              {speedup:.2f}x")
    logs.append("")
    logs.append(f"Sequential throughput: {seq_throughput:.2f} audio-sec/wall-sec")
    logs.append(f"Batch throughput:      {batch_throughput:.2f} audio-sec/wall-sec")
    logs.append("")
    logs.append(f"Sequential RTF: {seq_rtf:.3f}")
    logs.append(f"Batch RTF:      {batch_rtf:.3f}")

    if speedup >= 1.5:
        verdict = "Batching gives meaningful speedup."
    elif speedup > 1.05:
        verdict = "Batching gives small speedup."
    else:
        verdict = "Batching gives little/no speedup; inference may be mostly sequential internally."

    logs.append("")
    logs.append(f"Result: {verdict}")

    result_summary = (
        f"Sequential: {seq_wall_time:.2f}s | "
        f"Batch: {batch_wall_time:.2f}s | "
        f"Speedup: {speedup:.2f}x | "
        f"Result: {verdict}"
    )

    # Per-row text labels for easier one-to-one comparison.
    row_labels = []
    for idx, text in enumerate(texts, start=1):
        seq_time = seq_times[idx - 1] if idx - 1 < len(seq_times) else 0.0
        seq_dur = seq_durations[idx - 1] if idx - 1 < len(seq_durations) else 0.0
        batch_dur = batch_durations[idx - 1] if idx - 1 < len(batch_durations) else 0.0

        row_labels.append(
            f"Text {idx}\n"
            f"{text}\n\n"
            f"Sequential generation time: {seq_time:.2f}s\n"
            f"Sequential audio duration: {seq_dur:.2f}s\n"
            f"Batch audio duration: {batch_dur:.2f}s\n"
            f"Batch generated inside one shared batch call: {batch_wall_time:.2f}s total"
        )

    row_labels = pad_items(row_labels, fill_value="")
    seq_outputs = pad_items(sequential_audio_files)
    batch_outputs = pad_items(batch_audio_files)

    # Return:
    # summary, logs,
    # then 8 rows x [label, sequential audio, batch audio]
    row_outputs = []
    for i in range(MAX_UI_AUDIOS):
        row_outputs.extend([row_labels[i], seq_outputs[i], batch_outputs[i]])

    return (
        result_summary,
        "\n".join(logs),
        *row_outputs,
    )


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="Indic Parler-TTS Batch Benchmark UI") as demo:
    gr.Markdown("# Indic Parler-TTS Sequential vs Batch Benchmark")
    gr.Markdown(
        "Paste Bengali texts, one per line. "
        "Each row below compares the sequential output and the batched output for the same text."
    )

    with gr.Row():
        with gr.Column():
            texts_input = gr.Textbox(
                label="Texts, one per line",
                value=DEFAULT_TEXTS,
                lines=8,
            )

            description_input = gr.Textbox(
                label="Voice Description",
                value=DEFAULT_DESCRIPTION,
                lines=4,
            )

            warmup_checkbox = gr.Checkbox(
                label="Run warmup before benchmark",
                value=True,
            )

            run_button = gr.Button("Run Sequential vs Batch Benchmark")

        with gr.Column():
            summary_output = gr.Textbox(
                label="Summary",
                lines=3,
            )

            logs_output = gr.Textbox(
                label="Full Benchmark Logs",
                lines=18,
            )

    gr.Markdown("## One-to-One Audio Comparison")
    gr.Markdown("Left = input text/details, middle = sequential audio, right = batch audio.")

    row_components = []

    for i in range(1, MAX_UI_AUDIOS + 1):
        with gr.Row():
            text_box = gr.Textbox(
                label=f"Text {i} Details",
                lines=7,
                interactive=False,
            )

            seq_audio = gr.Audio(
                label=f"Sequential Audio {i}",
                type="filepath",
            )

            batch_audio = gr.Audio(
                label=f"Batch Audio {i}",
                type="filepath",
            )

        row_components.extend([text_box, seq_audio, batch_audio])

    run_button.click(
        fn=benchmark,
        inputs=[texts_input, description_input, warmup_checkbox],
        outputs=[
            summary_output,
            logs_output,
            *row_components,
        ],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=6011,
        share=False,
    )
