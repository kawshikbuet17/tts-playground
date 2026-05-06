import argparse
import gc
import time
from pathlib import Path
from statistics import mean

import numpy as np
import soundfile as sf
import torch

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_CONFIG = "configs/ljs_base.json"
DEFAULT_CHECKPOINT = "checkpoints/pretrained_drive/pretrained_ljs.pth"
DEFAULT_OUTPUT_DIR = "outputs/benchmark"

SILENCE_PEAK_THRESHOLD = 1e-4
SILENCE_RMS_THRESHOLD = 1e-5


# -----------------------------
# Warmup texts
# -----------------------------
WARMUP_TEXTS = [
    "This is a short warmup test.",
    "The text to speech model is preparing for the benchmark.",
    "This final warmup sentence helps stabilize the first few GPU calls.",
    "Testing the model with different sentences.",
    "This is another test sentence for warmup.",
    "Let's see how the model performs with these sentences.",
    "The model should generate consistent audio quality across all tests.",
    "We are now ready to start the actual benchmark tests.",
    "Let's begin the benchmark tests now.",
    "This is the final warmup sentence before starting the benchmark.",
]


# -----------------------------
# LJ English benchmark texts
# Approx target generated audio range:
#   very short: ~2-4 sec
#   short:      ~4-8 sec
#   medium:     ~8-15 sec
#   long:       ~15-30 sec
# -----------------------------
TEXTS = [
    {
        "label": "en_very_short_01",
        "target_range": "2-4s",
        "text": "I am speaking now.",
    },
    {
        "label": "en_very_short_02",
        "target_range": "2-4s",
        "text": "This is a short test.",
    },
    {
        "label": "en_very_short_03",
        "target_range": "2-4s",
        "text": "Audio is being generated.",
    },

    {
        "label": "en_short_01",
        "target_range": "4-8s",
        "text": "This is a simple test of text to speech synthesis.",
    },
    {
        "label": "en_short_02",
        "target_range": "4-8s",
        "text": "The model converts written English text into natural sounding speech.",
    },
    {
        "label": "en_short_03",
        "target_range": "4-8s",
        "text": "Today we are testing the speed and quality of a VITS speech model.",
    },

    {
        "label": "en_medium_01",
        "target_range": "8-15s",
        "text": (
            "Text to speech systems are useful for voice assistants, accessibility tools, "
            "automated announcements, and many other spoken language applications."
        ),
    },
    {
        "label": "en_medium_02",
        "target_range": "8-15s",
        "text": (
            "A good speech synthesis model should produce clear pronunciation, stable rhythm, "
            "natural pauses, and consistent audio quality across different sentence lengths."
        ),
    },
    {
        "label": "en_medium_03",
        "target_range": "8-15s",
        "text": (
            "In this benchmark, we measure generation time, audio duration, real time factor, "
            "and basic signal statistics such as peak amplitude and root mean square energy."
        ),
    },

    {
        "label": "en_long_01",
        "target_range": "15-30s",
        "text": (
            "During this test, we evaluate how the VITS model handles short, medium, and longer "
            "English inputs on a server equipped with a graphics processing unit. The goal is to "
            "measure both runtime behavior and generated audio validity."
        ),
    },
    {
        "label": "en_long_02",
        "target_range": "15-30s",
        "text": (
            "When deploying a text to speech model on a server, it is important to measure more "
            "than only subjective audio quality. We also need to record model loading time, "
            "generation latency, output duration, real time factor, and whether any silent or "
            "invalid audio files were produced."
        ),
    },
    {
        "label": "en_long_03",
        "target_range": "15-30s",
        "text": (
            "Real world text to speech systems must handle many kinds of input. Sometimes the "
            "user provides a very short command, sometimes a complete sentence, and sometimes "
            "a longer paragraph. A useful benchmark should therefore include examples across "
            "several text lengths."
        ),
    },
]


# -----------------------------
# Helpers
# -----------------------------
def clear_gpu_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)

    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)

    return torch.LongTensor(text_norm)


def normalize_audio(audio):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    audio = np.asarray(audio)
    audio = np.squeeze(audio)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0

    if peak > 1.0:
        audio = audio / peak

    return audio


def get_audio_stats(audio):
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        return {
            "peak": 0.0,
            "rms": 0.0,
            "mean_abs": 0.0,
            "is_silent": True,
        }

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    mean_abs = float(np.mean(np.abs(audio)))

    is_silent = (
        peak < SILENCE_PEAK_THRESHOLD
        or rms < SILENCE_RMS_THRESHOLD
    )

    return {
        "peak": peak,
        "rms": rms,
        "mean_abs": mean_abs,
        "is_silent": is_silent,
    }


def load_model(config_path, checkpoint_path, device):
    hps = utils.get_hparams_from_file(config_path)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)

    net_g.eval()

    _ = utils.load_checkpoint(checkpoint_path, net_g, None)

    sample_rate = int(hps.data.sampling_rate)

    return net_g, hps, sample_rate


def generate_one(
    model,
    hps,
    text,
    out_path,
    sample_rate,
    device,
    noise_scale,
    noise_scale_w,
    length_scale,
):
    stn_tst = get_text(text, hps)
    x_tst = stn_tst.to(device).unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    infer_start = time.monotonic()

    with torch.inference_mode():
        audio = model.infer(
            x_tst,
            x_tst_lengths,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )[0][0, 0].data.cpu().float().numpy()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    infer_elapsed = time.monotonic() - infer_start

    post_start = time.monotonic()

    audio = normalize_audio(audio)
    audio_stats = get_audio_stats(audio)

    duration = len(audio) / sample_rate if audio.size else 0.0

    sf.write(
        str(out_path),
        audio,
        samplerate=sample_rate,
    )

    post_elapsed = time.monotonic() - post_start
    total_elapsed = infer_elapsed + post_elapsed
    rtf = total_elapsed / duration if duration > 0 else 0.0

    return {
        "elapsed": total_elapsed,
        "infer_elapsed": infer_elapsed,
        "post_elapsed": post_elapsed,
        "duration": duration,
        "rtf": rtf,
        **audio_stats,
    }


def summarize_by_range(results):
    by_range = {}

    for r in results:
        by_range.setdefault(r["target_range"], []).append(r)

    lines = []

    for target_range, rows in sorted(by_range.items()):
        total_elapsed = sum(r["elapsed"] for r in rows)
        total_audio = sum(r["duration"] for r in rows)
        avg_rtf = mean(r["rtf"] for r in rows)
        effective_rtf = total_elapsed / total_audio if total_audio > 0 else 0.0
        silent_count = sum(1 for r in rows if r["is_silent"])

        lines.append(
            f"{target_range}: items={len(rows)}, "
            f"silent={silent_count}, "
            f"audio={total_audio:.2f}s, "
            f"generation={total_elapsed:.2f}s, "
            f"effective_rtf={effective_rtf:.3f}, "
            f"avg_rtf={avg_rtf:.3f}"
        )

    return lines


def write_summary_file(summary, warmup_results, results, output_dir):
    summary_path = output_dir / "summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("VITS LJ Speech Benchmark Summary\n")
        f.write("=" * 120 + "\n\n")

        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

        f.write("\nWarmup files:\n")

        for r in warmup_results:
            f.write(
                f"warmup {r['idx']:02d} | "
                f"elapsed={r['elapsed']:6.2f}s | "
                f"infer={r['infer_elapsed']:6.2f}s | "
                f"post={r['post_elapsed']:5.2f}s | "
                f"audio={r['duration']:6.2f}s | "
                f"RTF={r['rtf']:6.3f} | "
                f"peak={r['peak']:.8f} | "
                f"rms={r['rms']:.8f} | "
                f"silent={r['is_silent']} | "
                f"{r['path']}\n"
            )

        f.write("\nBenchmark files:\n")

        for r in results:
            f.write(
                f"item {r['idx']:02d} | "
                f"label={r['label']:<18} | "
                f"range={r['target_range']:<6} | "
                f"chars={r['chars']:4d} | "
                f"elapsed={r['elapsed']:6.2f}s | "
                f"infer={r['infer_elapsed']:6.2f}s | "
                f"post={r['post_elapsed']:5.2f}s | "
                f"audio={r['duration']:6.2f}s | "
                f"RTF={r['rtf']:6.3f} | "
                f"peak={r['peak']:.8f} | "
                f"rms={r['rms']:.8f} | "
                f"silent={r['is_silent']} | "
                f"{r['path']}\n"
            )

    return summary_path


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("VITS LJ Speech benchmark")
    print("=" * 120)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    print(f"Warmup runs: {len(WARMUP_TEXTS)}")
    print(f"Benchmark texts: {len(TEXTS)}")
    print("Language: English")
    print("Target audio ranges: 2s to 30s")
    print(f"Output dir: {output_dir}")

    if torch.cuda.is_available():
        print(f"Visible GPU: {torch.cuda.get_device_name(0)}")

    clear_gpu_memory()

    print("\nLoading VITS model once...")
    load_start = time.monotonic()

    model, hps, sample_rate = load_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    load_elapsed = time.monotonic() - load_start

    print(f"Model loaded in {load_elapsed:.2f}s")
    print(f"Sample rate: {sample_rate}")

    # -----------------------------
    # Warmup
    # -----------------------------
    print("\nWarmup...")
    warmup_results = []

    for idx, text in enumerate(WARMUP_TEXTS, start=1):
        out_path = output_dir / f"warmup_{idx}.wav"

        result = generate_one(
            model=model,
            hps=hps,
            text=text,
            out_path=out_path,
            sample_rate=sample_rate,
            device=device,
            noise_scale=args.noise_scale,
            noise_scale_w=args.noise_scale_w,
            length_scale=args.length_scale,
        )

        row = {
            "idx": idx,
            "text": text,
            "path": str(out_path),
            **result,
        }

        warmup_results.append(row)

        print(
            f"warmup {idx}: "
            f"elapsed={result['elapsed']:.2f}s, "
            f"infer={result['infer_elapsed']:.2f}s, "
            f"post={result['post_elapsed']:.2f}s, "
            f"audio={result['duration']:.2f}s, "
            f"RTF={result['rtf']:.3f}, "
            f"peak={result['peak']:.8f}, "
            f"rms={result['rms']:.8f}, "
            f"silent={result['is_silent']}, "
            f"path={out_path}"
        )

    # -----------------------------
    # Benchmark
    # -----------------------------
    results = []

    print("\nBenchmark...")
    print("-" * 120)

    benchmark_start = time.monotonic()

    for idx, item in enumerate(TEXTS, start=1):
        label = item["label"]
        target_range = item["target_range"]
        text = item["text"]

        out_path = output_dir / f"bench_{idx:02d}_{label}.wav"

        result = generate_one(
            model=model,
            hps=hps,
            text=text,
            out_path=out_path,
            sample_rate=sample_rate,
            device=device,
            noise_scale=args.noise_scale,
            noise_scale_w=args.noise_scale_w,
            length_scale=args.length_scale,
        )

        row = {
            "idx": idx,
            "label": label,
            "target_range": target_range,
            "text": text,
            "chars": len(text),
            "path": str(out_path),
            **result,
        }

        results.append(row)

        print(
            f"item {idx:02d}: "
            f"label={label:<18}, "
            f"range={target_range:<6}, "
            f"chars={len(text):4d}, "
            f"elapsed={result['elapsed']:6.2f}s, "
            f"infer={result['infer_elapsed']:6.2f}s, "
            f"post={result['post_elapsed']:5.2f}s, "
            f"audio={result['duration']:6.2f}s, "
            f"RTF={result['rtf']:6.3f}, "
            f"peak={result['peak']:.8f}, "
            f"rms={result['rms']:.8f}, "
            f"silent={result['is_silent']}, "
            f"path={out_path}"
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    benchmark_wall = time.monotonic() - benchmark_start

    total_elapsed = sum(r["elapsed"] for r in results)
    total_infer = sum(r["infer_elapsed"] for r in results)
    total_post = sum(r["post_elapsed"] for r in results)
    total_audio = sum(r["duration"] for r in results)

    effective_rtf = total_elapsed / total_audio if total_audio > 0 else 0.0
    infer_rtf = total_infer / total_audio if total_audio > 0 else 0.0
    wall_rtf = benchmark_wall / total_audio if total_audio > 0 else 0.0

    avg_rtf = mean(r["rtf"] for r in results) if results else 0.0
    avg_audio = mean(r["duration"] for r in results) if results else 0.0
    avg_elapsed = mean(r["elapsed"] for r in results) if results else 0.0
    avg_infer = mean(r["infer_elapsed"] for r in results) if results else 0.0
    avg_post = mean(r["post_elapsed"] for r in results) if results else 0.0
    avg_chars = mean(r["chars"] for r in results) if results else 0.0
    avg_peak = mean(r["peak"] for r in results) if results else 0.0
    avg_rms = mean(r["rms"] for r in results) if results else 0.0
    min_peak = min(r["peak"] for r in results) if results else 0.0
    min_rms = min(r["rms"] for r in results) if results else 0.0

    silent_count = sum(1 for r in results if r["is_silent"])
    warmup_silent_count = sum(1 for r in warmup_results if r["is_silent"])

    fastest = min(results, key=lambda r: r["rtf"]) if results else None
    slowest = max(results, key=lambda r: r["rtf"]) if results else None

    range_summary_lines = summarize_by_range(results)

    summary = {
        "status": "SUCCESS",
        "model_family": "VITS",
        "checkpoint": args.checkpoint,
        "config": args.config,
        "language": "English",
        "dataset_style": "LJ Speech",
        "sample_rate": str(sample_rate),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch": torch.__version__,
        "noise_scale": str(args.noise_scale),
        "noise_scale_w": str(args.noise_scale_w),
        "length_scale": str(args.length_scale),
        "model_load_time_sec": f"{load_elapsed:.2f}",
        "warmup_runs": str(len(warmup_results)),
        "warmup_silent_items": str(warmup_silent_count),
        "benchmark_items": str(len(results)),
        "silent_items": str(silent_count),
        "benchmark_wall_time_sec": f"{benchmark_wall:.2f}",
        "total_generation_time_sec": f"{total_elapsed:.2f}",
        "total_inference_time_sec": f"{total_infer:.2f}",
        "total_postprocess_time_sec": f"{total_post:.2f}",
        "total_audio_duration_sec": f"{total_audio:.2f}",
        "average_generation_time_sec": f"{avg_elapsed:.2f}",
        "average_inference_time_sec": f"{avg_infer:.2f}",
        "average_postprocess_time_sec": f"{avg_post:.2f}",
        "average_audio_duration_sec": f"{avg_audio:.2f}",
        "average_text_chars": f"{avg_chars:.1f}",
        "average_peak": f"{avg_peak:.8f}",
        "average_rms": f"{avg_rms:.8f}",
        "min_peak": f"{min_peak:.8f}",
        "min_rms": f"{min_rms:.8f}",
        "effective_rtf": f"{effective_rtf:.3f}",
        "inference_only_rtf": f"{infer_rtf:.3f}",
        "wall_clock_rtf": f"{wall_rtf:.3f}",
        "average_per_item_rtf": f"{avg_rtf:.3f}",
    }

    if fastest:
        summary["fastest_item"] = (
            f"item {fastest['idx']} | "
            f"label={fastest['label']} | "
            f"range={fastest['target_range']} | "
            f"RTF={fastest['rtf']:.3f} | "
            f"audio={fastest['duration']:.2f}s | "
            f"peak={fastest['peak']:.8f} | "
            f"rms={fastest['rms']:.8f} | "
            f"silent={fastest['is_silent']}"
        )

    if slowest:
        summary["slowest_item"] = (
            f"item {slowest['idx']} | "
            f"label={slowest['label']} | "
            f"range={slowest['target_range']} | "
            f"RTF={slowest['rtf']:.3f} | "
            f"audio={slowest['duration']:.2f}s | "
            f"peak={slowest['peak']:.8f} | "
            f"rms={slowest['rms']:.8f} | "
            f"silent={slowest['is_silent']}"
        )

    for i, line in enumerate(range_summary_lines, start=1):
        summary[f"range_summary_{i}"] = line

    summary_path = write_summary_file(
        summary=summary,
        warmup_results=warmup_results,
        results=results,
        output_dir=output_dir,
    )

    print()
    print("=" * 120)
    print("Summary")
    print("=" * 120)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"summary_file: {summary_path}")

    if silent_count > 0 or warmup_silent_count > 0:
        print()
        print("WARNING:")
        print(f"warmup_silent_items={warmup_silent_count}")
        print(f"benchmark_silent_items={silent_count}")
        print("Check generated WAV files.")

    print()
    print("Generated files saved under:")
    print(output_dir)


if __name__ == "__main__":
    main()
