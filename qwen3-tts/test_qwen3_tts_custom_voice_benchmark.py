import argparse
import gc
import time
from pathlib import Path
from statistics import mean

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


SILENCE_PEAK_THRESHOLD = 1e-4
SILENCE_RMS_THRESHOLD = 1e-5

WARMUP_TEXTS = [
    "This is a short warmup test.",
    "The text to speech model is preparing for the benchmark.",
    "This final warmup sentence helps stabilize the first few GPU calls.",
]

TEXTS = [
    ("en_very_short_01", "2-4s", "I am speaking now."),
    ("en_very_short_02", "2-4s", "This is a short test."),
    ("en_short_01", "4-8s", "This is a simple test of Qwen three text to speech synthesis."),
    ("en_short_02", "4-8s", "The model converts written English text into natural sounding speech."),
    (
        "en_medium_01",
        "8-15s",
        "Text to speech systems are useful for voice assistants, accessibility tools, automated announcements, and spoken language applications.",
    ),
    (
        "en_medium_02",
        "8-15s",
        "A good speech synthesis model should produce clear pronunciation, stable rhythm, natural pauses, and consistent audio quality.",
    ),
    (
        "en_long_01",
        "15-30s",
        "During this test, we evaluate how the Qwen three text to speech model handles short, medium, and longer English inputs on a server equipped with a graphics processing unit.",
    ),
    (
        "en_long_02",
        "15-30s",
        "When deploying a text to speech model on a server, it is important to measure loading time, generation latency, output duration, real time factor, and whether any silent audio files were produced.",
    ),
]


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def normalize_audio(audio):
    audio = np.asarray(audio).squeeze().astype(np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio


def get_audio_stats(audio):
    if audio.size == 0:
        return 0.0, 0.0, True

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    silent = peak < SILENCE_PEAK_THRESHOLD or rms < SILENCE_RMS_THRESHOLD
    return peak, rms, silent


def generate_one(model, text, language, speaker, instruct, out_path):
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    infer_start = time.monotonic()

    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct=instruct,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    infer_elapsed = time.monotonic() - infer_start

    post_start = time.monotonic()

    audio = normalize_audio(wavs[0])
    peak, rms, silent = get_audio_stats(audio)
    duration = len(audio) / sr if len(audio) else 0.0

    sf.write(str(out_path), audio, sr)

    post_elapsed = time.monotonic() - post_start
    total_elapsed = infer_elapsed + post_elapsed
    rtf = total_elapsed / duration if duration > 0 else 0.0

    return {
        "sr": sr,
        "elapsed": total_elapsed,
        "infer_elapsed": infer_elapsed,
        "post_elapsed": post_elapsed,
        "duration": duration,
        "rtf": rtf,
        "peak": peak,
        "rms": rms,
        "silent": silent,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--language", default="English")
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--instruct", default="Speak clearly with a neutral tone.")
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument("--attn", default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Qwen3-TTS CustomVoice benchmark")
    print("=" * 120)
    print("model:", args.model)
    print("language:", args.language)
    print("speaker:", args.speaker)
    print("attn:", args.attn)
    print("dtype:", args.dtype)
    print("output_dir:", output_dir)
    print("cuda:", torch.cuda.is_available())
    print("gpu:", torch.cuda.get_device_name(0))

    clear_gpu_memory()

    load_start = time.monotonic()
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=dtype,
        attn_implementation=args.attn,
    )
    torch.cuda.synchronize()
    load_elapsed = time.monotonic() - load_start

    print(f"model_load_time_sec: {load_elapsed:.2f}")

    warmup_results = []
    print("\nWarmup...")
    for idx, text in enumerate(WARMUP_TEXTS, start=1):
        out_path = output_dir / f"warmup_{idx}.wav"
        r = generate_one(model, text, args.language, args.speaker, args.instruct, out_path)
        r["path"] = str(out_path)
        warmup_results.append(r)
        print(
            f"warmup {idx}: elapsed={r['elapsed']:.2f}s, audio={r['duration']:.2f}s, "
            f"RTF={r['rtf']:.3f}, peak={r['peak']:.8f}, rms={r['rms']:.8f}, silent={r['silent']}, path={out_path}"
        )

    results = []
    print("\nBenchmark...")
    benchmark_start = time.monotonic()

    for idx, (label, target_range, text) in enumerate(TEXTS, start=1):
        out_path = output_dir / f"bench_{idx:02d}_{label}.wav"
        r = generate_one(model, text, args.language, args.speaker, args.instruct, out_path)
        row = {
            "idx": idx,
            "label": label,
            "target_range": target_range,
            "chars": len(text),
            "text": text,
            "path": str(out_path),
            **r,
        }
        results.append(row)

        print(
            f"item {idx:02d}: label={label:<18}, range={target_range:<6}, chars={len(text):4d}, "
            f"elapsed={r['elapsed']:.2f}s, infer={r['infer_elapsed']:.2f}s, post={r['post_elapsed']:.2f}s, "
            f"audio={r['duration']:.2f}s, RTF={r['rtf']:.3f}, peak={r['peak']:.8f}, rms={r['rms']:.8f}, "
            f"silent={r['silent']}, path={out_path}"
        )

    torch.cuda.synchronize()
    wall_elapsed = time.monotonic() - benchmark_start

    total_elapsed = sum(r["elapsed"] for r in results)
    total_infer = sum(r["infer_elapsed"] for r in results)
    total_post = sum(r["post_elapsed"] for r in results)
    total_audio = sum(r["duration"] for r in results)

    silent_count = sum(1 for r in results if r["silent"])
    warmup_silent_count = sum(1 for r in warmup_results if r["silent"])

    summary = {
        "status": "SUCCESS" if silent_count == 0 else "WARNING_SILENT_OUTPUT",
        "model_family": "Qwen3-TTS",
        "model": args.model,
        "language": args.language,
        "speaker": args.speaker,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "attn": args.attn,
        "model_load_time_sec": f"{load_elapsed:.2f}",
        "warmup_runs": str(len(warmup_results)),
        "warmup_silent_items": str(warmup_silent_count),
        "benchmark_items": str(len(results)),
        "silent_items": str(silent_count),
        "benchmark_wall_time_sec": f"{wall_elapsed:.2f}",
        "total_generation_time_sec": f"{total_elapsed:.2f}",
        "total_inference_time_sec": f"{total_infer:.2f}",
        "total_postprocess_time_sec": f"{total_post:.2f}",
        "total_audio_duration_sec": f"{total_audio:.2f}",
        "effective_rtf": f"{total_elapsed / total_audio if total_audio > 0 else 0.0:.3f}",
        "inference_only_rtf": f"{total_infer / total_audio if total_audio > 0 else 0.0:.3f}",
        "wall_clock_rtf": f"{wall_elapsed / total_audio if total_audio > 0 else 0.0:.3f}",
        "average_per_item_rtf": f"{mean(r['rtf'] for r in results):.3f}",
        "average_audio_duration_sec": f"{mean(r['duration'] for r in results):.2f}",
        "average_peak": f"{mean(r['peak'] for r in results):.8f}",
        "average_rms": f"{mean(r['rms'] for r in results):.8f}",
        "min_peak": f"{min(r['peak'] for r in results):.8f}",
        "min_rms": f"{min(r['rms'] for r in results):.8f}",
    }

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Qwen3-TTS CustomVoice Benchmark Summary\n")
        f.write("=" * 120 + "\n\n")

        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

        f.write("\nBenchmark files:\n")
        for r in results:
            f.write(
                f"item {r['idx']:02d} | label={r['label']} | range={r['target_range']} | "
                f"chars={r['chars']} | elapsed={r['elapsed']:.2f}s | infer={r['infer_elapsed']:.2f}s | "
                f"post={r['post_elapsed']:.2f}s | audio={r['duration']:.2f}s | RTF={r['rtf']:.3f} | "
                f"peak={r['peak']:.8f} | rms={r['rms']:.8f} | silent={r['silent']} | {r['path']}\n"
            )

    print("\nSummary")
    print("=" * 120)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("summary_file:", summary_path)


if __name__ == "__main__":
    main()
