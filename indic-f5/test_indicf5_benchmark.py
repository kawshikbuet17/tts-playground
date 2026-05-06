import gc
import time
from pathlib import Path
from statistics import mean

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel


# -----------------------------
# Config
# -----------------------------
MODEL_ID = "ai4bharat/IndicF5"
SAMPLE_RATE = 24000

# Keep same benchmark output folder.
OUTPUT_DIR = Path("outputs/benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REF_AUDIO_PATH = "prompts/PAN_F_HAPPY_00001.wav"

REF_TEXT = (
    "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ "
    "ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ ਹਨ।"
)

SET_MATMUL_PRECISION = True
MATMUL_PRECISION = "high"

SILENCE_PEAK_THRESHOLD = 1e-4
SILENCE_RMS_THRESHOLD = 1e-5


# -----------------------------
# Warmup texts
# -----------------------------
WARMUP_TEXTS = [
    "এটি একটি ছোট প্রস্তুতিমূলক পরীক্ষা।",
    "বাংলা বাক্য থেকে স্বাভাবিক কণ্ঠ তৈরি করার পরীক্ষা চলছে।",
    "এখন মূল পরীক্ষা শুরু করার আগে শেষ প্রস্তুতিমূলক ধাপ চালানো হচ্ছে।",
]


# -----------------------------
# Bengali benchmark texts
# -----------------------------
# Approx target generated audio range:
#   very short: ~2-4 sec
#   short:      ~4-8 sec
#   medium:     ~8-15 sec
#   long:       ~15-30 sec
TEXTS = [
    {
        "label": "bn_very_short_01",
        "target_range": "2-4s",
        "text": "আমি বাংলা বলছি।",
    },
    {
        "label": "bn_very_short_02",
        "target_range": "2-4s",
        "text": "এটি একটি ছোট পরীক্ষা।",
    },
    {
        "label": "bn_very_short_03",
        "target_range": "2-4s",
        "text": "বাংলা অডিও তৈরি হচ্ছে।",
    },

    {
        "label": "bn_short_01",
        "target_range": "4-8s",
        "text": "এটি বাংলা বাক্য থেকে কণ্ঠ তৈরির একটি পরীক্ষা।",
    },
    {
        "label": "bn_short_02",
        "target_range": "4-8s",
        "text": "বাংলাদেশ একটি সুন্দর দেশ। এখানে মানুষ বাংলা ভাষায় কথা বলে।",
    },
    {
        "label": "bn_short_03",
        "target_range": "4-8s",
        "text": "আজ আমরা বাংলা ভাষার কণ্ঠ তৈরির গতি এবং মান পরীক্ষা করছি।",
    },

    {
        "label": "bn_medium_01",
        "target_range": "8-15s",
        "text": (
            "কৃত্রিম বুদ্ধিমত্তা এখন ভাষা প্রযুক্তির উন্নয়নে গুরুত্বপূর্ণ ভূমিকা রাখছে। "
            "বাংলা ভাষার জন্য উন্নত বক্তৃতা প্রযুক্তি তৈরি করা হচ্ছে।"
        ),
    },
    {
        "label": "bn_medium_02",
        "target_range": "8-15s",
        "text": (
            "লিখিত ভাষাকে স্বাভাবিক মানুষের কণ্ঠে রূপান্তর করা একটি গুরুত্বপূর্ণ কাজ। "
            "একটি ভালো পদ্ধতি বাক্যের গতি, স্বর এবং বিরতি ঠিকভাবে প্রকাশ করার চেষ্টা করে।"
        ),
    },
    {
        "label": "bn_medium_03",
        "target_range": "8-15s",
        "text": (
            "বাংলা ভাষার জন্য ভালো মানের কণ্ঠ সংশ্লেষণ তৈরি করা একটি গুরুত্বপূর্ণ গবেষণা সমস্যা। "
            "কারণ বাংলা ভাষায় উচ্চারণ, যুক্তবর্ণ এবং বাক্যের ছন্দ অনেক সময় জটিল হতে পারে।"
        ),
    },

    {
        "label": "bn_long_01",
        "target_range": "15-30s",
        "text": (
            "আজকের পরীক্ষায় আমরা বাংলা কণ্ঠ তৈরির গতি, উচ্চারণের স্বাভাবিকতা এবং দীর্ঘ বাক্য সামলানোর ক্ষমতা যাচাই করছি। "
            "প্রথমে ছোট বাক্য দিয়ে পরীক্ষা করা হয়েছে, তারপর মাঝারি দৈর্ঘ্যের বাক্য ব্যবহার করা হয়েছে, "
            "এবং এখন তুলনামূলকভাবে বড় অনুচ্ছেদ ব্যবহার করা হচ্ছে।"
        ),
    },
    {
        "label": "bn_long_02",
        "target_range": "15-30s",
        "text": (
            "সার্ভার ভিত্তিক কণ্ঠ তৈরির ব্যবস্থা তৈরি করার সময় শুধু অডিওর মান দেখলেই হয় না। "
            "এর পাশাপাশি মডেল লোডিং সময়, তৈরি করার সময়, অডিওর দৈর্ঘ্য, বাস্তব সময়ের তুলনা, "
            "গ্রাফিক্স প্রসেসরের মেমরি ব্যবহার এবং বারবার চালালে কার্যক্ষমতা কতটা স্থিতিশীল থাকে, সেটিও পরীক্ষা করা দরকার।"
        ),
    },
    {
        "label": "bn_long_03",
        "target_range": "15-30s",
        "text": (
            "বাস্তব ব্যবহারের ক্ষেত্রে একটি কণ্ঠ তৈরির ব্যবস্থাকে বিভিন্ন দৈর্ঘ্যের ইনপুট সামলাতে হয়। "
            "কখনো ব্যবহারকারী একটি ছোট বাক্য দেয়, কখনো একটি মাঝারি অনুচ্ছেদ দেয়, আবার কখনো বড় ব্যাখ্যামূলক লেখা থেকে অডিও তৈরি করতে চায়। "
            "এই কারণে পরীক্ষায় ছোট, মাঝারি এবং বড় সব ধরনের বাংলা লেখা রাখা হয়েছে।"
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


def generate_one(model, text, out_path):
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    infer_start = time.monotonic()

    with torch.inference_mode():
        audio = model(
            text,
            ref_audio_path=REF_AUDIO_PATH,
            ref_text=REF_TEXT,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    infer_elapsed = time.monotonic() - infer_start

    post_start = time.monotonic()

    audio = normalize_audio(audio)
    audio_stats = get_audio_stats(audio)

    duration = len(audio) / SAMPLE_RATE if audio.size else 0.0

    sf.write(
        str(out_path),
        audio,
        samplerate=SAMPLE_RATE,
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


def write_summary_file(summary, warmup_results, results):
    summary_path = OUTPUT_DIR / "summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("IndicF5 Bengali FP32 Benchmark Summary\n")
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
    if SET_MATMUL_PRECISION:
        torch.set_float32_matmul_precision(MATMUL_PRECISION)

    print("IndicF5 Bengali FP32 benchmark")
    print("=" * 120)
    print(f"Model: {MODEL_ID}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MATMUL_PRECISION: {MATMUL_PRECISION if SET_MATMUL_PRECISION else 'disabled'}")
    print(f"Warmup runs: {len(WARMUP_TEXTS)}")
    print(f"Benchmark texts: {len(TEXTS)}")
    print("Language: Bengali only")
    print("Target audio ranges: 2s to 30s")
    print(f"Output dir: {OUTPUT_DIR}")

    if torch.cuda.is_available():
        print(f"Visible GPU: {torch.cuda.get_device_name(0)}")

    clear_gpu_memory()

    print("\nLoading IndicF5 model once...")
    load_start = time.monotonic()

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    load_elapsed = time.monotonic() - load_start

    print(f"Model loaded in {load_elapsed:.2f}s")

    # -----------------------------
    # Warmup
    # -----------------------------
    print("\nWarmup...")
    warmup_results = []

    for idx, text in enumerate(WARMUP_TEXTS, start=1):
        out_path = OUTPUT_DIR / f"warmup_{idx}.wav"

        result = generate_one(
            model=model,
            text=text,
            out_path=out_path,
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

        out_path = OUTPUT_DIR / f"bench_{idx:02d}_{label}.wav"

        result = generate_one(
            model=model,
            text=text,
            out_path=out_path,
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
        "precision": "FP32",
        "language": "Bengali",
        "target_audio_range": "2s_to_30s",
        "model": MODEL_ID,
        "sample_rate": str(SAMPLE_RATE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
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
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
