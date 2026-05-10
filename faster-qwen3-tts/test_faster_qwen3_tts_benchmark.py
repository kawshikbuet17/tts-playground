#!/usr/bin/env python3
"""
Faster Qwen3-TTS voice-clone benchmark.

This is intentionally shaped like a VITS benchmark script:
- load model once
- run warmups
- run benchmark texts across short/medium/long inputs
- write WAVs
- compute audio duration, elapsed time, VITS-style RTF, throughput, peak/RMS/silence
- write summary.txt and results.json

Run example:
  CUDA_VISIBLE_DEVICES=1 python test_faster_qwen3_tts_benchmark.py \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --ref-audio ref_audio.wav \
    --ref-text "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed." \
    --dtype bf16 \
    --streaming

Notes:
- rtf here is elapsed / audio_duration, matching many VITS-style benchmarks.
  Lower is better. Example: 0.10 means generation took 10% of real-time.
- throughput_x is audio_duration / elapsed, matching faster-qwen3 README's RTF style.
  Higher is better. Example: 10.0x means 10x faster than real-time.
"""

import argparse
import gc
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from faster_qwen3_tts import FasterQwen3TTS


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_REF_AUDIO = "ref_audio.wav"
DEFAULT_REF_TEXT = (
    "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up "
    "reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach "
    "of training on verifiable outcomes is doomed."
)
DEFAULT_OUTPUT_DIR = "outputs/faster_qwen3_tts_benchmark"

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
# English benchmark texts
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
        "text": "Today we are testing the speed and quality of a Qwen three speech model.",
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
            "time to first audio, and basic signal statistics such as peak amplitude and root mean square energy."
        ),
    },
    {
        "label": "en_long_01",
        "target_range": "15-30s",
        "text": (
            "During this test, we evaluate how the Qwen three text to speech model handles short, medium, "
            "and longer English inputs on a server equipped with a graphics processing unit. The goal is to "
            "measure both runtime behavior and generated audio validity."
        ),
    },
    {
        "label": "en_long_02",
        "target_range": "15-30s",
        "text": (
            "When deploying a text to speech model on a server, it is important to measure more than only "
            "subjective audio quality. We also need to record model loading time, generation latency, output "
            "duration, real time factor, time to first audio, and whether any silent or invalid audio files were produced."
        ),
    },
    {
        "label": "en_long_03",
        "target_range": "15-30s",
        "text": (
            "Real world text to speech systems must handle many kinds of input. Sometimes the user provides a very "
            "short command, sometimes a complete sentence, and sometimes a longer paragraph. A useful benchmark should "
            "therefore include examples across several text lengths."
        ),
    },
]


# -----------------------------
# Helpers
# -----------------------------
def clear_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def parse_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower().strip()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}. Use bf16, fp16, or fp32.")
    return mapping[normalized]


def normalize_audio(audio: Any) -> np.ndarray:
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


def get_audio_stats(audio: np.ndarray) -> Dict[str, Any]:
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

    is_silent = peak < SILENCE_PEAK_THRESHOLD or rms < SILENCE_RMS_THRESHOLD

    return {
        "peak": peak,
        "rms": rms,
        "mean_abs": mean_abs,
        "is_silent": is_silent,
    }


def concat_audio_chunks(chunks: Iterable[Any]) -> np.ndarray:
    arrays = []
    for chunk in chunks:
        arr = normalize_audio(chunk)
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return np.zeros(1, dtype=np.float32)
    return normalize_audio(np.concatenate(arrays, axis=0))


def generate_one(
    model: FasterQwen3TTS,
    text: str,
    out_path: Path,
    language: str,
    ref_audio: str,
    ref_text: str,
    streaming: bool,
    chunk_size: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    xvec_only: bool,
    non_streaming_mode: Optional[bool],
    append_silence: bool,
    instruct: Optional[str],
) -> Dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.monotonic()
    ttfa: Optional[float] = None
    timing_last: Dict[str, Any] = {}

    with torch.inference_mode():
        if streaming:
            audio_chunks: List[np.ndarray] = []
            sr: Optional[int] = None

            for audio_chunk, chunk_sr, timing in model.generate_voice_clone_streaming(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                chunk_size=chunk_size,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                xvec_only=xvec_only,
                non_streaming_mode=non_streaming_mode,
                append_silence=append_silence,
                instruct=instruct,
            ):
                if ttfa is None:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    ttfa = time.monotonic() - start

                audio_chunks.append(audio_chunk)
                sr = int(chunk_sr)
                timing_last = dict(timing) if isinstance(timing, dict) else {}

            audio = concat_audio_chunks(audio_chunks)
            sample_rate = int(sr or model.sample_rate)
        else:
            audio_list, sample_rate = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                xvec_only=xvec_only,
                non_streaming_mode=non_streaming_mode,
                append_silence=append_silence,
                instruct=instruct,
            )
            audio = concat_audio_chunks(audio_list)
            sample_rate = int(sample_rate)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    infer_elapsed = time.monotonic() - start

    post_start = time.monotonic()
    audio = normalize_audio(audio)
    audio_stats = get_audio_stats(audio)
    duration = len(audio) / sample_rate if audio.size else 0.0
    sf.write(str(out_path), audio, samplerate=sample_rate)
    post_elapsed = time.monotonic() - post_start

    total_elapsed = infer_elapsed + post_elapsed

    # VITS-style RTF: lower is better.
    rtf = total_elapsed / duration if duration > 0 else 0.0

    # Faster-Qwen README style throughput: higher is better.
    throughput_x = duration / total_elapsed if total_elapsed > 0 else 0.0

    return {
        "elapsed": total_elapsed,
        "infer_elapsed": infer_elapsed,
        "post_elapsed": post_elapsed,
        "duration": duration,
        "rtf": rtf,
        "throughput_x": throughput_x,
        "ttfa": ttfa,
        "sample_rate": sample_rate,
        "timing_last": timing_last,
        **audio_stats,
    }


def summarize_by_range(results: List[Dict[str, Any]]) -> List[str]:
    by_range: Dict[str, List[Dict[str, Any]]] = {}

    for r in results:
        by_range.setdefault(r["target_range"], []).append(r)

    lines = []

    for target_range, rows in sorted(by_range.items()):
        total_elapsed = sum(r["elapsed"] for r in rows)
        total_audio = sum(r["duration"] for r in rows)
        avg_rtf = mean(r["rtf"] for r in rows)
        effective_rtf = total_elapsed / total_audio if total_audio > 0 else 0.0
        throughput_x = total_audio / total_elapsed if total_elapsed > 0 else 0.0
        silent_count = sum(1 for r in rows if r["is_silent"])
        ttfas = [r["ttfa"] for r in rows if r.get("ttfa") is not None]
        avg_ttfa = mean(ttfas) if ttfas else None

        line = (
            f"{target_range}: items={len(rows)}, "
            f"silent={silent_count}, "
            f"audio={total_audio:.2f}s, "
            f"generation={total_elapsed:.2f}s, "
            f"effective_rtf={effective_rtf:.3f}, "
            f"throughput_x={throughput_x:.2f}, "
            f"avg_rtf={avg_rtf:.3f}"
        )
        if avg_ttfa is not None:
            line += f", avg_ttfa={avg_ttfa:.3f}s"

        lines.append(line)

    return lines


def write_summary_file(
    summary: Dict[str, str],
    warmup_results: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    summary_path = output_dir / "summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Faster Qwen3-TTS Voice Clone Benchmark Summary\n")
        f.write("=" * 120 + "\n\n")

        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

        f.write("\nWarmup files:\n")
        for r in warmup_results:
            ttfa_text = "n/a" if r.get("ttfa") is None else f"{r['ttfa']:.3f}s"
            f.write(
                f"warmup {r['idx']:02d} | "
                f"elapsed={r['elapsed']:6.2f}s | "
                f"infer={r['infer_elapsed']:6.2f}s | "
                f"post={r['post_elapsed']:5.2f}s | "
                f"ttfa={ttfa_text:>8} | "
                f"audio={r['duration']:6.2f}s | "
                f"RTF_elapsed_over_audio={r['rtf']:6.3f} | "
                f"throughput_x={r['throughput_x']:6.2f} | "
                f"peak={r['peak']:.8f} | "
                f"rms={r['rms']:.8f} | "
                f"silent={r['is_silent']} | "
                f"{r['path']}\n"
            )

        f.write("\nBenchmark files:\n")
        for r in results:
            ttfa_text = "n/a" if r.get("ttfa") is None else f"{r['ttfa']:.3f}s"
            f.write(
                f"item {r['idx']:02d} | "
                f"label={r['label']:<18} | "
                f"range={r['target_range']:<6} | "
                f"chars={r['chars']:4d} | "
                f"elapsed={r['elapsed']:6.2f}s | "
                f"infer={r['infer_elapsed']:6.2f}s | "
                f"post={r['post_elapsed']:5.2f}s | "
                f"ttfa={ttfa_text:>8} | "
                f"audio={r['duration']:6.2f}s | "
                f"RTF_elapsed_over_audio={r['rtf']:6.3f} | "
                f"throughput_x={r['throughput_x']:6.2f} | "
                f"peak={r['peak']:.8f} | "
                f"rms={r['rms']:.8f} | "
                f"silent={r['is_silent']} | "
                f"{r['path']}\n"
            )

    return summary_path


def write_json_file(
    summary: Dict[str, str],
    warmup_results: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    json_path = output_dir / "results.json"

    def make_json_safe(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return obj

    payload = {
        "summary": summary,
        "warmup_results": warmup_results,
        "benchmark_results": results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=make_json_safe)

    return json_path


def optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.lower().strip()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    if v in {"none", "null", "default"}:
        return None
    raise argparse.ArgumentTypeError("Use true, false, or none/default.")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--language", default="English")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--xvec-only", action="store_true")
    parser.add_argument("--append-silence", action="store_true", default=True)
    parser.add_argument("--no-append-silence", dest="append_silence", action="store_false")
    parser.add_argument("--non-streaming-mode", type=optional_bool, default=None)
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--warmup-runs", type=int, default=len(WARMUP_TEXTS))
    parser.add_argument("--benchmark-runs", type=int, default=len(TEXTS))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False.")

    device = args.device
    dtype = parse_dtype(args.dtype)

    ref_audio_path = Path(args.ref_audio)
    if not ref_audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    print("Faster Qwen3-TTS voice clone benchmark")
    print("=" * 120)
    print(f"Model: {args.model}")
    print(f"Reference audio: {args.ref_audio}")
    print(f"Reference text chars: {len(args.ref_text)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    print(f"dtype: {args.dtype}")
    print(f"Streaming: {args.streaming}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Warmup runs: {min(args.warmup_runs, len(WARMUP_TEXTS))}")
    print(f"Benchmark texts: {min(args.benchmark_runs, len(TEXTS))}")
    print(f"Language: {args.language}")
    print("Target audio ranges: 2s to 30s")
    print(f"Output dir: {output_dir}")

    if torch.cuda.is_available():
        print(f"Visible GPU: {torch.cuda.get_device_name(0)}")

    clear_gpu_memory()

    print("\nLoading Faster Qwen3-TTS model once...")
    load_start = time.monotonic()

    model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=device,
        dtype=dtype,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    load_elapsed = time.monotonic() - load_start
    sample_rate = int(getattr(model, "sample_rate", 24000))

    print(f"Model loaded in {load_elapsed:.2f}s")
    print(f"Sample rate: {sample_rate}")
    print("Note: first generation may include CUDA graph capture cost.")

    common_kwargs = dict(
        model=model,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        streaming=args.streaming,
        chunk_size=args.chunk_size,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        xvec_only=args.xvec_only,
        non_streaming_mode=args.non_streaming_mode,
        append_silence=args.append_silence,
        instruct=args.instruct,
    )

    # -----------------------------
    # Warmup
    # -----------------------------
    print("\nWarmup...")
    warmup_results: List[Dict[str, Any]] = []
    selected_warmups = WARMUP_TEXTS[: max(0, min(args.warmup_runs, len(WARMUP_TEXTS)))]

    for idx, text in enumerate(selected_warmups, start=1):
        out_path = output_dir / f"warmup_{idx:02d}.wav"

        result = generate_one(
            text=text,
            out_path=out_path,
            **common_kwargs,
        )

        row = {
            "idx": idx,
            "text": text,
            "path": str(out_path),
            **result,
        }
        warmup_results.append(row)

        ttfa_text = "n/a" if result.get("ttfa") is None else f"{result['ttfa']:.3f}s"
        print(
            f"warmup {idx}: "
            f"elapsed={result['elapsed']:.2f}s, "
            f"infer={result['infer_elapsed']:.2f}s, "
            f"post={result['post_elapsed']:.2f}s, "
            f"ttfa={ttfa_text}, "
            f"audio={result['duration']:.2f}s, "
            f"RTF={result['rtf']:.3f}, "
            f"throughput_x={result['throughput_x']:.2f}, "
            f"peak={result['peak']:.8f}, "
            f"rms={result['rms']:.8f}, "
            f"silent={result['is_silent']}, "
            f"path={out_path}"
        )

    # -----------------------------
    # Benchmark
    # -----------------------------
    print("\nBenchmark...")
    print("-" * 120)

    results: List[Dict[str, Any]] = []
    selected_texts = TEXTS[: max(0, min(args.benchmark_runs, len(TEXTS)))]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    benchmark_start = time.monotonic()

    for idx, item in enumerate(selected_texts, start=1):
        label = item["label"]
        target_range = item["target_range"]
        text = item["text"]
        out_path = output_dir / f"bench_{idx:02d}_{label}.wav"

        result = generate_one(
            text=text,
            out_path=out_path,
            **common_kwargs,
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

        ttfa_text = "n/a" if result.get("ttfa") is None else f"{result['ttfa']:.3f}s"
        print(
            f"item {idx:02d}: "
            f"label={label:<18}, "
            f"range={target_range:<6}, "
            f"chars={len(text):4d}, "
            f"elapsed={result['elapsed']:6.2f}s, "
            f"infer={result['infer_elapsed']:6.2f}s, "
            f"post={result['post_elapsed']:5.2f}s, "
            f"ttfa={ttfa_text:>8}, "
            f"audio={result['duration']:6.2f}s, "
            f"RTF={result['rtf']:6.3f}, "
            f"throughput_x={result['throughput_x']:6.2f}, "
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
    throughput_x = total_audio / total_elapsed if total_elapsed > 0 else 0.0
    wall_throughput_x = total_audio / benchmark_wall if benchmark_wall > 0 else 0.0

    avg_rtf = mean(r["rtf"] for r in results) if results else 0.0
    avg_throughput_x = mean(r["throughput_x"] for r in results) if results else 0.0
    avg_audio = mean(r["duration"] for r in results) if results else 0.0
    avg_elapsed = mean(r["elapsed"] for r in results) if results else 0.0
    avg_infer = mean(r["infer_elapsed"] for r in results) if results else 0.0
    avg_post = mean(r["post_elapsed"] for r in results) if results else 0.0
    avg_chars = mean(r["chars"] for r in results) if results else 0.0
    avg_peak = mean(r["peak"] for r in results) if results else 0.0
    avg_rms = mean(r["rms"] for r in results) if results else 0.0
    min_peak = min((r["peak"] for r in results), default=0.0)
    min_rms = min((r["rms"] for r in results), default=0.0)

    ttfas = [r["ttfa"] for r in results if r.get("ttfa") is not None]
    avg_ttfa = mean(ttfas) if ttfas else None
    min_ttfa = min(ttfas) if ttfas else None
    max_ttfa = max(ttfas) if ttfas else None

    silent_count = sum(1 for r in results if r["is_silent"])
    warmup_silent_count = sum(1 for r in warmup_results if r["is_silent"])

    fastest = min(results, key=lambda r: r["rtf"]) if results else None
    slowest = max(results, key=lambda r: r["rtf"]) if results else None

    range_summary_lines = summarize_by_range(results)

    summary: Dict[str, str] = {
        "status": "SUCCESS",
        "model_family": "Faster Qwen3-TTS",
        "model": args.model,
        "language": args.language,
        "mode": "voice_clone_streaming" if args.streaming else "voice_clone_non_streaming",
        "ref_audio": args.ref_audio,
        "ref_text_chars": str(len(args.ref_text)),
        "sample_rate": str(sample_rate),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch": torch.__version__,
        "dtype": args.dtype,
        "chunk_size": str(args.chunk_size),
        "xvec_only": str(args.xvec_only),
        "append_silence": str(args.append_silence),
        "non_streaming_mode": str(args.non_streaming_mode),
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
        "effective_rtf_elapsed_over_audio": f"{effective_rtf:.3f}",
        "inference_only_rtf_elapsed_over_audio": f"{infer_rtf:.3f}",
        "wall_clock_rtf_elapsed_over_audio": f"{wall_rtf:.3f}",
        "average_per_item_rtf_elapsed_over_audio": f"{avg_rtf:.3f}",
        "throughput_x_audio_over_elapsed": f"{throughput_x:.2f}",
        "wall_clock_throughput_x_audio_over_wall": f"{wall_throughput_x:.2f}",
        "average_per_item_throughput_x": f"{avg_throughput_x:.2f}",
    }

    if avg_ttfa is not None:
        summary["average_ttfa_sec"] = f"{avg_ttfa:.3f}"
        summary["min_ttfa_sec"] = f"{min_ttfa:.3f}"
        summary["max_ttfa_sec"] = f"{max_ttfa:.3f}"

    if fastest:
        summary["fastest_item_by_elapsed_over_audio_rtf"] = (
            f"item {fastest['idx']} | "
            f"label={fastest['label']} | "
            f"range={fastest['target_range']} | "
            f"RTF={fastest['rtf']:.3f} | "
            f"throughput_x={fastest['throughput_x']:.2f} | "
            f"audio={fastest['duration']:.2f}s | "
            f"peak={fastest['peak']:.8f} | "
            f"rms={fastest['rms']:.8f} | "
            f"silent={fastest['is_silent']}"
        )

    if slowest:
        summary["slowest_item_by_elapsed_over_audio_rtf"] = (
            f"item {slowest['idx']} | "
            f"label={slowest['label']} | "
            f"range={slowest['target_range']} | "
            f"RTF={slowest['rtf']:.3f} | "
            f"throughput_x={slowest['throughput_x']:.2f} | "
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
    json_path = write_json_file(
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
    print(f"json_file: {json_path}")

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
