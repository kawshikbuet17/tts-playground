#!/usr/bin/env python3
"""
Full Faster Qwen3-TTS benchmark suite.

Covers these repo modes:
  1. clone_streaming
  2. clone_non_streaming
  3. clone_xvec_streaming
  4. clone_xvec_non_streaming
  5. custom_streaming
  6. custom_non_streaming
  7. design_streaming
  8. design_non_streaming

It follows the same style as the VITS benchmark:
  - load each required model once
  - run warmups per mode
  - run benchmark texts
  - write WAV files
  - compute elapsed, infer/post time, audio duration, RTF, throughput, TTFA, peak/RMS/silence
  - write summary.txt + results.json

Recommended first run on your RTX 3090:

  cd /home/fuji/projects/faster-qwen3-tts
  conda activate faster_qwen3
  export CUDA_VISIBLE_DEVICES=1

  python test_faster_qwen3_tts_full_benchmark.py \
    --suite clone \
    --base-model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --ref-audio ref_audio.wav \
    --ref-text "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed." \
    --dtype bf16

Full suite including 1.7B CustomVoice and VoiceDesign:

  python test_faster_qwen3_tts_full_benchmark.py \
    --suite all \
    --dtype bf16 \
    --ref-audio ref_audio.wav \
    --ref-text "..." \
    --speaker aiden \
    --output-dir outputs/faster_qwen3_tts_full_benchmark

Notes:
  - RTF here = elapsed / generated_audio_duration. Lower is better.
  - throughput_x = generated_audio_duration / elapsed. Higher is better.
  - TTFA is only meaningful for streaming modes; non-streaming modes report n/a.
  - fp16 may be unstable on this repo/codepath. bf16 is recommended from your tests.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch

from faster_qwen3_tts import FasterQwen3TTS


DEFAULT_BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_CUSTOM_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_REF_AUDIO = "ref_audio.wav"
DEFAULT_REF_TEXT = (
    "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up "
    "reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach "
    "of training on verifiable outcomes is doomed."
)
DEFAULT_OUTPUT_DIR = "outputs/faster_qwen3_tts_full_benchmark"
DEFAULT_DESIGN_INSTRUCT = "Warm, confident narrator with clear pronunciation and a natural conversational rhythm."
DEFAULT_LANGUAGE = "English"
DEFAULT_SPEAKER = "aiden"

SILENCE_PEAK_THRESHOLD = 1e-4
SILENCE_RMS_THRESHOLD = 1e-5

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

TEXTS = [
    {"label": "en_very_short_01", "bucket": "very_short", "text": "I am speaking now."},
    {"label": "en_very_short_02", "bucket": "very_short", "text": "This is a short test."},
    {"label": "en_very_short_03", "bucket": "very_short", "text": "Audio is being generated."},
    {"label": "en_short_01", "bucket": "short", "text": "This is a simple test of text to speech synthesis."},
    {"label": "en_short_02", "bucket": "short", "text": "The model converts written English text into natural sounding speech."},
    {"label": "en_short_03", "bucket": "short", "text": "Today we are testing the speed and quality of a Qwen three speech model."},
    {
        "label": "en_medium_01",
        "bucket": "medium",
        "text": (
            "Text to speech systems are useful for voice assistants, accessibility tools, "
            "automated announcements, and many other spoken language applications."
        ),
    },
    {
        "label": "en_medium_02",
        "bucket": "medium",
        "text": (
            "A good speech synthesis model should produce clear pronunciation, stable rhythm, "
            "natural pauses, and consistent audio quality across different sentence lengths."
        ),
    },
    {
        "label": "en_medium_03",
        "bucket": "medium",
        "text": (
            "In this benchmark, we measure generation time, audio duration, real time factor, "
            "time to first audio, and basic signal statistics such as peak amplitude and root mean square energy."
        ),
    },
    {
        "label": "en_long_01",
        "bucket": "long",
        "text": (
            "During this test, we evaluate how the Qwen three text to speech model handles short, medium, "
            "and longer English inputs on a server equipped with a graphics processing unit. The goal is to "
            "measure both runtime behavior and generated audio validity."
        ),
    },
    {
        "label": "en_long_02",
        "bucket": "long",
        "text": (
            "When deploying a text to speech model on a server, it is important to measure more than only "
            "subjective audio quality. We also need to record model loading time, generation latency, output "
            "duration, real time factor, time to first audio, and whether any silent or invalid audio files were produced."
        ),
    },
    {
        "label": "en_long_03",
        "bucket": "long",
        "text": (
            "Real world text to speech systems must handle many kinds of input. Sometimes the user provides a very "
            "short command, sometimes a complete sentence, and sometimes a longer paragraph. A useful benchmark should "
            "therefore include examples across several text lengths."
        ),
    },
]


@dataclass
class ModeSpec:
    name: str
    family: str  # clone/custom/design
    streaming: bool
    model_key: str  # base/custom/design
    xvec_only: bool = False


ALL_MODES: List[ModeSpec] = [
    ModeSpec("clone_streaming", "clone", True, "base", xvec_only=False),
    ModeSpec("clone_non_streaming", "clone", False, "base", xvec_only=False),
    ModeSpec("clone_xvec_streaming", "clone", True, "base", xvec_only=True),
    ModeSpec("clone_xvec_non_streaming", "clone", False, "base", xvec_only=True),
    ModeSpec("custom_streaming", "custom", True, "custom"),
    ModeSpec("custom_non_streaming", "custom", False, "custom"),
    ModeSpec("design_streaming", "design", True, "design"),
    ModeSpec("design_non_streaming", "design", False, "design"),
]

SUITES: Dict[str, List[str]] = {
    "clone": ["clone_streaming", "clone_non_streaming", "clone_xvec_streaming", "clone_xvec_non_streaming"],
    "clone_icl": ["clone_streaming", "clone_non_streaming"],
    "clone_xvec": ["clone_xvec_streaming", "clone_xvec_non_streaming"],
    "custom": ["custom_streaming", "custom_non_streaming"],
    "design": ["design_streaming", "design_non_streaming"],
    "streaming": ["clone_streaming", "clone_xvec_streaming", "custom_streaming", "design_streaming"],
    "non_streaming": ["clone_non_streaming", "clone_xvec_non_streaming", "custom_non_streaming", "design_non_streaming"],
    "all": [m.name for m in ALL_MODES],
}

MODE_BY_NAME = {m.name: m for m in ALL_MODES}


def clear_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = dtype_name.lower().strip()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}; use bf16, fp16, or fp32.")
    return mapping[key]


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


def concat_audio(chunks: Sequence[Any]) -> np.ndarray:
    arrays = [normalize_audio(c) for c in chunks if c is not None]
    arrays = [a for a in arrays if a.size > 0]
    if not arrays:
        return np.zeros(0, dtype=np.float32)
    return normalize_audio(np.concatenate(arrays, axis=0))


def audio_stats(audio: np.ndarray) -> Dict[str, Any]:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return {"peak": 0.0, "rms": 0.0, "mean_abs": 0.0, "is_silent": True}
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    mean_abs = float(np.mean(np.abs(audio)))
    return {
        "peak": peak,
        "rms": rms,
        "mean_abs": mean_abs,
        "is_silent": bool(peak < SILENCE_PEAK_THRESHOLD or rms < SILENCE_RMS_THRESHOLD),
    }


def safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        if np.isnan(value) or np.isinf(value):
            return None
    except TypeError:
        return None
    return float(value)


def fmt(value: Optional[float], width: int = 6, decimals: int = 2) -> str:
    if value is None:
        return "n/a".rjust(width)
    return f"{value:{width}.{decimals}f}"


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, samplerate=int(sample_rate))


def generate_one(
    *,
    model: FasterQwen3TTS,
    spec: ModeSpec,
    text: str,
    out_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.monotonic()
    ttfa: Optional[float] = None
    sample_rate = int(getattr(model, "sample_rate", 24000) or 24000)
    chunks: List[np.ndarray] = []

    try:
        with torch.inference_mode():
            if spec.family == "clone" and spec.streaming:
                gen = model.generate_voice_clone_streaming(
                    text=text,
                    language=args.language,
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=not args.no_sample,
                    repetition_penalty=args.repetition_penalty,
                    chunk_size=args.chunk_size,
                    xvec_only=spec.xvec_only,
                    non_streaming_mode=args.non_streaming_mode,
                    append_silence=not args.no_append_silence,
                    instruct=args.clone_instruct or None,
                )
                for audio_chunk, sr, timing in gen:
                    if ttfa is None:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        ttfa = time.monotonic() - start
                    sample_rate = int(sr)
                    chunks.append(normalize_audio(audio_chunk))
                audio = concat_audio(chunks)

            elif spec.family == "clone":
                audio_list, sr = model.generate_voice_clone(
                    text=text,
                    language=args.language,
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=not args.no_sample,
                    repetition_penalty=args.repetition_penalty,
                    xvec_only=spec.xvec_only,
                    non_streaming_mode=args.non_streaming_mode,
                    append_silence=not args.no_append_silence,
                    instruct=args.clone_instruct or None,
                )
                sample_rate = int(sr)
                audio = concat_audio(audio_list)

            elif spec.family == "custom" and spec.streaming:
                gen = model.generate_custom_voice_streaming(
                    text=text,
                    speaker=args.speaker,
                    language=args.language,
                    instruct=args.custom_instruct or None,
                    non_streaming_mode=args.non_streaming_mode,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=not args.no_sample,
                    repetition_penalty=args.repetition_penalty,
                    chunk_size=args.chunk_size,
                )
                for audio_chunk, sr, timing in gen:
                    if ttfa is None:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        ttfa = time.monotonic() - start
                    sample_rate = int(sr)
                    chunks.append(normalize_audio(audio_chunk))
                audio = concat_audio(chunks)

            elif spec.family == "custom":
                audio_list, sr = model.generate_custom_voice(
                    text=text,
                    speaker=args.speaker,
                    language=args.language,
                    instruct=args.custom_instruct or None,
                    non_streaming_mode=args.non_streaming_mode,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=not args.no_sample,
                    repetition_penalty=args.repetition_penalty,
                )
                sample_rate = int(sr)
                audio = concat_audio(audio_list)

            elif spec.family == "design" and spec.streaming:
                gen = model.generate_voice_design_streaming(
                    text=text,
                    instruct=args.design_instruct,
                    language=args.language,
                    non_streaming_mode=args.non_streaming_mode,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=not args.no_sample,
                    repetition_penalty=args.repetition_penalty,
                    chunk_size=args.chunk_size,
                )
                for audio_chunk, sr, timing in gen:
                    if ttfa is None:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        ttfa = time.monotonic() - start
                    sample_rate = int(sr)
                    chunks.append(normalize_audio(audio_chunk))
                audio = concat_audio(chunks)

            elif spec.family == "design":
                audio_list, sr = model.generate_voice_design(
                    text=text,
                    instruct=args.design_instruct,
                    language=args.language,
                    non_streaming_mode=args.non_streaming_mode,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=not args.no_sample,
                    repetition_penalty=args.repetition_penalty,
                )
                sample_rate = int(sr)
                audio = concat_audio(audio_list)
            else:
                raise ValueError(f"Unsupported mode spec: {spec}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_elapsed = time.monotonic() - start

        post_start = time.monotonic()
        audio = normalize_audio(audio)
        stats = audio_stats(audio)
        duration = float(len(audio) / sample_rate) if audio.size else 0.0
        write_wav(out_path, audio, sample_rate)
        post_elapsed = time.monotonic() - post_start
        elapsed = infer_elapsed + post_elapsed

        return {
            "status": "ok",
            "error": None,
            "elapsed": elapsed,
            "infer_elapsed": infer_elapsed,
            "post_elapsed": post_elapsed,
            "ttfa": safe_float(ttfa),
            "duration": duration,
            "sample_rate": sample_rate,
            "rtf": elapsed / duration if duration > 0 else 0.0,
            "throughput_x": duration / elapsed if elapsed > 0 else 0.0,
            "path": str(out_path),
            **stats,
        }

    except Exception as exc:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        elapsed = time.monotonic() - start
        return {
            "status": "error",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed": elapsed,
            "infer_elapsed": elapsed,
            "post_elapsed": 0.0,
            "ttfa": safe_float(ttfa),
            "duration": 0.0,
            "sample_rate": sample_rate,
            "rtf": 0.0,
            "throughput_x": 0.0,
            "path": str(out_path),
            "peak": 0.0,
            "rms": 0.0,
            "mean_abs": 0.0,
            "is_silent": True,
        }


def resolve_modes(args: argparse.Namespace) -> List[ModeSpec]:
    selected: List[str] = []
    for suite in args.suite:
        if suite in SUITES:
            selected.extend(SUITES[suite])
        elif suite in MODE_BY_NAME:
            selected.append(suite)
        else:
            valid = sorted(set(SUITES) | set(MODE_BY_NAME))
            raise ValueError(f"Unknown suite/mode {suite!r}. Valid values: {', '.join(valid)}")

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for name in selected:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return [MODE_BY_NAME[name] for name in deduped]


def model_name_for_spec(spec: ModeSpec, args: argparse.Namespace) -> str:
    return {
        "base": args.base_model,
        "custom": args.custom_model,
        "design": args.design_model,
    }[spec.model_key]


def load_model_once(model_name: str, args: argparse.Namespace) -> Tuple[FasterQwen3TTS, float]:
    clear_gpu_memory()
    dtype = parse_dtype(args.dtype)
    start = time.monotonic()
    model = FasterQwen3TTS.from_pretrained(
        model_name,
        device=args.device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        max_seq_len=args.max_seq_len,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_elapsed = time.monotonic() - start
    return model, load_elapsed


def limited_texts(items: Sequence[Dict[str, str]], limit: Optional[int]) -> List[Dict[str, str]]:
    if limit is None or limit <= 0:
        return list(items)
    return list(items[:limit])


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    errors = [r for r in rows if r.get("status") != "ok"]
    audio = sum(float(r.get("duration") or 0.0) for r in ok)
    elapsed = sum(float(r.get("elapsed") or 0.0) for r in ok)
    infer = sum(float(r.get("infer_elapsed") or 0.0) for r in ok)
    post = sum(float(r.get("post_elapsed") or 0.0) for r in ok)
    ttfas = [float(r["ttfa"]) for r in ok if r.get("ttfa") is not None]
    return {
        "items": len(rows),
        "ok_items": len(ok),
        "error_items": len(errors),
        "silent_items": sum(1 for r in ok if r.get("is_silent")),
        "total_audio_duration_sec": audio,
        "total_generation_time_sec": elapsed,
        "total_inference_time_sec": infer,
        "total_postprocess_time_sec": post,
        "effective_rtf_elapsed_over_audio": elapsed / audio if audio > 0 else 0.0,
        "inference_only_rtf_elapsed_over_audio": infer / audio if audio > 0 else 0.0,
        "throughput_x_audio_over_elapsed": audio / elapsed if elapsed > 0 else 0.0,
        "average_per_item_rtf_elapsed_over_audio": mean([r["rtf"] for r in ok]) if ok else 0.0,
        "average_per_item_throughput_x": mean([r["throughput_x"] for r in ok]) if ok else 0.0,
        "average_audio_duration_sec": mean([r["duration"] for r in ok]) if ok else 0.0,
        "average_generation_time_sec": mean([r["elapsed"] for r in ok]) if ok else 0.0,
        "average_peak": mean([r["peak"] for r in ok]) if ok else 0.0,
        "average_rms": mean([r["rms"] for r in ok]) if ok else 0.0,
        "min_peak": min([r["peak"] for r in ok]) if ok else 0.0,
        "min_rms": min([r["rms"] for r in ok]) if ok else 0.0,
        "average_ttfa_sec": mean(ttfas) if ttfas else None,
        "min_ttfa_sec": min(ttfas) if ttfas else None,
        "max_ttfa_sec": max(ttfas) if ttfas else None,
    }


def summarize_by_bucket(rows: List[Dict[str, Any]]) -> List[str]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            buckets.setdefault(row.get("bucket", "unknown"), []).append(row)

    lines = []
    for bucket in sorted(buckets):
        s = summarize_rows(buckets[bucket])
        ttfa = s["average_ttfa_sec"]
        ttfa_part = f", avg_ttfa={ttfa:.3f}s" if ttfa is not None else ""
        lines.append(
            f"{bucket}: items={s['items']}, silent={s['silent_items']}, "
            f"audio={s['total_audio_duration_sec']:.2f}s, generation={s['total_generation_time_sec']:.2f}s, "
            f"effective_rtf={s['effective_rtf_elapsed_over_audio']:.3f}, "
            f"throughput_x={s['throughput_x_audio_over_elapsed']:.2f}{ttfa_part}"
        )
    return lines


def print_row(prefix: str, row: Dict[str, Any]) -> None:
    if row.get("status") != "ok":
        print(f"{prefix}: ERROR elapsed={row.get('elapsed', 0.0):.2f}s error={row.get('error')}")
        return
    print(
        f"{prefix}: "
        f"elapsed={row['elapsed']:6.2f}s, "
        f"infer={row['infer_elapsed']:6.2f}s, "
        f"post={row['post_elapsed']:5.2f}s, "
        f"ttfa={fmt(row.get('ttfa'), width=6, decimals=3)}s, "
        f"audio={row['duration']:6.2f}s, "
        f"RTF={row['rtf']:6.3f}, "
        f"throughput_x={row['throughput_x']:5.2f}, "
        f"peak={row['peak']:.8f}, "
        f"rms={row['rms']:.8f}, "
        f"silent={row['is_silent']}, "
        f"path={row['path']}"
    )


def run_mode(spec: ModeSpec, model: FasterQwen3TTS, load_elapsed: float, args: argparse.Namespace) -> Dict[str, Any]:
    mode_dir = Path(args.output_dir) / spec.name
    mode_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 120)
    print(f"Mode: {spec.name}")
    print("=" * 120)
    print(f"Family: {spec.family}")
    print(f"Streaming: {spec.streaming}")
    print(f"xvec_only: {spec.xvec_only}")
    print(f"Model: {model_name_for_spec(spec, args)}")
    print(f"Output dir: {mode_dir}")
    print(f"Model load time for this model: {load_elapsed:.2f}s")

    warmups = []
    bench = []

    warmup_texts = limited_texts(
        [{"label": f"warmup_{i:02d}", "bucket": "warmup", "text": t} for i, t in enumerate(WARMUP_TEXTS, start=1)],
        args.warmup_runs,
    )

    print("\nWarmup...")
    for idx, item in enumerate(warmup_texts, start=1):
        out_path = mode_dir / f"warmup_{idx:02d}.wav"
        row = generate_one(model=model, spec=spec, text=item["text"], out_path=out_path, args=args)
        row.update({"idx": idx, "phase": "warmup", "mode": spec.name, "label": item["label"], "bucket": "warmup", "text": item["text"], "chars": len(item["text"])})
        warmups.append(row)
        print_row(f"warmup {idx:02d}", row)
        if row.get("status") != "ok" and not args.continue_on_error:
            raise RuntimeError(f"Mode {spec.name} failed during warmup: {row.get('error')}")

    print("\nBenchmark...")
    print("-" * 120)
    bench_items = limited_texts(TEXTS, args.benchmark_runs)
    wall_start = time.monotonic()
    for idx, item in enumerate(bench_items, start=1):
        out_path = mode_dir / f"bench_{idx:02d}_{item['label']}.wav"
        row = generate_one(model=model, spec=spec, text=item["text"], out_path=out_path, args=args)
        row.update({"idx": idx, "phase": "benchmark", "mode": spec.name, "label": item["label"], "bucket": item["bucket"], "text": item["text"], "chars": len(item["text"])})
        bench.append(row)
        prefix = f"item {idx:02d} | label={item['label']:<18} | bucket={item['bucket']:<10} | chars={len(item['text']):4d}"
        print_row(prefix, row)
        if row.get("status") != "ok" and not args.continue_on_error:
            raise RuntimeError(f"Mode {spec.name} failed during benchmark: {row.get('error')}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_elapsed = time.monotonic() - wall_start

    summary = summarize_rows(bench)
    summary.update({
        "mode": spec.name,
        "family": spec.family,
        "streaming": spec.streaming,
        "xvec_only": spec.xvec_only,
        "model": model_name_for_spec(spec, args),
        "model_load_time_sec": load_elapsed,
        "benchmark_wall_time_sec": wall_elapsed,
        "wall_clock_rtf_elapsed_over_audio": wall_elapsed / summary["total_audio_duration_sec"] if summary["total_audio_duration_sec"] > 0 else 0.0,
        "wall_clock_throughput_x_audio_over_wall": summary["total_audio_duration_sec"] / wall_elapsed if wall_elapsed > 0 else 0.0,
        "warmup_runs": len(warmups),
        "warmup_silent_items": sum(1 for r in warmups if r.get("status") == "ok" and r.get("is_silent")),
        "warmup_error_items": sum(1 for r in warmups if r.get("status") != "ok"),
        "bucket_summaries": summarize_by_bucket(bench),
    })

    fastest = min([r for r in bench if r.get("status") == "ok"], key=lambda r: r["rtf"], default=None)
    slowest = max([r for r in bench if r.get("status") == "ok"], key=lambda r: r["rtf"], default=None)
    if fastest:
        summary["fastest_item_by_elapsed_over_audio_rtf"] = f"item {fastest['idx']} | {fastest['label']} | RTF={fastest['rtf']:.3f} | throughput_x={fastest['throughput_x']:.2f} | audio={fastest['duration']:.2f}s"
    if slowest:
        summary["slowest_item_by_elapsed_over_audio_rtf"] = f"item {slowest['idx']} | {slowest['label']} | RTF={slowest['rtf']:.3f} | throughput_x={slowest['throughput_x']:.2f} | audio={slowest['duration']:.2f}s"

    print("\nMode summary")
    for k, v in summary.items():
        if k == "bucket_summaries":
            for i, line in enumerate(v, start=1):
                print(f"bucket_summary_{i}: {line}")
        else:
            print(f"{k}: {v}")

    return {"spec": asdict(spec), "summary": summary, "warmups": warmups, "benchmarks": bench}


def write_summary_file(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("Faster Qwen3-TTS Full Benchmark Summary\n")
        f.write("=" * 120 + "\n\n")

        meta = payload["meta"]
        for key, value in meta.items():
            f.write(f"{key}: {value}\n")

        f.write("\nOverall summary\n")
        f.write("-" * 120 + "\n")
        for key, value in payload["overall_summary"].items():
            f.write(f"{key}: {value}\n")

        for mode_result in payload["modes"]:
            s = mode_result["summary"]
            f.write("\n" + "=" * 120 + "\n")
            f.write(f"Mode: {s['mode']}\n")
            f.write("=" * 120 + "\n")
            for key, value in s.items():
                if key == "bucket_summaries":
                    for i, line in enumerate(value, start=1):
                        f.write(f"bucket_summary_{i}: {line}\n")
                else:
                    f.write(f"{key}: {value}\n")

            f.write("\nBenchmark files:\n")
            for r in mode_result["benchmarks"]:
                f.write(
                    f"item {r['idx']:02d} | status={r['status']} | label={r['label']:<18} | bucket={r['bucket']:<10} | "
                    f"chars={r['chars']:4d} | elapsed={r['elapsed']:.2f}s | infer={r['infer_elapsed']:.2f}s | "
                    f"post={r['post_elapsed']:.2f}s | ttfa={r.get('ttfa')} | audio={r['duration']:.2f}s | "
                    f"RTF={r['rtf']:.3f} | throughput_x={r['throughput_x']:.2f} | peak={r['peak']:.8f} | "
                    f"rms={r['rms']:.8f} | silent={r['is_silent']} | path={r['path']} | error={r.get('error')}\n"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Full Faster Qwen3-TTS benchmark suite")
    parser.add_argument("--suite", nargs="+", default=["clone"], help="Suite/mode names. Examples: clone, custom, design, streaming, non_streaming, all, clone_streaming")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--custom-model", default=DEFAULT_CUSTOM_MODEL)
    parser.add_argument("--design-model", default=DEFAULT_DESIGN_MODEL)
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--design-instruct", default=DEFAULT_DESIGN_INSTRUCT)
    parser.add_argument("--custom-instruct", default="")
    parser.add_argument("--clone-instruct", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-sample", action="store_true", help="Use deterministic decode when supported.")
    parser.add_argument("--no-append-silence", action="store_true", help="Disable trailing silence on clone ref audio.")
    parser.add_argument("--non-streaming-mode", type=lambda x: {"true": True, "false": False, "none": None}[x.lower()], default=None, help="Override repo non_streaming_mode: true/false/none")
    parser.add_argument("--warmup-runs", type=int, default=10, help="Use 0 for all warmups? Actually 0 means no warmups here.")
    parser.add_argument("--benchmark-runs", type=int, default=0, help="0 means all benchmark texts; otherwise first N.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a mode/item fails. Useful for fp16 experiments.")
    parser.add_argument("--unload-between-models", action="store_true", default=True, help="Unload each model family after use to reduce VRAM.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = resolve_modes(args)
    required_models: List[Tuple[str, str]] = []
    for spec in specs:
        name = model_name_for_spec(spec, args)
        if not any(existing_name == name for _, existing_name in required_models):
            required_models.append((spec.model_key, name))

    print("Faster Qwen3-TTS full benchmark suite")
    print("=" * 120)
    print(f"Suites/modes requested: {args.suite}")
    print(f"Resolved modes: {[s.name for s in specs]}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"dtype: {args.dtype}")
    print(f"Language: {args.language}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Benchmark runs: {'all' if args.benchmark_runs <= 0 else args.benchmark_runs}")
    print(f"Required model loads: {required_models}")
    if torch.cuda.is_available():
        print(f"Visible GPU: {torch.cuda.get_device_name(0)}")

    results_by_mode: List[Dict[str, Any]] = []
    loaded_models: Dict[str, Tuple[FasterQwen3TTS, float]] = {}

    for model_key, model_name in required_models:
        print("\n" + "#" * 120)
        print(f"Loading model group: {model_key} | {model_name}")
        print("#" * 120)
        model, load_elapsed = load_model_once(model_name, args)
        loaded_models[model_name] = (model, load_elapsed)
        print(f"Loaded in {load_elapsed:.2f}s; sample_rate={getattr(model, 'sample_rate', 'unknown')}")

        for spec in [s for s in specs if model_name_for_spec(s, args) == model_name]:
            mode_result = run_mode(spec, model, load_elapsed, args)
            results_by_mode.append(mode_result)

        if args.unload_between_models:
            print(f"\nUnloading model group: {model_key}")
            del model
            loaded_models.pop(model_name, None)
            clear_gpu_memory()

    all_bench_rows = [r for m in results_by_mode for r in m["benchmarks"]]
    overall = summarize_rows(all_bench_rows)
    overall.update({
        "status": "SUCCESS" if overall["error_items"] == 0 else "PARTIAL_WITH_ERRORS",
        "modes_run": len(results_by_mode),
        "mode_names": ", ".join([m["summary"]["mode"] for m in results_by_mode]),
    })

    payload = {
        "meta": {
            "script": Path(__file__).name,
            "device": args.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "torch": torch.__version__,
            "dtype": args.dtype,
            "language": args.language,
            "base_model": args.base_model,
            "custom_model": args.custom_model,
            "design_model": args.design_model,
            "speaker": args.speaker,
            "ref_audio": args.ref_audio,
            "ref_text_chars": len(args.ref_text or ""),
            "design_instruct": args.design_instruct,
            "chunk_size": args.chunk_size,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "non_streaming_mode": args.non_streaming_mode,
            "append_silence": not args.no_append_silence,
            "do_sample": not args.no_sample,
        },
        "overall_summary": overall,
        "modes": results_by_mode,
    }

    json_path = output_dir / "results.json"
    summary_path = output_dir / "summary.txt"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    write_summary_file(summary_path, payload)

    print("\n" + "=" * 120)
    print("Overall summary")
    print("=" * 120)
    for key, value in overall.items():
        print(f"{key}: {value}")
    print(f"summary_file: {summary_path}")
    print(f"json_file: {json_path}")
    print(f"generated_files_dir: {output_dir}")

    return 0 if overall["error_items"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
