#!/usr/bin/env python3
"""
Full OmniVoice benchmark suite.

Covers these OmniVoice modes:
1. auto
2. design
3. clone

Mechanism follows the same style as your Faster Qwen3-TTS benchmark:
- force/use CUDA_VISIBLE_DEVICES=1
- load model once
- run warmups per mode
- run benchmark texts
- write WAV files
- compute elapsed, infer/post time, audio duration, RTF, throughput, peak/RMS/silence
- write summary.txt + results.json

Notes:
- OmniVoice current public Python API is non-streaming model.generate().
- TTFA is therefore reported as n/a / None.
- RTF = elapsed / generated_audio_duration. Lower is better.
- throughput_x = generated_audio_duration / elapsed. Higher is better.

Recommended first run:

cd /home/kawshik/tts-testing/OmniVoice
conda activate omnivoice

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

python test_omnivoice_full_benchmark.py \
  --suite auto \
  --model k2-fsa/OmniVoice \
  --dtype fp16 \
  --num-step 16 \
  --output-dir outputs/omnivoice_full_benchmark

Voice design:

CUDA_VISIBLE_DEVICES=1 python test_omnivoice_full_benchmark.py \
  --suite design \
  --instruct "female, young adult, clear voice" \
  --dtype fp16 \
  --num-step 16

Voice clone:

CUDA_VISIBLE_DEVICES=1 python test_omnivoice_full_benchmark.py \
  --suite clone \
  --ref-audio ref_audio.wav \
  --ref-text "আমি বাংলায় কথা বলছি।" \
  --dtype fp16 \
  --num-step 16

Full suite:

CUDA_VISIBLE_DEVICES=1 python test_omnivoice_full_benchmark.py \
  --suite all \
  --ref-audio ref_audio.wav \
  --ref-text "আমি বাংলায় কথা বলছি।" \
  --instruct "female, young adult, clear voice" \
  --dtype fp16 \
  --num-step 16
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Force physical GPU 1 unless the user already starts the process differently.
# This must happen before importing torch.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice


DEFAULT_MODEL = "k2-fsa/OmniVoice"
DEFAULT_REF_AUDIO = "ref_audio.wav"
DEFAULT_REF_TEXT = "আমি বাংলায় কথা বলছি।"
DEFAULT_INSTRUCT = "female, young adult, clear voice"
DEFAULT_OUTPUT_DIR = "outputs/omnivoice_full_benchmark"
DEFAULT_SAMPLE_RATE = 24000

SILENCE_PEAK_THRESHOLD = 1e-4
SILENCE_RMS_THRESHOLD = 1e-5


WARMUP_TEXTS = [
    "আমি বাংলায় কথা বলছি।",
    "আপনার পেমেন্ট সফল হয়েছে।",
    "This is a short warmup test.",
    "আমি API integration নিয়ে কাজ করছি।",
    "The text to speech model is preparing for the benchmark.",
    "আজ আমরা একটি বাংলা টেক্সট টু স্পিচ মডেল পরীক্ষা করছি।",
    "Also I can talk in English.",
    "আপনার account successfully created হয়েছে।",
    "This final warmup sentence helps stabilize the first few GPU calls.",
    "এখন আমরা মূল benchmark শুরু করব।",
]


TEXTS = [
    {
        "label": "bn_very_short_01",
        "bucket": "very_short",
        "text": "আমি বাংলায় কথা বলছি।",
    },
    {
        "label": "bn_very_short_02",
        "bucket": "very_short",
        "text": "আপনার পেমেন্ট সফল হয়েছে।",
    },
    {
        "label": "bn_very_short_03",
        "bucket": "very_short",
        "text": "আজ আবহাওয়া ভালো।",
    },
    {
        "label": "bn_short_01",
        "bucket": "short",
        "text": "আজ আমরা একটি বাংলা টেক্সট টু স্পিচ মডেল পরীক্ষা করছি।",
    },
    {
        "label": "bn_short_02",
        "bucket": "short",
        "text": "ঢাকা থেকে চট্টগ্রাম যেতে কত সময় লাগে?",
    },
    {
        "label": "bn_short_03",
        "bucket": "short",
        "text": "আপনার অনুরোধটি সফলভাবে সম্পন্ন হয়েছে।",
    },
    {
        "label": "bn_code_mix_01",
        "bucket": "code_mix",
        "text": "আমি বাংলায় কথা বলতে চাই। Also I can talk in English.",
    },
    {
        "label": "bn_code_mix_02",
        "bucket": "code_mix",
        "text": "আপনার account successfully created হয়েছে। এখন login করে profile update করুন।",
    },
    {
        "label": "bn_code_mix_03",
        "bucket": "code_mix",
        "text": "এই API integration এর জন্য database migration complete করতে হবে।",
    },
    {
        "label": "bn_code_mix_04",
        "bucket": "code_mix",
        "text": "Payment gateway response না আসলে retry mechanism ব্যবহার করুন।",
    },
    {
        "label": "bn_number_01",
        "bucket": "number",
        "text": "আপনার OTP হলো চার পাঁচ ছয় সাত আট নয়।",
    },
    {
        "label": "bn_number_02",
        "bucket": "number",
        "text": "আজ বারো ডিসেম্বর দুই হাজার ছাব্বিশ, তাপমাত্রা ঊনত্রিশ দশমিক পাঁচ ডিগ্রি।",
    },
    {
        "label": "bn_number_03",
        "bucket": "number",
        "text": "আপনার মোট বিল পাঁচশ টাকা এবং পঞ্চাশ পয়সা।",
    },
    {
        "label": "bn_medium_01",
        "bucket": "medium",
        "text": (
            "একটি ভালো টেক্সট টু স্পিচ সিস্টেমের উচ্চারণ পরিষ্কার হওয়া উচিত, "
            "বাক্যের ছন্দ স্থিতিশীল হওয়া উচিত, এবং বিভিন্ন দৈর্ঘ্যের লেখায় "
            "অডিওর মান একই রকম থাকা দরকার।"
        ),
    },
    {
        "label": "bn_medium_02",
        "bucket": "medium",
        "text": (
            "এই বেঞ্চমার্কে আমরা জেনারেশন টাইম, অডিও ডিউরেশন, রিয়েল টাইম ফ্যাক্টর, "
            "থ্রুপুট, পিক অ্যাম্পলিটিউড এবং আর এম এস এনার্জি মাপছি।"
        ),
    },
    {
        "label": "bn_medium_03",
        "bucket": "medium",
        "text": (
            "প্রোডাকশন সার্ভারে মডেল চালানোর সময় শুধু গড় latency দেখলেই হয় না, "
            "বরং silent audio, invalid output এবং দীর্ঘ বাক্যে stability ও পরীক্ষা করতে হয়।"
        ),
    },
    {
        "label": "bn_long_01",
        "bucket": "long",
        "text": (
            "প্রোডাকশন সার্ভারে টেক্সট টু স্পিচ মডেল চালানোর সময় শুধু অডিওর মান দেখলেই হয় না। "
            "মডেল লোড হতে কত সময় লাগে, প্রতিটি অনুরোধে কত সময় লাগে, তৈরি হওয়া অডিওর দৈর্ঘ্য কত, "
            "রিয়েল টাইম ফ্যাক্টর কত, এবং কোনো সাইলেন্ট বা invalid audio তৈরি হচ্ছে কি না, "
            "এসব বিষয়ও নিয়মিতভাবে পরীক্ষা করা দরকার।"
        ),
    },
    {
        "label": "bn_long_02",
        "bucket": "long",
        "text": (
            "বাস্তব অ্যাপ্লিকেশনে ব্যবহারকারী কখনো খুব ছোট বাক্য দেয়, কখনো মাঝারি দৈর্ঘ্যের নির্দেশনা দেয়, "
            "আবার কখনো বড় paragraph দেয়। তাই একটি কার্যকর benchmark suite এ বিভিন্ন দৈর্ঘ্য, "
            "ভাষা মিশ্রণ, সংখ্যা, তারিখ, এবং domain specific শব্দ অন্তর্ভুক্ত করা উচিত।"
        ),
    },
    {
        "label": "en_short_01",
        "bucket": "english",
        "text": "This is a simple English text to speech benchmark.",
    },
    {
        "label": "en_medium_01",
        "bucket": "english",
        "text": (
            "A good speech synthesis model should produce clear pronunciation, stable rhythm, "
            "natural pauses, and consistent audio quality across different sentence lengths."
        ),
    },
    {
        "label": "en_long_01",
        "bucket": "english",
        "text": (
            "When deploying a text to speech model on a server, it is important to measure more than only "
            "subjective audio quality. We also need to record model loading time, generation latency, output "
            "duration, real time factor, throughput, and whether any silent or invalid audio files were produced."
        ),
    },
]


@dataclass
class ModeSpec:
    name: str
    family: str  # auto/design/clone


ALL_MODES: List[ModeSpec] = [
    ModeSpec("auto", "auto"),
    ModeSpec("design", "design"),
    ModeSpec("clone", "clone"),
]

SUITES: Dict[str, List[str]] = {
    "auto": ["auto"],
    "design": ["design"],
    "clone": ["clone"],
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

    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    return audio.astype(np.float32)


def concat_audio(audio_obj: Any) -> np.ndarray:
    """
    OmniVoice model.generate() returns list[np.ndarray] according to README.
    This also handles np.ndarray / torch.Tensor defensively.
    """
    if isinstance(audio_obj, (list, tuple)):
        arrays = [normalize_audio(item) for item in audio_obj if item is not None]
        arrays = [a for a in arrays if a.size > 0]
        if not arrays:
            return np.zeros(0, dtype=np.float32)
        return normalize_audio(np.concatenate(arrays, axis=0))

    return normalize_audio(audio_obj)


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


def get_gpu_memory() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
            "gpu_max_memory_allocated_mb": None,
            "gpu_max_memory_reserved_mb": None,
        }

    return {
        "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
        "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
        "gpu_max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        "gpu_max_memory_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 / 1024, 2),
    }


def build_generate_kwargs(
    *,
    spec: ModeSpec,
    text: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "text": text,
        "num_step": args.num_step,
        "speed": args.speed,
    }

    if args.duration is not None and args.duration > 0:
        kwargs["duration"] = args.duration

    if args.language_id:
        kwargs["language_id"] = args.language_id

    if spec.family == "clone":
        if not args.ref_audio:
            raise ValueError("clone mode requires --ref-audio")
        kwargs["ref_audio"] = args.ref_audio
        if args.ref_text:
            kwargs["ref_text"] = args.ref_text

    elif spec.family == "design":
        if not args.instruct:
            raise ValueError("design mode requires --instruct")
        kwargs["instruct"] = args.instruct

    elif spec.family == "auto":
        pass

    else:
        raise ValueError(f"Unsupported mode family: {spec.family}")

    return kwargs


def generate_one(
    *,
    model: OmniVoice,
    spec: ModeSpec,
    text: str,
    out_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.monotonic()
    sample_rate = int(args.sample_rate)
    ttfa: Optional[float] = None  # non-streaming API; no TTFA

    try:
        kwargs = build_generate_kwargs(spec=spec, text=text, args=args)

        with torch.inference_mode():
            audio_obj = model.generate(**kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        infer_elapsed = time.monotonic() - start

        post_start = time.monotonic()

        audio = concat_audio(audio_obj)
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
            "infer_rtf": infer_elapsed / duration if duration > 0 else 0.0,
            "throughput_x": duration / elapsed if elapsed > 0 else 0.0,
            "infer_throughput_x": duration / infer_elapsed if infer_elapsed > 0 else 0.0,
            "path": str(out_path),
            **stats,
            **get_gpu_memory(),
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
            "infer_rtf": 0.0,
            "throughput_x": 0.0,
            "infer_throughput_x": 0.0,
            "path": str(out_path),
            "peak": 0.0,
            "rms": 0.0,
            "mean_abs": 0.0,
            "is_silent": True,
            **get_gpu_memory(),
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

    deduped: List[str] = []
    seen = set()

    for name in selected:
        if name not in seen:
            deduped.append(name)
            seen.add(name)

    return [MODE_BY_NAME[name] for name in deduped]


def load_model_once(args: argparse.Namespace) -> Tuple[OmniVoice, float]:
    clear_gpu_memory()

    dtype = parse_dtype(args.dtype)

    start = time.monotonic()

    model = OmniVoice.from_pretrained(
        args.model,
        device_map=args.device_map,
        dtype=dtype,
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
        "inference_only_throughput_x": audio / infer if infer > 0 else 0.0,
        "average_per_item_rtf_elapsed_over_audio": mean([r["rtf"] for r in ok]) if ok else 0.0,
        "average_per_item_infer_rtf_elapsed_over_audio": mean([r["infer_rtf"] for r in ok]) if ok else 0.0,
        "average_per_item_throughput_x": mean([r["throughput_x"] for r in ok]) if ok else 0.0,
        "average_audio_duration_sec": mean([r["duration"] for r in ok]) if ok else 0.0,
        "average_generation_time_sec": mean([r["elapsed"] for r in ok]) if ok else 0.0,
        "average_inference_time_sec": mean([r["infer_elapsed"] for r in ok]) if ok else 0.0,
        "average_postprocess_time_sec": mean([r["post_elapsed"] for r in ok]) if ok else 0.0,
        "average_peak": mean([r["peak"] for r in ok]) if ok else 0.0,
        "average_rms": mean([r["rms"] for r in ok]) if ok else 0.0,
        "min_peak": min([r["peak"] for r in ok]) if ok else 0.0,
        "min_rms": min([r["rms"] for r in ok]) if ok else 0.0,
        "average_ttfa_sec": mean(ttfas) if ttfas else None,
        "min_ttfa_sec": min(ttfas) if ttfas else None,
        "max_ttfa_sec": max(ttfas) if ttfas else None,
        **get_gpu_memory(),
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
            f"audio={s['total_audio_duration_sec']:.2f}s, "
            f"generation={s['total_generation_time_sec']:.2f}s, "
            f"infer={s['total_inference_time_sec']:.2f}s, "
            f"effective_rtf={s['effective_rtf_elapsed_over_audio']:.3f}, "
            f"infer_rtf={s['inference_only_rtf_elapsed_over_audio']:.3f}, "
            f"throughput_x={s['throughput_x_audio_over_elapsed']:.2f}"
            f"{ttfa_part}"
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
        f"infer_RTF={row['infer_rtf']:6.3f}, "
        f"throughput_x={row['throughput_x']:5.2f}, "
        f"peak={row['peak']:.8f}, "
        f"rms={row['rms']:.8f}, "
        f"silent={row['is_silent']}, "
        f"path={row['path']}"
    )


def run_mode(spec: ModeSpec, model: OmniVoice, load_elapsed: float, args: argparse.Namespace) -> Dict[str, Any]:
    mode_dir = Path(args.output_dir) / spec.name
    mode_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 120)
    print(f"Mode: {spec.name}")
    print("=" * 120)
    print(f"Family: {spec.family}")
    print(f"Model: {args.model}")
    print(f"Output dir: {mode_dir}")
    print(f"Model load time: {load_elapsed:.2f}s")

    warmups: List[Dict[str, Any]] = []
    bench: List[Dict[str, Any]] = []

    warmup_texts = [
        {"label": f"warmup_{i:02d}", "bucket": "warmup", "text": t}
        for i, t in enumerate(WARMUP_TEXTS, start=1)
    ]
    warmup_texts = limited_texts(warmup_texts, args.warmup_runs)

    print("\nWarmup...")

    for idx, item in enumerate(warmup_texts, start=1):
        out_path = mode_dir / f"warmup_{idx:02d}.wav"

        row = generate_one(
            model=model,
            spec=spec,
            text=item["text"],
            out_path=out_path,
            args=args,
        )

        row.update(
            {
                "idx": idx,
                "phase": "warmup",
                "mode": spec.name,
                "label": item["label"],
                "bucket": "warmup",
                "text": item["text"],
                "chars": len(item["text"]),
            }
        )

        warmups.append(row)
        print_row(f"warmup {idx:02d}", row)

        if row.get("status") != "ok" and not args.continue_on_error:
            raise RuntimeError(f"Mode {spec.name} failed during warmup: {row.get('error')}")

    print("\nBenchmark...")
    print("-" * 120)

    bench_items = limited_texts(TEXTS, args.benchmark_runs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    wall_start = time.monotonic()

    for idx, item in enumerate(bench_items, start=1):
        out_path = mode_dir / f"bench_{idx:02d}_{item['label']}.wav"

        row = generate_one(
            model=model,
            spec=spec,
            text=item["text"],
            out_path=out_path,
            args=args,
        )

        row.update(
            {
                "idx": idx,
                "phase": "benchmark",
                "mode": spec.name,
                "label": item["label"],
                "bucket": item["bucket"],
                "text": item["text"],
                "chars": len(item["text"]),
            }
        )

        bench.append(row)

        prefix = (
            f"item {idx:02d} | "
            f"label={item['label']:<18} | "
            f"bucket={item['bucket']:<10} | "
            f"chars={len(item['text']):4d}"
        )

        print_row(prefix, row)

        if row.get("status") != "ok" and not args.continue_on_error:
            raise RuntimeError(f"Mode {spec.name} failed during benchmark: {row.get('error')}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    wall_elapsed = time.monotonic() - wall_start

    summary = summarize_rows(bench)
    summary.update(
        {
            "mode": spec.name,
            "family": spec.family,
            "model": args.model,
            "model_load_time_sec": load_elapsed,
            "benchmark_wall_time_sec": wall_elapsed,
            "wall_clock_rtf_elapsed_over_audio": (
                wall_elapsed / summary["total_audio_duration_sec"]
                if summary["total_audio_duration_sec"] > 0
                else 0.0
            ),
            "wall_clock_throughput_x_audio_over_wall": (
                summary["total_audio_duration_sec"] / wall_elapsed if wall_elapsed > 0 else 0.0
            ),
            "warmup_runs": len(warmups),
            "warmup_silent_items": sum(
                1 for r in warmups if r.get("status") == "ok" and r.get("is_silent")
            ),
            "warmup_error_items": sum(1 for r in warmups if r.get("status") != "ok"),
            "bucket_summaries": summarize_by_bucket(bench),
        }
    )

    ok_bench = [r for r in bench if r.get("status") == "ok"]

    fastest = min(ok_bench, key=lambda r: r["rtf"], default=None)
    slowest = max(ok_bench, key=lambda r: r["rtf"], default=None)

    if fastest:
        summary["fastest_item_by_elapsed_over_audio_rtf"] = (
            f"item {fastest['idx']} | {fastest['label']} | "
            f"RTF={fastest['rtf']:.3f} | "
            f"infer_RTF={fastest['infer_rtf']:.3f} | "
            f"throughput_x={fastest['throughput_x']:.2f} | "
            f"audio={fastest['duration']:.2f}s"
        )

    if slowest:
        summary["slowest_item_by_elapsed_over_audio_rtf"] = (
            f"item {slowest['idx']} | {slowest['label']} | "
            f"RTF={slowest['rtf']:.3f} | "
            f"infer_RTF={slowest['infer_rtf']:.3f} | "
            f"throughput_x={slowest['throughput_x']:.2f} | "
            f"audio={slowest['duration']:.2f}s"
        )

    print("\nMode summary")
    for k, v in summary.items():
        if k == "bucket_summaries":
            for i, line in enumerate(v, start=1):
                print(f"bucket_summary_{i}: {line}")
        else:
            print(f"{k}: {v}")

    return {
        "spec": asdict(spec),
        "summary": summary,
        "warmups": warmups,
        "benchmarks": bench,
    }


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_summary_file(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("OmniVoice Full Benchmark Summary\n")
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

            f.write("\nWarmup files:\n")
            for r in mode_result["warmups"]:
                f.write(
                    f"warmup {r['idx']:02d} | status={r['status']} | label={r['label']:<12} | "
                    f"chars={r['chars']:4d} | elapsed={r['elapsed']:.2f}s | "
                    f"infer={r['infer_elapsed']:.2f}s | post={r['post_elapsed']:.2f}s | "
                    f"ttfa={r.get('ttfa')} | audio={r['duration']:.2f}s | "
                    f"RTF={r['rtf']:.3f} | infer_RTF={r.get('infer_rtf', 0.0):.3f} | "
                    f"throughput_x={r['throughput_x']:.2f} | peak={r['peak']:.8f} | "
                    f"rms={r['rms']:.8f} | silent={r['is_silent']} | "
                    f"path={r['path']} | error={r.get('error')}\n"
                )

            f.write("\nBenchmark files:\n")
            for r in mode_result["benchmarks"]:
                f.write(
                    f"item {r['idx']:02d} | status={r['status']} | label={r['label']:<18} | "
                    f"bucket={r['bucket']:<10} | chars={r['chars']:4d} | "
                    f"elapsed={r['elapsed']:.2f}s | infer={r['infer_elapsed']:.2f}s | "
                    f"post={r['post_elapsed']:.2f}s | ttfa={r.get('ttfa')} | "
                    f"audio={r['duration']:.2f}s | RTF={r['rtf']:.3f} | "
                    f"infer_RTF={r.get('infer_rtf', 0.0):.3f} | "
                    f"throughput_x={r['throughput_x']:.2f} | peak={r['peak']:.8f} | "
                    f"rms={r['rms']:.8f} | silent={r['is_silent']} | "
                    f"path={r['path']} | error={r.get('error')}\n"
                )


def write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=make_json_safe)


def summarize_overall(mode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_rows: List[Dict[str, Any]] = []

    for mode_result in mode_results:
        all_rows.extend(mode_result["benchmarks"])

    s = summarize_rows(all_rows)

    mode_lines = []
    for mode_result in mode_results:
        ms = mode_result["summary"]
        mode_lines.append(
            f"{ms['mode']}: ok={ms['ok_items']}/{ms['items']}, "
            f"silent={ms['silent_items']}, "
            f"audio={ms['total_audio_duration_sec']:.2f}s, "
            f"generation={ms['total_generation_time_sec']:.2f}s, "
            f"effective_rtf={ms['effective_rtf_elapsed_over_audio']:.3f}, "
            f"infer_rtf={ms['inference_only_rtf_elapsed_over_audio']:.3f}, "
            f"throughput_x={ms['throughput_x_audio_over_elapsed']:.2f}"
        )

    s["mode_summaries"] = mode_lines
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description="Full OmniVoice benchmark suite")

    parser.add_argument(
        "--suite",
        nargs="+",
        default=["auto"],
        help="Suite/mode names. Examples: auto, design, clone, all",
    )

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)

    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["bf16", "fp16", "fp32", "bfloat16", "float16", "float32"],
    )

    parser.add_argument("--num-step", type=int, default=16)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--language-id", default=None)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
        help="0 means no warmups; otherwise first N warmup texts.",
    )

    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=0,
        help="0 means all benchmark texts; otherwise first N.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a mode/item fails.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("OmniVoice full benchmark")
    print("=" * 120)
    print(f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"model={args.model}")
    print(f"suite={args.suite}")
    print(f"output_dir={output_dir}")
    print(f"device_map={args.device_map}")
    print(f"dtype={args.dtype}")
    print(f"num_step={args.num_step}")
    print(f"speed={args.speed}")
    print(f"duration={args.duration}")
    print(f"language_id={args.language_id}")
    print(f"sample_rate={args.sample_rate}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_version={torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"visible_gpu_count={torch.cuda.device_count()}")
        print(f"visible_gpu_0={torch.cuda.get_device_name(0)}")

    modes = resolve_modes(args)

    if any(m.family == "clone" for m in modes):
        if not args.ref_audio:
            raise ValueError("clone suite requires --ref-audio")
        if not Path(args.ref_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {args.ref_audio}")

    if any(m.family == "design" for m in modes) and not args.instruct:
        raise ValueError("design suite requires --instruct")

    model, load_elapsed = load_model_once(args)

    mode_results: List[Dict[str, Any]] = []

    for spec in modes:
        result = run_mode(spec, model, load_elapsed, args)
        mode_results.append(result)

    overall_summary = summarize_overall(mode_results)

    meta = {
        "model_family": "OmniVoice",
        "model": args.model,
        "suites": args.suite,
        "modes": [m.name for m in modes],
        "output_dir": str(output_dir),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_map": args.device_map,
        "dtype": args.dtype,
        "num_step": args.num_step,
        "speed": args.speed,
        "duration": args.duration,
        "language_id": args.language_id,
        "sample_rate": args.sample_rate,
        "ref_audio": args.ref_audio if any(m.family == "clone" for m in modes) else None,
        "ref_text_chars": len(args.ref_text or ""),
        "instruct": args.instruct if any(m.family == "design" for m in modes) else None,
        "model_load_time_sec": load_elapsed,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        **get_gpu_memory(),
    }

    payload = {
        "meta": meta,
        "overall_summary": overall_summary,
        "modes": mode_results,
    }

    summary_path = output_dir / "summary.txt"
    json_path = output_dir / "results.json"

    write_summary_file(summary_path, payload)
    write_json_file(json_path, payload)

    print("\n" + "=" * 120)
    print("Overall summary")
    print("=" * 120)

    for key, value in overall_summary.items():
        if key == "mode_summaries":
            for i, line in enumerate(value, start=1):
                print(f"mode_summary_{i}: {line}")
        else:
            print(f"{key}: {value}")

    print(f"\nsummary_file: {summary_path}")
    print(f"json_file: {json_path}")

    print("\nGenerated files saved under:")
    print(output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
