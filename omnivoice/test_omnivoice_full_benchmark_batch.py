#!/usr/bin/env python3
"""
Full OmniVoice benchmark suite.

Covers these OmniVoice modes:
1. auto
2. design
3. clone

Runs benchmark variants:
1. nonbatch
2. fixed-size batch
3. duration-based batch

Notes:
- OmniVoice current public Python API is non-streaming model.generate().
- TTFA is therefore reported as n/a / None.
- RTF = elapsed / generated_audio_duration. Lower is better.
- throughput_x = generated_audio_duration / elapsed. Higher is better.
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

try:
    from omnivoice.utils.duration import RuleDurationEstimator
except Exception:
    RuleDurationEstimator = None


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
        "label": "telco_dialogue_01",
        "bucket": "greeting",
        "text": "স্বাগতম। আপনি টেলিকম কাস্টমার কেয়ারে কল করেছেন। আমি কীভাবে আপনাকে সাহায্য করতে পারি?",
    },
    {
        "label": "telco_dialogue_02",
        "bucket": "customer_reply",
        "text": "হ্যালো, আমার মোবাইল ডাটা কাজ করছে না। আমি সকাল থেকে ইন্টারনেট ব্যবহার করতে পারছি না।",
    },
    {
        "label": "telco_dialogue_03",
        "bucket": "question",
        "text": "দুঃখিত। আপনি কি আপনার মোবাইল নম্বরটি বলবেন, যাতে আমি অ্যাকাউন্টটি চেক করতে পারি?",
    },
    {
        "label": "telco_dialogue_04",
        "bucket": "customer_reply",
        "text": "অবশ্যই। আমার নম্বর হলো শূন্য এক সাত এক দুই তিন চার পাঁচ ছয় সাত আট।",
    },
    {
        "label": "telco_dialogue_05",
        "bucket": "normal",
        "text": "ধন্যবাদ। আমি এখন আপনার নম্বরের সার্ভিস স্ট্যাটাস, ব্যালেন্স এবং ডাটা প্যাক যাচাই করছি।",
    },
    {
        "label": "telco_dialogue_06",
        "bucket": "customer_reply",
        "text": "ঠিক আছে। কিন্তু একটু দ্রুত দেখবেন? আমার জরুরি অনলাইন মিটিং আছে।",
    },
    {
        "label": "telco_dialogue_07",
        "bucket": "surprise",
        "text": "ওহ! আপনার সক্রিয় ইন্টারনেট প্যাকের মেয়াদ আজ সকাল আটটা ত্রিশ মিনিটে শেষ হয়ে গেছে।",
    },
    {
        "label": "telco_dialogue_08",
        "bucket": "customer_reply",
        "text": "সত্যি? আমি তো ভেবেছিলাম প্যাকটি আজ রাত পর্যন্ত চলবে!",
    },
    {
        "label": "telco_dialogue_09",
        "bucket": "apology",
        "text": "আমি বুঝতে পারছি। আপনার আগের প্যাকটি সাত দিনের ছিল, এবং এটি নির্ধারিত সময়েই শেষ হয়েছে।",
    },
    {
        "label": "telco_dialogue_10",
        "bucket": "question",
        "text": "আপনি কি এখন নতুন ইন্টারনেট প্যাক চালু করতে চান?",
    },
    {
        "label": "telco_dialogue_11",
        "bucket": "customer_reply",
        "text": "হ্যাঁ, এমন কোনো প্যাক আছে কি, যেখানে কম খরচে বেশি ডাটা পাব?",
    },
    {
        "label": "telco_dialogue_12",
        "bucket": "offer",
        "text": "আপনার জন্য তিনটি অফার আছে। এক জিবি এক দিনের জন্য উনিশ টাকা, পাঁচ জিবি সাত দিনের জন্য একশ টাকা, এবং দশ জিবি ত্রিশ দিনের জন্য দুইশ নিরানব্বই টাকা।",
    },
    {
        "label": "telco_dialogue_13",
        "bucket": "customer_reply",
        "text": "পাঁচ জিবি সাত দিনের প্যাকটি ভালো মনে হচ্ছে। এতে কি ভ্যাটসহ একশ টাকা লাগবে?",
    },
    {
        "label": "telco_dialogue_14",
        "bucket": "clarification",
        "text": "জি। ভ্যাটসহ মোট চার্জ একশ টাকা। প্যাকটি চালু হলে আপনি এস এম এস কনফার্মেশন পাবেন।",
    },
    {
        "label": "telco_dialogue_15",
        "bucket": "customer_reply",
        "text": "ঠিক আছে, প্যাকটি চালু করে দিন। আমার ব্যালেন্সে যথেষ্ট টাকা আছে তো?",
    },
    {
        "label": "telco_dialogue_16",
        "bucket": "normal",
        "text": "আপনার বর্তমান মূল ব্যালেন্স একশ পঁচিশ টাকা পঞ্চাশ পয়সা, তাই প্যাকটি চালু করা যাবে।",
    },
    {
        "label": "telco_dialogue_17",
        "bucket": "confirmation_question",
        "text": "আপনি কি নিশ্চিত করছেন যে পাঁচ জিবি সাত দিনের ইন্টারনেট প্যাকটি একশ টাকায় চালু করা হবে?",
    },
    {
        "label": "telco_dialogue_18",
        "bucket": "customer_reply",
        "text": "জি। আমি নিশ্চিত করছি। অনুগ্রহ করে প্যাকটি এখনই চালু করুন।",
    },
    {
        "label": "telco_dialogue_19",
        "bucket": "success",
        "text": "চমৎকার। আপনার পাঁচ জিবি সাত দিনের ইন্টারনেট প্যাক সফলভাবে চালু হয়েছে।",
    },
    {
        "label": "telco_dialogue_20",
        "bucket": "customer_reply",
        "text": "ধন্যবাদ। কিন্তু এখনো মোবাইল ডাটা চালু করলে ইন্টারনেট আসছে না কেন?",
    },
    {
        "label": "telco_dialogue_21",
        "bucket": "instruction",
        "text": "অনুগ্রহ করে মোবাইল ডাটা বন্ধ করে আবার চালু করুন। তারপর ফোনের airplane mode দশ সেকেন্ডের জন্য অন করে অফ করুন।",
    },
    {
        "label": "telco_dialogue_22",
        "bucket": "customer_reply",
        "text": "আমি airplane mode অন করে অফ করেছি, কিন্তু এখনো সমস্যা আছে।",
    },
    {
        "label": "telco_dialogue_23",
        "bucket": "question",
        "text": "আপনার ফোনে কি ফোর জি বা এল টি ই নেটওয়ার্ক সিলেক্ট করা আছে?",
    },
    {
        "label": "telco_dialogue_24",
        "bucket": "customer_reply",
        "text": "না, এখানে শুধু থ্রি জি দেখাচ্ছে। এটা কি সমস্যার কারণ হতে পারে?",
    },
    {
        "label": "telco_dialogue_25",
        "bucket": "explanation",
        "text": "জি। দুর্বল নেটওয়ার্ক মোডের কারণে ডাটা স্পিড কমে যেতে পারে বা সংযোগ স্থিতিশীল নাও থাকতে পারে।",
    },
    {
        "label": "telco_dialogue_26",
        "bucket": "instruction",
        "text": "আপনার ফোনের network settings থেকে preferred network type অপশনে গিয়ে ফোর জি বা এল টি ই নির্বাচন করুন।",
    },
    {
        "label": "telco_dialogue_27",
        "bucket": "customer_reply",
        "text": "ঠিক আছে। এখন ফোর জি দেখাচ্ছে। এক মিনিট, ওহ! ইন্টারনেট আবার কাজ করছে!",
    },
    {
        "label": "telco_dialogue_28",
        "bucket": "success",
        "text": "খুব ভালো। আপনার ডাটা সার্ভিস এখন স্বাভাবিকভাবে কাজ করছে দেখে ভালো লাগছে।",
    },
    {
        "label": "telco_dialogue_29",
        "bucket": "customer_reply",
        "text": "আরেকটা প্রশ্ন আছে। আমার প্যাকের মেয়াদ কখন শেষ হবে?",
    },
    {
        "label": "telco_dialogue_30",
        "bucket": "answer",
        "text": "আপনার পাঁচ জিবি ইন্টারনেট প্যাকের মেয়াদ শেষ হবে সাত দিন পর রাত এগারোটা ঊনষাট মিনিটে।",
    },
    {
        "label": "telco_dialogue_31",
        "bucket": "customer_reply",
        "text": "আমি কি অ্যাপ থেকে বাকি ডাটা দেখতে পারব?",
    },
    {
        "label": "telco_dialogue_32",
        "bucket": "instruction",
        "text": "জি। আপনি আমাদের mobile app খুলে usage section থেকে বাকি ডাটা, মেয়াদ এবং ব্যালেন্স দেখতে পারবেন।",
    },
    {
        "label": "telco_dialogue_33",
        "bucket": "customer_reply",
        "text": "অ্যাপে লগইন করতে গেলে OTP আসতে দেরি হলে কী করব?",
    },
    {
        "label": "telco_dialogue_34",
        "bucket": "instruction",
        "text": "OTP আসতে দেরি হলে এক মিনিট অপেক্ষা করুন, resend OTP চাপুন, এবং নিশ্চিত করুন যে আপনার ইনবক্স পূর্ণ নয়।",
    },
    {
        "label": "telco_dialogue_35",
        "bucket": "customer_reply",
        "text": "ঠিক আছে। আজকের সমস্যার জন্য কি কোনো complaint number পাব?",
    },
    {
        "label": "telco_dialogue_36",
        "bucket": "confirmation",
        "text": "জি। আপনার complaint number হলো এক দুই তিন চার পাঁচ ছয়। ভবিষ্যতে follow up করার সময় এই নম্বরটি ব্যবহার করতে পারবেন।",
    },
    {
        "label": "telco_dialogue_37",
        "bucket": "customer_reply",
        "text": "ধন্যবাদ। এখন সব পরিষ্কার। আপনার সাহায্যটা সত্যিই ভালো ছিল।",
    },
    {
        "label": "telco_dialogue_38",
        "bucket": "closing",
        "text": "আপনাকে ধন্যবাদ। আমাদের সাথে থাকার জন্য কৃতজ্ঞ। আপনার দিনটি শুভ হোক!",
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


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


OPTIONAL_GENERATE_KWARGS = (
    "guidance_scale",
    "t_shift",
    "audio_chunk_duration",
    "audio_chunk_threshold",
    "preprocess_prompt",
    "postprocess_output",
    "layer_penalty_factor",
    "position_temperature",
    "class_temperature",
    "denoise",
)


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


def as_batch_value(value: Any, batch_size: int, *, force_list: bool = False) -> Any:
    """
    OmniVoice batch CLI passes per-sample fields as lists.

    force_list=True is important for a final partial batch of size 1:
    text is still passed as [text], so aligned fields should also be [value].
    """
    if not force_list and batch_size <= 1:
        return value
    return [value for _ in range(batch_size)]


def build_generate_kwargs(
    *,
    spec: ModeSpec,
    text: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    is_batch = isinstance(text, list)
    batch_size = len(text) if is_batch else 1

    kwargs: Dict[str, Any] = {
        "text": text,
        "num_step": args.num_step,
    }

    # Extra generation knobs from OmniVoice official batch CLI.
    # These are generation-level scalars, not per-sample lists.
    for name in OPTIONAL_GENERATE_KWARGS:
        value = getattr(args, name, None)
        if value is not None:
            kwargs[name] = value

    # Match OmniVoice infer_batch.py: batch inputs are lists.
    # It uses language=..., not language_id=..., for generate().
    if args.speed is not None:
        kwargs["speed"] = as_batch_value(args.speed, batch_size, force_list=is_batch)

    if args.duration is not None and args.duration > 0:
        kwargs["duration"] = as_batch_value(args.duration, batch_size, force_list=is_batch)

    if args.language_id:
        kwargs["language"] = as_batch_value(args.language_id, batch_size, force_list=is_batch)

    if spec.family == "clone":
        if not args.ref_audio:
            raise ValueError("clone mode requires --ref-audio")
        kwargs["ref_audio"] = as_batch_value(args.ref_audio, batch_size, force_list=is_batch)
        if args.ref_text:
            kwargs["ref_text"] = as_batch_value(args.ref_text, batch_size, force_list=is_batch)

    elif spec.family == "design":
        if not args.instruct:
            raise ValueError("design mode requires --instruct")
        kwargs["instruct"] = as_batch_value(args.instruct, batch_size, force_list=is_batch)

    elif spec.family == "auto":
        pass

    else:
        raise ValueError(f"Unsupported mode family: {spec.family}")

    return kwargs


def make_error_row(
    *,
    error: Exception,
    traceback_text: str,
    elapsed: float,
    out_path: Path,
    sample_rate: int,
) -> Dict[str, Any]:
    return {
        "status": "error",
        "error": repr(error),
        "traceback": traceback_text,
        "elapsed": elapsed,
        "infer_elapsed": elapsed,
        "post_elapsed": 0.0,
        "ttfa": None,
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


def make_ok_row(
    *,
    audio_obj: Any,
    out_path: Path,
    sample_rate: int,
    infer_elapsed: float,
    per_item_elapsed: float,
    batch_elapsed: Optional[float] = None,
    batch_size: int = 1,
) -> Dict[str, Any]:
    post_start = time.monotonic()

    audio = concat_audio(audio_obj)
    audio = normalize_audio(audio)

    stats = audio_stats(audio)
    duration = float(len(audio) / sample_rate) if audio.size else 0.0

    write_wav(out_path, audio, sample_rate)
    post_elapsed = time.monotonic() - post_start
    elapsed = per_item_elapsed + post_elapsed

    return {
        "status": "ok",
        "error": None,
        "elapsed": elapsed,
        "infer_elapsed": infer_elapsed,
        "post_elapsed": post_elapsed,
        "ttfa": None,
        "duration": duration,
        "sample_rate": sample_rate,
        "rtf": elapsed / duration if duration > 0 else 0.0,
        "infer_rtf": infer_elapsed / duration if duration > 0 else 0.0,
        "throughput_x": duration / elapsed if elapsed > 0 else 0.0,
        "infer_throughput_x": duration / infer_elapsed if infer_elapsed > 0 else 0.0,
        "path": str(out_path),
        "batch_size": batch_size,
        "batch_elapsed": batch_elapsed,
        **stats,
        **get_gpu_memory(),
    }


def looks_like_batch_audio(audio_obj: Any, expected_items: int) -> bool:
    """Heuristic for model.generate(text=[...]) returning one audio object per input text."""
    return isinstance(audio_obj, (list, tuple)) and len(audio_obj) == expected_items


def generate_batch(
    *,
    model: OmniVoice,
    spec: ModeSpec,
    items: Sequence[Dict[str, str]],
    out_paths: Sequence[Path],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    Batch inference.

    Important:
    Each item gets the full batch inference time.
    We do NOT divide batch_infer_elapsed by batch_size because the batch output
    becomes available only after the whole batch finishes.
    """
    if len(items) != len(out_paths):
        raise ValueError("items and out_paths must have the same length")

    batch_size = len(items)
    sample_rate = int(args.sample_rate)

    if batch_size == 0:
        return []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.monotonic()

    try:
        kwargs = build_generate_kwargs(spec=spec, text=[item["text"] for item in items], args=args)
        
        start = time.monotonic()

        with torch.inference_mode():
            audio_obj = model.generate(**kwargs)
        
        batch_infer_elapsed = time.monotonic() - start
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        

        if not looks_like_batch_audio(audio_obj, batch_size):
            raise RuntimeError(
                "Batch inference expected model.generate(text=[...]) to return one audio item per text. "
                f"Got {type(audio_obj).__name__} with length "
                f"{len(audio_obj) if isinstance(audio_obj, (list, tuple)) else 'n/a'}."
            )

        rows: List[Dict[str, Any]] = []

        for item_audio, out_path in zip(audio_obj, out_paths):
            rows.append(
                make_ok_row(
                    audio_obj=item_audio,
                    out_path=out_path,
                    sample_rate=sample_rate,
                    infer_elapsed=batch_infer_elapsed,
                    per_item_elapsed=batch_infer_elapsed,
                    batch_elapsed=batch_infer_elapsed,
                    batch_size=batch_size,
                )
            )

        return rows

    except Exception as exc:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        elapsed = time.monotonic() - start
        tb = traceback.format_exc()

        return [
            make_error_row(
                error=exc,
                traceback_text=tb,
                elapsed=elapsed,
                out_path=out_path,
                sample_rate=sample_rate,
            )
            for out_path in out_paths
        ]


def generate_one(
    *,
    model: OmniVoice,
    spec: ModeSpec,
    text: str,
    out_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Original non-batch path: one scalar text per model.generate() call.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.monotonic()
    sample_rate = int(args.sample_rate)
    ttfa: Optional[float] = None

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
            "batch_size": 1,
            "batch_elapsed": None,
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

        return make_error_row(
            error=exc,
            traceback_text=traceback.format_exc(),
            elapsed=elapsed,
            out_path=out_path,
            sample_rate=sample_rate,
        )


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


def iter_batches(items: Sequence[Dict[str, str]], batch_size: int) -> List[List[Dict[str, str]]]:
    batch_size = max(1, int(batch_size))
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def get_reference_audio_duration_sec(path: Optional[str]) -> float:
    if not path:
        return 0.0
    try:
        return float(sf.info(path).duration)
    except Exception:
        return 0.0


def estimate_item_total_duration_sec(
    *,
    item: Dict[str, str],
    spec: ModeSpec,
    args: argparse.Namespace,
    duration_estimator: Any = None,
) -> float:
    text = item.get("text", "")

    ref_audio = args.ref_audio if spec.family == "clone" else None
    ref_text = args.ref_text if spec.family == "clone" else None
    ref_duration = get_reference_audio_duration_sec(ref_audio)

    if args.duration is not None and args.duration > 0:
        gen_duration = float(args.duration)
    elif duration_estimator is not None:
        try:
            if spec.family == "clone":
                gen_duration = float(
                    duration_estimator.estimate_duration(
                        text,
                        ref_text or "",
                        ref_duration,
                        low_threshold=2.0,
                    )
                )
            else:
                gen_duration = float(
                    duration_estimator.estimate_duration(
                        text,
                        "Nice to meet you.",
                        0.5,
                        low_threshold=2.0,
                    )
                )
        except Exception:
            gen_duration = max(1.0, len(text) / 14.0)
    else:
        gen_duration = max(1.0, len(text) / 14.0)

    return max(0.0, ref_duration + gen_duration)


def iter_duration_batches(
    *,
    items: Sequence[Dict[str, str]],
    spec: ModeSpec,
    args: argparse.Namespace,
    batch_duration: float,
) -> List[List[Dict[str, str]]]:
    if not items:
        return []

    if batch_duration <= 0:
        return [list(items)]

    duration_estimator = RuleDurationEstimator() if RuleDurationEstimator is not None else None

    item_with_duration = [
        (
            item,
            estimate_item_total_duration_sec(
                item=item,
                spec=spec,
                args=args,
                duration_estimator=duration_estimator,
            ),
        )
        for item in items
    ]
    item_with_duration.sort(key=lambda x: x[1], reverse=True)

    batches: List[List[Dict[str, str]]] = []
    current_batch: List[Dict[str, str]] = []
    current_total = 0.0

    for item, duration in item_with_duration:
        if duration > batch_duration:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_total = 0.0
            batches.append([item])
            continue

        if current_batch and current_total + duration > batch_duration:
            batches.append(current_batch)
            current_batch = [item]
            current_total = duration
        else:
            current_batch.append(item)
            current_total += duration

    if current_batch:
        batches.append(current_batch)

    return batches


def make_batches(
    *,
    items: Sequence[Dict[str, str]],
    spec: ModeSpec,
    args: argparse.Namespace,
    batching_strategy: str,
    configured_batch_size: int,
) -> List[List[Dict[str, str]]]:
    if batching_strategy == "none":
        return iter_batches(items, 1)
    if batching_strategy == "fixed":
        return iter_batches(items, configured_batch_size)
    if batching_strategy == "duration":
        return iter_duration_batches(
            items=items,
            spec=spec,
            args=args,
            batch_duration=float(args.batch_duration),
        )
    raise ValueError(f"Unknown batching strategy: {batching_strategy}")


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


def run_mode(
    spec: ModeSpec,
    model: OmniVoice,
    load_elapsed: float,
    args: argparse.Namespace,
    *,
    benchmark_variant: str,
    batching_enabled: bool,
    configured_batch_size: int,
    batching_strategy: str,
) -> Dict[str, Any]:
    mode_dir = Path(args.output_dir) / benchmark_variant / spec.name
    mode_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 120)
    print(f"Mode: {spec.name}")
    print(f"Benchmark variant: {benchmark_variant}")
    print("=" * 120)
    print(f"Family: {spec.family}")
    print(f"Batching enabled: {batching_enabled}")
    print(f"Batching strategy: {batching_strategy}")
    print(f"Configured batch size: {configured_batch_size}")
    print(f"Batch duration: {getattr(args, 'batch_duration', None)}")
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

    warmup_batches = make_batches(
        items=warmup_texts,
        spec=spec,
        args=args,
        batching_strategy=batching_strategy,
        configured_batch_size=configured_batch_size,
    )

    for batch in warmup_batches:
        batch_start_idx = len(warmups) + 1
        out_paths = [
            mode_dir / f"warmup_{batch_start_idx + offset:02d}.wav"
            for offset, _ in enumerate(batch)
        ]

        if batching_enabled:
            rows = generate_batch(
                model=model,
                spec=spec,
                items=batch,
                out_paths=out_paths,
                args=args,
            )
        else:
            rows = [
                generate_one(
                    model=model,
                    spec=spec,
                    text=item["text"],
                    out_path=out_path,
                    args=args,
                )
                for item, out_path in zip(batch, out_paths)
            ]

        for offset, (item, row) in enumerate(zip(batch, rows), start=0):
            idx = batch_start_idx + offset
            row.update(
                {
                    "idx": idx,
                    "phase": "warmup",
                    "mode": spec.name,
                    "benchmark_variant": benchmark_variant,
                    "batching_enabled": batching_enabled,
                    "configured_batch_size": configured_batch_size,
                    "batching_strategy": batching_strategy,
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

    benchmark_batches = make_batches(
        items=bench_items,
        spec=spec,
        args=args,
        batching_strategy=batching_strategy,
        configured_batch_size=configured_batch_size,
    )

    for batch in benchmark_batches:
        batch_start_idx = len(bench) + 1
        out_paths = [
            mode_dir / f"bench_{batch_start_idx + offset:02d}_{item['label']}.wav"
            for offset, item in enumerate(batch)
        ]

        if batching_enabled:
            rows = generate_batch(
                model=model,
                spec=spec,
                items=batch,
                out_paths=out_paths,
                args=args,
            )
        else:
            rows = [
                generate_one(
                    model=model,
                    spec=spec,
                    text=item["text"],
                    out_path=out_path,
                    args=args,
                )
                for item, out_path in zip(batch, out_paths)
            ]

        for offset, (item, row) in enumerate(zip(batch, rows), start=0):
            idx = batch_start_idx + offset
            row.update(
                {
                    "idx": idx,
                    "phase": "benchmark",
                    "mode": spec.name,
                    "benchmark_variant": benchmark_variant,
                    "batching_enabled": batching_enabled,
                    "configured_batch_size": configured_batch_size,
                    "batching_strategy": batching_strategy,
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
            "benchmark_variant": benchmark_variant,
            "batching_enabled": batching_enabled,
            "configured_batch_size": configured_batch_size,
            "batching_strategy": batching_strategy,
            "batch_duration": getattr(args, "batch_duration", None),
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
        "benchmark_variant": benchmark_variant,
        "batching_enabled": batching_enabled,
        "configured_batch_size": configured_batch_size,
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
            f"{ms.get('benchmark_variant', 'default')}/{ms['mode']}: ok={ms['ok_items']}/{ms['items']}, "
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
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--t-shift", type=float, default=None)
    parser.add_argument("--audio-chunk-duration", type=float, default=None)
    parser.add_argument("--audio-chunk-threshold", type=float, default=None)
    parser.add_argument("--preprocess-prompt", type=str2bool, default=None)
    parser.add_argument("--postprocess-output", type=str2bool, default=None)
    parser.add_argument("--layer-penalty-factor", type=float, default=None)
    parser.add_argument("--position-temperature", type=float, default=None)
    parser.add_argument("--class-temperature", type=float, default=None)
    parser.add_argument("--denoise", type=str2bool, default=None)

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
        "--batch-duration",
        type=float,
        default=1000.0,
        help="Maximum estimated total duration per duration-based batch, in seconds.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Fixed batch size for fixed-size batch benchmark runs.",
    )

    parser.add_argument(
        "--benchmark-kind",
        choices=["nonbatch", "fixed", "duration", "batch", "both", "all"],
        default="all",
        help=(
            "Which benchmark variants to run. "
            "nonbatch=single item calls, fixed=fixed-size batching, "
            "duration=duration-budget batching, batch=fixed alias, "
            "both=nonbatch+fixed legacy alias, all=nonbatch+fixed+duration."
        ),
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
    for name in OPTIONAL_GENERATE_KWARGS:
        print(f"{name}={getattr(args, name, None)}")
    print(f"speed={args.speed}")
    print(f"duration={args.duration}")
    print(f"language_id={args.language_id}")
    print(f"sample_rate={args.sample_rate}")
    print(f"batch_size={args.batch_size}")
    print(f"batch_duration={args.batch_duration}")
    print(f"benchmark_kind={args.benchmark_kind}")
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

    benchmark_variants: List[Tuple[str, bool, int, str]] = []
    fixed_batch_size = max(1, int(args.batch_size))

    if args.benchmark_kind in ("nonbatch", "both", "all"):
        benchmark_variants.append(("nonbatch", False, 1, "none"))
    if args.benchmark_kind in ("fixed", "batch", "both", "all"):
        benchmark_variants.append((f"batch_fixed_bs{fixed_batch_size}", True, fixed_batch_size, "fixed"))
    if args.benchmark_kind in ("duration", "all"):
        safe_batch_duration = str(args.batch_duration).replace(".", "p")
        benchmark_variants.append((f"batch_duration_{safe_batch_duration}s", True, fixed_batch_size, "duration"))

    for variant_name, batching_enabled, configured_batch_size, batching_strategy in benchmark_variants:
        for spec in modes:
            result = run_mode(
                spec,
                model,
                load_elapsed,
                args,
                benchmark_variant=variant_name,
                batching_enabled=batching_enabled,
                configured_batch_size=configured_batch_size,
                batching_strategy=batching_strategy,
            )
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
        "generate_kwargs": {name: getattr(args, name, None) for name in OPTIONAL_GENERATE_KWARGS},
        "speed": args.speed,
        "duration": args.duration,
        "language_id": args.language_id,
        "sample_rate": args.sample_rate,
        "batch_size": args.batch_size,
        "batch_duration": args.batch_duration,
        "benchmark_kind": args.benchmark_kind,
        "benchmark_variants": [r[0] for r in benchmark_variants],
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