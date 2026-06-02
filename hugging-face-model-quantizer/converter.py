import os

# Optional hard-code. You can also run with:
# CUDA_VISIBLE_DEVICES=1 python converter.py
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import shutil
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel


def copy_full_snapshot(src_dir: str, dst_dir: str):
    """
    Copy the complete HF snapshot to dst_dir.

    Important:
    - Do NOT preserve HF cache symlinks.
    - Dereference symlinks and copy actual files.
    - Otherwise nested files like speech_tokenizer/config.json may become broken/missing.
    """
    if os.path.exists(dst_dir):
        print(f"🧹 Removing old output directory: {dst_dir}")
        shutil.rmtree(dst_dir)

    os.makedirs(dst_dir, exist_ok=True)

    for item in os.listdir(src_dir):
        src = os.path.join(src_dir, item)
        dst = os.path.join(dst_dir, item)

        if os.path.isdir(src):
            shutil.copytree(
                src,
                dst,
                symlinks=False,  # critical: copy real files, not HF cache symlinks
            )
        else:
            shutil.copy2(
                src,
                dst,
                follow_symlinks=True,  # critical: dereference root-level symlinks too
            )


def validate_local_snapshot(save_dir: str):
    """
    Basic sanity checks for Qwen3-TTS local loading.
    """
    required_paths = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
        "speech_tokenizer/model.safetensors",
    ]

    missing = []

    for rel_path in required_paths:
        abs_path = os.path.join(save_dir, rel_path)
        if not os.path.exists(abs_path):
            missing.append(rel_path)

    # This is the exact file your error complained about.
    speech_config = os.path.join(save_dir, "speech_tokenizer", "config.json")
    speech_preprocessor = os.path.join(save_dir, "speech_tokenizer", "preprocessor_config.json")

    if not os.path.exists(speech_config):
        if os.path.exists(speech_preprocessor):
            print("⚠️ speech_tokenizer/config.json missing.")
            print("⚠️ Found speech_tokenizer/preprocessor_config.json instead.")
            print("⚠️ Copying preprocessor_config.json -> config.json as compatibility fallback.")
            shutil.copy2(speech_preprocessor, speech_config)
        else:
            missing.append("speech_tokenizer/config.json")

    if missing:
        print("❌ Missing required files:")
        for item in missing:
            print(f"   - {item}")
        raise FileNotFoundError(f"Local snapshot is incomplete: {save_dir}")

    print("✅ Local snapshot validation passed.")


def get_device_map():
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if torch.cuda.is_available():
        print(f"✅ CUDA available. Visible GPU count: {torch.cuda.device_count()}")
        print("✅ Using logical cuda:0. With CUDA_VISIBLE_DEVICES=1, this maps to physical GPU 1.")
        return "cuda:0"

    print("⚠️ CUDA not available. Falling back to CPU.")
    return "cpu"


def convert_qwen_tts_customvoice(
    model_id: str,
    save_bf16_dir: str,
    save_fp8_dir: str | None = None,
    make_fp8: bool = False,
):
    device_map = get_device_map()

    print(f"📥 Downloading/loading model from {model_id}...")

    model_path = snapshot_download(
        repo_id=model_id,
        local_files_only=False,
    )

    print(f"📁 Local HF cache snapshot: {model_path}")

    print("📦 Creating local BF16-loadable model directory...")
    copy_full_snapshot(model_path, save_bf16_dir)

    print(f"✅ Model files copied to: {save_bf16_dir}")

    validate_local_snapshot(save_bf16_dir)

    print(f"🧪 Testing reload with dtype=torch.bfloat16 on {device_map}...")

    _ = Qwen3TTSModel.from_pretrained(
        save_bf16_dir,
        device_map=device_map,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    print(f"✅ BF16 reload test successful from: {save_bf16_dir}")

    if make_fp8:
        print("⚠️ FP8 conversion skipped.")
        print("⚠️ Qwen3TTSModel is a wrapper, not a plain torch.nn.Module.")
        print("⚠️ Use official/runtime-supported FP8 quantization later.")

        if save_fp8_dir is not None:
            print(f"📦 Copying original snapshot to FP8 placeholder dir: {save_fp8_dir}")
            copy_full_snapshot(model_path, save_fp8_dir)
            validate_local_snapshot(save_fp8_dir)

    print("🎉 Conversion complete!")


if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    SAVE_BF16 = "/home/kawshik/Quantized_models/qwen3-tts-12hz-1.7b-customvoice-bf16"
    SAVE_FP8 = "/home/kawshik/Quantized_models/qwen3-tts-12hz-1.7b-customvoice-fp8"

    convert_qwen_tts_customvoice(
        model_id=MODEL_ID,
        save_bf16_dir=SAVE_BF16,
        save_fp8_dir=SAVE_FP8,
        make_fp8=True,
    )
