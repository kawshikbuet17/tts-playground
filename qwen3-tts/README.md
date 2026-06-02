# Qwen3-TTS Execution Documentation

**Qwen3-TTS** is the native inference implementation for Qwen3-TTS based on the standard `qwen_tts` package. This document records the Linux server setup, execution flow, benchmark scripts (`test_qwen3_tts_voice_clone_benchmark.py` and `test_qwen3_tts_custom_voice_benchmark.py`), successful English inference results, sdpa/flash_attention_2 benchmark comparison, CustomVoice benchmark suite results, and output storage pattern for Qwen3-TTS.

**Repository:**

```text
https://github.com/QwenLM/Qwen3-TTS
```

---

## 1. System Assumptions

Tested environment:

```text
Server OS: Linux
Project path: /home/kawshik/tts-testing/Qwen3-TTS
Python env: conda environment named qwen3-tts
Python: 3.12.13
GPU selected for test: NVIDIA GeForce RTX 3090
CUDA visible device: 1
Runtime logical device inside Python: cuda:0
Model family: Qwen3-TTS (native qwen_tts)
Primary language tested: English
Audio sample rate: 24000 Hz
Torch: 2.5.1+cu121
Runtime device: cuda
Stable dtype for benchmark: bf16 (bfloat16)
Stable fallback dtype: fp32 (float32)
Stable attention implementation: flash_attention_2
Standard fallback attention: sdpa
```
---

## 2. Go to Project Folder

From the server:

```bash
cd /home/kawshik/tts-testing/Qwen3-TTS
```

If starting from scratch:

```bash
cd /home/kawshik/tts-testing/

# Clone the native repository
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
```

---

## 3. Create and Activate Python Environment

Create a fresh conda environment:

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
```

Upgrade base Python tooling if needed:

```bash
pip install --upgrade pip setuptools wheel
```

---

## 4. Select GPU

Use physical GPU 1:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```


Verify from Python:

```bash
python - <<'PY'
import torch, sys

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)

if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    print("visible gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

Expected successful GPU check:

```text
python: 3.12.13
python cuda available: True
torch: 2.5.1+cu121
cuda: 12.1
visible gpu: NVIDIA GeForce RTX 3090
capability: (8, 6)
```

---

## 5. Install Qwen3-TTS Dependencies

From the local repository:

```bash
cd /home/kawshik/tts-testing/Qwen3-TTS
conda activate qwen3-tts

pip install -r requirements.txt
```

Check script execution and CLI options:

For Voice Cloning (ICL):
```bash
python test_qwen3_tts_voice_clone_benchmark.py --help
```

For Custom Voice synthesis:
```bash
python test_qwen3_tts_custom_voice_benchmark.py --help
```

Supported scripts:

```text
[test_qwen3_tts_voice_clone_benchmark.py](test_qwen3_tts_voice_clone_benchmark.py)  - benchmarking standard voice cloning using reference audio and transcript
[test_qwen3_tts_custom_voice_benchmark.py](test_qwen3_tts_custom_voice_benchmark.py) - benchmarking custom predefined voices (like Ryan) with style descriptions
```

Supported arguments for `test_qwen3_tts_voice_clone_benchmark.py`:

```text
--model               - model path or HuggingFace repo (default: Qwen/Qwen3-TTS-12Hz-0.6B-Base)
--ref-audio           - reference audio file path
--ref-text            - reference audio transcript text
--dtype               - precision format (bf16, fp16, fp32)
--xvec-only           - x‑vector‑only mode (no ICL reference text)
--warmup-runs         - number of warmup runs
--benchmark-runs      - number of benchmark runs to execute
```

Supported arguments for `test_qwen3_tts_custom_voice_benchmark.py`:

```text
--model               - path to CustomVoice model (default: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
--language            - target language (default: English)
--speaker             - speaker ID (default: Ryan)
--instruct            - optional speech style instruction (default: "Speak clearly with a neutral tone.")
--attn                - attention implementation (flash_attention_2, sdpa, eager)
--dtype               - precision format (bfloat16, float16, float32)
--output-dir          - target directory for outputs (default: outputs/benchmark)
```

---

## 6. Attention and Precision Decisions

The following attention and precision modes were tested:

```text
bfloat16 + flash_attention_2: stable, fast, and achieves optimal RTF (approx. 2.715)
bfloat16 + sdpa: stable fallback, slightly slower RTF (approx. 3.273)
float32: stable fallback, slower than bfloat16
```

Final runtime decision:

```text
Use bfloat16 with flash_attention_2 for normal Qwen3-TTS experiments and deployment benchmarking.
Use bfloat16 with sdpa only as a stable fallback when flash_attention_2 is not supported.
Use float32 only as a stable debugging/reference mode.
```

---

## 7. First Working Single Inference: 0.6B Base Voice Clone

The first successful manual run used the included reference audio:

```text
ref_audio.wav
```

Run:

```bash
cd /home/kawshik/tts-testing/Qwen3-TTS
conda activate qwen3-tts
export CUDA_VISIBLE_DEVICES=1

python test_qwen3_tts_voice_clone_benchmark.py \
  --device cuda \
  --dtype bf16 \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio ref_audio.wav \
  --ref-text "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed." \
  --output-dir outputs/qwen3_tts_voice_clone_benchmark \
  --benchmark-runs 1
```

Successful result:

[Benchmark Summary](outputs/qwen3_tts_voice_clone_benchmark/summary.txt)

---

## 8. Single Clone Benchmark: bf16

Script: [test_qwen3_tts_voice_clone_benchmark.py](test_qwen3_tts_voice_clone_benchmark.py)

Run:

```bash
cd /home/kawshik/tts-testing/Qwen3-TTS
conda activate qwen3-tts
export CUDA_VISIBLE_DEVICES=1

python test_qwen3_tts_voice_clone_benchmark.py \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio ref_audio.wav \
  --ref-text "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed." \
  --dtype bf16 \
  --output-dir outputs/qwen3_tts_voice_clone_benchmark
```

To run other datatypes (like fp32/fp16), change `--dtype bf16` to `--dtype fp32` or `--dtype fp16`.

Documentation/archive folder:

```text
outputs/qwen3_tts_voice_clone_benchmark/
```

Expected important files:

```text
outputs/qwen3_tts_voice_clone_benchmark/summary.txt
outputs/qwen3_tts_voice_clone_benchmark/results.json
outputs/qwen3_tts_voice_clone_benchmark/*.wav
```

Successful bf16 clone benchmark result:

[Benchmark Summary](outputs/qwen3_tts_voice_clone_benchmark/summary.txt)

Important interpretation:

```text
RTF = generation_time / generated_audio_duration
Lower is better.
RTF < 1.0 means faster than real-time.
Throughput = generated_audio_duration / generation_time
Higher is better.
```

---

## 9. CustomVoice Benchmark Suite: SDPA vs Flash Attention 2

Script: [test_qwen3_tts_custom_voice_benchmark.py](test_qwen3_tts_custom_voice_benchmark.py)

Run for SDPA (Standard Attention):

```bash
cd /home/kawshik/tts-testing/Qwen3-TTS
conda activate qwen3-tts
export CUDA_VISIBLE_DEVICES=1

python test_qwen3_tts_custom_voice_benchmark.py \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Ryan \
  --dtype bfloat16 \
  --attn sdpa \
  --output-dir outputs/benchmark/qwen3_0p6b_custom_voice_en
```

Run for Flash Attention 2:

```bash
cd /home/kawshik/tts-testing/Qwen3-TTS
conda activate qwen3-tts
export CUDA_VISIBLE_DEVICES=1

python test_qwen3_tts_custom_voice_benchmark.py \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Ryan \
  --dtype bfloat16 \
  --attn flash_attention_2 \
  --output-dir outputs/benchmark/qwen3_0p6b_custom_voice_en_flash_bf16
```

Attention implementations:

```text
sdpa             = PyTorch Standard Scaled Dot Product Attention
flash_attention_2 = Optimized FlashAttention-2 GPU kernel (faster & lower VRAM usage)
```

Model modes:

```text
custom = predefined speaker ID e.g. Ryan (0.6B CustomVoice model)
```

Must-have arguments:

```text
--model               = path to CustomVoice model (Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
--speaker             = speaker ID for custom mode (e.g., Ryan)
--dtype               = precision (bfloat16/float16/float32)
--attn                - attention implementation (flash_attention_2/sdpa/eager)
--output-dir          = output directory for results
```

Documentation/archive folder:

```text
outputs/benchmark/
```

Expected important files:

```text
outputs/benchmark/qwen3_0p6b_custom_voice_en/summary.txt
outputs/benchmark/qwen3_0p6b_custom_voice_en_flash_bf16/summary.txt
outputs/benchmark/qwen3_0p6b_custom_voice_en/*.wav
outputs/benchmark/qwen3_0p6b_custom_voice_en_flash_bf16/*.wav
```

Successful bf16 CustomVoice benchmark result:

[Benchmark Summary (Flash Attention 2)](outputs/benchmark/qwen3_0p6b_custom_voice_en_flash_bf16/summary.txt)
[Benchmark Summary (SDPA Attention)](outputs/benchmark/qwen3_0p6b_custom_voice_en/summary.txt)

Important interpretation:

```text
RTF = generation_time / generated_audio_duration
Lower is better.
RTF < 1.0 means faster than real-time.
Throughput = generated_audio_duration / generation_time
Higher is better.
```
---

## Prepared By

**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com
