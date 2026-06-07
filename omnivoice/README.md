# OmniVoice Execution Documentation

**OmniVoice** is a multilingual TTS system from k2-fsa. This document records the Linux server setup, execution flow, smoke test, full benchmark command, benchmark summary, and output storage pattern for OmniVoice.

**Repository:**

```text
https://github.com/k2-fsa/OmniVoice
```

---

## 1. System Assumptions

Tested environment:

```text
Server OS: Linux
Project path: /home/kawshik/tts-testing/OmniVoice
Python env: conda environment named omnivoice
Python: 3.10.20
GPU selected for test: NVIDIA GeForce RTX 3090
CUDA visible device: 1
Runtime logical device inside Python: cuda:0
Model family: OmniVoice
Primary language tested: Bengali / Bangla
Code-mix tested: Bengali + English
Audio sample rate: 24000 Hz
Torch: 2.8.0+cu128
CUDA: 12.8
Runtime device: cuda
Stable dtype for benchmark: fp16
```

---

## 2. Go to Project Folder

From the server:

```bash
cd /home/kawshik/tts-testing/OmniVoice
```

If starting from scratch:

```bash
cd /home/kawshik/tts-testing
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
```

---

## 3. Create and Activate Python Environment

Create a fresh conda environment:

```bash
conda create -n omnivoice python=3.10 -y
conda activate omnivoice
```

Upgrade base Python tooling:

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
cuda available: True
visible gpu: NVIDIA GeForce RTX 3090
Runtime logical device inside Python: cuda:0
```

---

## 5. Install PyTorch

Install CUDA 12.8 PyTorch:

```bash
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

Expected:

```text
torch: 2.8.0+cu128
cuda available: True
cuda: 12.8
gpu: NVIDIA GeForce RTX 3090
```

---

## 6. Install OmniVoice

From the local repository:

```bash
cd /home/kawshik/tts-testing/OmniVoice
conda activate omnivoice
pip install -e .
```

Check import:

```bash
python - <<'PY'
from omnivoice import OmniVoice
print("OmniVoice import OK")
PY
```

Expected:

```text
OmniVoice import OK
```

---

## 7. First Working Smoke Test

[test_omnivoice_smoke.py](test_omnivoice_smoke.py)

Run:

```bash
cd /home/kawshik/tts-testing/OmniVoice
conda activate omnivoice

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

CUDA_VISIBLE_DEVICES=1 python test_omnivoice_smoke.py
```

Successful result:

```text
Fetching 13 files: 100%
Download complete: 3.27G/3.27G
Loading weights: 100%
saved smoke_omnivoice.wav
```

Generated output:

```text
smoke_omnivoice.wav
```

---

## 8. Full Benchmark Suite: fp16

[test_omnivoice_full_benchmark.py](test_omnivoice_full_benchmark.py)

Run:

```bash
cd /home/kawshik/tts-testing/OmniVoice
conda activate omnivoice

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

CUDA_VISIBLE_DEVICES=1 python test_omnivoice_full_benchmark.py \
  --suite all \
  --model k2-fsa/OmniVoice \
  --ref-audio ref_audio.wav \
  --ref-text "বসের নির্দেশ, অন্য কাগজের কাছে খবরটা যাওয়ার আগেই আমাদেরকে কোনোভাবে স্পটে পৌছাতে হবে। অগত্যা তাই যেতেই হলো।" \
  --instruct "female, young adult" \
  --dtype fp16 \
  --num-step 16 \
  --speed 1.0 \
  --output-dir outputs/omnivoice_full_benchmark_all_step16
```

Suite types:

```text
--suite all    = run auto, design, and clone modes
--suite auto   = run auto mode only
--suite design = run voice design mode only
--suite clone  = run voice cloning mode only
```

Model modes:

```text
auto   = generate speech directly from text
design = generate speech from text + voice instruction
clone  = generate speech using reference audio + reference transcript
```

Must-have arguments:

```text
--suite      = benchmark suite type
--model      = OmniVoice model name or local model path
--ref-audio  = reference audio path for clone mode
--ref-text   = reference transcript for clone mode
--instruct   = valid voice design instruction for design mode
--dtype      = precision, e.g. fp16
--num-step   = generation step count
--speed      = speech speed
--output-dir = output directory for results
```

Valid example design instructions:

```text
female
female, young adult
female, young adult, high pitch
male, elderly, low pitch, whisper
female, young adult, high pitch, british accent
```

Important design instruction note:
```text
Check supported voice design attributes in:
https://github.com/k2-fsa/OmniVoice/blob/master/docs/voice-design.md

Only use attributes listed in that file.
```
Documentation/archive folder:

```text
outputs/omnivoice_full_benchmark_all_step16/
```

Expected important files:

```text
outputs/omnivoice_full_benchmark_all_step16/summary.txt
outputs/omnivoice_full_benchmark_all_step16/results.json
outputs/omnivoice_full_benchmark_all_step16/auto/*.wav
outputs/omnivoice_full_benchmark_all_step16/design/*.wav
outputs/omnivoice_full_benchmark_all_step16/clone/*.wav
```

---

## 9. Benchmark Summary Observed

Observed result on RTX 3090 with `fp16`, `num_step=16`, and `--suite all`:

[Benchmark Summary](outputs/omnivoice_full_benchmark_all_step16/summary.txt)

Important interpretation:

```text
RTF = generation_time / generated_audio_duration
Lower is better.
RTF < 1.0 means faster than real-time.
Throughput = generated_audio_duration / generation_time
Higher is better.
```

---

## 10. Run Auto Mode Only

```bash
CUDA_VISIBLE_DEVICES=1 python test_omnivoice_full_benchmark.py \
  --suite auto \
  --model k2-fsa/OmniVoice \
  --dtype fp16 \
  --num-step 16 \
  --speed 1.0 \
  --output-dir outputs/omnivoice_benchmark_auto_step16
```

---

## 11. Run Design Mode Only

```bash
CUDA_VISIBLE_DEVICES=1 python test_omnivoice_full_benchmark.py \
  --suite design \
  --model k2-fsa/OmniVoice \
  --instruct "female, young adult" \
  --dtype fp16 \
  --num-step 16 \
  --speed 1.0 \
  --output-dir outputs/omnivoice_benchmark_design_step16
```

---

## 12. Run Clone Mode Only

```bash
CUDA_VISIBLE_DEVICES=1 python test_omnivoice_full_benchmark.py \
  --suite clone \
  --model k2-fsa/OmniVoice \
  --ref-audio ref_audio.wav \
  --ref-text "বসের নির্দেশ, অন্য কাগজের কাছে খবরটা যাওয়ার আগেই আমাদেরকে কোনোভাবে স্পটে পৌছাতে হবে। অগত্যা তাই যেতেই হলো।" \
  --dtype fp16 \
  --num-step 16 \
  --speed 1.0 \
  --output-dir outputs/omnivoice_benchmark_clone_step16
```

---

## 13. Precision and Parameter Notes

Current tested precision:

```text
fp16: working and stable in this setup
bf16: working better, providing better rtf
```

Main speed-quality parameter:

```text
num_step=4 = very fast but not a good quality
num_step=8 = faster than 16 and moderate quality
num_step=16 = faster than 32
num_step=32 = potentially better quality but slower
```
---

## Prepared By

**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com
