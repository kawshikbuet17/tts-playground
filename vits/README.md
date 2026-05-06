# VITS Execution Documentation

**VITS** is a neural text-to-speech system based on a conditional variational autoencoder with adversarial learning. This document records the complete Linux server setup, execution flow, troubleshooting, successful LJ Speech inference result, benchmark result, and output storage pattern for VITS.

**Official Repository:**

```text
https://github.com/jaywalnut310/vits
```

---

## 1. System Assumptions

Tested environment:

```text
Server OS: Linux
Project path: ~/tts-testing/vits
Python env: conda environment named vits
Python: 3.8
GPU selected for test: NVIDIA GeForce RTX 3090
CUDA visible device: 1
Model family: VITS
Checkpoint: checkpoints/pretrained_drive/pretrained_ljs.pth
Config: configs/ljs_base.json
Dataset style: LJ Speech
Language: English
Audio sample rate: 22050 Hz
Torch: 2.2.0+cu121
Runtime device: cuda
```

Important GPU mapping note:

```text
If CUDA_VISIBLE_DEVICES=1 is set, physical GPU 1 becomes logical cuda:0 inside Python.
So seeing cuda:0 in logs is expected.
```

Stable GPU ordering was set using:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

---

## 2. Go to Project Folder

From the server:

```bash
cd ~/tts-testing

git clone https://github.com/jaywalnut310/vits.git
cd vits
```

---

## 3. Create and Activate Python Environment

Create a fresh conda environment:

```bash
conda create -n vits python=3.8 -y
conda activate vits
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

Verify:

```bash
echo $CUDA_VISIBLE_DEVICES
```

Expected:

```text
1
```

Verify from Python:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("visible gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

Expected successful GPU check:

```text
cuda available: True
cuda device count: 1
visible gpu: NVIDIA GeForce RTX 3090
capability: (8, 6)
```

---

## 5. Install VITS Dependencies

The project uses the local repository and the maintained [`requirements.txt`](requirements.txt).

Install:

```bash
cd ~/tts-testing/vits
conda activate vits

pip install -r requirements.txt
```

If [`requirements.txt`](requirements.txt) is updated later, reinstall/update the environment using:

```bash
pip install -r requirements.txt --upgrade
```

---

## 6. Torch Version Fix for RTX 3090

The original VITS requirements installed an old PyTorch version:

```text
torch==1.6.0
torchvision==0.7.0
```

This version detected the RTX 3090 but showed an architecture compatibility warning:

```text
NVIDIA GeForce RTX 3090 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75.
```

Therefore, PyTorch was upgraded to a CUDA 12.1 build:

```bash
pip uninstall -y torch torchvision torchaudio

pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA matmul:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("cuda matmul OK:", y.shape)
PY
```

Successful result:

```text
torch: 2.2.0+cu121
cuda available: True
cuda device count: 1
gpu: NVIDIA GeForce RTX 3090
capability: (8, 6)
cuda matmul OK: torch.Size([1024, 1024])
```

---

## 7. Build Monotonic Alignment Extension

VITS uses a compiled monotonic alignment extension.

Initial direct build from inside `monotonic_align` failed because the setup script expected an output folder path that did not exist.

Working build sequence:

```bash
cd ~/tts-testing/vits
conda activate vits

rm -rf build
rm -rf monotonic_align/build
rm -rf monotonic_align/monotonic_align
rm -f monotonic_align/*.c
rm -f monotonic_align/*.so

mkdir -p monotonic_align/monotonic_align

cd monotonic_align
python setup.py build_ext --inplace
cd ..
```

Verify compiled files:

```bash
find monotonic_align -name "*.so" -type f -print
```

Successful result:

```text
monotonic_align/build/lib.linux-x86_64-cpython-38/monotonic_align/core.cpython-38-x86_64-linux-gnu.so
monotonic_align/monotonic_align/core.cpython-38-x86_64-linux-gnu.so
```

Copy compiled extension to the top-level import folder:

```bash
cp monotonic_align/monotonic_align/core*.so monotonic_align/
```

Verify import:

```bash
python - <<'PY'
import monotonic_align
from monotonic_align import maximum_path

print("monotonic_align import OK")
print("maximum_path:", maximum_path)
PY
```

Expected:

```text
monotonic_align import OK
maximum_path: <function maximum_path ...>
```

---

## 8. Verify Python Imports

Run:

```bash
cd ~/tts-testing/vits
conda activate vits

python - <<'PY'
import torch
import numpy as np
import scipy
import librosa
import soundfile as sf

import commons
import utils
import models
import text
import monotonic_align
from monotonic_align import maximum_path

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("librosa:", librosa.__version__)
print("soundfile OK")
print("vits imports OK")
print("monotonic_align OK")
print("maximum_path:", maximum_path)
PY
```

Successful result:

```text
torch: 2.2.0+cu121
cuda: True
gpu: NVIDIA GeForce RTX 3090
numpy: 1.18.5
scipy: 1.5.2
librosa: 0.8.0
soundfile OK
vits imports OK
monotonic_align OK
maximum_path: <function maximum_path ...>
```

---

## 9. Download Official Pretrained Checkpoints

The official VITS pretrained checkpoints are hosted on Google Drive.

Install `gdown`:

```bash
pip install gdown
```

Download pretrained checkpoint folder:

```bash
cd ~/tts-testing/vits

gdown --folder "https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing" \
  -O checkpoints/pretrained_drive
```

Downloaded files:

```text
checkpoints/pretrained_drive/pretrained_ljs.pth
checkpoints/pretrained_drive/pretrained_vctk.pth
```

For this benchmark, the LJ Speech checkpoint was used:

```text
checkpoints/pretrained_drive/pretrained_ljs.pth
```

Optional cleaner checkpoint copy:

```bash
mkdir -p checkpoints/ljs_base
cp checkpoints/pretrained_drive/pretrained_ljs.pth checkpoints/ljs_base/pretrained_ljs.pth
```

---

## 10. Confirm LJ Speech Config

Config file:

```text
configs/ljs_base.json
```

Check:

```bash
ls -lh configs/ljs_base.json
```

Inspect important values:

```bash
python - <<'PY'
import json

config_path = "configs/ljs_base.json"

with open(config_path, "r", encoding="utf-8") as f:
    c = json.load(f)

print("sampling_rate:", c["data"]["sampling_rate"])
print("filter_length:", c["data"]["filter_length"])
print("hop_length:", c["data"]["hop_length"])
print("win_length:", c["data"]["win_length"])
print("text_cleaners:", c["data"]["text_cleaners"])
print("add_blank:", c["data"]["add_blank"])
print("n_speakers:", c["data"].get("n_speakers"))
PY
```

Successful result:

```text
sampling_rate: 22050
filter_length: 1024
hop_length: 256
win_length: 1024
text_cleaners: ['english_cleaners2']
add_blank: True
n_speakers: 0
```

Important note:

```text
This LJ Speech checkpoint is English-only.
It should not be used for Bengali evaluation.
For Bengali VITS, a Bengali-compatible checkpoint, config, symbols, and text cleaner are required.
```

---

## 11. Working Single-Inference Script

File: [`infer_ljs.py`](infer_ljs.py)

Purpose:

```text
Run one English LJ Speech VITS inference using the official pretrained_ljs checkpoint.
The script loads the model, converts input text to sequence, generates audio, saves WAV output, and reports timing/audio statistics.
```

Important implementation note:

```text
In this VITS repo, symbols are not stored inside hps.
So the script imports symbols using:

from text.symbols import symbols

and initializes SynthesizerTrn with:

len(symbols)
```

Run:

```bash
cd ~/tts-testing/vits
conda activate vits

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

python infer_ljs.py \
  --config configs/ljs_base.json \
  --checkpoint checkpoints/pretrained_drive/pretrained_ljs.pth \
  --text "This is a test of the VITS text to speech system." \
  --out outputs/test_ljs.wav
```

Successful single-inference results:

Run 1:

```text
device: cuda
sampling_rate: 22050
output: outputs/test_ljs.wav
elapsed_sec: 1.600
audio_duration_sec: 3.204
RTF: 0.499
peak: 0.40953934
rms: 0.05263351
silent: False
```

Run 2:

```text
device: cuda
sampling_rate: 22050
output: outputs/test_ljs.wav
elapsed_sec: 0.901
audio_duration_sec: 3.274
RTF: 0.275
peak: 0.62824976
rms: 0.06051065
silent: False
```

Run 3:

```text
device: cuda
sampling_rate: 22050
output: outputs/test_ljs.wav
elapsed_sec: 1.045
audio_duration_sec: 3.239
RTF: 0.323
peak: 0.37964833
rms: 0.05354407
silent: False
```

Generated output:

```text
outputs/test_ljs.wav
```

---

## 12. Working Benchmark Script

File: [`test_vits_ljs_benchmark.py`](test_vits_ljs_benchmark.py)

Purpose:

```text
Run an English-only benchmark for the official pretrained LJ Speech VITS model.
The script loads the model once, runs 3 warmup generations, then generates English audio samples from very short to long text.
It measures inference time, post-processing time, total generation time, audio duration, RTF, audio peak, RMS, and silent-output validation.
```

Benchmark text design:

```text
Language: English only
Warmup runs: 3
Benchmark items: 12
Target generated audio range: approximately 2s to 30s
Checkpoint: pretrained_ljs.pth
Config: ljs_base.json
Sample rate: 22050 Hz
```

Run:

```bash
cd ~/tts-testing/vits
conda activate vits

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

python test_vits_ljs_benchmark.py \
  --config configs/ljs_base.json \
  --checkpoint checkpoints/pretrained_drive/pretrained_ljs.pth \
  --output-dir outputs/benchmark
```

Output directory:

```text
outputs/benchmark/
```

Generated output samples:

```text
outputs/benchmark/warmup_1.wav
outputs/benchmark/warmup_2.wav
outputs/benchmark/warmup_3.wav
outputs/benchmark/bench_01_en_very_short_01.wav
outputs/benchmark/bench_02_en_very_short_02.wav
outputs/benchmark/bench_03_en_very_short_03.wav
outputs/benchmark/bench_04_en_short_01.wav
outputs/benchmark/bench_05_en_short_02.wav
outputs/benchmark/bench_06_en_short_03.wav
outputs/benchmark/bench_07_en_medium_01.wav
outputs/benchmark/bench_08_en_medium_02.wav
outputs/benchmark/bench_09_en_medium_03.wav
outputs/benchmark/bench_10_en_long_01.wav
outputs/benchmark/bench_11_en_long_02.wav
outputs/benchmark/bench_12_en_long_03.wav
outputs/benchmark/summary.txt
```

Recommended files to listen after a benchmark run:

```text
outputs/benchmark/bench_01_en_very_short_01.wav
outputs/benchmark/bench_04_en_short_01.wav
outputs/benchmark/bench_07_en_medium_01.wav
outputs/benchmark/bench_10_en_long_01.wav
outputs/benchmark/bench_12_en_long_03.wav
```

---

## 13. Final LJ Speech Benchmark Result

Summary file: [`outputs/benchmark/summary.txt`](outputs/benchmark/summary.txt)

Successful run summary:

```text
status: SUCCESS
model_family: VITS
checkpoint: checkpoints/pretrained_drive/pretrained_ljs.pth
config: configs/ljs_base.json
language: English
dataset_style: LJ Speech
sample_rate: 22050
device: cuda
gpu: NVIDIA GeForce RTX 3090
torch: 2.2.0+cu121
noise_scale: 0.667
noise_scale_w: 0.8
length_scale: 1.0
model_load_time_sec: 0.90
warmup_runs: 10
warmup_silent_items: 0
benchmark_items: 12
silent_items: 0
benchmark_wall_time_sec: 8.14
total_generation_time_sec: 4.01
total_inference_time_sec: 3.07
total_postprocess_time_sec: 0.94
total_audio_duration_sec: 102.86
average_generation_time_sec: 0.33
average_inference_time_sec: 0.26
average_postprocess_time_sec: 0.08
average_audio_duration_sec: 8.57
average_text_chars: 123.4
average_peak: 0.56449918
average_rms: 0.06637839
min_peak: 0.42001870
min_rms: 0.05135309
effective_rtf: 0.039
inference_only_rtf: 0.030
wall_clock_rtf: 0.079
average_per_item_rtf: 0.074
```

Fastest item:

```text
item 12 | label=en_long_03 | range=15-30s | RTF=0.018 | audio=18.99s | peak=0.58393854 | rms=0.06149209 | silent=False
```

Slowest item:

```text
item 1 | label=en_very_short_01 | range=2-4s | RTF=0.179 | audio=1.46s | peak=0.42001870 | rms=0.06902470 | silent=False
```

Range-wise result:

```text
15-30s: items=3, silent=0, audio=52.74s, generation=1.13s, effective_rtf=0.021, avg_rtf=0.022
2-4s:   items=3, silent=0, audio=4.75s,  generation=0.80s, effective_rtf=0.169, avg_rtf=0.170
4-8s:   items=3, silent=0, audio=13.04s, generation=0.86s, effective_rtf=0.066, avg_rtf=0.067
8-15s:  items=3, silent=0, audio=32.33s, generation=1.21s, effective_rtf=0.038, avg_rtf=0.038
```

Important observation:

```text
Very short clips are slower because fixed inference/post-processing overhead dominates.
Medium and long clips produce much better effective RTF because the overhead is amortized over longer generated audio.
```

The benchmark produced valid audio for every item:

```text
warmup_silent_items: 0
silent_items: 0
min_peak: 0.42001870
min_rms: 0.05135309
```

---

## 14. Generated Audio Storage

Main benchmark outputs:

```text
outputs/benchmark/
```

Summary file:

```text
outputs/benchmark/summary.txt
```

The summary file contains:

```text
Warmup file timings
Benchmark file timings
Audio duration
RTF
Audio peak
Audio RMS
Silent-output validation
Range-wise summary
```

Single inference output:

```text
outputs/test_ljs.wav
```

---

## 15. Warning Notes

During inference and benchmark, PyTorch prints this warning:

```text
torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.
```

This is not an execution failure.

Explanation:

```text
The original VITS code uses torch.nn.utils.weight_norm.
PyTorch 2.x keeps it functional but marks it deprecated.
Inference and benchmark completed successfully despite the warning.
```

---

## 16. Precision and Runtime Notes

Final runtime decision:

```text
Use torch 2.2.0+cu121 for RTX 3090 compatibility.
Use CUDA inference.
Do not use the original torch 1.6.0 package on RTX 3090 because it lacks sm_86 compatibility.
```

VITS inference uses stochastic noise parameters:

```text
noise_scale: 0.667
noise_scale_w: 0.8
length_scale: 1.0
```

Therefore, repeated runs can produce slightly different audio duration, peak, RMS, and RTF values.

---

## 17. Bengali VITS Note

The current successful setup is for the official English LJ Speech pretrained model.

Current checkpoint/config pair:

```text
checkpoint: checkpoints/pretrained_drive/pretrained_ljs.pth
config: configs/ljs_base.json
text_cleaners: ['english_cleaners2']
language: English
```

For Bengali VITS execution, the following are required:

```text
Bengali-compatible VITS checkpoint
matching config JSON
matching symbols
matching Bengali text cleaner or phonemizer/G2P pipeline
correct sample rate
speaker ID handling if the checkpoint is multi-speaker
```

Do not evaluate Bengali text using the LJ Speech checkpoint, because it is English-only.

---

## 18. Files to Keep

The following files should be preserved for reproducibility:

```text
[requirements.txt](requirements.txt)
[infer_ljs.py](infer_ljs.py)
[test_vits_ljs_benchmark.py](test_vits_ljs_benchmark.py)
[outputs/benchmark/summary.txt](outputs/benchmark/summary.txt)
```

Recommended optional files to keep:

```text
outputs/test_ljs.wav
outputs/benchmark/*.wav
checkpoints/pretrained_drive/pretrained_ljs.pth
configs/ljs_base.json
```

---

## Prepared By

**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com
