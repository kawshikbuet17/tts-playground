# IndicF5 Execution Documentation

**IndicF5** is a multilingual Indian-language text-to-speech system from AI4Bharat. This document records the complete Linux server setup, execution flow, troubleshooting, successful Bengali benchmark result, and output storage pattern for IndicF5.

**Official Repository:**

```text
https://github.com/AI4Bharat/IndicF5
```

---

## 1. System Assumptions

Tested environment:

```text
Server OS: Linux
Project path: ~/tts-testing/IndicF5
Python env: conda environment named indicf5
Python: 3.10
GPU selected for test: NVIDIA GeForce RTX 3090
CUDA visible device: 1
Model: ai4bharat/IndicF5
Reference audio: prompts/PAN_F_HAPPY_00001.wav
Audio sample rate: 24000 Hz
Runtime precision used for final benchmark: FP32
Benchmark language: Bengali
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

git clone https://github.com/AI4Bharat/IndicF5.git
cd IndicF5
```

Create output/cache folders:

```bash
mkdir -p ./outputs ./outputs/benchmark ./samples ./logs ./debug_outputs
```

---

## 3. Create and Activate Python Environment

Create a fresh conda environment:

```bash
conda create -n indicf5 python=3.10 -y
conda activate indicf5
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
PY
```

Expected successful GPU check:

```text
cuda available: True
cuda device count: 1
visible gpu: NVIDIA GeForce RTX 3090
```

---

## 5. Install IndicF5 Dependencies

The project uses the local repository and the maintained [`requirements.txt`](requirements.txt).

Install:

```bash
cd ~/tts-testing/IndicF5
conda activate indicf5

pip install -r requirements.txt
```

If `requirements.txt` is updated later, reinstall/update the environment using:

```bash
pip install -r requirements.txt --upgrade
```

---

## 6. Verify Python Imports

Run:

```bash
python - <<'PY'
import sys
import torch
import transformers
import soundfile
import numpy as np

print("python:", sys.version)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("numpy:", np.__version__)
print("soundfile import OK")
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("visible gpu:", torch.cuda.get_device_name(0))
PY
```

Expected:

```text
soundfile import OK
cuda available: True
visible gpu: NVIDIA GeForce RTX 3090
```

---

## 7. Hugging Face Model Access

The model is loaded from:

```text
ai4bharat/IndicF5
```

Login if needed:

```bash
huggingface-cli login
huggingface-cli whoami
```

Verify access:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="ai4bharat/IndicF5",
    filename="config.json",
)

print("Access OK:", path)
PY
```

---

## 8. Reference Audio Setup

IndicF5 inference uses:

```text
1. Text to synthesize
2. Reference prompt audio
3. Transcript of the reference prompt audio
```

Reference audio used in testing:

```text
prompts/PAN_F_HAPPY_00001.wav
```

Reference transcript used in the benchmark script:

```text
ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ ਹਨ।
```

Check reference audio:

```bash
ls -lh prompts/PAN_F_HAPPY_00001.wav
```

Recommended reference audio properties:

```text
Format: WAV
Speech: clean, single speaker
Noise: low background noise
Transcript: should match the spoken reference audio
```

---

## 9. Working Benchmark Script

File: [`test_indicf5_benchmark.py`](test_indicf5_benchmark.py)

Purpose:

```text
Run a Bengali-only FP32 benchmark for IndicF5.
The script loads the model once, runs 3 warmup generations, then generates Bengali audio samples from very short to long text.
It measures inference time, post-processing time, total generation time, audio duration, RTF, audio peak, RMS, and silent-output validation.
```

Benchmark text design:

```text
Language: Bengali only
Warmup runs: 3
Benchmark items: 12
Target generated audio range: approximately 2s to 30s
Precision: FP32
```

Run:

```bash
cd ~/tts-testing/IndicF5
conda activate indicf5

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

python test_indicf5_benchmark.py
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
outputs/benchmark/bench_01_bn_very_short_01.wav
outputs/benchmark/bench_02_bn_very_short_02.wav
outputs/benchmark/bench_03_bn_very_short_03.wav
outputs/benchmark/bench_04_bn_short_01.wav
outputs/benchmark/bench_05_bn_short_02.wav
outputs/benchmark/bench_06_bn_short_03.wav
outputs/benchmark/bench_07_bn_medium_01.wav
outputs/benchmark/bench_08_bn_medium_02.wav
outputs/benchmark/bench_09_bn_medium_03.wav
outputs/benchmark/bench_10_bn_long_01.wav
outputs/benchmark/bench_11_bn_long_02.wav
outputs/benchmark/bench_12_bn_long_03.wav
outputs/benchmark/summary.txt
```

Recommended files to listen after a benchmark run:

```text
outputs/benchmark/bench_01_bn_very_short_01.wav
outputs/benchmark/bench_04_bn_short_01.wav
outputs/benchmark/bench_07_bn_medium_01.wav
outputs/benchmark/bench_10_bn_long_01.wav
outputs/benchmark/bench_12_bn_long_03.wav
```

---

## 10. Final Bengali FP32 Benchmark Result

Summary file: [`summary.txt`](summary.txt)

Successful run summary:

```text
Status: SUCCESS
Precision: FP32
Language: Bengali
Target audio range: 2s to 30s
Model: ai4bharat/IndicF5
Sample rate: 24000 Hz
GPU: NVIDIA GeForce RTX 3090
Model load time: 8.06 sec
Warmup runs: 3
Warmup silent items: 0
Benchmark items: 12
Silent items: 0
Benchmark wall time: 62.96 sec
Total generation time: 62.96 sec
Total inference time: 62.10 sec
Total postprocess time: 0.86 sec
Total audio duration: 120.70 sec
Average generation time: 5.25 sec
Average inference time: 5.18 sec
Average postprocess time: 0.07 sec
Average audio duration: 10.06 sec
Average text chars: 122.9
Average peak: 0.94971212
Average RMS: 0.09996591
Minimum peak: 0.69686890
Minimum RMS: 0.09959835
Effective RTF: 0.522
Inference-only RTF: 0.515
Wall-clock RTF: 0.522
Average per-item RTF: 0.764
```


Range-wise result:

```text
15-30s: items=3, silent=0, audio=66.10s, generation=31.15s, effective_rtf=0.471, avg_rtf=0.471
2-4s:   items=3, silent=0, audio=4.81s,  generation=7.04s,  effective_rtf=1.464, avg_rtf=1.500
4-8s:   items=3, silent=0, audio=13.09s, generation=8.20s,  effective_rtf=0.627, avg_rtf=0.633
8-15s:  items=3, silent=0, audio=36.70s, generation=16.56s, effective_rtf=0.451, avg_rtf=0.451
```

Important observation:

```text
Very short clips are slower because fixed inference overhead dominates.
Medium and long Bengali clips stabilize around 0.45 to 0.47 RTF.
```

---

## 11. Generated Audio Storage

Main Bengali benchmark outputs:

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


---

## 12. Precision Notes

FP32 was used for the final benchmark because it produced valid, non-silent audio.

FP16 autocast was tested during optimization, but generated muted/silent audio in this setup. Therefore it was not used for the final benchmark.

Full half precision conversion was also not used because the model path produced a dtype mismatch.

Final runtime decision:

```text
Use FP32 inference.
Do not use FP16 autocast for this setup.
Do not use full model.half() for this setup.
```


---

## Prepared By

**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com
