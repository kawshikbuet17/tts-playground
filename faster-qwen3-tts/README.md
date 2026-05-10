# Faster Qwen3-TTS Execution Documentation

**Faster Qwen3-TTS** is an optimized inference implementation for Qwen3-TTS. This document records the Linux server setup, execution flow, benchmark scripts, successful English inference results, bf16/fp32 benchmark comparison, full benchmark suite results, and output storage pattern for Faster Qwen3-TTS.

**Repository:**

```text
https://github.com/andimarafioti/faster-qwen3-tts
```

---

## 1. System Assumptions

Tested environment:

```text
Server OS: Linux
Project path: /home/kawshik/projects/faster-qwen3-tts
Python env: conda environment named faster_qwen3
Python: 3.10.20
GPU selected for test: NVIDIA GeForce RTX 3090
CUDA visible device: 1
Runtime logical device inside Python: cuda:0
Model family: Faster Qwen3-TTS
Primary language tested: English
Audio sample rate: 24000 Hz
Torch: 2.5.1+cu121
Runtime device: cuda
Stable dtype for benchmark: bf16
Stable fallback dtype: fp32
Unstable dtype observed: fp16 for streaming generation
```

Important GPU mapping note:

```text
If CUDA_VISIBLE_DEVICES=1 is set, physical GPU 1 becomes logical cuda:0 inside Python.
So seeing cuda:0 in logs is expected.
```

Stable GPU ordering can be set using:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

---

## 2. Go to Project Folder

From the server:

```bash
cd /home/kawshik/projects/faster-qwen3-tts
```

If starting from scratch:

```bash
cd /home/kawshik/projects

git clone https://github.com/andimarafioti/faster-qwen3-tts.git
cd faster-qwen3-tts
```

---

## 3. Create and Activate Python Environment

Create a fresh conda environment:

```bash
conda create -n faster_qwen3 python=3.10 -y
conda activate faster_qwen3
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
python: 3.10.20
python cuda available: True
torch: 2.5.1+cu121
cuda: 12.1
visible gpu: NVIDIA GeForce RTX 3090
capability: (8, 6)
```

---

## 5. Install Faster Qwen3-TTS Dependencies

From the local repository:

```bash
cd /home/kawshik/projects/faster-qwen3-tts
conda activate faster_qwen3

pip install -r [requirements.txt](requirements.txt)
```

For demo/API server dependencies:

```bash
pip install -e ".[demo]"
```

Check CLI:

```bash
which faster-qwen3-tts
faster-qwen3-tts --help
```

Successful CLI check:

```text
/opt/miniconda3/envs/faster_qwen3/bin/faster-qwen3-tts
usage: faster-qwen3-tts [-h] [--device DEVICE] [--dtype {bf16,fp16,fp32}] {clone,custom,design,serve} ...
```

Supported CLI modes:

```text
clone   - voice cloning using reference audio and transcript
custom  - predefined speaker ID / CustomVoice model
design  - instruction-based VoiceDesign model
serve   - keep model hot and generate multiple requests from stdin
```

---

## 6. Precision Decision

The following precision modes were tested:

```text
bf16: stable and fastest in this setup
fp32: stable fallback, slower than bf16
fp16: failed during streaming generation due to invalid probability tensor / CUDA device-side assert
```

Observed fp16 failure pattern:

```text
Assertion `probability tensor contains either `inf`, `nan` or element < 0` failed.
RuntimeError: CUDA error: device-side assert triggered
```

Final runtime decision:

```text
Use bf16 for normal Faster Qwen3-TTS experiments and deployment benchmarking.
Use fp32 only as a stable debugging/reference mode.
Avoid fp16 for the current streaming path unless generation code is modified.
```

---

## 7. First Working Single Inference: 0.6B Base Voice Clone

The first successful manual run used the included reference audio:

```text
ref_audio.wav
```

Run:

```bash
cd /home/kawshik/projects/faster-qwen3-tts
conda activate faster_qwen3
export CUDA_VISIBLE_DEVICES=1

faster-qwen3-tts --device cuda --dtype bf16 clone \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello, this is a quick test from faster Qwen three TTS." \
  --language English \
  --ref-audio ref_audio.wav \
  --ref-text "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed." \
  --output out_0_6b.wav \
  --streaming
```

Successful result:

```text
Warming up predictor (3 runs)...
Capturing CUDA graph for predictor...
CUDA graph captured!
Warming up talker graph (3 runs)...
Capturing CUDA graph for talker decode...
Talker CUDA graph captured!
Wrote out_0_6b.wav (dur 2.96s, RTF 0.07)
```

Generated output:

```text
out_0_6b.wav
```

Important observation:

```text
First run downloads model files and captures CUDA graphs.
Later hot-path generations are much faster.
```

---

## 8. Single Clone Benchmark: bf16

Script: [`test_faster_qwen3_tts_benchmark.py`](test_faster_qwen3_tts_benchmark.py)

Run:

```bash
cd /home/kawshik/projects/faster-qwen3-tts
conda activate faster_qwen3
export CUDA_VISIBLE_DEVICES=1

python test_faster_qwen3_tts_benchmark.py \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio ref_audio.wav \
  --ref-text "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed." \
  --dtype bf16 \
  --streaming \
  --output-dir outputs/faster_qwen3_tts_benchmark_bf16
```

To run fp32 instead, change `--dtype bf16` to `--dtype fp32` and update the output directory accordingly.

Documentation/archive folder:

```text
faster_qwen3_tts_benchmark_bf16/
```

Expected important files:

```text
faster_qwen3_tts_benchmark_bf16/summary.txt
faster_qwen3_tts_benchmark_bf16/results.json
faster_qwen3_tts_benchmark_bf16/*.wav
```

Successful bf16 clone benchmark result:

```text
status: SUCCESS
model_family: Faster Qwen3-TTS
model: Qwen/Qwen3-TTS-12Hz-0.6B-Base
mode: voice_clone_streaming
dtype: bf16
gpu: NVIDIA GeForce RTX 3090
sample_rate: 24000
warmup_runs: 10
benchmark_items: 12
silent_items: 0
total_audio_duration_sec: 82.00
total_generation_time_sec: 30.72
effective_rtf_elapsed_over_audio: 0.375
throughput_x_audio_over_elapsed: 2.67
average_ttfa_sec: 0.312
min_ttfa_sec: 0.306
max_ttfa_sec: 0.329
```

---

## 9. Full Benchmark Suite: bf16

Script: [`test_faster_qwen3_tts_full_benchmark.py`](test_faster_qwen3_tts_full_benchmark.py)

Run:

```bash
cd /home/kawshik/projects/faster-qwen3-tts
conda activate faster_qwen3
export CUDA_VISIBLE_DEVICES=1

python test_faster_qwen3_tts_full_benchmark.py \
  --suite all \
  --base-model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --custom-model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --design-model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --speaker aiden \
  --ref-audio ref_audio.wav \
  --ref-text "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed." \
  --dtype bf16 \
  --output-dir outputs/full_benchmark_all_bf16
```

To run fp32 instead, change `--dtype bf16` to `--dtype fp32` and update the output directory accordingly.

Suite types:

```text
--suite all           = run all 8 modes (clone, xvec, custom, design × streaming/non-streaming)
--suite clone         = run clone streaming and non-streaming only
--suite clone_xvec    = run x-vector clone streaming and non-streaming only
--suite custom        = run custom voice streaming and non-streaming only
--suite design        = run voice design streaming and non-streaming only
```

Model modes:

```text
clone  = voice clone using reference audio + transcript (0.6B Base model)
xvec   = clone using speaker embedding path (0.6B Base model)
custom = predefined speaker ID e.g. aiden (0.6B CustomVoice model)
design = create voice from text description (1.7B VoiceDesign model)
```

Must-have arguments:

```text
--suite               = benchmark suite type (all/clone/clone_xvec/custom/design)
--base-model          = path to Base model for clone/xvec modes
--custom-model        = path to CustomVoice model for custom mode
--design-model        = path to VoiceDesign model for design mode
--speaker             = speaker ID for custom mode (e.g., aiden)
--ref-audio           = reference audio path for clone modes
--ref-text            = reference transcript for clone modes
--dtype               = precision (bf16/fp32/fp16)
--output-dir          = output directory for results
```

Documentation/archive folder:

```text
full_benchmark_all_bf16/
```

Expected important files:

```text
full_benchmark_all_bf16/summary.txt
full_benchmark_all_bf16/results.json
full_benchmark_all_bf16/clone_streaming/*.wav
full_benchmark_all_bf16/clone_non_streaming/*.wav
full_benchmark_all_bf16/clone_xvec_streaming/*.wav
full_benchmark_all_bf16/clone_xvec_non_streaming/*.wav
full_benchmark_all_bf16/custom_streaming/*.wav
full_benchmark_all_bf16/custom_non_streaming/*.wav
full_benchmark_all_bf16/design_streaming/*.wav
full_benchmark_all_bf16/design_non_streaming/*.wav
```

Successful bf16 full-suite result:

```text
status: SUCCESS
modes_run: 8
items: 96
ok_items: 96
error_items: 0
silent_items: 0
total_audio_duration_sec: 757.36
total_generation_time_sec: 298.73
total_inference_time_sec: 256.68
total_postprocess_time_sec: 42.05
effective_rtf_elapsed_over_audio: 0.394
inference_only_rtf_elapsed_over_audio: 0.339
throughput_x_audio_over_elapsed: 2.535
average_ttfa_sec: 0.311
min_ttfa_sec: 0.276
max_ttfa_sec: 0.600
```

bf16 per-mode summary:

| Mode | Effective RTF | Throughput | Avg TTFA |
|---|---:|---:|---:|
| clone_streaming | 0.377 | 2.65x | 0.312s |
| clone_non_streaming | 0.331 | 3.02x | n/a |
| clone_xvec_streaming | 0.355 | 2.81x | 0.283s |
| clone_xvec_non_streaming | 0.321 | 3.11x | n/a |
| custom_streaming | 0.375 | 2.67x | 0.316s |
| custom_non_streaming | 0.335 | 2.98x | n/a |
| design_streaming | 0.608 | 1.64x | 0.331s |
| design_non_streaming | 0.434 | 2.31x | n/a |


---

## Prepared By

**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com
