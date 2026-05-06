# Spark-TTS Execution Documentation

**Spark-TTS** is an advanced text-to-speech system supporting voice cloning and voice creation with controllable attributes (gender, pitch, speed). It offers high-quality speech synthesis for Chinese and English with a lightweight 0.5B parameter model.

**Links:**
- Hugging Face: https://huggingface.co/SparkAudio/Spark-TTS-0.5B
- GitHub Repository: https://github.com/SparkAudio/Spark-TTS

---

A complete A-to-Z guide for setting up and running **Spark-TTS** for local TTS inference with:

- Local Python CLI inference
- Custom Gradio single-inference UI
- Voice cloning mode
- Voice creation mode
- Hugging Face model download
- GPU setup notes

> This documentation covers the setup up to the working custom Gradio UI: `spark_tts_custom_ui.py`.
> Triton/TensorRT-LLM runtime setup is intentionally not included here and should be documented separately.

## 1. System Assumptions

Tested environment:

```text
GPU: NVIDIA GeForce RTX 3090
CUDA visible device: 1
Python env: sparktts
Python: 3.12
PyTorch: 2.5.1+cu121
Transformers: 4.46.2
Gradio: working after Hugging Face Hub pin fix
```

Important GPU mapping note:

```text
If CUDA_VISIBLE_DEVICES=1 is set, physical GPU 1 becomes logical cuda:0 inside Python.
So seeing cuda:0 in PyTorch logs is expected and correct.
```

Important model note:

```text
Spark-TTS officially supports Chinese and English by default.
Bangla/Bengali is not officially supported by the default Spark-TTS checkpoint.
```

Important execution path note:

```text
This document covers local PyTorch/Python Spark-TTS inference.
It does not cover Triton/TensorRT-LLM optimized serving.
```

## 2. Clone Spark-TTS Repository

From `/home/kawshik/tts-testing`:

```bash
cd /home/kawshik/tts-testing
git clone https://github.com/SparkAudio/Spark-TTS.git
cd Spark-TTS
```

Create required folders:

```bash
mkdir -p outputs benchmark_outputs gradio_tmp pretrained_models prompt_wavs screenshots
```

## 3. Create and Activate Python Environment

Create a dedicated conda environment:

```bash
conda create -n sparktts python=3.12 -y
conda activate sparktts
```

Verify Python:

```bash
python --version
```

## 4. Select GPU

Use physical GPU 1:

```bash
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

Inside Python, this GPU will appear as `cuda:0`.

## 5. Install PyTorch CUDA Packages

Install/upgrade base packaging tools:

```bash
pip install --upgrade pip setuptools wheel
```

Install PyTorch with CUDA 12.1 wheels:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify PyTorch and GPU:

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

Expected output:

```text
torch: 2.5.1+cu121
cuda available: True
cuda device count: 1
visible gpu: NVIDIA GeForce RTX 3090
```

## 6. Install Spark-TTS Requirements

From the Spark-TTS repository root:

```bash
cd /home/kawshik/tts-testing/Spark-TTS
conda activate sparktts
pip install -r requirements.txt
```

## 7. Verify Basic Imports

Run:

```bash
python - <<'PY'
import torch
import transformers
import soundfile

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("basic imports OK")
PY
```

Example successful output:

```text
torch: 2.5.1+cu121
transformers: 4.46.2
cuda: True
gpu: NVIDIA GeForce RTX 3090
basic imports OK
```

## 8. Download Spark-TTS Model Checkpoint

Model:

```text
SparkAudio/Spark-TTS-0.5B
```

Download into the local project folder:

```bash
cd /home/kawshik/tts-testing/Spark-TTS
conda activate sparktts

python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path

model_dir = Path("pretrained_models/Spark-TTS-0.5B")

snapshot_download(
    repo_id="SparkAudio/Spark-TTS-0.5B",
    local_dir=str(model_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
)

print("Downloaded to:", model_dir.resolve())
PY
```

Check model files:

```bash
ls -lh pretrained_models/Spark-TTS-0.5B
du -sh pretrained_models/Spark-TTS-0.5B
```

Expected model path:

```text
/home/kawshik/tts-testing/Spark-TTS/pretrained_models/Spark-TTS-0.5B
```

Unlike Indic Parler-TTS, Spark-TTS model download did not require gated Hugging Face access.

---

## 9. Find Example Prompt Audio

Run:

```bash
cd /home/kawshik/tts-testing/Spark-TTS
find . -maxdepth 5 -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \)
```

Example useful prompt audio:

```text
./example/prompt_audio.wav
```

The default prompt audio used in this setup:

```text
example/prompt_audio.wav
```

---

## 10. Check CLI Help

Run:

```bash
python -m cli.inference --help
```

Expected options:

```text
--model_dir MODEL_DIR
--save_dir SAVE_DIR
--device DEVICE
--text TEXT
--prompt_text PROMPT_TEXT
--prompt_speech_path PROMPT_SPEECH_PATH
--gender {male,female}
--pitch {very_low,low,moderate,high,very_high}
--speed {very_low,low,moderate,high,very_high}
```

---

## 11. Run Official Example

Check official example script:

```bash
cat example/infer.sh
```

The official example uses:

```text
text="身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。"
prompt_text="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"
prompt_speech_path="example/prompt_audio.wav"
```

Run official example:

```bash
cd /home/kawshik/tts-testing/Spark-TTS
conda activate sparktts
export CUDA_VISIBLE_DEVICES=1
bash example/infer.sh
```

Example successful output:

```text
Using model from: pretrained_models/Spark-TTS-0.5B
Saving audio to: example/results
Using CUDA device: cuda:0
Starting inference...
Audio saved at: example/results/20260504050520.wav
```

Check outputs:

```bash
ls -lh example/results
find example/results -type f -name "*.wav" | tail -10
```

---

## 12. Run Custom English Voice-Cloning Inference

Use the Chinese prompt audio and its correct Chinese transcript, but generate English target text:

```bash
cd /home/kawshik/tts-testing/Spark-TTS
conda activate sparktts
export CUDA_VISIBLE_DEVICES=1

python -m cli.inference \
  --text "Today is a beautiful day. I am testing Spark TTS voice cloning." \
  --device 0 \
  --save_dir "outputs" \
  --model_dir "pretrained_models/Spark-TTS-0.5B" \
  --prompt_text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
  --prompt_speech_path "example/prompt_audio.wav"
```

Example successful output:

```text
Using model from: pretrained_models/Spark-TTS-0.5B
Saving audio to: outputs
Using CUDA device: cuda:0
Starting inference...
Audio saved at: outputs/20260504050606.wav
```

Check:

```bash
ls -lh outputs
find outputs -type f -name "*.wav" | tail -10
```

---

## 13. Important Voice-Cloning Note

For voice cloning mode:

```text
prompt_text must match the transcript of prompt_speech_path.
```

A wrong transcript may cause failed or unstable inference.

Example failed case:

```bash
--prompt_text "This is a prompt audio for testing."
--prompt_speech_path "example/prompt_audio.wav"
```

This failed because `example/prompt_audio.wav` is Chinese, but the prompt text was English and did not match the audio.

Correct prompt transcript:

```text
吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。
```

---

## 14. Test Voice Creation Mode

Spark-TTS also supports voice creation using gender, pitch, and speed controls.

Run:

```bash
cd /home/kawshik/tts-testing/Spark-TTS
conda activate sparktts
export CUDA_VISIBLE_DEVICES=1

python -m cli.inference \
  --text "Hello, this is Spark TTS speaking with a generated male voice." \
  --device 0 \
  --save_dir "outputs" \
  --model_dir "pretrained_models/Spark-TTS-0.5B" \
  --gender male \
  --pitch moderate \
  --speed moderate
```

Check output:

```bash
ls -lh outputs
```

---

## 15. Optional Bangla Test

Spark-TTS does not officially support Bangla by default. This can be tested only as an experiment:

```bash
cd /home/kawshik/tts-testing/Spark-TTS
conda activate sparktts
export CUDA_VISIBLE_DEVICES=1

python -m cli.inference \
  --text "আজকের আবহাওয়া খুব সুন্দর। আমি বাংলা টেক্সট টু স্পিচ পরীক্ষা করছি।" \
  --device 0 \
  --save_dir "outputs" \
  --model_dir "pretrained_models/Spark-TTS-0.5B" \
  --gender male \
  --pitch moderate \
  --speed moderate
```

Expected:

```text
The command may generate audio, but pronunciation/quality is not guaranteed.
```

---

## 16. Gradio Package Compatibility Fix

The official `webui.py` and the first custom Gradio UI attempts hit a Gradio/FastAPI schema issue:

```text
TypeError: argument of type 'bool' is not iterable
```

The Gradio stack was fixed by pinning compatible versions:

```bash
pip install --force-reinstall \
  "gradio==5.4.0" \
  "gradio_client==1.4.2" \
  "fastapi==0.115.6" \
  "starlette==0.41.3" \
  "uvicorn==0.30.0" \
  "pydantic==2.7.0" \
  "pydantic_core==2.18.1" \
  "python-multipart==0.0.12"
```

This command may upgrade `huggingface_hub` incorrectly. If import fails with:

```text
ImportError: cannot import name 'HfFolder' from 'huggingface_hub'
```

fix it with:

```bash
pip install --force-reinstall "huggingface_hub==0.36.2"
```

Verify:

```bash
python - <<'PY'
import huggingface_hub
import transformers
import tokenizers
import gradio
import gradio_client

print("huggingface_hub:", huggingface_hub.__version__)
print("transformers:", transformers.__version__)
print("tokenizers:", tokenizers.__version__)
print("gradio:", gradio.__version__)
print("gradio_client:", gradio_client.__version__)
print("imports OK")
PY
```

Expected important versions:

```text
huggingface_hub: 0.36.2
gradio: 5.4.0
gradio_client: 1.4.2
```

---

## 17. Custom Gradio UI

File: [`spark_tts_custom_ui.py`](spark_tts_custom_ui.py)

Purpose:

```text
A custom browser UI for testing Spark-TTS local inference.
Supports voice cloning and voice creation with generation stats.
```

Create required folders:

```bash
mkdir -p outputs gradio_tmp
```

Run:

```bash
cd /home/kawshik/tts-testing/Spark-TTS
conda activate sparktts

export CUDA_VISIBLE_DEVICES=1
export TMPDIR=/home/kawshik/tts-testing/Spark-TTS/gradio_tmp
export GRADIO_TEMP_DIR=/home/kawshik/tts-testing/Spark-TTS/gradio_tmp
mkdir -p gradio_tmp

python spark_tts_custom_ui.py
```

Expected:

```text
Loading Spark-TTS from: /home/kawshik/tts-testing/Spark-TTS/pretrained_models/Spark-TTS-0.5B
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
Spark-TTS model loaded.
Running on local URL: http://0.0.0.0:6012
```

Open in browser:

```text
http://YOUR_SERVER_IP:6012
```

---

## 18. Generated Audio Storage

Custom UI output files are saved in:

```bash
/home/kawshik/tts-testing/Spark-TTS/outputs/
```

Check:

```bash
ls -lh /home/kawshik/tts-testing/Spark-TTS/outputs/
```

Gradio temporary files are stored in:

```bash
/home/kawshik/tts-testing/Spark-TTS/gradio_tmp/
```

Clear outputs:

```bash
rm -rf /home/kawshik/tts-testing/Spark-TTS/outputs/*
rm -rf /home/kawshik/tts-testing/Spark-TTS/gradio_tmp/*
```

---

## 19. Common Warnings and Errors

### `weight_norm` FutureWarning

```text
FutureWarning: torch.nn.utils.weight_norm is deprecated
```

This is not a correctness issue. Inference can still work.

### Missing mel transformer tensors

```text
Missing tensor: mel_transformer.spectrogram.window
Missing tensor: mel_transformer.mel_scale.fb
```

This appeared during model loading and did not block successful inference.

### `pad_token_id` warning

```text
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
```

This appears during generation. It is not a blocker if audio is saved successfully.

### Prompt transcript mismatch

Error example:

```text
RuntimeError: Calculated padded input size per channel: (0). Kernel size: (1).
```

Likely cause:

```text
prompt_text does not match prompt_speech_path
```

Fix:

```text
Use the exact transcript of the prompt/reference audio.
```

For `example/prompt_audio.wav`, use:

```text
吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。
```

### Gradio schema error

Error:

```text
TypeError: argument of type 'bool' is not iterable
```

Fix:

```bash
pip install --force-reinstall \
  "gradio==5.4.0" \
  "gradio_client==1.4.2" \
  "fastapi==0.115.6" \
  "starlette==0.41.3" \
  "uvicorn==0.30.0" \
  "pydantic==2.7.0" \
  "pydantic_core==2.18.1" \
  "python-multipart==0.0.12"
```

Then fix Hugging Face Hub if needed:

```bash
pip install --force-reinstall "huggingface_hub==0.36.2"
```

### Gradio temp permission issue

If this appears:

```text
PermissionError: [Errno 13] Permission denied: '/tmp/gradio/...'
```

Use project-local temp folders:

```bash
cd /home/kawshik/tts-testing/Spark-TTS
mkdir -p gradio_tmp

export TMPDIR=/home/kawshik/tts-testing/Spark-TTS/gradio_tmp
export GRADIO_TEMP_DIR=/home/kawshik/tts-testing/Spark-TTS/gradio_tmp
```

---

## 20. What This Setup Does Not Cover

This README does not cover:

```text
Triton Inference Server
TensorRT-LLM runtime
runtime/triton_trtllm
Docker deployment
dynamic batching
in-flight batching
production serving benchmark
```

Those should be documented separately because the setup is much more complex and version-sensitive.

Current confirmed status:

```text
Spark-TTS local Python CLI inference: working
Spark-TTS custom Gradio UI: working
Voice cloning: working
Voice creation: supported
Default Python list-input batch: not supported
Triton/TensorRT-LLM runtime: separate future documentation
```

---

## Prepared By

**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com
