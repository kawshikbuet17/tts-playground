import time
import torch
import soundfile as sf

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


MODEL_ID = "ai4bharat/indic-parler-tts"
OUT_PATH = "indic_parler_bn_test.wav"


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading model...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )

    prompt = "আজকের আবহাওয়া খুব সুন্দর। আমি বাংলা টেক্সট টু স্পিচ পরীক্ষা করছি।"

    description = (
        "Aditi speaks in a natural Bengali voice, at a moderate pace, "
        "with clear pronunciation and very clear audio."
    )

    print("Tokenizing...")
    description_inputs = description_tokenizer(
        description,
        return_tensors="pt",
    ).to(device)

    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    print("Generating...")
    start = time.perf_counter()

    with torch.inference_mode():
        generation = model.generate(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )

    elapsed = time.perf_counter() - start

    audio_arr = generation.detach().cpu().float().numpy().squeeze()
    sampling_rate = model.config.sampling_rate

    sf.write(OUT_PATH, audio_arr, sampling_rate)

    audio_duration = len(audio_arr) / sampling_rate
    rtf = elapsed / audio_duration if audio_duration > 0 else None

    print(f"Saved: {OUT_PATH}")
    print(f"Sampling rate: {sampling_rate}")
    print(f"Audio duration: {audio_duration:.2f} sec")
    print(f"Generation time: {elapsed:.2f} sec")
    print(f"RTF: {rtf:.3f}")


if __name__ == "__main__":
    main()
