import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)

    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)

    return torch.LongTensor(text_norm)


def normalize_audio(audio):
    audio = np.asarray(audio).squeeze().astype(np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ljs_base.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", default="outputs/test_ljs.wav")
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hps = utils.get_hparams_from_file(args.config)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)

    net_g.eval()

    print("Loading checkpoint:", args.checkpoint)
    _ = utils.load_checkpoint(args.checkpoint, net_g, None)

    stn_tst = get_text(args.text, hps)
    x_tst = stn_tst.to(device).unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.monotonic()

    with torch.inference_mode():
        audio = net_g.infer(
            x_tst,
            x_tst_lengths,
            noise_scale=args.noise_scale,
            noise_scale_w=args.noise_scale_w,
            length_scale=args.length_scale,
        )[0][0, 0].data.cpu().float().numpy()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.monotonic() - start

    audio = normalize_audio(audio)
    sr = int(hps.data.sampling_rate)
    duration = len(audio) / sr if len(audio) else 0.0
    rtf = elapsed / duration if duration > 0 else 0.0

    sf.write(str(out_path), audio, sr)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    silent = peak < 1e-4 or rms < 1e-5

    print("device:", device)
    print("sampling_rate:", sr)
    print("output:", out_path)
    print(f"elapsed_sec: {elapsed:.3f}")
    print(f"audio_duration_sec: {duration:.3f}")
    print(f"RTF: {rtf:.3f}")
    print(f"peak: {peak:.8f}")
    print(f"rms: {rms:.8f}")
    print(f"silent: {silent}")


if __name__ == "__main__":
    main()
