import torch
import soundfile as sf
from omnivoice import OmniVoice

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16,
)

audio = model.generate(
    text="আমি বাংলায় কথা বলতে চাই। Also I can talk in English.",
    num_step=16,
    speed=1.0,
)

sf.write("smoke_omnivoice.wav", audio[0], 24000)
print("saved smoke_omnivoice.wav")
