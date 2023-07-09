"""Module used to check the embedding's output size"""
# pylint: disable=E1101:no-member
from pathlib import Path

import torch

from src.models.encodec.encoder import EncodecEncoder

CKPT_PATH = Path("C:/Users/paolo/PROG/ML/encodec_test/checkpoints")
SAMPLE_RATE = 24_000
AUDIO_LENGHT = 1
SEGMENT = 1.0
OVERLAP = 0.0

PRINT_MODEL = True

if SAMPLE_RATE == 48_000:
    encoder = EncodecEncoder.encodec_model_48khz(segment=SEGMENT, overlap=OVERLAP)
    encoder.load_state_dict(torch.load(CKPT_PATH / "encodec_encoder_48khz.pt"))
elif SAMPLE_RATE == 24_000:
    encoder = EncodecEncoder.encodec_model_24khz(segment=SEGMENT, overlap=OVERLAP)
    encoder.load_state_dict(torch.load(CKPT_PATH / "encodec_encoder_24khz.pt"))
else:
    raise NotImplementedError(f"Sample rate must be `48_000` or `24_000` but {SAMPLE_RATE} was given")

encoder.eval()

# generate a single random data
wav = torch.rand((1, encoder.channels, int(encoder.sample_rate * AUDIO_LENGHT)))

# compare outputs from the original and extracted encoder
with torch.no_grad():
    embedding = encoder(wav)  # extracted encoder output

frames_size_str = ""
for i, frame in enumerate(embedding):
    frames_size_str += f"    frame {i}: {frame.shape} \n"

to_print = (
    f"model: {encoder.sample_rate} \n"
    f"channels: {encoder.channels} \n"
    f"segment: {encoder.segment} \n"
    f"overlap: {encoder.overlap} \n"
    f"embedding size:\n{frames_size_str}"
)

print(to_print)

if PRINT_MODEL:
    print(encoder.encoder.model)
