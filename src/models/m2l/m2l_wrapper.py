"""
Wrapper class around the music2latent [1]model.

Official repo: https://github.com/SonyCSLParis/music2latent 

[1] Pasini, M., Lattner, S., & Fazekas, G. (2024). 
Music2Latent: Consistency Autoencoders for Latent Audio Compression. 
arXiv. https://doi.org/10.48550/arXiv.2408.06500

"""

import os
from pathlib import Path
from dotenv import load_dotenv

import torch
from torch import nn
from music2latent import EncoderDecoder

load_dotenv()  # take environment variables from .env for checkpoints folder
torch.hub.set_dir(Path(os.environ["PROJECT_ROOT"]))  # path to download/load checkpoints


class M2LWrapper(nn.Module):
    """
    Wrapper class around the music2latent [1]model.

    Official repo: https://github.com/SonyCSLParis/music2latent

    [1] Pasini, M., Lattner, S., & Fazekas, G. (2024).
    Music2Latent: Consistency Autoencoders for Latent Audio Compression.
    arXiv. https://doi.org/10.48550/arXiv.2408.06500
    """

    def __init__(self, extract_features: bool = True, max_batch_size: int = 512) -> None:
        super().__init__()
        self.model = EncoderDecoder()
        self.extract_features = extract_features
        self.max_batch_size = max_batch_size

        self.model.eval()

    @property
    def sample_rate(self) -> int:
        """Return the required input audio signal's sample rate."""
        return 44_100

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return f"m2l_{'features' if self.extract_features else ''}"

    @property
    def in_channels(self) -> int:
        """Return the required number of input audio channels."""
        return 1

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters of the model."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        return self.model.encode(
            x, max_batch_size=self.max_batch_size, extract_features=self.extract_features
        )


if __name__ == "__main__":
    encoder = M2LWrapper(extract_features=True)
    audio = torch.empty((2, 1, 44_100 * 2)).uniform_(-1, 1)
    embeddings = encoder(audio)
    print(f"Output shape: {embeddings.shape}")
