"""
Wrapper class around a torchopenl3 model for integration into the current pipeline.
torchopenl3 repo: https://github.com/torchopenl3/torchopenl3 
"""
import os
from dotenv import load_dotenv
from functools import partial

import torch
import torchopenl3
from torch import nn

load_dotenv()  # take environment variables from .env for checkpoints folder
torch.hub.set_dir(os.environ["PROJECT_ROOT"])  # path to download/load checkpoints


class TorchOpenL3Wrapper(nn.Module):
    def __init__(
        self,
        input_repr: str,
        content_type: str,
        embedding_size: int,
        hop_size: float,
        center: bool = False,
    ) -> None:
        super().__init__()
        self.input_repr = input_repr
        self.content_type = content_type
        self.embedding_size = embedding_size
        self.hop_size = hop_size
        self.center = center

        # load the model
        base_model = torchopenl3.core.load_audio_embedding_model(
            input_repr=self.input_repr,
            content_type=self.content_type,
            embedding_size=self.embedding_size,
        )

        # partial init such that only the audio must be passed in the forward function
        self.encoder = partial(
            torchopenl3.core.get_audio_embedding,
            model=base_model,
            hop_size=self.hop_size,
            center=self.center,
        )

    @property
    def segment(self) -> None:
        return None

    @property
    def channels(self) -> int:
        return 1

    @property
    def sample_rate(self) -> int:
        return 48_000

    @property
    def name(self) -> str:
        return f"openl3_{self.input_repr}_{self.content_type}_{self.embedding_size}"

    def forward(self, x: torch.Tensor, sample_rate: int) -> torch.Tensor:
        embeddings, _ = self.encoder(audio=x, sr=sample_rate)
        embeddings.swapdims_(-1, -2)  # swap time and channel dims
        return embeddings
