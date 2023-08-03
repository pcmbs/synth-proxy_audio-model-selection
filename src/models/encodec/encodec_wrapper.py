"""
Wrapper class around the encodec models for integration into the current pipeline.
encodec repo: https://github.com/facebookresearch/encodec/tree/main 
"""
from pathlib import Path
from typing import Optional, Union

import torch
from . import EncodecEncoder
from torch import nn


class EncodecWrapper(nn.Module):
    def __init__(
        self,
        model: str,
        repository: Optional[Union[Path, str]] = None,
        segment: Optional[float] = None,
        overlap: float = 0.0,
    ) -> None:
        super().__init__()
        if model == "24khz":
            self.encoder = EncodecEncoder.encodec_model_24khz(
                repository=repository, segment=segment, overlap=overlap
            )
        elif model == "48khz":
            self.encoder = EncodecEncoder.encodec_model_24khz(
                repository=repository, segment=segment, overlap=overlap
            )
        else:
            raise ValueError(
                f"model needs to be '24khz' or '48khz', but '{model}' was given"
            )

    @property
    def segment(self) -> None:
        return self.encoder.segment

    @property
    def channels(self) -> int:
        return self.encoder.channels

    @property
    def sample_rate(self) -> int:
        return self.encoder.sample_rate

    @property
    def name(self) -> str:
        return self.encoder.name

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        audio (torch.Tensor): input sounds of shape (n_sounds, n_channels, n_samples) in the range [-1, 1]
        """
        # returns a list with a single element since segment = audio input length
        return self.encoder(audio)[0]
