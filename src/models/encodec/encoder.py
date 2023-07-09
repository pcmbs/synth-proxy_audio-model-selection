# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EnCodec model implementation."""

import math
import typing as tp
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models.encodec.seanet import SEANetEncoder

EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]

# for preset embeddings:
# • set segment to 2.5 or 3secs and overlap to 0.
# this should allow to get a single frame embedding for the whole input which should
# be equivalent to the concatenation of several short frame (without overlap)
# normalize audio input (using the )


class EncodecEncoder(nn.Module):
    """EnCodec model operating on the raw waveform.
    Args:
        encoder (nn.Module): Encoder network.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization.
        segment (float or None): segment duration in sec. when doing overlap-add.
        Pass `None` to use the full input duration as segment.
        overlap (float): overlap between segment, given as a fraction of the segment duration.
        name (str): name of the model, used as metadata when compressing audio.
    """

    def __init__(
        self,
        encoder: SEANetEncoder,
        sample_rate: int,
        channels: int,
        normalize: bool = False,
        segment: tp.Optional[float] = None,
        overlap: float = 0.01,
        name: str = "unset",
    ):
        super().__init__()
        self.encoder = encoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios))
        self.name = name

    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))

    @property
    def output_shape_per_segment(self) -> tuple[int, int]:
        return (self.encoder.dimension, self.frame_rate * self.segment)

    def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        """Given a tensor `x`, returns a list of frames containing
        the embeddings for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.
        """
        assert x.dim() == 3
        _, channels, length = x.shape
        assert channels > 0 and channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: tp.List[EncodedFrame] = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset : offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        length = x.shape[-1]
        duration = length / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment

        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encoder(x)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = self.encode(x)
        return frames

    @staticmethod
    def encodec_model_48khz(
        pretrained: bool = True,
        repository: tp.Optional[tp.Union[Path, str]] = None,
        segment: float = 1.0,
        overlap: float = 0.01,
    ):
        """Return the pretrained 48khz model."""
        if repository:
            assert pretrained
        repository = Path(repository) if isinstance(repository, str) else repository
        checkpoint_name = "encodec_encoder_48khz.pt"
        sample_rate = 48_000
        channels = 2
        causal = False
        model_norm = "time_group_norm"
        audio_normalize = True
        name = "encodec_48khz" if pretrained else "unset"

        encoder = SEANetEncoder(channels=channels, norm=model_norm, causal=causal)

        model = EncodecEncoder(
            encoder=encoder,
            sample_rate=sample_rate,
            channels=channels,
            normalize=audio_normalize,
            segment=segment,
            overlap=overlap,
            name=name,
        )
        if pretrained:
            if not repository.is_dir():
                raise ValueError(f"{repository} must exist and be a directory.")
            state_dict = torch.load(repository / checkpoint_name)
            model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def encodec_model_24khz(
        pretrained: bool = True,
        repository: tp.Optional[tp.Union[Path, str]] = None,
        segment: float = None,
        overlap: float = 0.01,
    ):
        """Return the pretrained 24khz model."""
        if repository:
            assert pretrained
        repository = Path(repository) if isinstance(repository, str) else repository
        checkpoint_name = "encodec_encoder_24khz.pt"
        sample_rate = 24_000
        channels = 1
        causal = True
        model_norm = "weight_norm"
        audio_normalize = False
        name = "encodec_24khz" if pretrained else "unset"

        encoder = SEANetEncoder(channels=channels, norm=model_norm, causal=causal)

        model = EncodecEncoder(
            encoder=encoder,
            sample_rate=sample_rate,
            channels=channels,
            normalize=audio_normalize,
            segment=segment,
            overlap=overlap,
            name=name,
        )
        if pretrained:
            if not repository.is_dir():
                raise ValueError(f"{repository} must exist and be a directory.")
            state_dict = torch.load(repository / checkpoint_name)
            model.load_state_dict(state_dict)
        model.eval()
        return model
