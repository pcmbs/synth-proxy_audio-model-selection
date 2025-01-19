"""
Wrapper class around the DAC neural audio codec model [1].
Official Repo: https://github.com/LAION-AI/CLAP

[1] Kumar, R., Seetharaman, P., Luebs, A., Kumar, I., & Kumar, K. (2023). 
High-Fidelity Audio Compression with Improved RVQGAN (No. arXiv:2306.06546). arXiv. 
https://doi.org/10.48550/arXiv.2306.06546
"""

import os
from pathlib import Path
from dotenv import load_dotenv

import dac
import torch
from torch import nn

load_dotenv()  # take environment variables from .env for checkpoints folder
CKPT_PATH = Path(os.environ["PROJECT_ROOT"]) / "checkpoints"
torch.hub.set_dir(CKPT_PATH)  # path to download/load checkpoints


class DacWrapper(nn.Module):
    """
    Wrapper class around the DAC neural audio codec model [1].
    Official Repo: https://github.com/LAION-AI/CLAP

    [1] Kumar, R., Seetharaman, P., Luebs, A., Kumar, I., & Kumar, K. (2023).
    High-Fidelity Audio Compression with Improved RVQGAN (No. arXiv:2306.06546). arXiv.
    https://doi.org/10.48550/arXiv.2306.06546
    """

    # 630k_audioset, 630k_audioset_fusion, music_audioset, music_speech, music_speech_audioset
    def __init__(
        self,
        return_type: str = "z",
        model_type: str = "44khz",
    ) -> None:
        super().__init__()
        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path)

        self.model_type = model_type
        if model_type == "44khz":
            self._sample_rate = 44_100
        elif model_type == "24khz":
            self._sample_rate = 24_000
        else:  # 16khz
            self._sample_rate = 16_000

        if return_type not in ["z", "codes", "latents"]:
            raise ValueError(f"Unknown return type: {return_type}")
        self.return_type = return_type

        self.model.eval()

    @property
    def sample_rate(self) -> int:
        """Return the required input audio signal's sample rate."""
        return self._sample_rate

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return f"dac_{self.model_type}_{self.return_type}"

    @property
    def in_channels(self) -> int:
        """Return the required number of input audio channels."""
        return 1

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters of the model."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """
        Forward pass.
        audio (torch.Tensor): mono input sounds of shape (n_sounds, n_channels=1, n_samples) in the range [-1, 1]

        Returns: torch.Tensor. (M: n_out_channels D: codebook_size, N: n_codebooks, T: sequence length)
            If return_type="z":
                Quantized continuous representation of input, Tensor of shape (B, M, T)
                where M=1024 corresponds to the number of output channels of the last conv1d layer.
            If return_type="codes":
                Codebook indices for each codebook (quantized discrete representation of input),
                Tensor of shape (B, N, T), where N depends on the model's sample rate.
            If return_type="latents":
                Concatenation of the projected latents of each stage (continuous representation of input before quantization),
                Tensor of shape (B, N*D, T), where N depends on the model's sample rate, and where D is the codebook dimension
                (D=8 for all models).

        """
        assert x.shape[1] == 1
        x = self.model.preprocess(x, self.sample_rate)
        z, codes, latents, _, _ = self.model.encode(x)
        if self.return_type == "z":
            return z
        if self.return_type == "codes":
            return codes
        # self.return_type == "latents":
        return latents


if __name__ == "__main__":
    RETURN_TYPES = ["z", "codes", "latents"]
    MODEL_TYPES = ["44khz", "24khz", "16khz"]

    for model_name in MODEL_TYPES:
        print(f"\nSanity check for model: {model_name}")
        for return_type in RETURN_TYPES:
            encoder = DacWrapper(return_type=return_type, model_type=model_name)
            audio = torch.empty((1, 1, encoder.sample_rate * 2)).uniform_(-1, 1)
            embeddings = encoder(audio)
            print(f"{return_type} - output shape: {embeddings.shape}")
