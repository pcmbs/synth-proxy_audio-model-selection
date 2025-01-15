"""
"""

import torch
from torch import nn
import torchaudio.functional as Fa
import torchaudio.transforms as T


class MelModel(nn.Module):
    def __init__(self, n_mels=128, n_mfcc=40, min_db=-80) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.min_db = min_db

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            win_length=2048,
            hop_length=1024,
            f_min=20,
            f_max=16000,
            n_mels=self.n_mels,
            normalized="window",  # normalize the STFT by the window's L2 norm (for energy conservation)
            norm=None,
            mel_scale="htk",
        )
        if n_mfcc > 0:
            self.n_mfcc = n_mfcc
            # create DCT matrix
            self.register_buffer(
                "dct_mat",
                Fa.create_dct(n_mfcc=self.n_mfcc, n_mels=self.n_mels, norm="ortho"),
                persistent=False,
            )
        else:
            self.n_mfcc = None

        # normalization statistics (from AudioSet)
        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974

    @property
    def segment(self) -> None:
        return None

    @property
    def sample_rate(self) -> int:
        return 44_100

    @property
    def in_channels(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return f"mel{self.n_mels}_{self.n_mfcc}_{self.min_db}"

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        audio (torch.Tensor): mono input sounds @44,1khz of shape (n_sounds, n_channels=1, n_samples) in the range [-1, 1].

        Returns:
            torch.Tensor: audio embeddings of shape (n_sounds, n_mels, n_frames)
        """
        if self.n_mfcc is not None:
            return self._compute_mfcc(audio).squeeze_(1)

        return self._compute_mel(audio).squeeze_(1)

    def _compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        mel_specgram = self.mel_spectrogram(audio)  # mel scaled spectrogram
        # clamp magnitude by below to 10^(MIN_DB/10) (/10 and not /20 since squared spectrogram)
        mel_specgram = torch.maximum(mel_specgram, torch.tensor(10 ** (self.min_db / 10)))
        mel_specgram = 10 * torch.log10(mel_specgram)  # log magnitude spectrogram (in dB)
        mel_specgram = (mel_specgram - self.norm_mean) / self.norm_std
        return mel_specgram

    def _compute_mfcc(self, audio: torch.Tensor) -> torch.Tensor:
        mel_specgram = self._compute_mel(audio)  # mel spectrogram
        mfcc = torch.matmul(mel_specgram.transpose(-1, -2), self.dct_mat).transpose(-1, -2)  # compute MFCCs
        return mfcc
