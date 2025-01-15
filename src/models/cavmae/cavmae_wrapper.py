"""
Wrapper class around CAV-MAE [1] models for integration into the current pipeline.

Github Repo: https://github.com/YuanGongND/cav-mae/tree/master 

[1] @inproceedings{
gong2023contrastive,
title={Contrastive Audio-Visual Masked Autoencoder},
author={Yuan Gong and Andrew Rouditchenko and Alexander H. Liu and David Harwath and Leonid Karlinsky and Hilde Kuehne and James R. Glass},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=QPtMRyk5rb}
}

"""

import os
from pathlib import Path
from dotenv import load_dotenv
import torch
from torch import nn
import torchaudio

from models.cavmae.audio_mdl import CAVMAEFTAudio

load_dotenv()  # take environment variables from .env for checkpoints folder
# path to download/load checkpoints
torch.hub.set_dir(Path(os.environ["PROJECT_ROOT"]))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CAVMAEWrapper(nn.Module):
    def __init__(self, ckpt_name: str = "as_46.6") -> None:
        super().__init__()
        self.ckpt_name = ckpt_name
        # load model and weights
        state_dict = torch.hub.load_state_dict_from_url(
            url=f"https://www.dropbox.com/s/itfw7p0ueq7z9og/{ckpt_name}.pth?dl=1",
            map_location=DEVICE,
        )
        # convert the model to dataparallel object as all weights are
        # saved in dataparallel format (i.e., in module.xxx)
        self.audio_model = torch.nn.DataParallel(CAVMAEFTAudio(label_dim=527))
        miss, _ = self.audio_model.load_state_dict(state_dict, strict=False)
        print(f"Number of missing weights from checkpoint: {len(miss)}")

        # stats for mel spectrogram normalization
        # https://github.com/YuanGongND/cav-mae/blob/master/src/gen_audio_embedding_esc.py#L85C8-L85C9 (line 85)
        self.norm_mean = -5.081
        self.norm_std = 4.4849

    @property
    def segment(self) -> None:
        return 10

    @property
    def sample_rate(self) -> int:
        return 44_100

    @property
    def in_channels(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return f"cav-mae_{self.ckpt_name}"

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        audio (torch.Tensor): mono input sounds @44,1khz of shape (n_sounds, n_channels=1, n_samples) in the range [-1, 1]
        Note that the model's frame length is 10 seconds, such that shorted audio clips will be trimmed while
        longer ones will be padded.

        Returns:
            torch.Tensor: audio embeddings of shape (n_sounds, embed_size=768, num_patches=512)
        """
        # kaldi fbanks can only be computed for a single audio sample at a time
        fbanks_batch = torch.stack([self._wav2fbank(sample) for sample in audio], dim=0)
        return self.audio_model(fbanks_batch).transpose(-1, -2)

    def _wav2fbank(self, audio: torch.Tensor) -> torch.Tensor:
        # source: https://github.com/YuanGongND/cav-mae/blob/master/src/dataloader.py
        # see line 285 for fbanks normalization and line 164 for _wav2bank definition

        audio = audio - audio.mean()
        try:
            fbank = torchaudio.compliance.kaldi.fbank(
                audio,
                htk_compat=True,
                sample_frequency=self.sample_rate,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10,
            )
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print("there is a loading error")

        target_length = 1024
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        fbank = (fbank - self.norm_mean) / (self.norm_std)

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank
