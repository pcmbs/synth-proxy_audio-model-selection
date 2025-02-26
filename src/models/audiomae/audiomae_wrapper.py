"""
Wrapper class around AudioMAE [1] for integration into the current pipeline.

Github Repo: https://github.com/facebookresearch/AudioMAE/tree/main

[1] @inproceedings{huang2022amae,
  title = {Masked Autoencoders that Listen},
  author = {Huang, Po-Yao and Xu, Hu and Li, Juncheng and Baevski, Alexei and Auli, Michael and Galuba, Wojciech and Metze, Florian and Feichtenhofer, Christoph}
  booktitle = {NeurIPS},
  year = {2022}
}
"""

import os
import sys
from contextlib import contextmanager
from collections import OrderedDict
import pathlib
from pathlib import Path

import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from torch import nn

from models.audiomae.nets.models_mae_encoder_only import mae_vit_base_patch16

load_dotenv()  # take environment variables from .env for checkpoints folder
# path to download/load checkpoints
CKPT_FOLDER = Path(os.environ["PROJECT_ROOT"]) / "checkpoints"
torch.hub.set_dir(Path(os.environ["PROJECT_ROOT"]))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def set_posix_windows():
    """
    Context manager to temporarily set pathlib.PosixPath to WindowsPath
    to avoid pathlib error if required.
    source: https://stackoverflow.com/a/68796747
    """
    if sys.platform == "win32":
        posix_backup = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            yield
        finally:
            pathlib.PosixPath = posix_backup
    else:
        yield


# Adapted from
# https://github.com/facebookresearch/AudioMAE/blob/main/demo/aud_mae_visualize.ipynb
# learned positional embedding
# patch embedding: 2Dconv layer (C_in=1, C_out=768, kernel_size=16, stride=16) (results in 512 patches)
# input shape: (N, 1, 44_100)
# fbanks shape (after unsqueeze): (N, n_channels=1, target_len=1024, n_mel=128)
# patch embedding:
# (n_sounds, num_patches=512, embed_dim=768)
# contextual_embs:
# the encoder output is the mean over the "encoder_depth - contextual_depth" last transformer layers' output
# if contextual_depth=-1 then the output is the last transformer layer followed by a layer norm
# encoder output shape (after final transpose):
# (n_sounds, embed_size=768, num_patches=512)


class AudioMAEWrapper(nn.Module):
    def __init__(self, ckpt_name: str = "as-2M_pt", contextual_depth: int = 8) -> None:
        super().__init__()
        if ckpt_name == "as-2M_pt+ft":
            ckpt_id = "18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq"
        elif ckpt_name == "as-2M_pt":
            ckpt_id = "1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu"
        else:
            raise ValueError(
                f"Invalid checkpoint name: {ckpt_name}. Must be one of ['as_2M_pt+ft', 'as_2M_pt']"
            )

        self.ckpt_name = ckpt_name
        ckpt_path = CKPT_FOLDER / f"audio-mae_{ckpt_name}.pth"

        # download and load weights from google drive link
        if not ckpt_path.exists():
            torch.hub.download_url_to_file(
                url=f"https://drive.google.com/uc?export=download&confirm=s5vl&id={ckpt_id}",
                dst=ckpt_path,
            )
        with set_posix_windows():
            checkpoint = torch.load(
                ckpt_path,
                map_location=DEVICE,
            )
        if ckpt_name == "as-2M_pt+ft":
            # layer norm params starts with fc_norm instead of norm (probs since models_vit instead of models_mae)
            checkpoint = OrderedDict(
                [(k[3:], v) if k.startswith("fc_norm") else (k, v) for k, v in checkpoint["model"].items()]
            )
        else:  # ckpt_name == "as-2M_pt"
            checkpoint = checkpoint["model"]
        # build model
        self.model = mae_vit_base_patch16(
            in_chans=1, audio_exp=True, img_size=(1024, 128), contextual_depth=contextual_depth
        )
        miss, _ = self.model.load_state_dict(checkpoint, strict=False)
        print(f"Missing weights from checkpoint: {miss}")

        # stats for mel spectrogram normalization
        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974

    @property
    def segment(self) -> None:
        return 10

    @property
    def sample_rate(self) -> int:
        # should technically be 16_000 since it is what the model was trained on
        # but we use 44_100 to include high frequency content
        # small loss of performance on some sound attributes (probably due to the difference in feautures)
        # but non-marginal gain for sound attributes relying on HF.
        return 44_100

    @property
    def in_channels(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return f"audio-mae_{self.ckpt_name}"

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        audio (torch.Tensor): mono input sounds @44,1khz of shape (n_sounds, n_channels=1, n_samples) in the range [-1, 1].
        Note that the model's frame length is 10 seconds, such that shorted audio clips will be trimmed while
        longer ones will be padded.

        Returns:
            torch.Tensor: audio embeddings of shape (n_sounds, embed_size=768, num_patches=512)
        """
        # kaldi fbanks can only be computed for a single audio sample at a time
        fbanks_batch = torch.stack([self._wav2fbank(sample) for sample in audio], dim=0)
        # add the channel dimension: (n_sounds,n_channels=1, target_len=1024, n_mels=128)
        fbanks_batch.unsqueeze_(dim=1)
        embeddings = self.model.forward_encoder_no_mask(fbanks_batch).transpose(-1, -2)

        return embeddings[:, :, 1:]  # don't use class token

    def _wav2fbank(self, audio: torch.Tensor) -> torch.Tensor:
        TARGET_LEN = int(np.floor((10.26 - 0.025) / 0.01) + 1)  # 1024@16khz
        MELBINS = 128

        audio = audio - audio.mean()

        # 498 128
        fbank = torchaudio.compliance.kaldi.fbank(
            audio,
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=MELBINS,
            dither=0.0,
            frame_shift=10,
        )
        # AudioSet: 1024 (16K sr)
        # ESC: 512 (8K sr)
        n_frames = fbank.shape[0]
        p = TARGET_LEN - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:TARGET_LEN, :]

        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        return fbank
