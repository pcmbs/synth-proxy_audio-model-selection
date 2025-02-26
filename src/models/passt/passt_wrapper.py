"""
Wrapper class around PaSST models for integration into the current pipeline.

@inproceedings{koutini22passt,
  author       = {Khaled Koutini and
                  Jan Schl{\"{u}}ter and
                  Hamid Eghbal{-}zadeh and
                  Gerhard Widmer},
  title        = {Efficient Training of Audio Transformers with Patchout},
  booktitle    = {Interspeech 2022, 23rd Annual Conference of the International Speech
                  Communication Association, Incheon, Korea, 18-22 September 2022},
  pages        = {2753--2757},
  publisher    = {{ISCA}},
  year         = {2022},
  url          = {https://doi.org/10.21437/Interspeech.2022-227},
  doi          = {10.21437/Interspeech.2022-227},
}

stripped down version of the PaSST repo available at
https://github.com/kkoutini/passt_hear21/tree/main
"""

import os
from dotenv import load_dotenv
import torch
from torch import nn

from models.passt import base, base2level, base2levelmel

load_dotenv()  # take environment variables from .env for checkpoints folder
torch.hub.set_dir(os.environ["PROJECT_ROOT"])  # path to download/load checkpoints

# We use mono audio with a sampling rate of 32 kHz.
# We extract Mel features from a window of 25 ms with a hop length of 10 ms, resulting in 128 mel band

# default model seems to be `passt_s_swa_p16_128_ap476` meaning a passt model with:
# s means small
# swa means stochastic weight averaging
# p16 mean patch size is 16x16
# s16 means no overlap (stride=16)
# 128 mel bands,
# ap473 refers to the performance of this model on Audioset mAP=0.479.

# could try: "passt_s_swa_p16_128_ap476", "passt_l_kd_p16_128_ap47" or "passt_s_kd_p16_128_ap486"

# mode: "embed_only" (transformer encoder output only, embedding size = 768),
# "logits" (classification head's logits only, embedding size = 527), "all" (concatenation of both, embedding size = 1295).

# base model: basic model
# -> output shape for 1 seconds of audio: torch.Size([1, mode_out, 21])
# base2lvl: concatenate output from base model and the output of the same model with timestamp_window * 5
# -> output shape for 1 seconds of audio: torch.Size([1, mode_out*2, 21])
# base2lvlmel: same as base2lvl but also concatenates the flattened mel spectrogram


class PasstWrapper(nn.Module):
    def __init__(
        self,
        features: str = "base",
        arch: str = "passt_s_swa_p16_128_ap476",
        mode="all",
    ):
        """
        PaSST model wrapper.
        generate 20*n + 1 timestamps for n-seconds of input audio, each containing an embedding whose size depends on the current mode
        (see below)

        Args
        - `features` (str): determine which features will be concatenated in the final embedding.
        "base" output from base model, "base2level" concatenation of the output from base model and the output of the same model with
        timestamp_window * 5base2level model, "base2levelmel" same as base2lvl but also concatenates the flattened mel spectrogram
        - `arch` (str): pretrained model.
        See list at https://github.com/kkoutini/passt_hear21/blob/48610e7baaf913298906fcde0ca3c28d0b8277c7/hear21passt/models/passt.py#L868
        - `mode` (str): determines which embeddings to use. "embed_only" (transformer encoder output only, embedding size = 768),
        "logits" (classification head's logits only, embedding size = 527), "all" (concatenation of both, embedding size = 1295).
        """
        super().__init__()
        self.arch = arch
        self.mode = mode

        if features == "base":
            self.model = base.get_basic_model(arch=arch, mode=mode)
            self._module = base
        elif features == "base2level":
            self.model = base2level.get_concat_2level_model(arch=arch, mode=mode)
            self._module = base2level
        elif features == "base2levelmel":
            self.model = base2levelmel.get_concat_2levelmel_model(arch=arch, mode=mode)
            self._module = base2levelmel
        else:
            raise ValueError("features should be 'base', 'base2level', or 'base2levelmel'")
        self.features = features

    @property
    def segment(self) -> None:
        # should be 10 but seems to truncate results to original input length
        return None

    @property
    def sample_rate(self) -> int:
        return 32_000

    @property
    def in_channels(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return f"{self.arch}_{self.features}_{self.mode}"

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        audio (torch.Tensor): mono input sounds @32khz of shape (n_sounds, n_channels=1, n_samples) in the range [-1, 1]

        Returns:
            torch.Tensor: audio embeddings of shape (n_sounds, embed_size=768, n_timestamps) where n_timestamps
            is computed based on a window size of 160ms with a hop size of 50ms (5120 sand 1600 samples @32kHz, respectively)
            and depends on the input length.
        """
        # passt requires mono input audio of shape (n_sounds, n_samples)
        audio = audio.squeeze(-2)
        embeddings, _ = self._module.get_timestamp_embeddings(audio, self.model)
        return embeddings.swapdims_(-1, -2)  # swap time and channel dims
