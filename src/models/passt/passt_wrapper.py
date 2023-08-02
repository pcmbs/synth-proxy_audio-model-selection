# pylint: disable=E1101
"""
PaSST model from the paper: 
Efficient Training of Audio Transformers with Patchout (Koutini et al, 2022) 

based on the stripped down version of the PaSST repo available at
https://github.com/kkoutini/passt_hear21/tree/main
"""
import os
from dotenv import load_dotenv
import torch
from torch import nn

# dot from relative import needs to be removed when running as main
from .hear21passt.base import get_timestamp_embeddings, get_basic_model

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


class PasstWrapper(nn.Module):
    def __init__(self, arch: str = "passt_s_swa_p16_128_ap476", mode="all"):
        """
        PaSST model wrapper.
        generate 20*n + 1 timestamps for n-seconds of input audio, each containing an embedding whose size depends on the current mode
        (see below)

        Args
        - `arch` (str): pretrained model.
        See list at https://github.com/kkoutini/passt_hear21/blob/48610e7baaf913298906fcde0ca3c28d0b8277c7/hear21passt/models/passt.py#L868
        - `mode` (str): "embed_only" (transformer encoder output only, embedding size = 768),
        "logits" (classification head's logits only, embedding size = 527), "all" (concatenation of both, embedding size = 1295).
        """
        super().__init__()
        self.arch = arch
        self.mode = mode
        self.model = get_basic_model(arch=arch, mode=mode)

    @property
    def segment(self) -> None:
        # should be 10 but seems to truncate results to original input length
        return None

    @property
    def sample_rate(self) -> int:
        return 32_000

    @property
    def channels(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return self.arch

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        audio (torch.Tensor): mono input sounds @32khz of shape n_sounds x n_samples in the range [-1, 1]
        """
        embeddings, _ = get_timestamp_embeddings(audio, self.model)
        return embeddings.swapdims_(-1, -2)  # swap time and channel dims


# if __name__ == "__main__":
#     encoder = PasstWrapper(arch="passt_l_kd_p16_128_ap47", mode="all")
#     audio = torch.empty((1, 32_000 * 4)).uniform_(-1, 1)
#     embeddings = encoder(audio)
#     print(f"timestamp embeddings shape: {embeddings.shape}")
