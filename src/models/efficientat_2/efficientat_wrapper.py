"""
Wrapper class around EfficientAT [1,2] models.

Official EfficientAT repo available at
https://github.com/fschmid56/EfficientAT/ 

[1] Schmid, F., Koutini, K., & Widmer, G. (2022). 
Efficient Large-scale Audio Tagging via Transformer-to-CNN Knowledge Distillation.
http://arxiv.org/abs/2211.04772

[2] Schmid, F., Koutini, K., & Widmer, G. (2024). 
Dynamic Convolutional Neural Networks as Efficient Pre-Trained Audio Models. 
IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32, 2227–2241. 
https://doi.org/10.1109/TASLP.2024.3376984

"""

from contextlib import nullcontext
import os
from pathlib import Path
from dotenv import load_dotenv
import torch
from torch import nn, autocast

from models.efficientat_2.dymn.model import get_model as get_dymn
from models.efficientat_2.mn.model import get_model as get_mn
from models.efficientat_2.mn.utils import NAME_TO_WIDTH
from models.efficientat_2.preprocess import AugmentMelSTFT
from utils import reduce_fn


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

load_dotenv()  # take environment variables from .env for checkpoints folder
# path to download/load checkpoints
torch.hub.set_dir(Path(os.environ["PROJECT_ROOT"]))


class EfficientATWrapper(nn.Module):
    """
    Wrapper class around EfficientAT. This class is based on
    https://github.com/fschmid56/EfficientAT/blob/main/inference.py
    """

    def __init__(
        self, model_name: str, return_fmaps: bool = False, reduction: str = "identity", cuda: bool = True
    ) -> None:
        super().__init__()
        if model_name.startswith("dymn"):
            self.model = get_dymn(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name)
        elif model_name.startswith("mn"):
            self.model = get_mn(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.model_name = model_name
        self.model.eval()

        self.reduce_fn = getattr(reduce_fn, reduction)
        self.cuda = cuda
        self.return_fmaps = return_fmaps

        # model to preprocess waveform into mel spectrograms (hardcoded with values used for training)
        self.mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320)
        self.mel.eval()

    @property
    def sample_rate(self) -> int:
        """Return the required input audio signal's sample rate."""
        return 32_000

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return self.model_name

    @property
    def in_channels(self) -> int:
        """Return the required number of input audio channels."""
        return 1

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters of the model."""
        return sum(p.numel() for p in self.parameters())

    def _mel_forward(self, x):
        """
        Preprocess waveform into mel spectrograms.
        From https://github.com/fschmid56/EfficientAT/blob/main/ex_audioset.py
        """
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio (torch.Tensor): mono input sounds @32khz of shape (n_sounds, n_channels=1, n_samples)
            in the range [-1, 1].

        Returns: Union[list[torch.Tensor], torch.Tensor]:
            - if return_fmaps=False: torch.Tensor of shape (n_sounds, embed_size).
            - if return_fmaps=True: list[torch.Tensor] of length 17 where each Tensor has shape (B, C, H, W)
        """
        # our models are trained in half precision mode (torch.float16)
        # run on cuda with torch.float16 to get the best performance
        # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
        with torch.no_grad(), autocast(device_type=DEVICE.type) if self.cuda else nullcontext():
            spec = self._mel_forward(audio)
            _, features = self.model._forward_impl(spec, return_fmaps=self.return_fmaps)
        return features


# last feature map gets adaptive pool2d

if __name__ == "__main__":
    from models.efficientat_2.mn.model import pretrained_models
    from models.efficientat_2.dymn.model import pretrained_models

    encoder = EfficientATWrapper(model_name="dymn04_as", return_fmaps=False)
    audio = torch.empty((1, 1, 32_000 * 5)).uniform_(-1, 1)
    embeddings = encoder(audio)
    if isinstance(embeddings, torch.Tensor):
        print(f"Output shape: {embeddings.shape}")
    elif isinstance(embeddings, list):
        print(f"Number of fmaps: {len(embeddings)}")
        print(f"Output shape: {[e.shape for e in embeddings]}")
