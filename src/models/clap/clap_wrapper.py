"""
Wrapper class around the clap [1] model.
Official Repo: https://github.com/LAION-AI/CLAP

[1] Wu, Y., Chen, K., Zhang, T., Hui, Y., Nezhurina, M., Berg-Kirkpatrick, T., & Dubnov, S. (2024). 
Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation. 
arXiv. https://doi.org/10.48550/arXiv.2211.06687

"""

import os
from pathlib import Path
from dotenv import load_dotenv

import torch
from torch import nn
import laion_clap

load_dotenv()  # take environment variables from .env for checkpoints folder
CKPT_PATH = Path(os.environ["PROJECT_ROOT"]) / "checkpoints"
torch.hub.set_dir(CKPT_PATH)  # path to download/load checkpoints


# quantize functions
def int16_to_float32_torch(x):
    return (x / 32767.0).type(torch.float32)


def float32_to_int16_torch(x):
    x = torch.clamp(x, min=-1.0, max=1.0)
    return (x * 32767.0).type(torch.int16)


class ClapWrapper(nn.Module):
    """
    Wrapper class around the clap [1] model.
    Official Repo: https://github.com/LAION-AI/CLAP

    [1] Wu, Y., Chen, K., Zhang, T., Hui, Y., Nezhurina, M., Berg-Kirkpatrick, T., & Dubnov, S. (2024).
    Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation.
    arXiv. https://doi.org/10.48550/arXiv.2211.06687

    """

    # 630k_audioset, 630k_audioset_fusion, music_audioset, music_speech, music_speech_audioset
    def __init__(self, model_name: str, verbose=False) -> None:
        super().__init__()
        # 5 models available: more info at https://github.com/LAION-AI/CLAP?tab=readme-ov-file#pretrained-models
        # • Uses HTSAT-tiny as audio encode:
        #   - 630k_audioset: default/best non-fusion model trained on 630k and audioset datasets.
        #   - 630k_audioset_fusion: default/best fusion model trained on 630k and audioset datasets.
        # • Uses HTSAT-base as audio encoder (new models):
        #   - music_audioset: non-fusion model trained on music + speech + Audioset + LAION-Audio-630k.
        #   - music_speech: non-fusion model trained on music + Audioset + LAION-Audio-630k.
        #   - music_speech_audioset: non-fusion model trained on music + speech + LAION-Audio-630k.
        # load model
        requires_fusion = model_name.endswith("fusion")
        audio_model_size = "base" if model_name.startswith("music") else "tiny"
        self.model = laion_clap.CLAP_Module(enable_fusion=requires_fusion, amodel=f"HTSAT-{audio_model_size}")

        # load ckpt
        if model_name.startswith("630k"):  # default models
            self.model.load_ckpt(model_id=3 if requires_fusion else 1, verbose=verbose)
        elif model_name.startswith("music"):  # music models
            if model_name == "music_audioset":
                ckpt_name = f"{model_name}_epoch_15_esc_90.14.pt"
            elif model_name == "music_speech":
                ckpt_name = f"{model_name}_epoch_15_esc_89.25.pt"
            else:  # model_name == "music_speech_audioset"
                ckpt_name = f"{model_name}_epoch_15_esc_89.98.pt"
            self.model.load_ckpt(ckpt=CKPT_PATH / ckpt_name, verbose=verbose)
        else:  # invalid model name
            raise ValueError(f"Invalid model name: {model_name}")

        self._model_name = model_name
        self.model.eval()

    @property
    def sample_rate(self) -> int:
        """Return the required input audio signal's sample rate."""
        return 48_000

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return f"clap_{self._model_name}"

    @property
    def in_channels(self) -> int:
        """Return the required number of input audio channels."""
        return 1

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters of the model."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
            x = int16_to_float32_torch(float32_to_int16_torch(x))  # quantize
            # deterministic for len(x)<10 secs
        return self.model.get_audio_embedding_from_data(x, use_tensor=True)


if __name__ == "__main__":
    MODEL_NAMES = [
        "630k_audioset",
        "630k_audioset_fusion",
        "music_audioset",
        "music_speech",
        "music_speech_audioset",
    ]
    for model_name in MODEL_NAMES[0:2]:
        print(f"Sanity check for model: {model_name}")
        encoder = ClapWrapper(model_name=model_name)
        audio = torch.empty((2, 1, 48_000 * 2)).uniform_(-1, 1)
        embeddings = encoder(audio)
        print(f"Output shape: {embeddings.shape}")
