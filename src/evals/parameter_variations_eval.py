# pylint: disable=W1203,C0413,W0212

import logging
import sys
from pathlib import Path

from typing import Optional

import hydra
import torch
from dotenv import load_dotenv
from lightning import seed_everything
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# from torchmetrics import functional as tm_functional

# add parents directory to sys.path.
sys.path.insert(1, str(Path(__file__).parent.parent))

from data.tal_noisemaker.noisemaker_dataset import NoisemakerVariationsDataset
from models.encodec.encoder import EncodecEncoder
from utils.embeddings import get_embeddings

load_dotenv()  # take environment variables from .env for hydra configs

# logger for this file
log = logging.getLogger(__name__)

# set torch device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# PATH_TO_DATASET = (
#     Path(os.getenv("PROJECT_ROOT")) / "data" / "TAL-NoiseMaker" / "parameter_variations"
# )
# VARIATION_TYPE = "amp_attack"

# dataset = NoisemakerVariationsDataset(
#     root=PATH_TO_DATASET, variation_type=VARIATION_TYPE
# )

# dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# encoder = EncodecEncoder()

# print("breakpoint me!")


# @hydra.main(
#     version_base=None, config_path="../configs", config_name="evaluate_embeddings"
# )
# def main(cfg: DictConfig) -> Optional[float]:
#     #################### preparation
#     # set seed for random number generators in pytorch, numpy and python.random
#     if cfg.get("seed"):
#         seed_everything(cfg.seed, workers=True)

#     # instantiate encoder
#     log.info(f"Instantiating model <{cfg.model.encoder._target_}>")
#     encoder: EncodecEncoder = hydra.utils.call(cfg.model.encoder)
#     encoder.to(DEVICE)

#     #################### get embeddings

#     embeddings, indices_from_batch = get_embeddings(
#         encoder=encoder,
#         dataloader=nsynth_dataloader,
#         num_samples=cfg.data.num_samples,
#         data_sample_rate=nsynth_dataset.sample_rate,
#         encoder_sample_rate=encoder.sample_rate,
#         encoder_channels=encoder.channels,
#         encoder_frame_length=encoder.segment,
#         device=DEVICE,
#     )


# if __name__ == "__main__":
#     main()  # pylint: disable=no-value-for-parameter
