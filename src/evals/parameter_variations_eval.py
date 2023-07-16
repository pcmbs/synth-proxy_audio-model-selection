# pylint: disable=E1101,C0413

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchmetrics.functional import pairwise_manhattan_distance, pearson_corrcoef

# add parents directory to sys.path.
sys.path.insert(1, str(Path(__file__).parent.parent))

from data.tal_noisemaker.noisemaker_dataset import NoisemakerVariationsDataset
from models.encodec.encoder import EncodecEncoder
from utils.embeddings import compute_embeddings

load_dotenv()  # take environment variables from .env for hydra configs

# logger for this file
log = logging.getLogger(__name__)

# set torch device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

###### to move in hydra config
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
PATH_TO_CKPT = PROJECT_ROOT / "checkpoints"
PATH_TO_DATASET = PROJECT_ROOT / "data/TAL-NoiseMaker/parameter_variations"
AVAILABLE_VARIATIONS = [
    "amp_attack",
    "amp_decay",
    "filter_cutoff",
    "filter_decay",
    "filter_resonance",
    "frequency_mod",
    "lfo_amount_on_filter",
    "lfo_amount_on_volume",
    "lfo_rate_on_filter",
    "lfo_rate_on_volume",
    "pitch_coarse",
    "reverb",
]

###### functions definition


def parameter_variations_eval(
    variations: list[str], encoder: nn.Module, reduce_fn: str
) -> dict[float]:
    corrcoeffs = {}

    for variation in variations:
        log.info(f"Computing correlation coefficient for variation `{variation}`...")
        corrcoeffs[variation] = _compute_corrcoeff_for_variation(
            variation, encoder, reduce_fn
        )

    # compute mean and median correlation coefficient
    corrcoeffs["mean"] = np.mean(list(corrcoeffs.values()))
    corrcoeffs["median"] = np.median(list(corrcoeffs.values()))

    return corrcoeffs


def _compute_corrcoeff_for_variation(
    variation: str, encoder: nn.Module, reduce_fn: str
) -> float:
    dataset = NoisemakerVariationsDataset(
        root=PATH_TO_DATASET, variation_type=variation
    )

    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    encoder.segment = dataset.audio_length

    embeddings, rank = compute_embeddings(
        encoder=encoder,
        dataloader=dataloader,
        num_samples=-1,
        data_sample_rate=dataset.sample_rate,
        encoder_sample_rate=encoder.sample_rate,
        encoder_channels=encoder.channels,
        encoder_frame_length=encoder.segment,
        device=DEVICE,
    )

    # flatten embeddings
    embeddings = embeddings.flatten(start_dim=1)

    dist_mat = pairwise_manhattan_distance(embeddings)

    indices = torch.argsort(dist_mat, dim=1)

    ranking_from_first = indices[0, 1:].float()
    ranking_from_last = indices[-1, 1:].float()

    target_from_first = torch.tensor(rank[1:]).float()
    target_from_last = target_from_first.flip(0) - 1

    corrcoeff_from_first = pearson_corrcoef(ranking_from_first, target_from_first)
    corrcoeff_from_last = pearson_corrcoef(ranking_from_last, target_from_last)

    avg_corrcoeff = (corrcoeff_from_first + corrcoeff_from_last) / 2

    return avg_corrcoeff.item()


if __name__ == "__main__":
    VARIATION_TYPE = "amp_attack"

    encoder = EncodecEncoder.encodec_model_48khz(
        repository=PATH_TO_CKPT, segment=1.0, overlap=0.0
    )

    corrcoeff = parameter_variations_eval(AVAILABLE_VARIATIONS, encoder, "mean")

    print("breakpoint me!")
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
