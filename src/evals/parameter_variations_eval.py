# pylint: disable=E1101,C0413,W1203

import logging
from pathlib import Path
from typing import Union, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchmetrics import functional as tm_functional
from torchmetrics.functional import pearson_corrcoef

# add parents directory to sys.path if run as as main
if __name__ == "__main__":
    import sys

    sys.path.insert(1, str(Path(__file__).parents[1]))

from data.tal_noisemaker.noisemaker_dataset import NoisemakerVariationsDataset
from utils.embeddings import compute_embeddings
from utils import reduce_fn as r_fn

# logger for this file
log = logging.getLogger(__name__)


def parameter_variations_eval(
    path_to_dataset: Union[Path, str],
    variations: List[str],
    encoder: nn.Module,
    similarity: str,
    reduce_fn: str,
) -> Dict[str, float]:
    path_to_dataset = (
        Path(path_to_dataset) if isinstance(path_to_dataset, str) else path_to_dataset
    )

    corrcoeffs = {}

    for variation in variations:
        corrcoeff_from_first, corrcoeff_from_last = _compute_corrcoeff_for_variation(
            path_to_dataset, variation, encoder, similarity, reduce_fn
        )
        corrcoeffs[variation] = (corrcoeff_from_first + corrcoeff_from_last) / 2
        log.info(
            f"Correlation coefficient for variation `{variation}`: \n"
            f"    from first: {corrcoeff_from_first}\n"
            f"    from last: {corrcoeff_from_last}\n"
            f"    mean: {corrcoeffs[variation]}"
        )

    # compute the median correlation coefficient
    corrcoeffs["mean"] = np.mean(list(corrcoeffs.values()))
    corrcoeffs["median"] = np.median(list(corrcoeffs.values()))

    return corrcoeffs


def _compute_corrcoeff_for_variation(
    path_to_dataset: Union[Path, str],
    variation: str,
    encoder: nn.Module,
    similarity: str,
    reduce_fn: str,
) -> Tuple[float, float]:
    dataset = NoisemakerVariationsDataset(
        root=path_to_dataset, variation_type=variation
    )

    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    encoder.segment = (
        dataset.audio_length
    )  # to delete, will be set beforehand from config

    embeddings, rank = compute_embeddings(
        encoder=encoder,
        dataloader=dataloader,
        num_samples=-1,
        data_sample_rate=dataset.sample_rate,
        encoder_sample_rate=encoder.sample_rate,
        encoder_channels=encoder.channels,
        encoder_frame_length=encoder.segment,
        pbar=False,
    )

    # flatten embeddings
    embeddings = getattr(r_fn, reduce_fn)(embeddings)

    distance_fn = getattr(tm_functional, similarity)
    dist_mat = distance_fn(embeddings)

    indices = torch.argsort(
        dist_mat,
        dim=1,
        descending=similarity == "pairwise_cosine_similarity",
    )

    ranking_from_first = indices[0, 1:].float()
    ranking_from_last = indices[-1, 1:].float()

    target_from_first = torch.tensor(rank[1:]).float()
    target_from_last = target_from_first.flip(0) - 1

    corrcoeff_from_first = pearson_corrcoef(
        ranking_from_first, target_from_first
    ).item()
    corrcoeff_from_last = pearson_corrcoef(ranking_from_last, target_from_last).item()

    return corrcoeff_from_first, corrcoeff_from_last


if __name__ == "__main__":
    # import sys
    # import os
    # from models.encodec.encoder import EncodecEncoder

    # # set torch device
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ###### to move in hydra config
    # PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
    # PATH_TO_CKPT = PROJECT_ROOT / "checkpoints"
    # PATH_TO_DATASET = PROJECT_ROOT / "data/TAL-NoiseMaker/parameter_variations"
    # AVAILABLE_VARIATIONS = [
    #     "amp_attack",
    #     "amp_decay",
    #     "filter_cutoff",
    #     "filter_decay",
    #     "filter_resonance",
    #     "frequency_mod",
    #     "lfo_amount_on_filter",
    #     "lfo_amount_on_volume",
    #     "lfo_rate_on_filter",
    #     "lfo_rate_on_volume",
    #     "pitch_coarse",
    #     "reverb",
    # ]

    # SIMILARITY = "pairwise_manhattan_distance"

    # REDUC_FN = "flatten"

    # encoder = EncodecEncoder.encodec_model_48khz(
    #     repository=PATH_TO_CKPT, segment=1.0, overlap=0.0
    # )

    # corrcoeff = parameter_variations_eval(
    #     PATH_TO_DATASET, AVAILABLE_VARIATIONS, encoder, SIMILARITY, REDUC_FN, DEVICE
    # )
    print("breakpoint me!")
