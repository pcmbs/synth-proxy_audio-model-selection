# pylint: disable=E1101,C0413,W1203
"""
Module containing functions for the sound attributes ranking evaluation,
used to evaluate the ability of a model to order sounds subject to monotonic changes of parameter values
corresponding to different sound attributes.
"""

import logging
from pathlib import Path
from typing import Union, Dict, Tuple

import numpy as np
import torch
from torch import nn

from torch.utils.data import DataLoader
import torchmetrics.functional as tm_functional
from torchmetrics.functional import pearson_corrcoef

# add parents directory to sys.path if run as as main (for debugging purposes)
if __name__ == "__main__":
    import sys

    sys.path.insert(1, str(Path(__file__).parents[1]))

from data.tal_noisemaker.noisemaker_dataset import SoundAttributesDataset
from utils.embeddings import compute_embeddings
from utils import reduce_fn as r_fn

# logger for this file
log = logging.getLogger(__name__)

# set torch device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sound_attributes_ranking_eval(
    path_to_dataset: Union[Path, str],
    encoder: nn.Module,
    distance_fn: str,
    reduce_fn: str,
) -> Dict[str, float]:
    """
    Sound attributes ranking evaluation, used to evaluate the ability of a model to order sounds
    subject to monotonic changes of parameter values corresponding to different sound attributes.
    Given a dataset composed K groups, each composed of N presets. For each group, a single parameter
    is monotonically increased (or decreased) in order to modify a given sound attribute.
    For each sound attribute, the evaluation can be described as follows:
    - (1): get the representation of each sound from a given `encoder` and a reduction function `reduce_fn`
    - (2): compute the distance matrix for each preset using pairwise `distance_fn` (must be the name of a
    torchmetrics.functional function)
    - (3): sort the sounds by ascending order of distance to the sound with the lowest parameter value
    for each preset (lowest rank)
    - (4): sort the sounds by ascending order of distance to the sound with the highest parameter value
    for each preset (highest rank)
    - (5): compute the Spearman's rank correlation coefficients for the ranking obtained in (3) and (4)
    for each preset and take the mean.

    Args
    - `path_to_dataset` (Union[Path, str]): The path to the dataset containing the sound attributes dataset.
    - `encoder` (nn.Module): The encoder model to use to generate the embeddings.
    - `distance_fn` (str): The distance used for computing the distance matrix.
    - `reduce_fn` (str): The reduction function to use for reducing the embeddings dimensionality.

    Returns
        Dict[str, float]: A dictionary containing the correlation coefficients for each parameter variation,
        as well as the mean and median correlation coefficients.
    """
    path_to_dataset = (
        Path(path_to_dataset) if isinstance(path_to_dataset, str) else path_to_dataset
    )

    available_atributes = sorted([p.stem for p in path_to_dataset.iterdir()])
    if ".DS_Store" in available_atributes:
        available_atributes.remove(".DS_Store")

    corrcoeffs = {}

    for attribute in available_atributes:
        presets = sorted([p.stem for p in (path_to_dataset / attribute).iterdir()])
        if ".DS_Store" in presets:
            presets.remove(".DS_Store")

        corrcoeff_up = np.zeros(len(presets))
        corrcoeff_down = np.zeros(len(presets))

        for i, preset in enumerate(presets):
            path_to_audio = path_to_dataset / attribute / str(preset)

            corrcoeff_up[i], corrcoeff_down[i] = _compute_corrcoeff_for_preset(
                path_to_audio, encoder, distance_fn, reduce_fn
            )

        corrcoeff_up = corrcoeff_up.mean()
        corrcoeff_down = corrcoeff_down.mean()

        corrcoeffs[attribute] = (corrcoeff_up + corrcoeff_down) / 2

        log.info(
            f"Mean Spearmann correlation coefficient for attribute `{attribute}`: {corrcoeffs[attribute]}"
        )

    # compute the median correlation coefficient
    corrcoeffs["mean"] = np.mean(list(corrcoeffs.values()))
    corrcoeffs["median"] = np.median(list(corrcoeffs.values()))

    return corrcoeffs


def _compute_corrcoeff_for_preset(
    path_to_audio: Union[Path, str],
    encoder: nn.Module,
    distance_fn: str,
    reduce_fn: str,
) -> Tuple[float, float]:
    dataset = SoundAttributesDataset(path_to_audio=path_to_audio)

    embeddings, ranks = compute_embeddings(
        encoder=encoder,
        dataloader=DataLoader(dataset, batch_size=len(dataset), shuffle=False),
        num_samples=-1,
        data_sample_rate=dataset.sample_rate,
        encoder_sample_rate=encoder.sample_rate,
        encoder_channels=encoder.channels,
        encoder_frame_length=encoder.segment,
        pbar=False,
    )

    embeddings = getattr(r_fn, reduce_fn)(embeddings)

    distance_matrix = getattr(tm_functional, distance_fn)(embeddings)

    indices = torch.argsort(
        distance_matrix,
        dim=1,
        descending=distance_fn == "pairwise_cosine_similarity",
    )

    ranking_target = torch.tensor(ranks[1:]).float().to(DEVICE)

    corrcoeff_up = pearson_corrcoef(
        preds=indices[0, 1:].float(), target=ranking_target
    ).item()

    corrcoeff_down = pearson_corrcoef(
        preds=indices[-1, 1:].float(), target=ranking_target.flip(0) - 1
    ).item()

    return corrcoeff_up, corrcoeff_down


if __name__ == "__main__":
    import sys
    import os
    from models.encodec.encoder import EncodecEncoder

    # set torch device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ###### to move in hydra config
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
    PATH_TO_CKPT = PROJECT_ROOT / "checkpoints"
    PATH_TO_DATASET = (
        PROJECT_ROOT / "data" / "TAL-NoiseMaker" / "sound_attributes_ranking_dataset"
    )

    DISTANCE_FN = "pairwise_manhattan_distance"

    REDUC_FN = "flatten"

    encoder = EncodecEncoder.encodec_model_48khz(
        repository=PATH_TO_CKPT, segment=1.0, overlap=0.0
    )

    corrcoeff = sound_attributes_ranking_eval(
        PATH_TO_DATASET, encoder, DISTANCE_FN, REDUC_FN
    )
    print("breakpoint me!")
