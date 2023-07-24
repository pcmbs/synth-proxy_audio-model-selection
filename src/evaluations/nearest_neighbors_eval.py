# pylint: disable=E1101:no-member,W1203
"""
Module containing functions for the nearest neighbors evaluation, which is 
used to evaluate the ability of a model to order sounds by "similarity".
"""

import logging
from typing import Any, List, Sequence

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset

import wandb
from data.nsynth.nsynth_dataset import NSynthDataset
from utils import reduce_fn as r_fn
from utils.distances import nearest_neighbors, iterative_distance_matrix
from utils.embeddings import compute_embeddings

# logger for this file
log = logging.getLogger(__name__)

WandbRunner = Any


def nearest_neighbors_eval(
    cfg: DictConfig,
    encoder: nn.Module,
    distance_fn: str,
    reduce_fn: str,
    logger: WandbRunner,
) -> None:
    """
    Nearest Neighbors evaluation, used to evaluate the ability of a model to order sounds by "similarity".
    The evaluation can be described as follows:
    - (1): randomly pick K samples from a sound corpus (here a subset of the nsynth-train dataset)
    - (2): get the representation of each sound from a given `encoder` and a reduction function `reduce_fn`
    - (3): pick I<<K mples
    - (4): compute and sort the pairwise distances for those samples in ascending order using `distance_fn`
    - (5): take the N nearest neighbors as well as N linearly spaced neighbors ranking from the closest to
    the farthest for each of the I samples and log the result into a wandb.Table for listening.

    Args
    - `cfg` (DictConfig): The hydra config containing the evaluation parameters.
    Must be cfg.eval.nearest_neighbors
    - `encoder` (nn.Module): The encoder model to use to generate the embeddings.
    - `distance_fn` (str): The distance used for computing the distance matrix.
    - `reduce_fn` (str): The reduction function to use for reducing the embeddings dimensionality.

    Returns
    - `None`
    """
    nsynth_dataset = NSynthDataset(root=cfg.data.root, sources=cfg.data.sources)

    nsynth_dataloader = DataLoader(
        nsynth_dataset, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle
    )

    embeddings, sample_indices = compute_embeddings(
        encoder=encoder,
        dataloader=nsynth_dataloader,
        num_samples=cfg.data.num_samples,
        data_sample_rate=nsynth_dataset.sample_rate,
        encoder_sample_rate=encoder.sample_rate,
        encoder_channels=encoder.channels,
        encoder_frame_length=encoder.segment,
    )

    embeddings = getattr(r_fn, reduce_fn)(embeddings)

    distance_matrix = iterative_distance_matrix(embeddings, distance_fn)

    sorted_indices = nearest_neighbors(
        distance_matrix,
        num_samples=cfg.metric.relative_pairwise_dists.num_samples,
        descending=distance_fn == "pairwise_cosine_similarity",
    )

    _log_neighbors(
        logger=logger,
        dataset=nsynth_dataset,
        num_neighbors=cfg.metric.relative_pairwise_dists.num_neighbors,
        distance_matrix=distance_matrix,
        sorted_indices=sorted_indices,
        sample_indices=sample_indices,
        mode=cfg.metric.relative_pairwise_dists.mode,
    )


def _log_neighbors(
    logger: WandbRunner,
    dataset: Dataset,
    num_neighbors: int,
    distance_matrix: torch.Tensor,
    sorted_indices: torch.Tensor,
    sample_indices: List[int],
    mode: Sequence[str] = ("nearest", "linspace"),
) -> None:
    """
    Logs the nearest or farthest n neighbors of a given point in the dataset or a set of n neighbors
    equally spaced using `linspace` mode.

    Args
    - `logger` (WandbRunner): The wandb Run object used for logging.
    - `dataset` (Dataset): The dataset from which the embeddings are extracted.
    - `num_neighbors` (int): Number of neighbors to log.
    - `distance_matrix` (torch.Tensor): The distance matrix from which to get the pairwise distances.
    - `sample_indices` (list): The indices as returned by the dataloader used to retrieve the
    original samples from the dataset
    - `mode` (Sequence[str], optional): Sequence of modes to use when selecting the neighbors.
    Can include "nearest" to log the n nearest neighbors, "farthest" to log the n farthest neighbors
    or "linspace" to log n neighbors equally spaced between the first and the last element of the indices.
    (Defaults: ("nearest", "linspace")).

    Returns:
        None
    """
    for current_mode in mode:
        if current_mode == "nearest":
            indices = sorted_indices[:, :num_neighbors]
        elif current_mode == "farthest":
            indices = sorted_indices[:, -num_neighbors:].flip(1)
        elif current_mode == "linspace":
            indices = sorted_indices[
                :,
                torch.linspace(
                    0, len(sample_indices) - 2, num_neighbors, dtype=torch.long
                ),
            ]
        else:
            raise ValueError(
                f"mode must be 'nearest', 'farthest', or 'linspace', not '{current_mode}'"
            )

        for pairs in indices:
            table = _generate_log_table(
                dataset=dataset,
                distance_matrix=distance_matrix,
                pairs=pairs,
                sample_indices=sample_indices,
            )
            anchor_str = dataset.names[sample_indices[pairs[0, 0]]]
            logger.log({f"neighbors/{anchor_str}/{current_mode}": table})


def _generate_log_table(
    dataset: Dataset,
    distance_matrix: torch.Tensor,
    pairs: torch.Tensor,
    sample_indices: List[int],
) -> wandb.Table:
    data_to_log = []
    columns = ["distance", "sample"]
    # add a new row containing the orignal sample (anchor) distance and audio
    data_to_log.append([0.0])
    data_to_log[0].append(_retrieve_sample(dataset, sample_indices[pairs[0, 0]]))

    # add a new row for each neighbor
    for anchor, neighbor in pairs:
        # add distance between anchor and neighbor and the neighbor's audio
        data_to_log.append([distance_matrix[anchor, neighbor].item()])
        data_to_log[-1].append(_retrieve_sample(dataset, sample_indices[neighbor]))

    return wandb.Table(data=data_to_log, columns=columns)


def _retrieve_sample(dataset: Dataset, sample_id: int):
    wav, index = dataset[sample_id]
    if index != sample_id:
        raise ValueError("Index and sample_id do not match.")
    return wandb.Audio(
        wav.squeeze_().cpu().detach().numpy(),
        sample_rate=dataset.sample_rate,
        caption=dataset.names[sample_id],
    )
