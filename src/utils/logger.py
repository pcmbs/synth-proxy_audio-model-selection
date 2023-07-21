# pylint: disable=E1101:no-member,W1203
"""
Not used module. Might be deleted soon.
see evals/nearest_neighbors_eval.py instead
"""
import logging
from typing import Any, Sequence

import torch
import wandb
from torch.utils.data import Dataset
from utils.distances import global_argsort, nearest_neighbors

# logger for this file
log = logging.getLogger(__name__)

WandbRunner = Any


class LogAbsolutePairwiseDists:
    """
    Class to compute and sort absolute pairwise distances and log them in a wandb.Table
    """

    def __init__(
        self,
        logger: WandbRunner,
        distance_matrix: torch.Tensor,
        indices_from_batch: list,
        dataset: Dataset,
        descending: bool = False,
    ) -> None:
        """
        Args:
            logger (WandbRunner): The wandb Run object used for logging
            distance_matrix (torch.Tensor): The distance matrix from which to compute the pairwise distances.
            indices_from_batch (list): The indices as returned by the dataloader
            used to retrieve the original samples from the dataset
            metric_str (str): The metric used to compute the pairwise distances (only for logging purpose).
            dataset (Dataset): The dataset from which the embeddings are extracted.
            descending (bool): Whether to sort the distances in descending order. This should be set to true
            when using, e.g., `pairwise_cosine_similarity`.
        """
        self.logger = logger

        self.sorted_dists, self.sorted_indices = global_argsort(
            distance_matrix,
            return_sorted_values=True,
            descending=descending,
        )
        self.indices_from_batch = indices_from_batch

        self.dataset = dataset

    def log_n_pairs(self, n: int, mode: str = "similar") -> None:
        """
        Given an integer n and a mode, this function selects the top n most similar or dissimilar audio pairs.
        If mode is "similar", the top n most similar pairs are selected. If mode is "dissimilar",
        the n most dissimilar pairs are selected. If mode is "median", the n pairs around the median distance are selected.
        The function then generates a log table and logs it using wandb.
        If export_audio is True, the function also exports the audio files corresponding to the selected pairs.
        Args:
            n (int): The number of pairs to select.
            mode (str): The mode of selection. Must be "similar", "dissimilar", or "median". (Default: "similar").
        Returns:
            None.
        Raises:
            ValueError: If mode is not "similar", "dissimilar", or "median".
        """
        get_mode_indices = {
            "similar": self._get_n_most_similar(n),
            "dissimilar": self._get_n_most_dissimilar(n),
            "median": self._get_n_around_median(n),
        }
        for current_mode in mode:
            try:
                indices, distances = get_mode_indices[current_mode]
            except KeyError as exc:
                raise ValueError(
                    f"mode must be 'similar', 'dissimilar', or 'median', not '{current_mode}'"
                ) from exc

            table = self._generate_log_table(indices, distances)
            self.logger.log({f"absolute_pairwise_dist/{current_mode}": table})

    def _get_n_most_similar(self, n: int) -> torch.Tensor:
        indices = self.sorted_indices[:n]
        distances = self.sorted_dists[:n]
        return indices, distances

    def _get_n_most_dissimilar(self, n: int) -> torch.Tensor:
        indices = self.sorted_indices[-n:].flip(0)
        distances = self.sorted_dists[-n:].flip(0)
        return indices, distances

    def _get_n_around_median(self, n: int) -> torch.Tensor:
        index_median = (self.sorted_dists == self.sorted_dists.median()).argwhere()
        index_median = (
            index_median[0] if len(index_median) != 1 else index_median
        )  # if not unique
        lower_index = int(index_median - n // 2)
        upper_index = int(index_median + n // 2)
        indices_around_median = torch.arange(lower_index, upper_index)
        indices = self.sorted_indices[indices_around_median]
        distances = self.sorted_dists[indices_around_median]
        return indices, distances

    def _generate_log_table(
        self, indices: torch.Tensor, dists: torch.Tensor
    ) -> wandb.Table:
        data_to_log = []
        columns = ["distance", "sample 1", "sample_2"]
        for i, pair in enumerate(indices):
            data_to_log.append([])  # add a new row
            data_to_log[i].append(dists[i])  # add distance
            for k in pair:
                sample_id = self.indices_from_batch[k]
                wav, index = self.dataset[sample_id]
                if index != sample_id:
                    raise ValueError("Index and sample_id do not match.")
                data_to_log[i].append(  # add audio
                    wandb.Audio(
                        wav.squeeze_().cpu().detach().numpy(),
                        sample_rate=self.dataset.sample_rate,
                        caption=self.dataset.names[sample_id],
                    )
                )

        table = wandb.Table(data=data_to_log, columns=columns)
        return table


class LogRelativePairwiseDists:
    """
    Compute the nearest, farthest, or linspaced neighbors of `num_samples` randomly chosen samples
    and log them in a wandb.Table
    """

    def __init__(
        self,
        logger: WandbRunner,
        distance_matrix: torch.Tensor,
        indices_from_batch: list,
        num_samples: int,
        dataset: Dataset,
        descending: bool = False,
    ) -> None:
        """
        Args:
            logger (WandbRunner): The wandb Run object used for logging.
            distance_matrix (torch.Tensor): The distance matrix from which to compute the pairwise distances.
            indices_from_batch (list): The indices as returned by the dataloader
            used to retrieve the original samples from the dataset
            metric_str (str): The metric used to compute the pairwise distances (only for logging purpose).
            dataset (Dataset): The dataset from which the embeddings are extracted.
            descending (bool): Whether to sort the distances in descending order. This should be set to true
            when using, e.g., `pairwise_cosine_similarity`.
        """
        self.logger = logger

        self.distance_matrix = distance_matrix
        self.sorted_indices = nearest_neighbor(
            self.distance_matrix,
            num_samples=num_samples,
            descending=descending,
        )
        self.indices_from_batch = indices_from_batch

        self.dataset = dataset

    def log_n_neighbors(
        self, n: int, mode: Sequence[str] = ("nearest", "linspace")
    ) -> None:
        """
        Logs the nearest or farthest n neighbors of a given point in the dataset or a set of n neighbors
        equally spaced using `linspace` mode.

        Args:
            n (int): Number of neighbors to log.
            mode (Sequence[str], optional): Sequence of modes to use when selecting the neighbors.
            Can include "nearest" to log the n nearest neighbors, "farthest" to log the n farthest neighbors
            or "linspace" to log n neighbors equally spaced between the first and the last element of the indices.
            (Defaults: "nearest")

        Returns:
            None
        """
        get_mode_indices = {
            "nearest": self.sorted_indices[:, :n],
            "farthest": self.sorted_indices[:, -n:].flip(1),
            "linspace": self.sorted_indices[
                :,
                torch.linspace(
                    0, len(self.indices_from_batch) - 2, n, dtype=torch.long
                ),
            ],
        }

        for current_mode in mode:
            try:
                indices = get_mode_indices[current_mode]
            except KeyError as exc:
                raise ValueError(
                    f"mode must be 'nearest', 'farthest', or 'linspace', not '{current_mode}'"
                ) from exc

            for pairs in indices:
                table = self._generate_log_table(pairs)
                anchor_str = self.dataset.names[self.indices_from_batch[pairs[0, 0]]]
                self.logger.log({f"neighbors/{anchor_str}/{current_mode}": table})

    def _generate_log_table(self, pairs: torch.Tensor) -> wandb.Table:
        data_to_log = []
        columns = ["distance", "sample"]
        data_to_log.append([])  # add a new row for the orignal sample (anchor)
        data_to_log[0].append(0.0)  # add sample distance to itself
        # retrieve sample
        sample_id = self.indices_from_batch[pairs[0, 0]]
        wav, index = self.dataset[sample_id]
        if index != sample_id:
            raise ValueError("Index and sample_id do not match.")
        # add audio from sample
        data_to_log[0].append(
            wandb.Audio(
                wav.squeeze_().cpu().detach().numpy(),
                sample_rate=self.dataset.sample_rate,
                caption=self.dataset.names[sample_id],
            )
        )
        for i, (anchor, neighbor) in enumerate(pairs):
            data_to_log.append([])  # add a new row for each neighbor
            # add distance between anchor and neighbor
            data_to_log[i + 1].append(self.distance_matrix[anchor, neighbor].item())
            # retrieve neighbor
            sample_id = self.indices_from_batch[neighbor]
            wav, index = self.dataset[sample_id]
            if index != sample_id:
                raise ValueError("Index and sample_id do not match.")
            # add audio from sample
            data_to_log[i + 1].append(
                wandb.Audio(
                    wav.squeeze_().cpu().detach().numpy(),
                    sample_rate=self.dataset.sample_rate,
                    caption=self.dataset.names[sample_id],
                )
            )

        table = wandb.Table(data=data_to_log, columns=columns)
        return table


if __name__ == "__main__":
    print("logger.py run successfully")
