# pylint: disable=E1101:no-member
import pytest
import torch
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance

from src.utils.distances import nearest_neighbor


def test_raises_error_for_invalid_sim_mat():
    sim_mat = torch.tensor([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError):
        nearest_neighbor(sim_mat, num_samples=5)


def test_correctness_using_euclidean_distance():
    embeddings = torch.rand((100, 512))
    dist_mat = pairwise_euclidean_distance(embeddings)
    indices = nearest_neighbor(dist_mat, num_samples=20)

    for anchor in indices:
        current_dist = 0.0
        for i, j in anchor:
            assert dist_mat[i, j] >= current_dist
            current_dist = dist_mat[i, j]


def test_correctness_using_cosine_similarity():
    embeddings = torch.rand((100, 512))
    dist_mat = pairwise_cosine_similarity(embeddings)
    indices = nearest_neighbor(dist_mat, num_samples=20, descending=True)

    for anchor in indices:
        current_dist = 1.0
        for i, j in anchor:
            assert dist_mat[i, j] <= current_dist
            current_dist = dist_mat[i, j]
