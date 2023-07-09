# pylint: disable=E1101:no-member,C0116:missing-function-docstring
import torch
import pytest
from torchmetrics.functional import pairwise_euclidean_distance

from src.utils.distances import global_argsort


def test_raises_error_for_invalid_dist_mat():
    dist_mat = torch.tensor([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError):
        global_argsort(dist_mat)


def test_returns_indices_of_sorted_dist_matrix():
    dist_mat = torch.tensor([[0.0, 0.5, 0.8], [0.5, 0.0, 0.6], [0.8, 0.6, 0.0]])
    expected_indices = torch.tensor([[0, 1], [1, 2], [0, 2]])
    assert torch.all(global_argsort(dist_mat) == expected_indices)


def test_returns_sorted_values_if_requested():
    dist_mat = torch.tensor([[0.0, 0.5, 0.8], [0.5, 0.0, 0.6], [0.8, 0.6, 0.0]])
    expected_indices = torch.tensor([[0, 1], [1, 2], [0, 2]])
    expected_dists = torch.tensor([0.5, 0.6, 0.8])
    sorted_dists, sorted_indices = global_argsort(dist_mat, return_sorted_values=True)
    assert torch.all(sorted_indices == expected_indices)
    assert torch.all(sorted_dists == expected_dists)


def test_returns_sorted_indices_in_descending_order():
    dist_mat = torch.tensor([[0.0, 0.5, 0.8], [0.5, 0.0, 0.6], [0.8, 0.6, 0.0]])
    expected_indices = torch.tensor([[0, 2], [1, 2], [0, 1]])
    assert torch.all(global_argsort(dist_mat, descending=True) == expected_indices)


def test_sort_synthetic_data_correctly():
    samples = torch.rand((100, 512))
    pairwise_dists = pairwise_euclidean_distance(samples)
    sorted_indices = global_argsort(pairwise_dists)
    current_dist = 0.0
    for i, j in sorted_indices:
        assert pairwise_euclidean_distance(samples[None, i], samples[None, j]) >= current_dist
        current_dist = pairwise_euclidean_distance(samples[None, i], samples[None, j])


def test_returns_correct_number_of_indices():
    samples = torch.rand((100, 512))
    pairwise_dists = pairwise_euclidean_distance(samples)
    sorted_indices = global_argsort(pairwise_dists)
    assert sorted_indices.shape[0] == 100 * (100 - 1) / 2


def test_correct_returns_if_zero_euclidean_distance():
    dist_mat = torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, 0.6], [0.8, 0.6, 0.0]])
    expected_indices = torch.tensor([[0, 1], [1, 2], [0, 2]])
    expected_dists = torch.tensor([0.0, 0.6, 0.8])
    sorted_dists, sorted_indices = global_argsort(dist_mat, return_sorted_values=True)
    assert torch.all(sorted_indices == expected_indices)
    assert torch.all(sorted_dists == expected_dists)


def test_correct_returns_if_one_cosine_distance():
    dist_mat = torch.tensor([[0.0, 0.5, 1.0], [0.5, 0.0, 0.6], [1.0, 0.6, 0.0]])
    expected_indices = torch.tensor([[0, 2], [1, 2], [0, 1]])
    expected_dists = torch.tensor([1.0, 0.6, 0.5])
    sorted_dists, sorted_indices = global_argsort(dist_mat, return_sorted_values=True, descending=True)
    assert torch.all(sorted_indices == expected_indices)
    assert torch.all(sorted_dists == expected_dists)
