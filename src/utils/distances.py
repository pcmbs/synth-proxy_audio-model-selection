# pylint: disable=E1101:no-member
import torch


def global_argsort(
    distance_matrix: torch.Tensor,
    return_sorted_values: bool = False,
    descending: bool = False,
) -> torch.Tensor:
    """
    Given a distance matrix, this function returns the indices of the pairs ordered by distance.

    Args:
        distance_matrix (torch.Tensor): A symmetric 2D tensor representing the distance matrix
        return_sorted_values (bool): If True, returns a tuple of sorted distances and
        corresponding indices. Defaults to False.
        descending (bool): If True, returns the sorted indices in descending order.
        Defaults to False.

    Returns:
        torch.Tensor: A 2D tensor of shape (N*(N-1)/2, 2) where N=dist_mat.shape[0] and where
        the first and second column are the row and column indices of elements in the sorted order,
        respectively. If `return_sorted_values` is True, returns a tuple of sorted distances and
        corresponding indices.

    Raises:
        ValueError: If the given distance matrix is not 2D and symmetric.
    """
    _check_symmetry(distance_matrix)

    # generate a mask to only consider the upper triangular distance matrix
    # (diagonal excluded) to avoid duplicates
    mask = torch.triu(torch.ones_like(distance_matrix, dtype=torch.bool), diagonal=1)

    # compute the distance matrix's nonzero indices, order from left-to-right and top-to-bottom
    nonzero_indices = torch.nonzero(mask)
    # get the values to be sorted in a 1D tensor, i.e., the distance matrix's nonzero values
    pairwise_dists_to_sort = distance_matrix.masked_select(mask)

    # return the indices of the element from the 1D in the sorted order
    sorted_indices = pairwise_dists_to_sort.argsort(descending=descending)

    # return the indices of the element from the distance matrix in the sorted order
    # and additionally the sorted values if required
    if return_sorted_values:
        return pairwise_dists_to_sort[sorted_indices], nonzero_indices[sorted_indices]

    return nonzero_indices[sorted_indices]


def nearest_neighbor(
    distance_matrix: torch.Tensor, num_samples: int = None, descending: bool = False
) -> torch.Tensor:
    """
    Retrieve the nearest neighbors of `num_samples` randomly chosen samples from a distance matrix.

    Args:
        dist_mat (torch.Tensor): A 2D tensor representing the distance matrix.
        num_samples (int): The number of samples to compute nearest neighbors for. Compute for all if None
        (however it might be preferable to directly use torch.argosort for that). (Default: None)
        descending (bool, optional): A flag indicating to sort in descending order. (Default: False).

    Raises:
        ValueError: If the distance matrix is not 2D or symmetric.

    Returns:
        torch.Tensor: A 3D tensor of indices of the pairwise distances sorted for
        each randomly chosen samples. The tensor is of shape (num_samples, dist_mat.shape[0]-1, 2),
        where the last dimension's first and second indices are the row and column indices
        of elements in the distance matrix in the sorted order.

    """
    _check_symmetry(distance_matrix)

    num_rows = distance_matrix.shape[0]
    num_cols = distance_matrix.shape[1]

    # randomly choose samples to compute nearest neighbors for if required and
    # create a mask to omit diagonal entries and to avoid unnecessary sorting rows
    if num_samples is None:
        num_samples = num_rows
        mask = torch.ones_like(distance_matrix, dtype=torch.bool)
    else:
        sample_indices = torch.randperm(num_rows)[:num_samples]
        mask = torch.zeros_like(distance_matrix, dtype=torch.bool)
        mask[sample_indices] = True

    mask.fill_diagonal_(False)
    distances_to_sort = distance_matrix.masked_select(mask).reshape(
        num_samples, num_cols - 1
    )

    # retrieve indices of the pairwise distances to be sorted for each sample
    sorted_indices = torch.nonzero(mask).reshape(num_samples, num_cols - 1, 2)

    # sort the pairwise distances for each sample
    sorted_distance_indices = distances_to_sort.argsort(dim=-1, descending=descending)

    # retrieve the indices from the distance matrix in the sorted order for each anchor
    sorted_indices = sorted_indices[
        torch.arange(num_samples).unsqueeze(1), sorted_distance_indices
    ]

    return sorted_indices


def _check_symmetry(dist_mat: torch.Tensor) -> None:
    if dist_mat.dim() != 2 or dist_mat.shape[0] != dist_mat.shape[1]:
        raise ValueError("The given distance matrix must be 2D and square.")
    if not torch.allclose(dist_mat.transpose(0, 1), dist_mat):
        raise ValueError("The given distance matrix must be symmetric.")


if __name__ == "__main__":
    # TODO: move that in test should return 1
    from torchmetrics.functional import pairwise_manhattan_distance, pearson_corrcoef

    NUM_SAMPLES = 10

    embeddings = torch.rand((1, 20)) + torch.arange(NUM_SAMPLES).reshape(NUM_SAMPLES, 1)

    dist_mat = pairwise_manhattan_distance(embeddings)

    # indices = nearest_neighbor(dist_mat)

    # ranking_from_first = indices[0, :, 1].type(torch.float)
    # ranking_from_last = indices[-1, :, 1].type(torch.float)

    indices = torch.argsort(dist_mat, dim=1)

    ranking_from_first = indices[0, 1:].float()
    ranking_from_last = indices[-1, 1:].float()

    target_from_first = torch.arange(1, NUM_SAMPLES).type(torch.float)
    target_from_last = target_from_first.flip(0) - 1
    corrcoeff_from_first = pearson_corrcoef(ranking_from_first, target_from_first)
    corrcoeff_from_last = pearson_corrcoef(ranking_from_last, target_from_last)

    avg_corrcoeff = (corrcoeff_from_first + corrcoeff_from_last) / 2

    print("breakpoint me!")
