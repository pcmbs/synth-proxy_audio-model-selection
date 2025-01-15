"""
Embeddings reduction functions.
"""

import torch

# TODO: try to process the embeddings as follow:
# - concat features from different layers (see papers in Zotero)
# - random projection


def flatten(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Flatten the input tensor along along the batch axis.

    Args
    - `embeddings` (torch.Tensor): The input tensor to be flattened.

    Returns
    - `torch.Tensor`: Flattened tensor).
    """
    return embeddings.flatten(start_dim=1)


def global_avg_pool_channel(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Calculates the global average pooling over the channel dimension of the input tensor.

    Args
    - `embeddings` (torch.Tensor): The input tensor.

    Returns
    - `torch.Tensor`: The output tensor.
    """
    return embeddings.mean(-2)


def global_avg_pool_time(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute the global average pooling of the given embeddings.
    Note that when applied to a ViT-based model,
    this operation is equivalent to the average pooling over the patches.

    Args
    - `embeddings` (torch.Tensor): The input embeddings tensor.

    Returns:
        `torch.Tensor: Output tensor.
    """
    return embeddings.mean(-1)


def global_max_pool_channel(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Calculates the global average pooling over the channel dimension of the input tensor.

    Args
    - `embeddings` (torch.Tensor): The input tensor.

    Returns
    - `torch.Tensor`: The output tensor.
    """
    return embeddings.amax(-2)


def global_max_pool_time(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute the global max pooling of the given embeddings.
        Note that when applied to a ViT-based model,
    this operation is equivalent to the max pooling over the patches.

    Args
    - `embeddings` (torch.Tensor): The input embeddings tensor.

    Returns:
        `torch.Tensor: Output tensor.
    """
    return embeddings.amax(-1)


def identity(x: torch.Tensor) -> torch.Tensor:
    """
    Do not apply any reduction on the input tensor.
    """
    return x


if __name__ == "__main__":
    embeddings = torch.rand((10, 128, 500))

    print("original:", embeddings.shape)
    print("flatten:", flatten(embeddings).shape)
    print("global_avg_pool_channel:", global_avg_pool_channel(embeddings).shape)
    print("global_avg_pool_time:", global_avg_pool_time(embeddings).shape)
    print("global_max_pool_channel:", global_max_pool_channel(embeddings).shape)
    print("global_max_pool_time:", global_max_pool_time(embeddings).shape)
