# pylint: disable=E1101:no-member
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.audio import convert_audio


def get_embeddings(
    encoder: nn.Module,
    dataloader: DataLoader,
    num_samples: int,
    data_sample_rate: int,
    encoder_sample_rate: int,
    encoder_channels: int,
    encoder_frame_length: float,
    device: str,
) -> tuple[torch.Tensor, list[int]]:
    """
    Compute embeddings for a given encoder and dataloader.

    Args:
        encoder (nn.Module): The encoder module used to compute embeddings.
        dataloader (DataLoader): The dataloader containing the data samples.
        num_samples (int): The total number of samples to generate embeddings for.
        data_sample_rate (int): The sample rate of the data.
        encoder_sample_rate (int): The desired sample rate for the encoder.
        encoder_channels (int): The number of input channels for the encoder.
        encoder_frame_length (float): The desired frame length for the encoder.
        device (str): The device to perform computations on.

    Returns:
        tuple[torch.Tensor, list[int]]: A tuple containing the computed embeddings and a list of indices returned by the dataloader.
    """
    # instantiate progress bar
    pbar = tqdm(
        enumerate(dataloader),
        dynamic_ncols=True,
        desc="Computing embeddings",
        total=num_samples // dataloader.batch_size,
    )

    # initialize tensor to store embeddings and a list of indices returned by the dataloader
    embeddings = torch.tensor([]).to(device)
    indices_from_batch = []

    with torch.no_grad():
        for i, batch in pbar:
            if i * dataloader.batch_size >= num_samples:
                break

            audio, indices = batch
            indices_from_batch.extend(indices)
            audio = convert_audio(
                audio=audio,
                sample_rate=data_sample_rate,
                target_sample_rate=encoder_sample_rate,
                target_channels=encoder_channels,
                target_duration=encoder_frame_length,
            )
            audio = audio.to(device)

            embeddings_from_batch = encoder(audio)[0].detach()
            embeddings = torch.cat([embeddings, embeddings_from_batch], dim=0)

    return embeddings, indices_from_batch
