# pylint: disable=E1101:no-member
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from .audio import convert_audio


def compute_embeddings(
    encoder: nn.Module,
    dataloader: DataLoader,
    num_samples: int,
    data_sample_rate: int,
    encoder_sample_rate: int,
    encoder_channels: int,
    encoder_frame_length: float,
    device: str,
    pbar: bool = True,
) -> tuple[torch.Tensor, list[int]]:
    """
    Compute embeddings for a given encoder and dataloader.

    Args:
        encoder (nn.Module): The encoder module used to compute embeddings.
        dataloader (DataLoader): The dataloader containing the data samples.
        num_samples (int): The total number of samples to generate embeddings for.
        Pass a negative value to generate embeddings for all samples.
        data_sample_rate (int): The sample rate of the data.
        encoder_sample_rate (int): The desired sample rate for the encoder.
        encoder_channels (int): The number of input channels for the encoder.
        encoder_frame_length (float): The desired frame length for the encoder.
        device (str): The device to perform computations on.
        pbar (bool): Whether to display a progress bar.

    Returns:
        tuple[torch.Tensor, list[int]]: A tuple containing the computed embeddings
        and a list of metadata returned by the dataloader.
    """
    # initialize tensor to store embeddings and a list of indices returned by the dataloader
    embeddings = []
    metadata = []

    num_batches = (
        num_samples // dataloader.batch_size if num_samples > -1 else len(dataloader)
    )

    if pbar:
        pbar = tqdm(
            enumerate(dataloader),
            dynamic_ncols=True,
            desc="Computing embeddings",
            total=num_batches,
        )
    else:
        pbar = enumerate(dataloader)

    with torch.no_grad():
        for i, batch in pbar:
            if i >= num_batches:
                break

            audio, batch_metadata = batch
            audio = convert_audio(
                audio=audio,
                sample_rate=data_sample_rate,
                target_sample_rate=encoder_sample_rate,
                target_channels=encoder_channels,
                target_duration=encoder_frame_length,
            )
            audio = audio.to(device)

            batch_embeddings = encoder(audio)[0].detach()
            embeddings.append(batch_embeddings)
            metadata.extend(batch_metadata)

    return torch.cat(embeddings, dim=0), metadata
