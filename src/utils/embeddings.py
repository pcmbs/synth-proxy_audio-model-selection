# pylint: disable=E1101:no-member
from typing import Optional
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from .audio import convert_audio

# set torch device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_embeddings(
    encoder: nn.Module,
    dataloader: DataLoader,
    num_samples: int,
    data_sample_rate: int,
    encoder_sample_rate: int,
    encoder_channels: int,
    encoder_frame_length: Optional[float] = None,
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
        Pass None to if the desired frame length is the same as the input frame length.
        (Default: None)
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
            audio = audio.to(DEVICE)

            if encoder.name.startswith("encodec"):
                # returns a list of tensors (containing a single element since only a single frame is considered)
                # encodec requiress input audio of shape (n_sounds, n_channels, n_samples)
                # where n_channels=1 for encodec24khz and n_channels=2 for encodec48khz
                batch_embeddings = encoder(audio)[0]
            elif encoder.name.startswith("openl3"):
                # partial init such that only the input audio (and its sample rate) need to be passed
                # the audio was already resampled in convert_audio(), hence passing encoder_sample_rate
                # to not resample again
                # audio of shape (batch, time, channels) required
                batch_embeddings = encoder(audio.swapdims(-1, -2), encoder_sample_rate)
            elif encoder.name.startswith("passt"):
                # passt requires mono input audio of shape (n_sounds, n_samples)
                batch_embeddings = encoder(audio.squeeze(-2))
            else:
                raise NotImplementedError()

            embeddings.append(batch_embeddings.detach())
            metadata.extend(batch_metadata)

    return torch.cat(embeddings, dim=0), metadata
