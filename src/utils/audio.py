# pylint: disable=E1101:no-member
"""
Audio utils.
"""
import torch
from torch.nn.functional import pad
from torchaudio.functional import resample


def convert_audio(
    audio: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
    target_channels: int,
    target_duration: float,
    normalize: bool = False,
    fadeout_duration: float = 0.1,
) -> torch.Tensor:
    """
    Converts an audio tensor to the desired sample rate, number of channels, and duration using various transformations.

    Args:
        audio (torch.Tensor): The input audio tensor.
        sample_rate (int): The sample rate of the input audio tensor.
        target_sample_rate (int): The target sample rate to convert the input audio tensor to.
        target_channels (int): The target number of channels to convert the input audio tensor to.
        target_duration (float): The target duration of the output audio tensor in seconds.
        Note that the input audio tensor will be padded or truncated (with a fade out - see below)
        to the target length if necessary.
        normalize (bool): Whether to normalize the input audio tensor.
        fadeout_duration (float): The duration of the fadeout in seconds. If not specified, defaults to 100ms.

    Returns:
        torch.Tensor: The transformed audio tensor of the specified sample rate, number of channels, and duration.
    """
    assert audio.shape[-2] in [1, 2], "Audio must be mono or stereo."

    # convert to mono if required
    audio = audio.mean(-2, keepdim=True) if (target_channels == 1) and (audio.shape[-2] == 2) else audio

    # convert to stereo if required
    if (target_channels == 2) and (audio.shape[-2] == 1):
        audio = audio.expand(*audio.shape[:-2], target_channels, -1)

    # resample to target sample rate
    if sample_rate != target_sample_rate:
        audio = audio.clone()  # might raise an error without
        audio = resample(audio, sample_rate, target_sample_rate)
    audio = resample(audio, sample_rate, target_sample_rate)

    # truncate to target duration and apply fade out if required
    target_num_samples = int(target_duration * target_sample_rate)

    if audio.shape[-1] > target_num_samples:
        fadeout_num_samples = int(fadeout_duration * target_sample_rate)
        fadeout = torch.linspace(1, 0, fadeout_num_samples) if fadeout_num_samples > 0 else 1.0
        audio = audio[..., :target_num_samples]
        audio[..., -fadeout_num_samples:] *= fadeout

    # zero-pad to target duration if required
    elif audio.shape[-1] < target_num_samples:
        audio = pad(audio, (0, target_num_samples - audio.shape[-1]))

    # normalize if required
    audio = (audio / audio.abs().amax((-2, -1), keepdim=True)) * 0.99 if normalize else audio

    return audio


if __name__ == "__main__":
    print("utils/audio.py run successfully.")
