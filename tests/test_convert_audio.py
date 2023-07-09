# pylint: disable=E1101:no-member
import torch
import torchaudio
from src.utils.audio import convert_audio


def test_convert_audio():
    # Test if the function returns a tensor with the expected number of channels and duration.
    audio = torch.zeros((10, 2, 16_000))
    output = convert_audio(audio, 16_000, 16_000, 1, 1.0)
    assert output.shape == (10, 1, 16_000)

    # Test if the function returns a tensor with the expected sample rate.
    audio = torch.zeros((10, 2, 22_050))
    output = convert_audio(audio, 22_050, 16_000, 1, 1.0)
    assert output.shape == (10, 1, 16_000)

    # Test if the function returns a tensor with the expected shape when the input audio is mono.
    audio = torch.zeros((10, 1, 16_000))
    output = convert_audio(audio, 16_000, 16_000, 2, 1.0)
    assert output.shape == (10, 2, 16_000)

    # Test if the function returns a tensor with the expected shape when the input audio is stereo.
    audio = torch.zeros((10, 2, 16_000))
    output = convert_audio(audio, 16_000, 16_000, 1, 1.0)
    assert output.shape == (10, 1, 16_000)

    # Test if the function returns a tensor with the expected shape when the input audio is truncated.
    audio = torch.zeros((10, 1, 32_000))
    output = convert_audio(audio, 16_000, 16_000, 1, 1.0)
    assert output.shape == (10, 1, 16_000)

    # Test if the function returns a tensor with the expected shape when the input audio is padded.
    audio = torch.zeros((10, 1, 8000))
    output = convert_audio(audio, 16_000, 16_000, 1, 1.0)
    assert output.shape == (10, 1, 16_000)

    # Test if the function returns a tensor with the expected shape when the input audio is normalized.
    audio = torch.ones((10, 1, 16_000))
    output = convert_audio(audio, 16_000, 16_000, 2, 1.0, normalize=True)
    assert torch.equal(output.abs().amax(dim=(-2, -1)), torch.tensor(0.99).repeat(10))

    # Test if the function raises an assertion error when the input audio has more than 2 channels.
    audio = torch.zeros((10, 3, 16_000))
    try:
        output = convert_audio(audio, 16_000, 16_000, 1, 1.0)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError but function completed successfully.")


def test_convert_audio_trunc():
    audio = torch.ones((10, 2, 16_000))
    output = convert_audio(audio, 16_000, 16_000, 1, 0.5)
    fadeout_num_samples = int(0.1 * 16_000)
    assert torch.equal(
        output[0, -1, output.shape[-1] - fadeout_num_samples :], torch.linspace(1, 0, fadeout_num_samples)
    )

    audio = torch.ones((10, 2, 16_000))
    output = convert_audio(audio, 16_000, 16_000, 1, 2.0)

    assert torch.equal(output[0, -1, -16_000:], torch.zeros(16_000))
