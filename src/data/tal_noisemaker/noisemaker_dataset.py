# pylint: disable=E1101:no-member
"""
Torch Dataset class for the TAL-Noisemaker sound attributes dataset.
"""
from pathlib import Path
from typing import Union
import torch
import torchaudio
from torch.utils.data import Dataset


class SoundAttributesDataset(Dataset):
    """
    Torch Dataset class for the TAL-Noisemaker sound attributes dataset.
    """

    def __init__(
        self,
        path_to_audio: Union[Path, str],
    ):
        """
        Args
        - `path_to_audio` (Union[Path, str]): The path to a single preset of a particular sound attribute.
        """

        path_to_audio = (
            Path(path_to_audio) if isinstance(path_to_audio, str) else path_to_audio
        )
        if not path_to_audio.is_dir():
            raise ValueError(f"{path_to_audio} is not a directory.")

        self._path_to_audio = path_to_audio

        self._file_stems = sorted([p.stem for p in self._path_to_audio.glob("*.wav")])

        with open(self._path_to_audio / f"{self._file_stems[0]}.wav", "rb") as f:
            tmp_audio, self._sample_rate = torchaudio.load(f)

        self._audio_length = tmp_audio.shape[1] // self._sample_rate

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def audio_length(self):
        return self._audio_length

    @property
    def path_to_audio(self):
        return self._path_to_audio

    @property
    def file_stems(self):
        return self._file_stems

    def __len__(self):
        return len(self.file_stems)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        name = self.file_stems[index]
        rank = int(name.split("-")[-1])
        with open(self._path_to_audio / f"{name}.wav", "rb") as f:
            audio, _ = torchaudio.load(f)
        return audio, rank


if __name__ == "__main__":
    print("breakpoint me!")
