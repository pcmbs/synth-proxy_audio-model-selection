# pylint: disable=E1101:no-member
"""
Torch Dataset class for the TAL-Noisemaker sound attributes dataset.
"""
import logging
from pathlib import Path
from typing import Union
import torch
import torchaudio
from torch.utils.data import Dataset

# logger for this file
log = logging.getLogger(__name__)


class NoisemakerSoundAttributesDataset(Dataset):
    """
    Torch Dataset class for the TAL-Noisemaker sound attributes dataset.
    """

    def __init__(
        self,
        root: Union[Path, str],
        sound_attribute: str,
    ):
        """
        Args
        - `root` (Union[Path, str]): The path to the sound attributes dataset.
        - `sound_attribute` (str): name of the sound attribute used for evaluation.
        """

        root = Path(root) if isinstance(root, str) else root
        if not root.is_dir():
            raise ValueError(f"{root} is not a directory.")

        available_attributes = sorted([p.stem for p in root.iterdir()])
        if ".DS_Store" in available_attributes:
            available_attributes.remove(".DS_Store")

        if sound_attribute not in available_attributes:
            raise ValueError(
                f"'{sound_attribute}' is not a valid sound attribute. "
                f"Available attributes: {available_attributes}"
            )
        self._sound_attribute = sound_attribute

        self._path_to_audio = root / f"{sound_attribute}"

        self._file_stems = sorted([p.stem for p in self._path_to_audio.glob("*.wav")])

        with open(self._path_to_audio / f"{self._file_stems[0]}.wav", "rb") as f:
            tmp_audio, self._sample_rate = torchaudio.load(f)

        self._audio_length = tmp_audio.shape[1] // self._sample_rate

        log.info(str(self))

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def audio_length(self):
        return self._audio_length

    @property
    def sound_attribute(self):
        return self._sound_attribute

    @property
    def path_to_audio(self):
        return self._path_to_audio

    @property
    def file_stems(self):
        return self._file_stems

    def __len__(self):
        return len(self.file_stems)

    def __str__(self):
        return f"NoisemakerSoundAttributesDataset: {len(self)} samples found for attribute `{self.sound_attribute}`."

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        name = self.file_stems[index]
        rank = int(name.split("-")[-1])
        with open(self._path_to_audio / f"{name}.wav", "rb") as f:
            audio, _ = torchaudio.load(f)
        return audio, rank


if __name__ == "__main__":
    print("breakpoint me!")
