# pylint: disable=E1101:no-member
"""
Torch Dataset class for the TAL-Noisemaker parameter variations dataset
"""

import json
import logging
from pathlib import Path
from typing import Union, List, Optional

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

# logger for this file
log = logging.getLogger(__name__)


class NoisemakerVariationsDataset(Dataset):
    """
    Torch Dataset class for the TAL-Noisemaker parameter variations dataset.
    For each variation_type, a single parameter is modified to a specific interval.
    """

    def __init__(
        self,
        root: Union[Path, str],
        variation_type: str,
    ):
        """
        Args
        - `root` (Union[Path, str]): The path to the parameter variations dataset.
        - `variation_type` (str): name of the parameter used for evaluation.
        """

        root = Path(root) if isinstance(root, str) else root
        if not root.is_dir():
            raise ValueError(f"{root} is not a directory.")

        available_variations = sorted([p.stem for p in root.iterdir()])
        if ".DS_Store" in available_variations:
            available_variations.remove(".DS_Store")

        if variation_type not in available_variations:
            raise ValueError(
                f"'{variation_type}' is not a valid variation type. "
                f"Available variations: {available_variations}"
            )
        self._variation_type = variation_type

        self._path_to_audio = root / f"{variation_type}"

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
    def variation_type(self):
        return self._variation_type

    @property
    def path_to_audio(self):
        return self._path_to_audio

    @property
    def file_stems(self):
        return self._file_stems

    def __len__(self):
        return len(self.file_stems)

    def __str__(self):
        return f"NoisemakerVariationsDataset: {len(self)} samples for variation `{self.variation_type}`."

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        name = self.file_stems[index]
        rank = int(name.split("-")[-1])
        with open(self._path_to_audio / f"{name}.wav", "rb") as f:
            audio, _ = torchaudio.load(f)
        return audio, rank


if __name__ == "__main__":
    # import os
    # import sys
    # from dotenv import load_dotenv

    # # add parents directory to sys.path.
    # sys.path.insert(1, str(Path(__file__).parents[2]))
    # from src.utils.embeddings import get_embeddings
    # from src.models.encodec.encoder import EncodecEncoder

    # load_dotenv()
    # PATH_TO_DATASET = (
    #     Path(os.getenv("PROJECT_ROOT"))
    #     / "data"
    #     / "TAL-NoiseMaker"
    #     / "parameter_variations"
    # )
    # VARIATION_TYPE = "amp_attack"

    # dataset = NoisemakerVariationsDataset(
    #     root=PATH_TO_DATASET, variation_type=VARIATION_TYPE
    # )

    # dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # # encoder = EncodecEncoder()

    # # audio, ranks =

    print("breakpoint me!")
