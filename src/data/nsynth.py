# pylint: disable=E1101:no-member,W1203
"""
Torch Dataset class for NSynth dataset
Adapted from: https://github.com/morris-frank/nsynth-pytorch/blob/master/nsynth/data.py 
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


class NSynthDataset(Dataset):
    """
    Dataset to handle the NSynth data in json/wav format.
    """

    def __init__(
        self,
        root: Union[Path, str],
        subset: str = "train",
        families: Optional[Union[str, List[str]]] = None,
        sources: Optional[Union[str, List[str]]] = None,
    ):
        """
        Args
            root (Union[Path, str]): The path to the dataset.
            Should contain the sub-folders for the splits as extracted from the .tar.gz.
            subset (str): The subset to use. Must be one of "valid", "test", "train".
            families (Optional[Union[str, List[str]]]): Only keep those Instrument families.
            Valid families: "bass", "brass", "flute", "guitar", "keyboard", "mallet",
            "organ", "reed", "string", "synth_lead", "vocal"
            sources (Optional[Union[str, List[str]]]): Only keep those instrument sources
            Valid sources: "acoustic", "electric", "synthetic".
        """
        self.subset = subset.lower()

        if isinstance(families, str):
            families = [families]
        if isinstance(sources, str):
            sources = [sources]

        assert self.subset in ["valid", "test", "train"]

        self.root = root / f"nsynth-{subset}"
        if not self.root.is_dir():
            raise ValueError("The given root path is not a directory." f"\nI got {self.root}")

        if not (self.root / "examples.json").is_file():
            raise ValueError("The given root path does not contain an `examples.json` file.")

        log.info(f"Loading NSynth data from split {self.subset} at {self.root}")

        with open(self.root / "examples.json", "r", encoding="utf-8") as f:
            self.attrs = json.load(f)

        if families:
            self.attrs = {k: a for k, a in self.attrs.items() if a["instrument_family_str"] in families}
        if sources:
            self.attrs = {k: a for k, a in self.attrs.items() if a["instrument_source_str"] in sources}

        log.info(f"\tFound {len(self)} samples.")

        files_on_disk = set(map(lambda x: x.stem, self.root.glob("audio/*.wav")))
        if not set(self.attrs) <= files_on_disk:
            raise FileNotFoundError

        self.names = list(self.attrs.keys())

    @property
    def sample_rate(self):
        return 16_000

    def __len__(self):
        return len(self.attrs)

    def __str__(self):
        return f"NSynthDataset: {len(self):>7} samples in subset {self.subset}"

    def __getitem__(self, index: int):
        name = self.names[index]
        # returning attrs (for analysis) raised an error since the "quality_str"
        # key contains a list whose length vary from sample to sample
        # Hence, return the index and get the corresponding attrs using get_attrs()
        # attrs = self.attrs[name]
        with open(self.root / "audio" / f"{name}.wav", "rb") as f:
            audio, _ = torchaudio.load(f)

        return audio, index

    def get_attrs(self, indices: torch.Tensor) -> Union[dict, List[dict]]:
        """
        Returns a list of attributes corresponding to the given indices.

        Args:
            indices (torch.Tensor): A 1D tensor or a scalar tensor of indices.

        Returns:
            Union[dict, List[dict]]: The attributes corresponding to the given indices.
        """
        if indices.ndim == 0:
            return self.attrs[self.names[indices]]

        attrs = [self.attrs[self.names[i]] for i in indices]
        return attrs


if __name__ == "__main__":
    DATA_DIR = Path("C:/Users/paolo/Desktop/synth_datasets/nsynth")

    nsynth_dataset = NSynthDataset(root=DATA_DIR, sources="synthetic")
    nsynth_dataloader = DataLoader(nsynth_dataset, batch_size=512, shuffle=True)

    for i, batch in enumerate(nsynth_dataloader):
        wav, idxs = batch
        attributes = nsynth_dataset.get_attrs(idxs)

        if i == 10:
            break

    print("nsynth.py run successfully.")
