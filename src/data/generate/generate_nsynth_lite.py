"""
script to randomly select samples from NSynth dataset and export them
"""
import json
import shutil
from pathlib import Path
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

####################### Config

# number of samples to randomly select
N_DATA = 20_000

PROJECT_ROOT = os.getenv("PROJECT_ROOT")

# path to NSynth dataset
NSYNTH_PATH = Path(
    "/Users/paolocombes/Desktop/pc_docs/MASTER_THESIS/external_datasets/nsynth/nsynth-train"
)  # -> 953 different sources

# path to exported files
EXPORT_PATH = PROJECT_ROOT / "data" / "nsynth-lite" / "nsynth-train"

if not EXPORT_PATH.exists():  # create export path
    (EXPORT_PATH / "audio").mkdir(parents=True)

####################### NSynth lite dataset

samples_path = list((NSYNTH_PATH / "audio").glob("*.wav"))  # list paths to .wav files
print(f"Found {len(samples_path)} samples in {NSYNTH_PATH / 'audio'}")
# random permutation for selection
index_permutation = np.random.permutation(len(samples_path))

# NSynth attributes
with open(NSYNTH_PATH / "examples.json", "r", encoding="utf-8") as f:
    nsynth_attrs = json.load(f)
lite_attrs = {}  # init dict for NSynth lite dataset's attributes

print(f"Randomly picking {N_DATA} samples...")
for idx in index_permutation[: N_DATA - 1]:  # pick randomly N_DATA samples
    path = samples_path[idx]
    shutil.copyfile(
        NSYNTH_PATH / "audio" / path.name, EXPORT_PATH / "audio" / path.name
    )
    lite_attrs[path.stem] = nsynth_attrs[path.stem]  # copy attrs

# Export NSynth lite attributes
with open(EXPORT_PATH / "examples.json", "w", encoding="utf-8") as f:
    json.dump(lite_attrs, f)

print(f"Done exporting to {EXPORT_PATH}")

# check the number of different sources by iterating over the export folder
sources = []
for file in (EXPORT_PATH / "audio").glob("*.wav"):
    file = str(file.stem).split("-", maxsplit=1)[0]
    if file not in sources:
        sources.append(file)

print(
    f"final dataset contains samples from {len(sources)} different sources (out of 953)"
)
