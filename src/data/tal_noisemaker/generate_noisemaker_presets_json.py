"""
Script used to extract TAL-Noisemaker factory presets and store them in a json file.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pedalboard import load_plugin

load_dotenv()

####################### Useful links
### Pedalboard
# https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.load_plugin
# https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.VST3Plugin

# parameters are contained in the synth.parameters dictionary.
# However, parameters can be of different types (e.g., bool, float, string) which
# is not cool for dawdreamer, hence parameter float values are accessed using
# synth._parameters.raw_value

# TAL-NoiseMaker can be downloaded at:
# https://tal-software.com/products/tal-noisemaker

# TAL-NoiseMaker presets can be downloaded at:
# https://tal-software.com//downloads/presets/TAL-NoiseMaker%20vst3.zip

####################### Setup
EXPORT_PRESETS = False

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
PATH_TO_TAL_FOLDER = PROJECT_ROOT / "data" / "TAL-NoiseMaker"
PATH_TO_PLUGIN = PATH_TO_TAL_FOLDER / "TAL-NoiseMaker.vst3"
PATH_TO_PRESETS = PATH_TO_TAL_FOLDER / "TAL-NoiseMaker vst3"

# path to TAL-NoiseMaker presets
paths_to_presets = sorted(PATH_TO_PRESETS.glob("*.vstpreset"))
print(f"Found {len(paths_to_presets)} presets")

# Load TAL-NoiseMaker
synth = load_plugin(
    path_to_plugin_file=str(PATH_TO_PLUGIN),
    plugin_name="TAL-NoiseMaker",
)

####################### export presets

presets = {}

for preset in paths_to_presets:
    preset_name = "default" if preset.stem == "! Startup Juno Osc TAL" else preset.stem
    synth.load_preset(str(preset))
    presets[preset_name] = []
    for param in synth._parameters[:89]:  # pylint: disable=W0212
        presets[preset_name].append(
            {"index": param.index, "name": param.name, "value": param.raw_value}
        )

if EXPORT_PRESETS:
    with open(
        PATH_TO_TAL_FOLDER / "tal_noisemaker_presets.json", "w", encoding="utf-8"
    ) as f:
        json.dump(presets, f)

print(f"noisemaker_presets.json saved in {str(PATH_TO_TAL_FOLDER)}")
