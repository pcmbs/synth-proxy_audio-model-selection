"""
Script used to extract TAL-Noisemaker factory presets and store them in a json file.
"""
from pathlib import Path
from pedalboard import load_plugin
import json

####################### Useful links
### Pedalboard
# https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.load_plugin
# https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.VST3Plugin

# parameters are contained in the synth.parameters dictionary.
# However, parameters can be of different types (e.g., bool, float, string) which
# is not cool for dawdreamer, hence parameter float values are accessed using
# synth._parameters.raw_value


####################### Setup
ROOT = Path("/Users/paolocombes/Desktop/pc_docs/MASTER_THESIS/external_datasets")
PATH_TO_PLUGIN = ROOT / "TAL-NoiseMaker.vst3"
PRESET_FOLDER = ROOT / "TAL-NoiseMaker vst3"


# path to TAL-NoiseMaker presets
paths_to_presets = sorted(PRESET_FOLDER.glob("*.vstpreset"))
print(f"Found {len(paths_to_presets)} presets")

# Load TAL-NoiseMaker
synth = load_plugin(
    path_to_plugin_file=str(PATH_TO_PLUGIN),
    plugin_name="TAL-NoiseMaker",
)

####################### export presets
# for i in range(89):  # parameters outside this range are not necessary
#     param = synth._parameters[i]  # pylint: disable=W0212
#     print(f"index: {i}, name: {param.name}")

presets = {}

for preset in paths_to_presets:
    preset_name = "default" if preset.stem == "! Startup Juno Osc TAL" else preset.stem
    synth.load_preset(str(preset))
    presets[preset_name] = []
    for param in synth._parameters[:89]:  # pylint: disable=W0212
        presets[preset_name].append(
            {"index": param.index, "name": param.name, "value": param.raw_value}
        )

with open(ROOT / "noisemaker_presets.json", "w", encoding="utf-8") as f:
    json.dump(presets, f)

print(f"noisemaker_presets.json saved in {str(ROOT)}")
