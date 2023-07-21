"""
Module containing all parameter variations for the TAL-NoiseMaker plugin 
in order to evaluate audio models representation.

Run as main to print parameter indices.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ParameterVariation:
    """
    Dataclass to hold a single parameter variation for a single preset.

    Args:
    - `preset` (str): preset name
    - `base_param_name` (str):name of the parameter to modify
    - `param_idx` (int): index of the parameter to modify
    - `interval` (tuple[float, float]): interval on which to modify the parameter
    - `extra_params` (tuple[tuple[int, float],...]): extra parameters to modify
    (in addition to the ones in the ParameterVariationList object)
    """

    preset: str
    # base_param_name: str
    # param_idx: int
    interval: Tuple[float, float]
    extra_params: Optional[Tuple[Tuple[int, float], ...]] = None


@dataclass
class ParameterVariationList:
    """
    Dataclass to hold a list of ParameterVariation objects.

    Args:
    - `base_param_name` (str):name of the parameter to modify
    - `param_idx` (int): index of the parameter to modify
    - `variations` (list[ParameterVariation]): list of parameter variations.
    - `extra_params` (tuple[tuple[int, float],...]): extra parameters to modify common to all variations.
    """

    base_param_name: str
    param_idx: int
    variations: List[ParameterVariation]
    extra_params: Tuple[Tuple[int, float], ...]


amp_attack = ParameterVariationList(
    base_param_name="Amp Attack",
    param_idx=11,
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    extra_params=((46, 1.0), (48, 1.0)),
    variations=[
        ParameterVariation(  # amp_attack_0
            preset="default",
            interval=(0.0, 0.8),
        ),
        ParameterVariation(  # amp_attack_1
            preset="default",
            interval=(0.0, 0.8),
            # set `Osc 1 Waveform` to 0.5 (square wave)
            extra_params=((23, 0.5)),
        ),
        ParameterVariation(  # amp_attack_2
            preset="ARP 303 Like FN",
            interval=(0.0, 0.8),
        ),
        ParameterVariation(  # amp_attack_3
            preset="ARP C64 II FN",
            interval=(0.0, 0.8),
            # set `Reverb Wet` to 0
            extra_params=((60, 0.0)),
        ),
        ParameterVariation(  # amp_attack_4
            preset="BS 7th Gates FN",
            interval=(0.0, 0.8),
        ),
        ParameterVariation(  # amp_attack_5
            preset="BS Bass Deep TAL",
            interval=(0.0, 0.8),
        ),
        ParameterVariation(  # amp_attack_6
            preset="BS Dark Bass TAL",
            interval=(0.0, 0.8),
        ),
        ParameterVariation(  # amp_attack_7
            preset="CH Chordionator II FN",
            interval=(0.0, 0.8),
        ),
        ParameterVariation(  # amp_attack_8
            preset="DR Hat Short Moves TAL",
            interval=(0.0, 0.8),
        ),
        ParameterVariation(  # amp_attack_9
            preset="FX Noiz3machin3 FN",
            interval=(0.0, 0.8),
            # set `Reverb Wet` to 0
            extra_params=((60, 0.0)),
        ),
    ],
)


amp_decay = ParameterVariationList(
    base_param_name="Amp Decay",
    param_idx=12,
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Amp Sustain` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (13, 0.0)),
    variations=[
        ParameterVariation(  # amp_decay_0
            preset="CH Open Mind FN",
            interval=(0.2, 0.6),
            # set `Reverb Wet` to 0
            extra_params=((60, 0.0)),
        ),
        ParameterVariation(  # amp_decay_1
            preset="default",
            interval=(0.2, 0.6),
        ),
        ParameterVariation(  # amp_decay_2
            preset="default",
            interval=(0.2, 0.6),
            # set `Osc 1 Waveform` to 0.5 (square wave)
            extra_params=((23, 0.5)),
        ),
        ParameterVariation(  # amp_decay_3
            preset="KB Pop Pluck TUC",
            interval=(0.2, 0.6),
        ),
        ParameterVariation(  # amp_decay_4
            preset="KB Screetcher TUC",
            interval=(0.2, 0.6),
        ),
        ParameterVariation(  # amp_decay_5
            preset="LD Rasp Lead AS",
            interval=(0.2, 0.6),
        ),
        ParameterVariation(  # amp_decay_6
            preset="LD Steam Pulse AS",
            interval=(0.2, 0.6),
        ),
        ParameterVariation(  # amp_decay_7
            preset="PD Gated Pad TAL",
            interval=(0.2, 0.6),
            # set `Amp Attack` to 0.0
            extra_params=((11, 0.0)),
        ),
        ParameterVariation(  # amp_decay_8
            preset="DR Kick Old School TAL",
            interval=(0.2, 0.6),
            # set `Amp Attack` to 0.0
            extra_params=((11, 0.0)),
        ),
        ParameterVariation(  # amp_decay_9
            preset="FX Jumper TAL",
            interval=(0.2, 0.6),
            # set `Amp Attack` to 0.0
            extra_params=((11, 0.0)),
        ),
    ],
)

filter_cutoff = ParameterVariationList(
    base_param_name="Filter Cutoff",
    param_idx=3,
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0)),
    variations=[
        ParameterVariation(  # filter_cutoff_0
            preset="BS Deep Driver TAL",
            interval=(0.22, 0.8),
            # set `Filter Type` to 0.0 (LP 24)
            # set `Filter Decay` to 0.0
            extra_params=((2, 0.0), (8, 0.0)),
        ),
    ],
)


filter_decay = ParameterVariationList(
    base_param_name="Filter Decay",
    param_idx=8,
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0)),
    variations=[
        ParameterVariation(  # filter_decay_0
            preset="LD 8bitter II FN",
            interval=(0.4, 0.68),
            # set `Reverb Wet` to 0
            extra_params=((60, 0.0)),
        ),
    ],
)


filter_resonance = ParameterVariationList(
    base_param_name="Filter Resonance",
    param_idx=4,
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0)),
    variations=[
        ParameterVariation(  # filter_resonance_0
            preset="BS Clean Flat Bass TAL",
            interval=(0.2, 0.95),
            # set `Filter Type` to 0.0 (LP 24)
            extra_params=((2, 0.0)),
        ),
    ],
)
frequency_mod = ParameterVariationList(
    base_param_name="Osc 2 FM",
    param_idx=36,
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0)),
    variations=[
        ParameterVariation(  # frequency_mod_0
            preset="KB Smooth Sine TAL",
            interval=(0.0, 1.0),
            # set `Transpose` to 0.5 and `Delay Wet` to 0
            extra_params=((40, 0.5), (78, 0.0)),
        )
    ],
)

lfo_amount_on_filter = ParameterVariationList(
    base_param_name="Lfo 2 Amount",
    param_idx=31,
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0)),
    variations=[
        ParameterVariation(  # lfo_amount_on_filter_0
            preset="LD Power Lead TAL",
            interval=(0.33, 0.0),
            # set `Lfo 2 Rate` to 0.45
            extra_params=((29, 0.45)),
        ),
    ],
)


lfo_amount_on_volume = ParameterVariationList(
    base_param_name="Lfo 2 Amount",
    param_idx=31,
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0)),
    variations=[
        ParameterVariation(  # lfo_amount_on_volume_0
            preset="LD Mellow Chord TAL",
            interval=(0.75, 1.0),
            # set `Delay Wet` to 0.0
            extra_params=((78, 0.0)),
        ),
    ],
)


lfo_rate_on_filter = ParameterVariationList(
    base_param_name="Lfo 2 Rate",
    param_idx=29,
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0)),
    variations=[
        ParameterVariation(
            preset="LD Acid Dist Noisy TAL",
            interval=(0.14, 0.6),
            # set `Delay Wet` to 0.0
            extra_params=((78, 0.0)),
        ),
    ],
)


lfo_rate_on_volume = ParameterVariationList(
    base_param_name="Lfo 2 Rate",
    param_idx=29,
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0)),
    variations=[
        ParameterVariation(
            preset="LD Mellow Chord TAL",
            interval=(0.25, 0.66),
            # set `Lfo 2 Amount` to 0.95
            # set `Delay Wet` to 0.0
            extra_params=((31, 0.95), (78, 0.0)),
        ),
    ],
)

pitch_coarse = ParameterVariationList(
    base_param_name="Osc 1 Tune",
    param_idx=19,
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0)),
    variations=[
        ParameterVariation(
            preset="default",
            interval=(0.25, 0.5),
            # set `Osc 3 Volume` to 0
            extra_params=((17, 0.0)),
        ),
    ],
)

reverb = ParameterVariationList(
    base_param_name="Reverb Wet",
    param_idx=60,
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0)),
    variations=[
        ParameterVariation(
            preset="BS Mixmiddle FN",
            interval=(0.0, 1.0),
            # set `Lfo 1 Amount` to 0.5
            # set `Reverb Decay` to 0.5 and `Reverb Pre Delay` to 0
            extra_params=((30, 0.5), (61, 0.5), (62, 0.0)),
        ),
    ],
)

if __name__ == "__main__":
    # run as main to print parameter indices
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    from pedalboard import load_plugin

    load_dotenv()

    PATH_TO_PLUGIN = (
        Path(os.getenv("PROJECT_ROOT"))
        / "data"
        / "TAL-NoiseMaker"
        / "TAL-NoiseMaker.vst3"
    )

    # Load TAL-NoiseMaker
    synth = load_plugin(
        path_to_plugin_file=str(PATH_TO_PLUGIN),
        plugin_name="TAL-NoiseMaker",
    )

    parameter_indices = []
    for param in synth._parameters[:89]:  # pylint: disable=W0212
        print(f"index: {param.index}, name: {param.name}")
        parameter_indices.append({"index": param.index, "name": param.name})

    # print(parameter_indices)
