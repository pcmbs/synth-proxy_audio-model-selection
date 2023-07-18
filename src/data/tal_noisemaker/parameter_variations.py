"""
Module containing all parameter variations for the TAL-NoiseMaker plugin 
in order to evaluate audio models representation.
"""

from dataclasses import dataclass


@dataclass
class ParameterVariation:
    """
    Dataclass to hold parameter variations.

    Args:
    - `preset` (str): preset name
    - `base_param_name` (str):name of the parameter to modify
    - `param_idx` (int): index of the parameter to modify
    - `interval` (tuple[float, float]): interval on which to modify the parameter
    - `extra_params` (tuple[tuple[int, float],...]): extra parameters to modify
    """

    preset: str
    base_param_name: str
    param_idx: int
    interval: tuple[float, float]
    extra_params: tuple[tuple[int, float], ...]


amp_attack = ParameterVariation(
    preset="default",
    base_param_name="Amp Attack",
    param_idx=11,
    interval=(0.0, 0.8),
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0)),
)

amp_decay = ParameterVariation(
    preset="CH Open Mind FN",
    base_param_name="Amp Decay",
    param_idx=12,
    interval=(0.2, 0.6),
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    # set `Reverb Wet` to 0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0), (60, 0.0)),
)

filter_cutoff = ParameterVariation(
    preset="BS Deep Driver TAL",
    base_param_name="Filter Cutoff",
    param_idx=3,
    interval=(0.22, 0.8),
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    # set `Filter Type` to 0.0 (LP 24)
    # set `Filter Decay` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0), (2, 0.0), (8, 0.0)),
)

filter_decay = ParameterVariation(
    preset="LD 8bitter II FN",
    base_param_name="Filter Decay",
    param_idx=8,
    interval=(0.4, 0.68),
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    # set `Reverb Wet` to 0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0), (60, 0.0)),
)

filter_resonance = ParameterVariation(
    preset="BS Clean Flat Bass TAL",
    base_param_name="Filter Resonance",
    param_idx=4,
    interval=(0.2, 0.95),
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    # set `Filter Type` to 0.0 (LP 24)
    # set `Crush` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0), (2, 0.0)),
)

frequency_mod = ParameterVariation(
    preset="KB Smooth Sine TAL",
    base_param_name="Osc 2 FM",
    param_idx=36,
    interval=(0.0, 1.0),
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    # set `Transpose` to 0.5 and `Delay Wet` to 0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0), (40, 0.5), (78, 0.0)),
)

lfo_amount_on_filter = ParameterVariation(
    preset="LD Power Lead TAL",
    base_param_name="Lfo 2 Amount",
    param_idx=31,
    interval=(0.33, 0.0),
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    # set `Lfo 2 Rate` to 0.45
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0), (29, 0.45)),
)

lfo_amount_on_volume = ParameterVariation(
    preset="LD Mellow Chord TAL",
    base_param_name="Lfo 2 Amount",
    param_idx=31,
    interval=(0.75, 1.0),
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    # set `Delay Wet` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0), (78, 0.0)),
)

lfo_rate_on_filter = ParameterVariation(
    preset="LD Acid Dist Noisy TAL",
    base_param_name="Lfo 2 Rate",
    param_idx=29,
    interval=(0.14, 0.6),
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    # set `Delay Wet` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0), (78, 0.0)),
)

lfo_rate_on_volume = ParameterVariation(
    preset="LD Mellow Chord TAL",
    base_param_name="Lfo 2 Rate",
    param_idx=29,
    interval=(0.25, 0.66),
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    # set `Lfo 2 Amount` to 0.95
    # set `Delay Wet` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0), (31, 0.95), (78, 0.0)),
)

pitch_coarse = ParameterVariation(
    preset="default",
    base_param_name="Osc 1 Tune",
    param_idx=19,
    interval=(0.25, 0.5),
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    # set `Osc 3 Volume` to 0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0), (17, 0.0)),
)

reverb = ParameterVariation(
    preset="BS Mixmiddle FN",
    base_param_name="Reverb Wet",
    param_idx=60,
    interval=(0.0, 1.0),
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    # set `Lfo 1 Amount` to 0.5
    # set `Reverb Decay` to 0.5 and `Reverb Pre Delay` to 0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0), (30, 0.5), (61, 0.5), (62, 0.0)),
)
