"""
Module containing all modified TAL-NoiseMaker presets for the
sound attributes ranking evaluation.

Run as main to print parameter indices.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

SOUND_ATTRIBUTES = [
    "amp_attack",
    "amp_decay",
    "filter_cutoff",
    "filter_decay",
    "filter_resonance",
    "frequency_mod",
    "lfo_amount_on_amp",
    "lfo_amount_on_filter",
    "lfo_amount_on_pitch",
    "lfo_rate_on_amp",
    "lfo_rate_on_filter",
    "lfo_rate_on_pitch",
    "pitch_coarse",
    "reverb_wet",
]


@dataclass
class PresetMod:
    """
    Dataclass holding modifications for a single preset.
    It allows to specify an interval from which values of a given parameter
    (defined in the PresetModList holding this object) will be monotonically sampled,
    as well as additional parameter values to modifiy.

    Args:
    - `preset` (str): preset name
    - `interval` (tuple[float, float]): interval to monotonically sample the parameter values from.
    - `param_idx` (Optional[int]): allows to override the main parameter to modify.
    - `extra_params` (tuple[tuple[int, float],...]): extra parameters to modify
    (in addition to the ones in the ParameterVariationList holding this instance)
    """

    preset: str
    interval: Tuple[float, float]
    param_idx: Optional[int] = None
    extra_params: Optional[Tuple[Tuple[int, float], ...]] = None


@dataclass
class PresetModList:
    """
    Dataclass to hold a list of PresetMod instances.

    Args:
    - `sound_attribute` (str): name of the sound attribute of the current instance.
    - `param_idx` (int): index of the parameter to modify
    - `extra_params` (tuple[tuple[int, float],...]): extra parameters to modify (common to all modified presets).
    - `preset_mods` (list[PresetMod]): list of PresetMod instances.
    """

    sound_attribute: str
    param_idx: int
    extra_params: Tuple[Tuple[int, float], ...]
    preset_mods: List[PresetMod]


amp_attack = PresetModList(
    sound_attribute="amp_attack",
    param_idx=11,  # `Amp Attack`
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    extra_params=((46, 1.0), (48, 1.0)),
    preset_mods=[
        PresetMod(  # amp_attack_0
            preset="default",
            interval=(0.0, 0.8),
        ),
        PresetMod(  # amp_attack_1
            preset="default",
            interval=(0.0, 0.8),
            # set `Osc 1 Waveform` to 0.5 (square wave)
            extra_params=((23, 0.5),),
        ),
        PresetMod(  # amp_attack_2
            preset="ARP 303 Like FN",
            interval=(0.0, 0.8),
        ),
        PresetMod(  # amp_attack_3
            preset="ARP C64 II FN",
            interval=(0.0, 0.8),
            # set `Reverb Wet` to 0
            extra_params=((60, 0.0),),
        ),
        PresetMod(  # amp_attack_4
            preset="BS 7th Gates FN",
            interval=(0.0, 0.8),
        ),
        PresetMod(  # amp_attack_5
            preset="BS Bass Deep TAL",
            interval=(0.0, 0.8),
        ),
        PresetMod(  # amp_attack_6
            preset="BS Dark Bass TAL",
            interval=(0.0, 0.8),
        ),
        PresetMod(  # amp_attack_7
            preset="CH Chordionator II FN",
            interval=(0.0, 0.8),
        ),
        PresetMod(  # amp_attack_8
            preset="DR Hat Short Moves TAL",
            interval=(0.0, 0.8),
        ),
        PresetMod(  # amp_attack_9
            preset="FX Noiz3machin3 FN",
            interval=(0.0, 0.8),
            # set `Reverb Wet` to 0
            extra_params=((60, 0.0),),
        ),
    ],
)


amp_decay = PresetModList(
    sound_attribute="amp_decay",
    param_idx=12,  # `Amp Decay`
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Amp Sustain` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (13, 0.0)),
    preset_mods=[
        PresetMod(  # amp_decay_0
            preset="CH Open Mind FN",
            interval=(0.2, 0.6),
            # set `Reverb Wet` to 0
            extra_params=((60, 0.0),),
        ),
        PresetMod(  # amp_decay_1
            preset="default",
            interval=(0.2, 0.6),
        ),
        PresetMod(  # amp_decay_2
            preset="default",
            interval=(0.2, 0.6),
            # set `Osc 1 Waveform` to 0.5 (square wave)
            extra_params=((23, 0.5),),
        ),
        PresetMod(  # amp_decay_3
            preset="KB Pop Pluck TUC",
            interval=(0.2, 0.6),
        ),
        PresetMod(  # amp_decay_4
            preset="KB Screetcher TUC",
            interval=(0.2, 0.6),
        ),
        PresetMod(  # amp_decay_5
            preset="LD Rasp Lead AS",
            interval=(0.2, 0.6),
        ),
        PresetMod(  # amp_decay_6
            preset="LD Steam Pulse AS",
            interval=(0.2, 0.6),
        ),
        PresetMod(  # amp_decay_7
            preset="PD Gated Pad TAL",
            interval=(0.2, 0.6),
            # set `Amp Attack` to 0.0
            extra_params=((11, 0.0),),
        ),
        PresetMod(  # amp_decay_8
            preset="DR Kick Old School TAL",
            interval=(0.2, 0.6),
            # set `Amp Attack` to 0.0
            extra_params=((11, 0.0),),
        ),
        PresetMod(  # amp_decay_9
            preset="FX  Jumper TAL",
            interval=(0.2, 0.6),
            # set `Amp Attack` to 0.0
            extra_params=((11, 0.0),),
        ),
    ],
)

filter_cutoff = PresetModList(
    sound_attribute="filter_cutoff",
    param_idx=3,  # `Filter Cutoff`
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    extra_params=((46, 1.0), (48, 1.0)),
    preset_mods=[
        PresetMod(  # filter_cutoff_0
            preset="BS Deep Driver TAL",
            interval=(0.22, 0.8),
            # set `Filter Type` to 0.0 (LP 24)
            # set `Filter Decay` to 0.0
            extra_params=((2, 0.0), (8, 0.0)),
        ),
        PresetMod(  # filter_cutoff_1
            preset="default",
            interval=(0.22, 0.8),
        ),
        PresetMod(  # filter_cutoff_2
            preset="default",
            interval=(0.22, 0.8),
            # set `Osc 1 Waveform` to 0.5 (square wave)
            extra_params=((23, 0.5),),
        ),
        PresetMod(  # filter_cutoff_3
            preset="ARP C64 FN",
            interval=(0.22, 0.8),
            # set `Filter Decay` to 0.0
            extra_params=((8, 0.0),),
        ),
        PresetMod(  # filter_cutoff_4
            preset="ARP Phasing Saws TAL",
            interval=(0.0, 0.6),
            # set `Filter Type` to 0.0 (LP 24)
            # set `Filter Decay` to 0.0
            # set `Delay Wet` to 0.3
            extra_params=((2, 0.0), (8, 0.0), (78, 0.3)),
        ),
        PresetMod(  # filter_cutoff_5
            preset="BS Justice TAL",
            interval=(0.0, 0.6),
        ),
        PresetMod(  # filter_cutoff_6
            preset="BS Terminator TAL",
            interval=(0.0, 0.7),
        ),
        PresetMod(  # filter_cutoff_7
            preset="KB Piano House TAL",
            interval=(0.05, 0.75),
        ),
        PresetMod(  # filter_cutoff_8
            preset="LD Fuzzy Box TAL",
            interval=(0.0, 0.6),
            # set `Filter Attack` to 0.16
            # set `Filter Decay` to 0.38
            extra_params=((7, 0.16), (8, 0.38)),
        ),
        PresetMod(  # filter_cutoff_9
            preset="FX Metallica FN",
            interval=(0.0, 0.6),
        ),
    ],
)


filter_decay = PresetModList(
    sound_attribute="filter_decay",
    param_idx=8,  # `Filter Decay`
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    extra_params=((46, 1.0), (48, 1.0)),
    preset_mods=[
        PresetMod(  # filter_decay_0
            preset="LD 8bitter II FN",
            interval=(0.4, 0.68),
            # set `Reverb Wet` to 0
            extra_params=((60, 0.0),),
        ),
        PresetMod(  # filter_decay_1
            preset="ARP Demotrack ARP FN",
            interval=(0.2, 0.6),
            # set `Lfo 2 Amount` to 0.5
            extra_params=((31, 0.5),),
        ),
        PresetMod(  # filter_decay_2
            preset="BS 7th Gates FN",
            interval=(0.2, 0.4),
        ),
        PresetMod(  # filter_decay_3
            preset="BS Bassolead FN",
            interval=(0.2, 0.6),
            # set `Filter Attack` to 0.05
            extra_params=((7, 0.05),),
        ),
        PresetMod(  # filter_decay_4
            preset="BS Basspa FN",
            interval=(0.2, 0.6),
            # set `Filter Attack` to 0.05
            extra_params=((7, 0.05),),
        ),
        PresetMod(  # filter_decay_5
            preset="CH Chordionator II FN",
            interval=(0.2, 0.6),
            # set `Filter Cutoff` to 0.1
            # set `Reverb Wet` to 0.0
            extra_params=((3, 0.1), (60, 0.0)),
        ),
        PresetMod(  # filter_decay_6
            preset="CH Open Mind FN",
            interval=(0.2, 0.6),
            # set `Filter Cutoff` to 0.1
            extra_params=((3, 0.1),),
        ),
        PresetMod(  # filter_decay_7
            preset="DR 8bit  Kick III FN",
            interval=(0.2, 0.6),
        ),
        PresetMod(  # filter_decay_8
            preset="LD Acid Saw TAL",
            interval=(0.2, 0.6),
        ),
        PresetMod(  # filter_decay_9
            preset="LD Aggggggro TAL",
            interval=(0.2, 0.6),
            # set `Filter Attack` to 0.05
            extra_params=((7, 0.05),),
        ),
    ],
)


filter_resonance = PresetModList(
    sound_attribute="filter_resonance",
    param_idx=4,  # `Filter Resonance`
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    extra_params=((46, 1.0), (48, 1.0)),
    preset_mods=[
        PresetMod(  # filter_resonance_0
            preset="BS Clean Flat Bass TAL",
            interval=(0.2, 0.95),
            # set `Filter Type` to 0.0 (LP 24)
            extra_params=((2, 0.0),),
        ),
        PresetMod(  # filter_resonance_1
            preset="ARP 303 Like II FN",
            interval=(0.4, 1.0),
            # set `Filter Type` to 0.0 (LP 24)
            extra_params=((2, 0.0),),
        ),
        PresetMod(  # filter_resonance_2
            preset="ARP 2050 Punk TAL",
            interval=(0.6, 1.0),
        ),
        PresetMod(  # filter_resonance_3
            preset="DR Kick III FN",
            interval=(0.4, 1.0),
            # Set `Filter Decay` to 0.4
            extra_params=((8, 0.4),),
        ),
        PresetMod(  # filter_resonance_4
            preset="FX Clean Noise Ramp TAL",
            interval=(0.2, 0.95),
        ),
        PresetMod(  # filter_resonance_5
            preset="FX  Jumper TAL",
            interval=(0.2, 0.95),
        ),
        PresetMod(  # filter_resonance_6
            preset="FX Metallica FN",
            interval=(0.2, 0.95),
        ),
        PresetMod(  # filter_resonance_7
            preset="LD Acid Dist Noisy TAL",
            interval=(0.2, 0.95),
        ),
        PresetMod(  # filter_resonance_8
            preset="LD Analog Down Glider TAL",
            interval=(0.2, 0.95),
        ),
        PresetMod(  # filter_resonance_9
            preset="LD Technoshocker FN",
            interval=(0.2, 0.95),
        ),
    ],
)
frequency_mod = PresetModList(
    sound_attribute="frequency_mod",
    param_idx=36,  # `Osc 2 FM`
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    extra_params=((46, 1.0), (48, 1.0)),
    preset_mods=[
        PresetMod(  # frequency_mod_0
            preset="KB Smooth Sine TAL",
            interval=(0.0, 1.0),
            # set `Transpose` to 0.5 and `Delay Wet` to 0
            extra_params=((40, 0.5), (78, 0.0)),
        ),
        PresetMod(  # frequency_mod_1
            preset="ARP C64 FN",
            interval=(0.0, 0.95),
        ),
        PresetMod(  # frequency_mod_2
            preset="BS 7th Gates FN",
            interval=(0.0, 1.0),
        ),
        PresetMod(  # frequency_mod_3
            preset="BS Detuned FN",
            interval=(0.0, 0.95),
        ),
        PresetMod(  # frequency_mod_4
            preset="BS Jelly Mountain AS",
            interval=(0.0, 0.95),
        ),
        PresetMod(  # frequency_mod_5
            preset="CH Chordionator II FN",
            interval=(0.0, 1.0),
        ),
        PresetMod(  # frequency_mod_6
            preset="DR Snare Dry TAL",
            interval=(0.0, 1.0),
        ),
        PresetMod(  # frequency_mod_7
            preset="FX Metallica FN",
            interval=(0.0, 1.0),
        ),
        PresetMod(  # frequency_mod_8
            preset="FX Noiz3machin3 FN",
            interval=(0.0, 1.0),
        ),
        PresetMod(  # frequency_mod_9
            preset="LD Everglade Walk TAL",
            interval=(0.0, 1.0),
        ),
    ],
)

lfo_amount_on_amp = PresetModList(
    sound_attribute="lfo_amount_on_amp",
    param_idx=31,  # `Lfo 2 Amount`
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    extra_params=((46, 1.0), (48, 1.0)),
    preset_mods=[
        PresetMod(  # lfo_amount_on_amp_0
            preset="LD Mellow Chord TAL",
            interval=(0.75, 1.0),
            # set `Delay Wet` to 0.0
            extra_params=((78, 0.0),),
        ),
        PresetMod(  # lfo_amount_on_amp_1
            preset="ARP On The Run TUC",
            interval=(0.65, 1.0),
            # set `Lfo 1 Sync` and `Lfo 2 Sync` to 0.0
            # set `Lfo 2 Rate` to 0.53
            extra_params=((45, 0.0), (47, 0.0), (29, 0.53)),
        ),
        PresetMod(  # lfo_amount_on_amp_2
            preset="ARP Rumpelkammer FN",
            interval=(0.65, 1.0),
        ),
        PresetMod(  # lfo_amount_on_amp_3
            preset="BS Bassolead FN",
            interval=(0.65, 1.0),
            # set `Lfo 1 Sync` and `Lfo 2 Sync` to 0.0
            extra_params=((45, 0.0), (47, 0.0)),
        ),
        PresetMod(  # lfo_amount_on_amp_4
            preset="CH Chordionator II FN",
            interval=(0.65, 1.0),
            # set `Lfo 2 Sync` to 0.0
            # set `Lfo 2 Rate` to 0.48
            extra_params=((47, 0.0), (29, 0.48)),
        ),
        PresetMod(  # lfo_amount_on_amp_5
            preset="KB Glockenschlag FN",
            interval=(0.75, 1.0),
        ),
        PresetMod(  # lfo_amount_on_amp_6
            preset="LD Mod-U-Crush AS",
            interval=(0.65, 1.0),
            # set `Lfo 2 Sync` to 0.0
            # set `Lfo 2 Rate` to 0.43
            extra_params=((47, 0.0), (29, 0.43)),
        ),
        PresetMod(  # lfo_amount_on_amp_7
            preset="LD Thin Lead TAL",
            interval=(0.70, 1.0),
            # set `Lfo 2 Sync` to 0.0
            # set `Lfo 2 Rate` to 0.5
            extra_params=((47, 0.0), (29, 0.5)),
        ),
        PresetMod(  # lfo_amount_on_amp_8
            preset="PD Bellomatism FN",
            interval=(0.75, 1.0),
        ),
        PresetMod(  # lfo_amount_on_amp_9
            preset="PD Gated Pad TAL",
            interval=(0.70, 1.0),
            # set `Lfo 2 Sync` to 0.0
            # set `Lfo 2 Rate` to 0.53
            extra_params=((47, 0.0), (29, 0.53)),
        ),
    ],
)

lfo_amount_on_filter = PresetModList(
    sound_attribute="lfo_amount_on_filter",
    param_idx=31,  # `Lfo 2 Amount`
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    extra_params=((46, 1.0), (48, 1.0)),
    preset_mods=[
        PresetMod(  # lfo_amount_on_filter_0
            preset="LD Power Lead TAL",
            interval=(0.33, 0.0),
            # set `Lfo 2 Rate` to 0.45
            extra_params=((29, 0.45),),
        ),
        PresetMod(  # lfo_amount_on_filter_1
            preset="BS Big Starter TAL",
            interval=(0.5, 1.0),
            # set `Filter Cutoff` to 0.25
            # set `Lfo 2 Sync` to 0.0
            extra_params=((3, 0.25), (47, 0.0)),
        ),
        PresetMod(  # lfo_amount_on_filter_2
            preset="BS Flamming Bass FN",
            interval=(0.5, 1.0),
        ),
        PresetMod(  # lfo_amount_on_filter_3
            preset="BS Juicy Bass TUC",
            interval=(0.5, 1.0),
        ),
        PresetMod(  # lfo_amount_on_filter_4
            preset="BS LFO Roller FN",
            interval=(0.5, 1.0),
            # set `Lfo 1 Sync` and `Lfo 2 Sync` to 0.0
            extra_params=((45, 0.0), (47, 0.0)),
        ),
        PresetMod(  # lfo_amount_on_filter_5
            preset="BS Tripple Wobbler TAL",
            param_idx=30,  # `Lfo 1 Amount`
            interval=(0.5, 1.0),
        ),
        PresetMod(  # lfo_amount_on_filter_6
            preset="LD Sci Fi Organ TAL",
            interval=(0.5, 1.0),
        ),
        PresetMod(  # lfo_amount_on_filter_7
            preset="LD Noisy Sync Lead TAL",
            interval=(0.5, 1.0),
            # set `Lfo 2 Rate` to 0.5
            extra_params=((29, 0.5),),
        ),
        PresetMod(  # lfo_amount_on_filter_8
            preset="LD Resobells FN",
            interval=(0.5, 0.0),
        ),
        PresetMod(  # lfo_amount_on_filter_9
            preset="LD Acid Dist Noisy TAL",
            interval=(0.5, 0.0),
            # set `Lfo 2 Rate` to 0.5
            extra_params=((29, 0.3),),
        ),
    ],
)


lfo_amount_on_pitch = PresetModList(
    sound_attribute="lfo_amount_on_pitch",
    param_idx=31,  # `Lfo 2 Amount`
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    extra_params=((46, 1.0), (48, 1.0)),
    preset_mods=[
        PresetMod(  # lfo_amount_on_pitch_0
            preset="BS Eager Beaver AS",
            interval=(0.60, 1.0),
            # set `Osc 3 Volume` to 0.0
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            extra_params=((17, 0.0), (33, 1.0)),
        ),
        PresetMod(  # lfo_amount_on_pitch_1
            preset="BS Jelly Mountain AS",
            interval=(0.50, 0.0),
            # set `Osc 3 Volume` to 0.0
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            # set `Lfo 2 Rate` to 0.45
            extra_params=((17, 0.0), (33, 1.0), (29, 0.45)),
        ),
        PresetMod(  # lfo_amount_on_pitch_2
            preset="BS Mong AS",
            interval=(0.55, 1.0),
            # set `Osc 3 Volume` to 0.2
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            extra_params=((17, 0.2), (33, 1.0)),
        ),
        PresetMod(  # lfo_amount_on_pitch_3
            preset="BS Sci Fi TAL",
            interval=(0.55, 1.0),
            # set `Osc 3 Volume` to 0.1
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            extra_params=((17, 0.1), (33, 1.0)),
        ),
        PresetMod(  # lfo_amount_on_pitch_4
            preset="BS Tremolo Bass TAL",
            interval=(0.55, 1.0),
            # set `Osc 3 Volume` to 0.2
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            extra_params=((17, 0.2), (33, 1.0)),
        ),
        PresetMod(  # lfo_amount_on_pitch_5
            preset="CH Chordionator II FN",
            interval=(0.60, 1.0),
            # set `Lfo 2 Rate` to 0.45
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            # set `Lfo 2 Sync` to 0.0
            extra_params=((29, 0.45), (33, 1.0), (47, 0.0)),
        ),
        PresetMod(  # lfo_amount_on_pitch_6
            preset="CH SIDchord FN",
            interval=(0.45, 0.0),
            # set `Lfo 2 Rate` to 0.5
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            extra_params=((29, 0.5), (33, 1.0)),
        ),
        PresetMod(  # lfo_amount_on_pitch_7
            preset="FX Turntable FN",
            interval=(0.55, 1.0),
            # set `Lfo 2 Rate` to 0.5
            # set `Lfo 1 Amount` to 0.4
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            # set `Lfo 1 Sync` to 0.0
            # set `Lfo 2 Sync` to 0.0
            extra_params=((29, 0.5), (30, 0.4), (33, 1.0), (45, 0.0), (47, 0.0)),
        ),
        PresetMod(  # lfo_amount_on_pitch_8
            preset="KB Chimp Organ TUC",
            interval=(0.55, 1.0),
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            extra_params=((33, 1.0),),
        ),
        PresetMod(  # lfo_amount_on_pitch_9
            preset="KB Ghostly Glomp AS",
            interval=(0.55, 1.0),
            # set `Osc 3 Volume` to 0.5
            extra_params=((17, 1.0),),
        ),
    ],
)

lfo_rate_on_amp = PresetModList(
    sound_attribute="lfo_rate_on_amp",
    param_idx=29,  # `Lfo 2 Rate`
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    extra_params=((46, 1.0), (48, 1.0)),
    preset_mods=[
        PresetMod(  # lfo_rate_on_amp_0
            preset="LD Mellow Chord TAL",
            interval=(0.25, 0.66),
            # set `Lfo 2 Amount` to 0.95
            # set `Delay Wet` to 0.0
            extra_params=((31, 0.95), (78, 0.0)),
        ),
        PresetMod(  # lfo_rate_on_amp_1
            preset="ARP On The Run TUC",
            interval=(0.4, 0.6),
            # set `Lfo 1 Sync` and `Lfo 2 Sync` to 0.0
            # set `Lfo 2 Amount` to 1.0
            extra_params=((45, 0.0), (47, 0.0), (31, 1.0)),
        ),
        PresetMod(  # lfo_rate_on_amp_2
            preset="ARP Rumpelkammer FN",
            interval=(0.4, 0.6),
        ),
        PresetMod(  # lfo_rate_on_amp_3
            preset="BS Bassolead FN",
            interval=(0.25, 0.6),
            # set `Lfo 2 Amount` to 1.0
            # set `Lfo 1 Sync` and `Lfo 2 Sync` to 0.0
            extra_params=((31, 1.0), (45, 0.0), (47, 0.0)),
        ),
        PresetMod(  # lfo_rate_on_amp_4
            preset="CH Chordionator II FN",
            interval=(0.4, 0.6),
            # set `Lfo 2 Amount` to 0.0
            # set `Lfo 2 Sync` to 0.0
            extra_params=((31, 0.0), (47, 0.0)),
        ),
        PresetMod(  # lfo_rate_on_amp_5
            preset="KB Glockenschlag FN",
            interval=(0.4, 0.6),
            # set `Lfo 2 Amount` to 1.0
            extra_params=((31, 1.0),),
        ),
        PresetMod(  # lfo_rate_on_amp_6
            preset="LD Mod-U-Crush AS",
            interval=(0.4, 0.6),
            # set `Lfo 2 Sync` to 0.0
            extra_params=((47, 0.0),),
        ),
        PresetMod(  # lfo_rate_on_amp_7
            preset="LD Thin Lead TAL",
            interval=(0.4, 0.6),
            # set `Lfo 2 Sync` to 0.0
            # set `Lfo 2 Amount` to 1.0
            extra_params=((31, 1.0), (47, 0.0)),
        ),
        PresetMod(  # lfo_rate_on_amp_8
            preset="PD Bellomatism FN",
            interval=(0.4, 0.6),
            # set `Lfo 2 Amount` to 0.95
            extra_params=((31, 0.95),),
        ),
        PresetMod(  # lfo_rate_on_amp_9
            preset="PD Gated Pad TAL",
            interval=(0.4, 0.6),
            # set `Lfo 2 Sync` to 0.0
            # set `Lfo 2 Amount` to 1.0
            extra_params=((47, 0.0), (31, 1.0)),
        ),
    ],
)

lfo_rate_on_filter = PresetModList(
    sound_attribute="lfo_rate_on_filter",
    param_idx=29,  # `Lfo 2 Rate`
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    extra_params=((46, 1.0), (48, 1.0)),
    preset_mods=[
        PresetMod(  # lfo_rate_on_filter_0
            preset="LD Acid Dist Noisy TAL",
            interval=(0.14, 0.6),
            # set `Delay Wet` to 0.0
            extra_params=((78, 0.0),),
        ),
        PresetMod(  # lfo_rate_on_filter_1
            preset="BS Big Starter TAL",
            interval=(0.14, 0.6),
            # set `Filter Cutoff` to 0.25
            # set `Lfo 2 Amount` to 0.75
            # set `Envelope Amount` to 0.0
            # set `Lfo 2 Sync` to 0.0
            extra_params=((3, 0.25), (31, 0.75), (73, 0.0), (47, 0.0)),
        ),
        PresetMod(  # lfo_rate_on_filter_2
            preset="BS Flamming Bass FN",
            interval=(0.14, 0.6),
            # set `Lfo 2 Amount` to 0.70
            extra_params=((31, 0.70),),
        ),
        PresetMod(  # lfo_rate_on_filter_3
            preset="BS Juicy Bass TUC",
            interval=(0.14, 0.6),
            # set `Lfo 2 Amount` to 0.75
            extra_params=((31, 0.75),),
        ),
        PresetMod(  # lfo_rate_on_filter_4
            preset="BS LFO Roller FN",
            interval=(0.14, 0.6),
            # set `Lfo 2 Amount` to 0.70
            # set `Lfo 1 Sync` and `Lfo 2 Sync` to 0.0
            extra_params=(
                (31, 0.70),
                (45, 0.0),
                (47, 0.0),
            ),
        ),
        PresetMod(  # lfo_rate_on_filter_5
            preset="BS Tripple Wobbler TAL",
            param_idx=30,  # `Lfo 1 Amount`
            interval=(0.33, 0.6),
            # set `Lfo 1 Sync` to 0.0
            extra_params=((45, 0.0),),
        ),
        PresetMod(  # lfo_rate_on_filter_6
            preset="LD Sci Fi Organ TAL",
            interval=(0.14, 0.6),
            # set `Lfo 2 Amount` to 0.70
            extra_params=((31, 0.70),),
        ),
        PresetMod(  # lfo_rate_on_filter_7
            preset="LD Noisy Sync Lead TAL",
            interval=(0.14, 0.6),
            # set `Lfo 2 Amount` to 0.7
            extra_params=((31, 0.7),),
        ),
        PresetMod(  # lfo_rate_on_filter_8
            preset="LD Resobells FN",
            interval=(0.33, 0.6),
            # set `Filter Cutoff` to 0.3
            # set `Amp Attack` to 0.0
            extra_params=((3, 0.3), (11, 0.0)),
        ),
        PresetMod(  # lfo_rate_on_filter_9
            preset="LD Power Lead TAL",
            interval=(0.25, 0.6),
            # set `Filter Cutoff` to 0.2
            # set `Lfo 2 Amount` to 0.7
            extra_params=((3, 0.2), (31, 0.7)),
        ),
    ],
)


lfo_rate_on_pitch = PresetModList(
    sound_attribute="lfo_rate_on_pitch",
    param_idx=29,  # `Lfo 2 Rate`
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    extra_params=((46, 1.0), (48, 1.0)),
    preset_mods=[
        PresetMod(  # lfo_rate_on_pitch_0
            preset="BS Eager Beaver AS",
            interval=(0.2, 0.6),
            # set `Osc 3 Volume` to 0.0
            # set `Lfo 2 Amount` to 0.3
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            extra_params=((17, 0.0), (31, 0.3), (33, 1.0)),
        ),
        PresetMod(  # lfo_rate_on_pitch_1
            preset="BS Jelly Mountain AS",
            interval=(0.2, 0.6),
            # set `Osc 3 Volume` to 0.0
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            # set `Lfo 2 Amount` to 0.7
            extra_params=((17, 0.0), (33, 1.0), (31, 0.45)),
        ),
        PresetMod(  # lfo_rate_on_pitch_2
            preset="BS Mong AS",
            interval=(0.2, 0.6),
            # set `Osc 3 Volume` to 0.2
            # set `Lfo 2 Amount` to 0.7
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            extra_params=((17, 0.2), (31, 0.7), (33, 1.0)),
        ),
        PresetMod(  # lfo_rate_on_pitch_3
            preset="BS Sci Fi TAL",
            interval=(0.2, 0.6),
            # set `Osc 3 Volume` to 0.1
            # set `Lfo 2 Amount` to 0.7
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            extra_params=((17, 0.1), (31, 0.7), (33, 1.0)),
        ),
        PresetMod(  # lfo_rate_on_pitch_4
            preset="BS Tremolo Bass TAL",
            interval=(0.2, 0.6),
            # set `Osc 3 Volume` to 0.2
            # set `Lfo 2 Amount` to 0.7
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            extra_params=((17, 0.2), (31, 0.7), (33, 1.0)),
        ),
        PresetMod(  # lfo_rate_on_pitch_5
            preset="CH Chordionator II FN",
            interval=(0.40, 0.6),
            # set `Lfo 2 Amount` to 0.33
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            # set `Lfo 2 Sync` to 0.0
            extra_params=((31, 0.33), (33, 1.0), (47, 0.0)),
        ),
        PresetMod(  # lfo_rate_on_pitch_6
            preset="CH SIDchord FN",
            interval=(0.2, 0.6),
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            extra_params=((33, 1.0),),
        ),
        PresetMod(  # lfo_rate_on_pitch_7
            preset="FX Turntable FN",
            interval=(0.2, 0.6),
            # set `Lfo 1 Amount` to 0.5
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            # set `Lfo 1 Sync` and `Lfo 2 Sync` to 0.0
            extra_params=((30, 0.5), (33, 1.0), (45, 0.0), (47, 0.0)),
        ),
        PresetMod(  # lfo_rate_on_pitch_8
            preset="KB Chimp Organ TUC",
            interval=(0.2, 0.6),
            # set `Lfo 2 Amount` to 0.6
            # set `Lfo 2 Destination` to 1.0 (Osc1 & 2)
            extra_params=((31, 0.6), (33, 1.0)),
        ),
        PresetMod(  # lfo_rate_on_pitch_9
            preset="KB Ghostly Glomp AS",
            interval=(0.55, 1.0),
            # set `Osc 3 Volume` to 0.5
            # set `Lfo 2 Amount` to 0.25
            extra_params=((17, 1.0), (31, 0.25)),
        ),
    ],
)

pitch_coarse = PresetModList(
    sound_attribute="pitch_coarse",
    param_idx=19,  # `Osc 1 Tune`
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    # set `Envelop Amount` to 0.0
    extra_params=((46, 1.0), (48, 1.0), (73, 0.0)),
    preset_mods=[
        PresetMod(  # pitch_coarse_0
            preset="default",
            interval=(0.25, 0.5),
            # set `Osc 3 Volume` to 0
            extra_params=((17, 0.0),),
        ),
        PresetMod(  # pitch_coarse_1
            preset="default",
            interval=(0.25, 0.5),
            # set `Osc 3 Volume` to 0
            # set `Osc 1 Waveform` to 0.5 (square wave)
            extra_params=((17, 0.0), (23, 0.5)),
        ),
        PresetMod(  # pitch_coarse_2
            preset="ARP 303 Like FN",
            interval=(0.25, 0.5),
        ),
        PresetMod(  # pitch_coarse_3
            preset="ARP Super Sync TAL",
            interval=(0.25, 0.5),
        ),
        PresetMod(  # pitch_coarse_4
            preset="BS Clean Flat Bass TAL",
            interval=(0.25, 0.5),
            # set `Osc 3 Volume` to 0
            extra_params=((17, 0.0),),
        ),
        PresetMod(  # pitch_coarse_5
            preset="BS Goodspeed FN",
            interval=(0.25, 0.5),
        ),
        PresetMod(  # pitch_coarse_6
            preset="FX Bit Shuffla TAL",
            interval=(0.25, 0.5),
        ),
        PresetMod(  # pitch_coarse_7
            preset="FX  Jumper TAL",
            interval=(0.25, 0.5),
            # set `Osc 3 Volume` to 0
            extra_params=((17, 0.0),),
        ),
        PresetMod(  # pitch_coarse_8
            preset="LD Analog Down Glider TAL",
            interval=(0.25, 0.5),
        ),
        PresetMod(  # pitch_coarse_9
            preset="LD Drop In Pulse TAL",
            interval=(0.25, 0.5),
        ),
    ],
)

reverb_wet = PresetModList(
    sound_attribute="reverb_wet",
    param_idx=60,  # `Reverb Wet`
    # set `Lfo 1 Keytrigger` and `Lfo 2 Keytrigger` to 1.0
    extra_params=((46, 1.0), (48, 1.0)),
    preset_mods=[
        PresetMod(  # reverb_wet_0
            preset="BS Mixmiddle FN",
            interval=(0.0, 1.0),
            # set `Lfo 1 Amount` to 0.5
            # set `Reverb Decay` to 0.5 and `Reverb Pre Delay` to 0
            extra_params=((30, 0.5), (61, 0.5), (62, 0.0)),
        ),
        PresetMod(  # reverb_wet_1
            preset="ARP 303 Like II FN",
            interval=(0.0, 1.0),
        ),
        PresetMod(  # reverb_wet_2
            preset="ARP C64 III FN",
            interval=(0.0, 1.0),
        ),
        PresetMod(  # reverb_wet_3
            preset="BS 7th Gates FN",
            interval=(0.0, 1.0),
        ),
        PresetMod(  # reverb_wet_4
            preset="BS Eager Beaver AS",
            interval=(0.0, 1.0),
        ),
        PresetMod(  # reverb_wet_5
            preset="CD Universal FN",
            interval=(0.0, 1.0),
        ),
        PresetMod(  # reverb_wet_6
            preset="CH Chordionator IV FN",
            interval=(0.0, 1.0),
        ),
        PresetMod(  # reverb_wet_7
            preset="DR Kick II FN",
            interval=(0.0, 1.0),
        ),
        PresetMod(  # reverb_wet_8
            preset="FX Break FM",
            interval=(0.0, 1.0),
        ),
        PresetMod(  # reverb_wet_9
            preset="LD Bon Voyage FN",
            interval=(0.0, 1.0),
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
