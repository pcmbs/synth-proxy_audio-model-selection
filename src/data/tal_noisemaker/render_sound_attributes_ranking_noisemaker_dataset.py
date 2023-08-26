"""
Module used to render the TAL-NoiseMaker preset modifications used
for the sound attributes ranking evaluation.
"""
import json
import os
from pathlib import Path
from timeit import default_timer
from typing import List, Optional, Sequence, Tuple

import dawdreamer as daw
import numpy as np
from dotenv import load_dotenv
from scipy.io import wavfile

import data.tal_noisemaker.noisemaker_preset_mods as npm

load_dotenv()  # take environment variables from .env for checkpoints folder

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
DATA_PATH = Path(os.environ["PROJECT_ROOT"]) / "data" / "TAL-NoiseMaker"
EXPORT_PATH = DATA_PATH / "sound_attributes_ranking_dataset"


class PresetRenderer:
    """
    Class to rendered audio from TAL-Noisemaker using DawDreamer.
    """

    def __init__(
        self,
        synth_path: str = "TAL-NoiseMaker/TAL-NoiseMaker.vst3",
        sample_rate: int = 44_100,
        render_duration_s: float = 4,
        fadeout_duration_s: float = 0.1,
        convert_to_mono: bool = False,
        normalize_audio: bool = False,
    ):
        ### Paths and name related member variables
        self.synth_path = synth_path
        self.synth_name = Path(self.synth_path).stem

        ### DawDreamer related member variables
        self.sample_rate = sample_rate
        self.render_duration_s = render_duration_s  # rendering time in seconds
        self.engine = daw.RenderEngine(self.sample_rate, block_size=128)  # pylint: disable=E1101
        self.synth = self.engine.make_plugin_processor(self.synth_name, self.synth_path)

        ### MIDI related member variables
        self.midi_note: int
        self.midi_note_velocity: int
        self.midi_note_start: float
        self.midi_note_duration_s: float

        ### Rendering relative member variables
        # fadeout
        self.fadeout_duration_s = fadeout_duration_s
        self.fadeout_len = int(self.sample_rate * self.fadeout_duration_s)
        # avoid multiplication with an empty array (from linspace) if no fadeout_duration_s = 0
        if self.fadeout_len > 0:
            self.fadeout = np.linspace(1, 0, self.fadeout_len)
        else:  # hard-coding if fadeout_duration_s = 0
            self.fadeout = 1.0

        # export to mono
        self.convert_to_mono = convert_to_mono

        # normalize
        self.normalize_audio = normalize_audio

    def assign_preset(self, preset: list[dict]) -> None:
        """
        Assign a preset to the synthesizer.
        """
        self.current_preset = preset  # update instance's current parameters

        # individually set each parameters since DawDreamer does not accept
        # list of tuples (param_idx, value)
        for param in self.current_preset:
            self.synth.set_parameter(param["index"], param["value"])

    def set_midi_parameters(
        self,
        midi_note: int = 60,
        midi_note_velocity: int = 100,
        midi_note_start: float = 0.0,
        midi_note_duration: float = 2,
    ):
        """
        Set the instance's midi parameters.
        """
        self.midi_note = midi_note
        self.midi_note_velocity = midi_note_velocity
        self.midi_note_start = midi_note_start
        self.midi_note_duration_s = midi_note_duration

        # Generate a MIDI note, specifying a start time and duration, both in seconds
        self.synth.add_midi_note(
            self.midi_note, self.midi_note_velocity, self.midi_note_start, self.midi_note_duration_s
        )

    def render_note(self) -> Sequence:
        """
        Renders a midi note (for the currently set patch) and returns the generated audio as ndarray.
        """

        if self.current_preset is None:
            raise ValueError("No preset has been set yet. Please use `assign_preset()` first.")

        graph = [(self.synth, [])]  # Generate DAG of processes
        self.engine.load_graph(graph)  # load a DAG of processors
        self.engine.render(self.render_duration_s)  # Render audio.
        audio = self.engine.get_audio()  # get audio
        if self.convert_to_mono:  # convert to mono if required
            audio = np.mean(audio, axis=0, keepdims=True)
        if self.normalize_audio:  # normalize audio if required
            audio = audio / np.max(np.abs(audio))
        audio[..., -self.fadeout_len :] = audio[..., -self.fadeout_len :] * self.fadeout  # fadeout
        return audio


def generate_eval_audio(
    renderer: PresetRenderer,
    sound_attribute: npm.PresetModList,
    current_preset_idx: Optional[int] = None,
    num_samples: int = 10,
) -> Tuple[List[np.ndarray], List[str]]:
    # initialize list to store the rendered audio
    output = []
    # intialize a list to store the file names of the rendered audio
    filenames = []

    sound_attribute_str = sound_attribute.sound_attribute

    if current_preset_idx is not None:
        preset_mods_to_render = [sound_attribute.preset_mods[current_preset_idx]]
        preset_indices = [current_preset_idx]
    else:
        preset_mods_to_render = sound_attribute.preset_mods
        preset_indices = np.arange(len(preset_mods_to_render))

    for p_idx, p in zip(preset_indices, preset_mods_to_render):
        # assign base preset to generate audio
        renderer.assign_preset(presets[p.preset])

        # index of the main parameter
        main_param_idx = p.param_idx if p.param_idx else sound_attribute.param_idx

        # generate `num_samples` linearly equally spaced values for the main parameter
        main_param_values = np.linspace(*p.interval, num_samples)

        # set extra parameters

        extra_params = sound_attribute.extra_params
        if p.extra_params:
            extra_params += p.extra_params
        for param in extra_params:
            renderer.synth.set_parameter(*param)

        # generate an output for each parameter value and append it to the output list
        for val_idx, val in enumerate(main_param_values):
            renderer.synth.set_parameter(main_param_idx, val)

            output.append(renderer.render_note())
            filenames.append(f"{sound_attribute_str}-{p_idx}-{val_idx}.wav")

    return output, filenames


if __name__ == "__main__":
    EXPORT_PATH.mkdir()

    SAMPLE_RATE = 44_100

    MIDI_NOTE_CC = 60
    MIDI_NOTE_VEL = 100
    MIDI_NOTE_START = 0.0
    MIDI_NOTE_DUR = 2

    CONVERT_TO_MONO = True

    RENDER_DURATION_S = 4

    NUM_SAMPLES = 20

    print(
        f"Generating `sound_attributes_ranking_dataset` with the following parameters:\n"
        f"- sample rate: {SAMPLE_RATE}\n"
        f"- num samples: {NUM_SAMPLES}\n"
        f"- midi note CC: {MIDI_NOTE_CC}\n"
        f"- midi note VEL: {MIDI_NOTE_VEL}\n"
        f"- midi note START: {MIDI_NOTE_START}\n"
        f"- midi note DUR: {MIDI_NOTE_DUR}\n\n"
    )

    with open(DATA_PATH / "tal_noisemaker_presets.json", "rb") as f:
        presets = json.load(f)

    renderer = PresetRenderer(
        sample_rate=SAMPLE_RATE,
        convert_to_mono=CONVERT_TO_MONO,
        render_duration_s=RENDER_DURATION_S,
        synth_path=str(
            DATA_PATH / "TAL-NoiseMaker.vst3",
        ),
    )
    renderer.set_midi_parameters(MIDI_NOTE_CC, MIDI_NOTE_VEL, MIDI_NOTE_START, MIDI_NOTE_DUR)

    for sound_attribute in npm.SOUND_ATTRIBUTES:
        print(f"Exporting audio for `{sound_attribute}`...")
        start = default_timer()

        current_attribute = getattr(npm, sound_attribute)

        path_to_attribute = EXPORT_PATH / current_attribute.sound_attribute
        path_to_attribute.mkdir()

        for i, preset in enumerate(current_attribute.preset_mods):
            path_to_preset = path_to_attribute / str(i)
            path_to_preset.mkdir()

            output, filenames = generate_eval_audio(renderer, current_attribute, i, NUM_SAMPLES)

            for audio, filename in zip(output, filenames):
                wavfile.write(path_to_preset / filename, SAMPLE_RATE, audio.reshape(-1, audio.shape[0]))

        print(f"Audio for `{sound_attribute}` exported in {default_timer() - start:.2f} seconds")
