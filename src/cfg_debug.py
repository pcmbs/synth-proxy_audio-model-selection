"""
Module used to debug/print the hydra config
"""
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env for hydra config


@hydra.main(
    version_base=None, config_path="../configs", config_name="evaluate_audio_model"
)
def cfg_debug(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("breakpoint me!")


if __name__ == "__main__":
    # args = ["cfg_debug.py", "debug=default"]
    args = ["cfg_debug.py"]

    sys.argv = args

    gettrace = getattr(sys, "gettrace", None)
    if gettrace():
        sys.argv = args
    # sys.argv = ["cfg_debug.py", "-c", "job"]
    cfg_debug()  # pylint: disable=E1120:no-value-for-parameter̦
