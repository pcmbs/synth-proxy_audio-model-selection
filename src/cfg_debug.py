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
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("breakpoint me!")


if __name__ == "__main__":
    args = ["my_app.py", "debug=default"]

    sys.argv = args

    gettrace = getattr(sys, "gettrace", None)
    if gettrace():
        sys.argv = args
    # sys.argv = ["my_app.py", "-c", "job"]
    my_app()  # pylint: disable=E1120:no-value-for-parameter̦
