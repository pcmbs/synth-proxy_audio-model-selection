# pylint: disable=W1203,C0413,W0212
"""
Evaluate an audio model based on the provided configuration and log the result and used hyperparameters to wandb.
The available evaluations are:
- `nearest_neighbors_eval`: evaluate the ability of a model to order sounds by "similarity".
- `parameter_values_ranking_eval`: used to evaluate the ability of a model to order sounds
subject to monotonic changes of parameter values corresponding to different sound attributes.

See corresponding modules for more details.
"""
import logging

import hydra
import torch
import wandb
from dotenv import load_dotenv
from lightning import seed_everything
from omegaconf import DictConfig
from torch.nn import Module
from torchinfo import summary

from evaluations import nearest_neighbors_eval, sound_attributes_ranking_eval
from utils.hparams import log_hyperparameters

load_dotenv()  # take environment variables from .env for hydra configs

# logger for this file
log = logging.getLogger(__name__)

# set torch device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(
    version_base=None, config_path="../configs", config_name="evaluate_audio_model"
)
def evaluate_audio_model(cfg: DictConfig) -> None:
    """
    Evaluate an audio model based on the provided configuration and log the result and used hyperparameters to wandb.
    The available evaluations are:
    - `nearest_neighbors_eval`: evaluate the ability of a model to order sounds by "similarity".
    - `sound_attributes_ranking_eval`: used to evaluate the ability of a model to order sounds subject to monotonic
    changes of parameter values corresponding to different sound attributes.

    See corresponding modules for more details.

    Args
    - `cfg` (DictConfig): hydra configuration settings for the evaluation.

    Returns
    - `None`
    """
    #################### preparation
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # instantiate encoder
    log.info(f"Instantiating model <{cfg.model.encoder._target_}>")
    encoder: Module = hydra.utils.call(cfg.model.encoder)
    encoder.to(DEVICE)

    # instantiate wandb logger
    if cfg.get("wandb"):
        log.info("Instantiating wandb logger")
        logger = wandb.init(**cfg.wandb)

    # # print torchinfo model summary
    # torchinfo_summary = summary(
    #     encoder.encoder.model,
    #     input_size=(1, encoder.channels, int(encoder.segment_length)),
    # )

    #################### evaluations

    if cfg.eval.get("sound_attributes_ranking"):
        log.info("Running sound attributes ranking evaluation...")

        corrcoefs = sound_attributes_ranking_eval(
            path_to_dataset=cfg.eval.sound_attributes_ranking.root,
            encoder=encoder,
            distance_fn=cfg.distance_fn,
            reduce_fn=cfg.reduce_fn,
        )

    if cfg.eval.get("nearest_neighbors"):
        if not cfg.get("wandb"):
            log.info(
                "nearest neighbors evaluation requires a wandb logger, skipping..."
            )
        else:
            log.info("Running nearest neighbors evaluation...")

            nearest_neighbors_eval(
                cfg=cfg.eval.nearest_neighbors,
                encoder=encoder,
                distance_fn=cfg.distance_fn,
                reduce_fn=cfg.reduce_fn,
                logger=logger,
            )

    #################### Logging hparams

    if cfg.get("wandb"):
        log.info("Logging hyperparameters...")
        log_hyperparameters(
            cfg=cfg,
            corrcoefs=corrcoefs if cfg.eval.get("sound_attributes_ranking") else None,
            encoder=encoder,
            # torchinfo_summary=torchinfo_summary,
        )
        wandb.finish()  # required for hydra multirun


if __name__ == "__main__":
    import sys

    # sys.argv = ["evaluate_audio_model.py", "debug=with_logger"]
    # check if program in debug mode, and set the corresponding config if so
    gettrace = getattr(sys, "gettrace", None)
    if gettrace():
        # sys.argv = ["evaluate_audio_model.py", "debug=default"]
        # sys.argv = ["evaluate_audio_model.py", "debug=with_logger"]
        sys.argv = ["evaluate_audio_model.py"]
    evaluate_audio_model()  # pylint: disable=no-value-for-parameter
