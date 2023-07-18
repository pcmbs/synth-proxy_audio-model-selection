# pylint: disable=W1203,C0413,W0212

import logging

# import sys
from typing import Optional

import hydra
import torch
from torch.nn import Module
from dotenv import load_dotenv
from lightning import seed_everything
from omegaconf import DictConfig
from torchinfo import summary

from evals import parameter_variations_eval
from utils.hparams import log_hyperparameters


load_dotenv()  # take environment variables from .env for hydra configs

# logger for this file
log = logging.getLogger(__name__)

# set torch device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(
    version_base=None, config_path="../configs", config_name="evaluate_embeddings"
)
def main(cfg: DictConfig) -> Optional[float]:
    #################### preparation
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # instantiate encoder
    log.info(f"Instantiating model <{cfg.model.encoder._target_}>")
    encoder: Module = hydra.utils.call(cfg.model.encoder)
    encoder.to(DEVICE)

    # instantiate wandb logger
    if cfg.get("logger"):
        log.info("Instantiating wandb logger")
        logger = hydra.utils.call(cfg.logger)

    # print torchinfo model summary
    torchinfo_summary = summary(
        encoder.encoder.model,
        input_size=(1, encoder.channels, int(encoder.segment_length)),
    )

    #################### embeddings evaluations

    if cfg.eval.get("parameter_variations"):
        log.info("Running parameter variations evaluation...")

        corrcoefs = parameter_variations_eval(
            path_to_dataset=cfg.eval.parameter_variations.root,
            variations=cfg.eval.parameter_variations.available_variations,
            encoder=encoder,
            similarity=cfg.similarity_fn,
            reduce_fn=cfg.reduce_fn,
        )

    if cfg.eval.get("nearest_neighbors"):
        log.info("Running nearest neighbors evaluation...")

    #################### Logging hparams

    if logger:
        log.info("Logging hyperparameters...")
        log_hyperparameters(
            cfg=cfg,
            corrcoefs=corrcoefs,
            encoder=encoder,
            torchinfo_summary=torchinfo_summary,
            device=DEVICE,
        )


if __name__ == "__main__":
    # check if program in debug mode, and set the corresponding config if so
    # gettrace = getattr(sys, "gettrace", None)
    # if gettrace():
    #     # sys.argv = ["audio_model_analysis.py", "debug=default"]
    #     # sys.argv = ["audio_model_analysis.py", "debug=with_logger"]
    #     sys.argv = ["audio_model_analysis.py"]
    main()  # pylint: disable=no-value-for-parameter
