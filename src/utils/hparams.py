# pylint: disable=W1203,E1101
# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py
import os

import torch
import torch.nn as nn
from omegaconf import DictConfig

import wandb


def log_hyperparameters(cfg: DictConfig, encoder: nn.Module, device: str) -> None:
    """
    log hparams to wandb.
    """

    hparams = {}

    hparams["model"] = cfg.model.get("name")
    hparams["sim_metric"] = cfg.get("similarity")
    hparams["audio_length"] = cfg.data.get("audio_length")
    hparams["dataset"] = cfg.data.get("root").split("/")[-1]
    hparams["num_samples"] = cfg.data.get("num_samples")
    # save number of model parameters
    hparams["num_params"] = sum(p.numel() for p in encoder.parameters())

    hparams["seed"] = cfg.get("seed")

    # get embedding size for one second of audio
    one_second_input = torch.rand(
        (1, encoder.channels, encoder.sample_rate), device=device
    )
    hparams["repr_size_per_sec"] = list(
        encoder(one_second_input)[0].shape
    )  # TODO: account for reduction

    # send hparams to wandb
    wandb.config = hparams

    # save hydra config in wandb
    wandb.save(
        glob_str=os.path.join(cfg.paths.get("output_dir"), ".hydra", "*.log"),
        base_path="./hydra",  # maybe needs .hydra
        policy="end",
    )
