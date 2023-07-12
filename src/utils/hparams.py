# pylint: disable=W1203,E1101
# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py
import os

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchinfo import ModelStatistics
import wandb


def log_hyperparameters(
    cfg: DictConfig, encoder: nn.Module, torchinfo_summary: ModelStatistics, device: str
) -> None:
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

    # add infos from torchinfos
    hparams["total_mult_adds"] = f"{torchinfo_summary.total_mult_adds:g}"
    hparams["params_size_MB"] = ModelStatistics.to_megabytes(
        f"{torchinfo_summary.total_param_bytes:0.2f}"
    )
    hparams["forbackward_size_MB"] = ModelStatistics.to_megabytes(
        f"{torchinfo_summary.total_output_bytes:0.2}"
    )
    # send hparams to wandb
    wandb.config.update(hparams)

    # save hydra config in wandb
    wandb.save(
        glob_str=os.path.join(cfg.paths.get("output_dir"), ".hydra", "*.yaml"),
        base_path=cfg.paths.get("output_dir"),
    )
