# pylint: disable=W1203,E1101
# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py
import os
from typing import Dict, Optional
import torch
from torch.nn import Module
from omegaconf import DictConfig

# from torchinfo import ModelStatistics
import wandb
from utils import reduce_fn

# set torch device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log_hyperparameters(
    cfg: DictConfig,
    corrcoefs: Optional[Dict[str, float]],
    encoder: Module,
    # torchinfo_summary: ModelStatistics,
) -> None:
    """
    summary statistics and hparams to log to wandb.
    """

    hparams = {}

    ##### General hparams
    hparams["general/audio_length"] = cfg.get("audio_length")
    hparams["general/seed"] = cfg.get("seed")
    hparams["general/distance_fn"] = cfg.get("distance_fn")
    hparams["general/reduce_fn"] = cfg.get("reduce_fn")

    ##### Evaluation related hparams
    if cfg.eval.get("sound_attributes_ranking"):
        sub_cfg = cfg.eval.sound_attributes_ranking
        hparams["attr_ranking/synth"] = sub_cfg.get("root").split("/")[-2]
        for key, val in corrcoefs.items():
            wandb.run.summary[f"attr_ranking/{key}"] = val

    if cfg.eval.get("nearest_neighbors"):
        sub_cfg = cfg.eval.nearest_neighbors.data
        hparams["nearest_neighbors/dataset"] = sub_cfg.get("root").split("/")[-1]
        hparams["nearest_neighbors/num_samples"] = sub_cfg.get("num_samples")

    ##### Model related hparams
    hparams["model/name"] = cfg.model.get("name")
    hparams["model/input_repr"] = cfg.model.get("input_repr")
    # hparams["model/num_params"] = sum(p.numel() for p in encoder.parameters())
    # hparams["model/total_mult_adds"] = torchinfo_summary.total_mult_adds
    # hparams["model/params_size_MB"] = ModelStatistics.to_megabytes(
    #     torchinfo_summary.total_param_bytes
    # )
    # hparams["model/forbackward_size_MB"] = ModelStatistics.to_megabytes(
    #     torchinfo_summary.total_output_bytes
    # )
    # get embedding size for one second of audio
    one_second_input = torch.rand(
        (1, encoder.channels, encoder.sample_rate), device=DEVICE
    )
    if encoder.name.startswith("openl3"):
        embedding = encoder(one_second_input.swapdims(-1, -2), encoder.sample_rate)
    elif encoder.name.startswith("encodec"):
        embedding = encoder(one_second_input)[0]
    else:
        raise NotImplementedError()
    wandb.run.summary["model/emb_size_per_sec"] = getattr(reduce_fn, cfg.reduce_fn)(
        embedding.detach()
    ).shape[-1]

    ##### Save hparams and hydra config
    wandb.config.update(hparams)
    # hydra config is saved under <project_name>/Runs/<run_id>/Files/.hydra
    wandb.save(
        glob_str=os.path.join(cfg.paths.get("output_dir"), ".hydra", "*.yaml"),
        base_path=cfg.paths.get("output_dir"),
    )
