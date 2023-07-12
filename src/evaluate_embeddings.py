# pylint: disable=W1203,C0413,W0212

import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
import torch
from dotenv import load_dotenv
from lightning import seed_everything
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import functional as tm_functional

# add parents directory to sys.path.
sys.path.insert(1, str(Path(__file__).parent.parent))

from src.data.nsynth import NSynthDataset
from src.models.encodec.encoder import EncodecEncoder
from src.utils.embeddings import get_embeddings
from src.utils.hparams import log_hyperparameters
from src.utils.logger import LogAbsolutePairwiseDists, LogRelativePairwiseDists

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

    # instantiate dataset and dataloader
    nsynth_dataset = NSynthDataset(root=Path(cfg.data.root), sources=cfg.data.sources)
    nsynth_dataloader = DataLoader(
        nsynth_dataset, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle
    )

    # instantiate encoder
    log.info(f"Instantiating model <{cfg.model.encoder._target_}>")
    encoder: EncodecEncoder = hydra.utils.call(cfg.model.encoder)
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
    #################### get embeddings

    embeddings, indices_from_batch = get_embeddings(
        encoder=encoder,
        dataloader=nsynth_dataloader,
        num_samples=cfg.data.num_samples,
        data_sample_rate=nsynth_dataset.sample_rate,
        encoder_sample_rate=encoder.sample_rate,
        encoder_channels=encoder.channels,
        encoder_frame_length=encoder.segment,
        device=DEVICE,
    )

    #################### Process embeddings
    # flatten embeddings
    embeddings = embeddings.view(cfg.data.num_samples, -1)
    log.info(f"Embeddings shape: {embeddings[0].shape[0]}")
    # TODO: try to process the embeddings as follow:
    # - global pooling along the channel (and frequency?) dimensions to preserve temporal information.
    # - check inter channels correlation take the mean per group of correlated channel or so
    # - concat features from different layers (see papers in Zotero)
    # - random projection

    #################### compute distance metrics
    # cfg.similarity must be a valid torchmetrics.functional metric. The behavior has only been tested
    # for "pairwise_cosine_similarity", "pairwise_euclidean_distance", and "pairwise_manhattan_distance".
    distance_fn = getattr(tm_functional, cfg.similarity)
    log.info(f"Computing distance matrix using {cfg.similarity}...")
    distance_matrix = distance_fn(embeddings)

    #################### compute and log evaluation metrics
    if logger and cfg.metrics.get("absolute_pairwise_dists"):
        log.info("Computing and logging absolute pairwise distances based metrics...")

        abs_pairwise_dists_logger = LogAbsolutePairwiseDists(
            logger=logger,
            distance_matrix=distance_matrix,
            indices_from_batch=indices_from_batch,
            dataset=nsynth_dataset,
            descending=bool(cfg.similarity == "pairwise_cosine_similarity"),
        )

        abs_pairwise_dists_logger.log_n_pairs(
            n=cfg.metrics.absolute_pairwise_dists.num_pairs,
            mode=cfg.metrics.absolute_pairwise_dists.mode,
        )

    if logger and cfg.metrics.get("relative_pairwise_dists"):
        log.info("Computing and logging relative pairwise distances based metrics")
        sub_cfg = cfg.metrics.relative_pairwise_dists

        rel_pairwise_dists_logger = LogRelativePairwiseDists(
            logger=logger,
            distance_matrix=distance_matrix,
            indices_from_batch=indices_from_batch,
            num_samples=sub_cfg.num_samples,
            dataset=nsynth_dataset,
            descending=bool(cfg.similarity == "pairwise_cosine_similarity"),
        )
        rel_pairwise_dists_logger.log_n_neighbors(
            sub_cfg.n_neighbors, mode=sub_cfg.mode
        )

    #################### Logging hparams

    if logger:
        log.info("Logging hyperparameters...")
        log_hyperparameters(
            cfg=cfg, encoder=encoder, torchinfo_summary=torchinfo_summary, device=DEVICE
        )


if __name__ == "__main__":
    # check if program in debug mode, and set the corresponding config if so
    gettrace = getattr(sys, "gettrace", None)
    if gettrace():
        # sys.argv = ["audio_model_analysis.py", "debug=default"]
        sys.argv = ["audio_model_analysis.py", "debug=with_logger"]
    main()  # pylint: disable=no-value-for-parameter
