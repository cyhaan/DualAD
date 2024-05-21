import logging

from datasets.mvtec_dataset import build_mvtec_dataloader
from datasets.visa_dataset import build_visa_dataloader

logger = logging.getLogger("global")


def build(cfg, training):
    if training:
        cfg.update(cfg.get("train", {}))
    else:
        cfg.update(cfg.get("test", {}))

    dataset = cfg["type"]
    if dataset == "MVTecAD":
        data_loader = build_mvtec_dataloader(cfg, training)
    elif dataset == "VisA":
        data_loader = build_visa_dataloader(cfg, training)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader


def build_dataloader(cfg_dataset, training=True):
    train_loader = None
    if training:
        train_loader = build(cfg_dataset, training=True)

    test_loader = build(cfg_dataset, training=False)

    logger.info("build dataset done")
    return train_loader, test_loader
