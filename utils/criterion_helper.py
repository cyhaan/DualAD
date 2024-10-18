import torch
import torch.nn as nn


class FeatureMSELoss(nn.Module):
    def __init__(self, weight, return_type="sum"):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight
        self.return_type = return_type

    def forward(self, input):
        feats_rec = input["feature_rec"].split(input["outplanes"], dim=1)
        feats_align = input["feature_align"].split(input["outplanes"], dim=1)
        losses = [self.criterion_mse(fr, fa) for fa, fr in zip(feats_align, feats_rec)]
        if self.return_type == "sum":
            return sum(losses)
        else:
            return sum(losses) / len(losses)


class ImageMSELoss(nn.Module):
    """Train a decoder for visualization of reconstructed features"""

    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        image = input["image"]
        image_rec = input["image_rec"]
        return self.criterion_mse(image, image_rec)


def build_criterion(config):
    loss_dict = {}
    for i in range(len(config)):
        cfg = config[i]
        loss_name = cfg["name"]
        loss_dict[loss_name] = globals()[cfg["type"]](**cfg["kwargs"])
    return loss_dict
