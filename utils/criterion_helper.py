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


class CosinConsistentLoss(nn.Module):
    def __init__(self, weight, threshold=0., return_type="sum"):
        super().__init__()
        self.criterion_cos = nn.CosineSimilarity(dim=-1)
        self.weight = weight
        self.threshold = threshold
        self.return_type = return_type

    def forward(self, input):
        if input["mems_anomal"] is None:
            return 0

        losses = [
            torch.mean(torch.clip(1 - self.criterion_cos(f1, f2) - self.threshold, min=0))
            for f1, f2 in zip(input["mems_input"]["memories_attn"], input["mems_anomal"]["memories_attn"])
        ]
        if self.return_type == "sum":
            return sum(losses)
        else:
            return sum(losses) / len(losses)



def build_criterion(config):
    loss_dict = {}
    for i in range(len(config)):
        cfg = config[i]
        loss_name = cfg["name"]
        loss_dict[loss_name] = globals()[cfg["type"]](**cfg["kwargs"])
    return loss_dict
