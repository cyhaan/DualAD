import logging

import numpy as np
import tabulate
from sklearn.metrics import roc_auc_score


class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


def eval_pixel_auc(input):
    score = roc_auc_score(input["masks"].ravel(), input["preds_pixel"].ravel())
    return score


def eval_image_auc(input):
    score = roc_auc_score(input["labels"], input["preds_image"])
    return score




eval_lookup_table = {
    "image_auc": eval_image_auc,
    "pixel_auc": eval_pixel_auc,
}


def performances(image_infos, metrics):
    clsnames = set(image_infos['classes'])
    eval_list = {}

    for clsname in clsnames:
        eval_cls = {}
        items = image_infos['classes'] == clsname
        eval_cls["preds_pixel"] = image_infos["preds_pixel"][items]
        eval_cls["preds_image"] = image_infos["preds_image"][items]
        eval_cls["masks"] = image_infos["masks"][items]
        eval_cls["labels"] = image_infos["labels"][items]
        eval_list[clsname] = eval_cls

    ret_metrics = {}

    for metric in metrics:
        scores = []
        for cls in clsnames:
            score = eval_lookup_table[metric](eval_list[cls])
            scores.append(score)
            ret_metrics["{}_{}".format(cls, metric)] = score
        mean_score = np.mean(np.array(scores))
        ret_metrics["{}_{}".format("mean", metric)] = mean_score

    return ret_metrics


def log_metrics(ret_metrics, evalnames):
    logger = logging.getLogger("global_logger")
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = sorted(list(clsnames - set(["mean"]))) + ["mean"]
    record = Report(["clsname"] + evalnames)
    for clsname in clsnames:
        clsvalues = [
            ret_metrics["{}_{}".format(clsname, evalname)]
            for evalname in evalnames
        ]
        record.add_one_record([clsname] + clsvalues)

    logger.info(f"\n{record}")
