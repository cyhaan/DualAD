import logging

import numpy as np
import tabulate
from skimage import measure
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
import multiprocessing

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

def eval_pixel_ap(input):
    score = average_precision_score(input["masks"].ravel(), input["preds_pixel"].ravel())
    return score
def eval_image_ap(input):
    score = average_precision_score(input["labels"], input["preds_image"])
    return score

def eval_pixel_f1max(input):
    precisions, recalls, thresholds = precision_recall_curve(input["masks"].ravel(), input["preds_pixel"].ravel())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]
    return best_f1_score
def eval_image_f1max(input):
    precisions, recalls, thresholds = precision_recall_curve(input["labels"], input["preds_image"])
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]
    return best_f1_score

def eval_pixel_aupro(input):
    score = cal_pro_score(input["masks"].squeeze(), input["preds_pixel"].squeeze())
    return score

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc

eval_lookup_table = {
    "image_auc": eval_image_auc,
    "pixel_auc": eval_pixel_auc,
    "image_ap": eval_image_ap,
    "pixel_ap": eval_pixel_ap,
    "image_f1max": eval_image_f1max,
    "pixel_f1max": eval_pixel_f1max,
    "pixel_aupro": eval_pixel_aupro,
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
