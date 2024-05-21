import os

import cv2
import numpy as np


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def visualize_compound(image_infos, cfg_vis, img_dir):
    vis_dir = cfg_vis.save_dir
    max_score = cfg_vis.get("max_score", None)
    min_score = cfg_vis.get("min_score", None)
    preds = image_infos["preds_pixel"]
    masks = image_infos["masks"]
    max_score = preds.max() if not max_score else max_score
    min_score = preds.min() if not min_score else min_score

    filenames = image_infos["filenames"]
    classes = image_infos["classes"]
    image_types = image_infos["image_types"]
    for i, (filename, clsname, image_type) in enumerate(zip(filenames,classes,image_types)):
        save_dir = os.path.join(vis_dir, clsname, image_type)
        os.makedirs(save_dir, exist_ok=True)
        # read image
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[0], image.shape[1]
        pred = preds[i].squeeze()[:, :, None].repeat(3, 2)
        pred = cv2.resize(pred, (w, h))

        # self normalize just for analysis
        scoremap_self = apply_ad_scoremap(image, normalize(pred))
        # global normalize
        pred = np.clip(pred, min_score, max_score)
        pred = normalize(pred, max_score, min_score)
        scoremap_global = apply_ad_scoremap(image, pred)

        if masks is not None:
            mask = (masks[i] * 255).squeeze().astype(np.uint8)[:, :, None].repeat(3, 2)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            save_path = os.path.join(save_dir, os.path.split(filename)[1])
            if mask.sum() == 0:
                scoremap = np.vstack([image, scoremap_global])
            else:
                scoremap = np.vstack([image, mask, scoremap_global, scoremap_self])
        else:
            scoremap = np.vstack([image, scoremap_global, scoremap_self])

        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, scoremap)


