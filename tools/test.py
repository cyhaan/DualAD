import argparse
import logging
import os
import pprint
import time

import numpy as np
import torch
import torch.optim
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import ModelHelper
from tensorboardX import SummaryWriter
from utils.criterion_helper import build_criterion
from utils.eval_helper import  log_metrics, performances
from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
    update_config,
)
from utils.optimizer_helper import get_optimizer
from utils.vis_helper import visualize_compound

parser = argparse.ArgumentParser(description="MGCFR Framework")
parser.add_argument("--config", default="./config.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("-d", "--debug", action="store_true")


all_classes = {
    "MVTecAD": ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
               'pill', 'screw', 'toothbrush', 'transistor', 'zipper'],
    "VisA": ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
             'pcb3', 'pcb4', 'pipe_fryum']
}

def main():
    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config = update_config(config)

    if config.dataset.classes == 'all':
        config.dataset.classes = all_classes.get(config.dataset.type, None)

    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, config.evaluator.save_dir)
    os.makedirs(config.saver.save_dir, exist_ok=True)
    os.makedirs(config.saver.log_dir, exist_ok=True)

    if args.debug:
        config.dataset.num_workers = 4
        config.dataset.use_cache = False
        config.trainer.val_freq_epoch = 1
        config.trainer.print_freq_step = 1


    current_time = get_current_time()
    logger = create_logger(
        "global_logger", config.log_path + "/dec_{}.log".format(current_time)
    )
    logger.info("args: {}".format(pprint.pformat(args)))
    logger.info("config: \n{}".format(pprint.pformat(config)))


    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    # create model
    model = ModelHelper(config.net)
    model.cuda()

    layers = []
    for module in config.net:
        layers.append(module["name"])
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    logger.info("layers: {}".format(layers))
    logger.info("active layers: {}".format(active_layers))

    load_path = config.saver.get("load_path", None)
    load_state(load_path, model)

    # # parameters needed to be updated
    # parameters = [
    #     {"params": getattr(model, layer).parameters()} for layer in active_layers
    # ]
    #
    # optimizer = get_optimizer(parameters, config.trainer.optimizer)
    # lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)
    #
    # key_metric = config.evaluator["key_metric"]
    # best_metric = 0
    # last_epoch = 0
    #
    # # load model: auto_resume > resume_model > load_path
    # auto_resume = config.saver.get("auto_resume", True)
    # resume_model = config.saver.get("resume_model", None)
    # load_path = config.saver.get("load_path", None)
    #
    # if resume_model and not resume_model.startswith("/"):
    #     resume_model = os.path.join(config.exp_path, resume_model)
    # lastest_model = os.path.join(config.save_path, "ckpt.pth.tar")
    # if auto_resume and os.path.exists(lastest_model):
    #     resume_model = lastest_model
    # if resume_model:
    #     best_metric, last_epoch = load_state(resume_model, model, optimizer=optimizer)
    # elif load_path:
    #     if not load_path.startswith("/"):
    #         load_path = os.path.join(config.exp_path, load_path)
    #     load_state(load_path, model)

    _, val_loader = build_dataloader(config.dataset, training=False)
    validate(val_loader, model)
    save_checkpoint(
        {
        "state_dict": model.state_dict(),
        },
        False,
        config,
        )
    return



def train_one_epoch(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    criterion,
    frozen_layers,
):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    # freeze selected layers
    for layer in frozen_layers:
        module = getattr(model, layer)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    logger = logging.getLogger("global_logger")
    end = time.time()

    for i, input in enumerate(train_loader):
        curr_step = start_iter + i
        current_lr = lr_scheduler.get_lr()[0]

        # measure data loading time
        data_time.update(time.time() - end)

        # forward
        outputs = model(input)
        loss = 0
        loss_items = []
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            value = criterion_loss(outputs)
            loss += weight * value
            loss_items.append({
                'name': name,
                'val': value.item()
            })
        losses.update(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)

        if (curr_step + 1) % config.trainer.tb_freq_step == 0:
            for item in loss_items:
                tb_logger.add_scalar(item["name"], item["val"], curr_step + 1)
            tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
            tb_logger.add_scalar("lr", current_lr, curr_step + 1)

            tb_logger.flush()
        if (curr_step + 1) % config.trainer.print_freq_step == 0:
            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "LR {lr:.5f}\t".format(
                    epoch + 1,
                    config.trainer.max_epoch,
                    curr_step + 1,
                    len(train_loader) * config.trainer.max_epoch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=current_lr,
                )
            )

        end = time.time()


def validate(val_loader, model):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()

    preds_pixel = []
    preds_image = []
    masks = []
    labels = []
    classes = []
    filenames = []
    image_types = []


    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # forward
            outputs = model(input)
            # record loss
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)
            num = len(outputs["file_name"])
            losses.update(loss.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.trainer.print_freq_step == 0:
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(val_loader), batch_time=batch_time
                    )
                )

            preds_pixel.append(outputs["pred_pixel"].cpu().numpy())
            preds_image.append(outputs["pred_image"].cpu().numpy())
            masks.append(outputs["mask"].cpu().numpy().astype(int))
            labels.append(outputs["label"].cpu().numpy())
            classes.append(outputs["cls_name"])
            filenames.append(outputs["file_name"])
            image_types.append(outputs["type"])

    logger.info("Gathering final results ...")
    # total loss
    logger.info(" * Loss {:.5f}\ttotal_num={}".format(losses.avg, losses.count))
    # evaluate, log & vis
    image_infos = {}
    image_infos["preds_pixel"] = np.concatenate(np.asarray(preds_pixel), axis=0)  # N x H x W
    image_infos["masks"] = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    image_infos["preds_image"] = np.concatenate(np.asarray(preds_image), axis=0)  # N
    image_infos["labels"] = np.concatenate(np.asarray(labels), axis=0)  # N
    image_infos["classes"] = np.concatenate(np.asarray(classes), axis=0)  # N
    image_infos["filenames"] = np.concatenate(np.asarray(filenames), axis=0)  # N
    image_infos["image_types"] = np.concatenate(np.asarray(image_types), axis=0)  # N

    ret_metrics = performances(image_infos, config.evaluator.metrics)
    log_metrics(ret_metrics, config.evaluator.metrics)
    if args.evaluate and config.evaluator.get("vis_compound", None):
        visualize_compound(
            image_infos,
            config.evaluator.vis_compound,
            config.dataset.image_dir
        )
    return ret_metrics


if __name__ == "__main__":
    main()
