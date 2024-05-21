import argparse
import logging
import os
import pprint
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict

from datasets.visa_dataset import prepare_data
from models.model_helper import ModelHelper
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.criterion_helper import build_criterion
from utils.dist_helper import setup_distributed
from utils.eval_helper import dump, log_metrics, merge_together, performances
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
from utils.vis_helper import visualize_compound, visualize_single

parser = argparse.ArgumentParser(description="MGCFR Framework")
parser.add_argument("--config", default="./test.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("-d", "--debug", action="store_true")

# parser.add_argument("--local_rank", default=None, help="local rank for dist")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

all_classes = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
             'pcb3', 'pcb4', 'pipe_fryum']


def main():
    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # config.port = config.get("port", None)
    # rank, world_size = setup_distributed(port=config.port)
    config = update_config(config)

    exp_dir = os.path.dirname(args.config)
    # dir_name = "_".join(config.dataset.classes)
    dir_name = config.dataset.classes
    config.saver.vis_dir = os.path.join(exp_dir, config.saver.vis_dir)
    config.saver.log_dir = os.path.join(exp_dir, config.saver.log_dir, dir_name)
    config.saver.save_dir = os.path.join(exp_dir, config.saver.save_dir, dir_name)
    config.evaluator.eval_dir = os.path.join(exp_dir, config.evaluator.save_dir, dir_name)
    config.saver.load_path = os.path.join(config.saver.save_dir, config.saver.ckpt_name + ".pth.tar")

    if config.dataset.classes == 'all':
        config.dataset.classes = all_classes
    else:
        config.dataset.classes = [config.dataset.classes]

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
    logger = create_logger("global_logger",
                           os.path.join(config.saver.log_dir, config.saver.ckpt_name + "_{}.log".format(current_time)))
    logger.info("args: {}".format(pprint.pformat(args)))
    logger.info("config: \n{}".format(pprint.pformat(config)))


    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    # create model
    model = ModelHelper(config.net)
    model.cuda()
    # local_rank = int(os.environ["LOCAL_RANK"])
    # model = DDP(
    #     model,
    #     device_ids=[local_rank],
    #     output_device=local_rank,
    #     find_unused_parameters=True,
    # )

    layers = []
    for module in config.net:
        layers.append(module["name"])
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    logger.info("layers: {}".format(layers))
    logger.info("active layers: {}".format(active_layers))

    # parameters needed to be updated
    parameters = [
        {"params": getattr(model, layer).parameters()} for layer in active_layers
    ]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    key_metric = config.evaluator["key_metric"]

    # best_metric = 0
    # last_epoch = 0

    auto_resume = config.saver.get("auto_resume", True)
    load_path = config.saver.load_path
    if auto_resume and os.path.exists(load_path):
        resume_model = load_path
        best_metric, last_epoch = load_state(resume_model, model, optimizer=optimizer)
    else:
        # if not load_path.startswith("/"):
        #     load_path = os.path.join(config.exp_path, load_path)
        best_metric, last_epoch = load_state(load_path, model)

    if not args.evaluate:
        train_loader, val_loader = build_dataloader(config.dataset, training=True)
        tb_logger = SummaryWriter(
            os.path.join(config.saver.log_dir, "events_" + config.saver.get("ckpt_name"), current_time))
    else:
        _, val_loader = build_dataloader(config.dataset, training=False)
        validate(val_loader, model)
        return

    criterion = build_criterion(config.criterion)

    for epoch in range(last_epoch, config.trainer.max_epoch):
        # train_loader.sampler.set_epoch(epoch)
        # val_loader.sampler.set_epoch(epoch)
        last_iter = epoch * len(train_loader)
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            last_iter,
            tb_logger,
            criterion,
            frozen_layers,
        )
        lr_scheduler.step(epoch)

        if (epoch + 1) % config.trainer.val_freq_epoch == 0:
            ret_metrics = validate(val_loader, model)
            ret_key_metric = ret_metrics[key_metric]
            is_best = ret_key_metric >= best_metric
            best_metric = max(ret_key_metric, best_metric)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": config.net,
                    "state_dict": model.state_dict(),
                    "best_metric": best_metric,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                config,
            )
            for scalar in config.evaluator.scalar:
                tb_logger.add_scalar(scalar, ret_metrics[scalar], epoch + 1)
            tb_logger.flush()
        else:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "arch": config.net,
                    "state_dict": model.state_dict(),
                    "best_metric": best_metric,
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(config.saver.save_dir, config.saver.ckpt_name + ".pth.tar")
            )


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

    # world_size = dist.get_world_size()
    # rank = dist.get_rank()
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
        # reduced_loss = loss.clone()
        # dist.all_reduce(reduced_loss)
        # reduced_loss = reduced_loss / world_size
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
            info = (
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss ".format(
                    epoch + 1,
                    config.trainer.max_epoch,
                    curr_step + 1,
                    len(train_loader) * config.trainer.max_epoch,
                    batch_time=batch_time,
                    data_time=data_time,
                )
            )
            for item in loss_items:
                info += "{loss:.5f} ".format(loss=item["val"])
            info += "{loss.val:.5f} ({loss.avg:.5f}) \t".format(loss=losses)
            info += "LR {lr:.5f}\t".format(lr=current_lr)
            logger.info(info)

            # logger.info(
            #     "Epoch: [{0}/{1}]\t"
            #     "Iter: [{2}/{3}]\t"
            #     "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
            #     "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
            #     "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
            #     "LR {lr:.5f}\t".format(
            #         epoch + 1,
            #         config.trainer.max_epoch,
            #         curr_step + 1,
            #         len(train_loader) * config.trainer.max_epoch,
            #         batch_time=batch_time,
            #         data_time=data_time,
            #         loss=losses,
            #         lr=current_lr,
            #     )
            # )

        end = time.time()


def validate(val_loader, model):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    # rank = dist.get_rank()
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

    # if rank == 0:
    #     os.makedirs(config.evaluator.eval_dir, exist_ok=True)
    # # all threads write to config.evaluator.eval_dir, it must be made before every thread begin to write
    # dist.barrier()

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # forward
            outputs = model(input)
            # dump(config.evaluator.eval_dir, outputs)

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
            # mask[mask > 0.5] = 1
            # mask[mask <= 0.5] = 0

            preds_pixel.append(outputs["pred_pixel"].cpu().numpy())
            preds_image.append(outputs["pred_image"].cpu().numpy())
            masks.append(outputs["mask"].cpu().numpy().astype(int))
            labels.append(outputs["label"].cpu().numpy())
            classes.append(outputs["cls_name"])
            filenames.append(outputs["file_name"])
            image_types.append(outputs["type"])

    # # gather final results
    # dist.barrier()
    # total_num = torch.Tensor([losses.count]).cuda()
    # loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
    # dist.all_reduce(total_num, async_op=True)
    # dist.all_reduce(loss_sum, async_op=True)
    # final_loss = loss_sum.item() / total_num.item()

    # ret_metrics = {}  # only ret_metrics on rank0 is not empty

    logger.info("Gathering final results ...")
    # total loss
    logger.info(" * Loss {:.5f}\ttotal_num={}".format(losses.avg, losses.count))
    # fileinfos, preds, masks = merge_together(config.evaluator.eval_dir)
    # shutil.rmtree(config.evaluator.eval_dir)
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
    # if args.evaluate and config.evaluator.get("vis_single", None):
    #     visualize_single(
    #         fileinfos,
    #         preds,
    #         config.evaluator.vis_single,
    #         config.dataset.image_reader,
    #     )
    return ret_metrics


if __name__ == "__main__":
    main()
