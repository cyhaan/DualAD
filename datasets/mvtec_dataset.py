import glob
import logging
import os

import PIL
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


logger = logging.getLogger("global_logger")


def build_mvtec_dataloader(cfg, training):
    if training:
        return build_mvtec_train_dataloader(cfg)
    else:
        return build_mvtec_test_dataloader(cfg)

def build_mvtec_train_dataloader(cfg):
    cfg.update(cfg.get("train", {}))
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.normalize.mean, cfg.normalize.std)
    ])

    logger.info("building MVTec-AD training dataset for classes: {}".format(cfg.classes))
    dataset = MVTecDataset_train(
        image_dir=cfg.image_dir,
        classes=cfg.classes,
        input_size=cfg.input_size,
        data_transform=data_transforms,
        use_cache=cfg.get("use_cache", False)
    )
    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        shuffle=True,
    )

    return data_loader


def build_mvtec_test_dataloader(cfg):
    cfg.update(cfg.get("test", {}))
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.normalize.mean, cfg.normalize.std)
    ])
    gt_transforms = transforms.Compose([transforms.ToTensor()])
    logger.info("building MVTec-AD test dataset for classes: {}".format(cfg.classes))
    dataset = MVTecDataset_test(
        image_dir=cfg.image_dir,
        classes=cfg.classes,
        input_size=cfg.input_size,
        data_transform=data_transforms,
        gt_transform=gt_transforms,
        use_cache=cfg.get("use_cache", False)
    )
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        # shuffle=False,
    )

    return data_loader

def build_mvtec_vis_train_dataloader(cfg):
    cfg.update(cfg.get("test", {}))
    transform_fns = []
    if cfg.input_size[0] < cfg.reshape_size[0] or cfg.input_size[1] < cfg.reshape_size[1]:
        transform_fns.append(transforms.CenterCrop(cfg.input_size))
    transform_fns.append(transforms.ToTensor())
    transform_fns.append(transforms.Normalize(cfg.normalize.mean, cfg.normalize.std))
    data_transforms = transforms.Compose(transform_fns)
    transform_fns = []
    if cfg.input_size[0] < cfg.reshape_size[0] or cfg.input_size[1] < cfg.reshape_size[1]:
        transform_fns.append(transforms.CenterCrop(cfg.input_size))
    transform_fns.append(transforms.ToTensor())
    gt_transforms = transforms.Compose(transform_fns)
    logger.info("building MVTec-AD test datasets with classes: {}".format(cfg.classes))
    dataset = MVTecDataset_test(
        image_dir=cfg.image_dir,
        classes=cfg.classes,
        reshape_size=cfg.reshape_size,
        input_size=cfg.input_size,
        data_transform=data_transforms,
        gt_transform=gt_transforms,
        use_cache=cfg.get("use_cache", True)

    )
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    return data_loader

class MVTecDataset_train(torch.utils.data.Dataset):
    def __init__(self, image_dir, classes, input_size, data_transform=None, use_cache=True):
        self.input_size = input_size
        self.classes = classes
        self.image_dir = image_dir
        self.use_cache = use_cache
        # load datasets
        self.image_infos = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.data_transform = data_transform

    def load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size[0], self.input_size[1]))
        image = PIL.Image.fromarray(image, "RGB")

        return image

    def load_dataset(self):
        image_paths = []
        image_classes = []
        for cls in self.classes:
            paths = sorted(glob.glob(os.path.join(self.image_dir, cls + '/train/good/*.png')))
            image_paths.extend(paths)
            image_classes.extend([cls] * len(paths))
        if self.use_cache:
            images = []
            for path in image_paths:
                images.append(self.load_image(path))
        else:
            images = [None] * len(image_paths)
        return list(zip(image_paths, image_classes, images))



    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, idx):
        sample = {}
        image_path, image_cls, image = self.image_infos[idx]
        sample.update(
            {
                # "file_name": image_path.split('/')[-1],
                "file_name": image_path,
                "cls_name": image_cls,
            }
        )
        if image is None:
            image = self.load_image(image_path)
        image = self.data_transform(image)

        sample.update(
            {
                "image": image,
            }
        )
        return sample


class MVTecDataset_test(torch.utils.data.Dataset):
    def __init__(self, image_dir, classes, input_size, data_transform=None, gt_transform=None, use_cache=True):
        self.input_size = input_size
        self.classes = classes
        self.image_dir = image_dir
        # load datasets
        self.use_cache = use_cache
        self.image_infos = self.load_dataset()
        self.data_transform = data_transform
        self.gt_transform = gt_transform

    def load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size[0], self.input_size[1]))
        image = PIL.Image.fromarray(image, "RGB")
        return image

    def load_mask(self, mask_path):
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # mask = cv2.resize(mask / 255., (self.input_size[0], self.input_size[1]))
            mask = cv2.resize(mask, (self.input_size[0], self.input_size[1]), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((self.input_size[0], self.input_size[0])).astype(np.uint8)
        mask = PIL.Image.fromarray(mask, "L")
        return mask

    def load_dataset(self):
        image_paths = []
        image_classes = []
        mask_paths = []
        labels = []
        image_types = []
        for cls in self.classes:
            cls_dir = os.path.join(self.image_dir, cls + '/test')
            mask_dir = os.path.join(self.image_dir, cls + '/ground_truth')
            defect_types = os.listdir(cls_dir)
            for defect_type in defect_types:
                paths = sorted(glob.glob(os.path.join(cls_dir, defect_type + "/*.png")))
                image_paths.extend(paths)
                image_classes.extend([cls] * len(paths))
                image_types.extend([defect_type] * len(paths))
                if defect_type == 'good':
                    mask_paths.extend([None] * len(paths))
                    labels.extend([0] * len(paths))
                else:
                    paths = sorted(glob.glob(os.path.join(mask_dir, defect_type + "/*.png")))
                    mask_paths.extend(paths)
                    labels.extend([1] * len(paths))
        assert len(image_paths) == len(mask_paths), "Something wrong with test and ground truth pair!"

        if self.use_cache:
            images = []
            masks = []
            for image_path, mask_path in zip(image_paths, mask_paths):
                images.append(self.load_image(image_path))
                masks.append(self.load_mask(mask_path))
        else:
            images = [None] * len(image_paths)
            masks = [None] * len(mask_paths)

        return list(zip(image_paths, mask_paths, labels, image_types, image_classes, images, masks))

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, idx):
        sample = {}
        image_path, mask_path, label, image_type, image_cls, image, mask = self.image_infos[idx]
        sample.update(
            {
                # "file_name": image_path.split('/')[-1],
                "file_name": image_path,
                "cls_name": image_cls,
                "type": image_type,
                "label": label,
            }
        )
        if image is None:
            image = self.load_image(image_path)
            if mask_path is not None:
                mask = self.load_mask(mask_path)
        image = self.data_transform(image)
        if mask is None:
            mask = torch.zeros([1, self.input_size[0], self.input_size[1]])
        else:
            mask = self.gt_transform(mask)

        sample.update(
            {
                "image": image,
                "mask": mask,
            }
        )
        return sample
