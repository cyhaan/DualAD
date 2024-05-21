import csv
import glob
import logging
import os
import shutil

import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


logger = logging.getLogger("global_logger")


all_classes = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
             'pcb3', 'pcb4', 'pipe_fryum']
def prepare_data(cfg):
    save_folder = cfg.image_dir
    source_folder = cfg.download_dir
    logger.info("preparing data from : {}".format(source_folder))
    for cls in all_classes:
        train_folder = os.path.join(save_folder, cls, 'train')
        test_folder = os.path.join(save_folder, cls, 'test')
        mask_folder = os.path.join(save_folder, cls, 'ground_truth')

        train_img_good_folder = os.path.join(train_folder, 'good')
        test_img_good_folder = os.path.join(test_folder, 'good')
        test_img_bad_folder = os.path.join(test_folder, 'bad')
        test_mask_bad_folder = os.path.join(mask_folder, 'bad')

        os.makedirs(train_img_good_folder)
        os.makedirs(test_img_good_folder)
        os.makedirs(test_img_bad_folder)
        os.makedirs(test_mask_bad_folder)

    split_file = os.path.join(source_folder, 'split_csv', '1cls.csv')
    with open(split_file, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            object, set, label, image_path, mask_path = row
            if label == 'normal':
                label = 'good'
            else:
                label = 'bad'
            image_name = image_path.split('/')[-1]
            mask_name = mask_path.split('/')[-1]
            img_src_path = os.path.join(source_folder, image_path)
            msk_src_path = os.path.join(source_folder, mask_path)
            img_dst_path = os.path.join(save_folder, object, set, label, image_name)
            msk_dst_path = os.path.join(save_folder, object, 'ground_truth', label, mask_name)
            shutil.copyfile(img_src_path, img_dst_path)
            if set == 'test' and label == 'bad':
                mask = Image.open(msk_src_path)

                # binarize mask
                mask_array = np.array(mask)
                mask_array[mask_array != 0] = 255
                mask = Image.fromarray(mask_array)

                mask.save(msk_dst_path)

def build_visa_dataloader(cfg, training):
    if not os.path.exists(cfg.image_dir):
        prepare_data(cfg)
    if training:
        return build_visa_train_dataloader(cfg)
    else:
        return build_visa_test_dataloader(cfg)

def build_visa_train_dataloader(cfg):
    cfg.update(cfg.get("train", {}))
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.normalize.mean, cfg.normalize.std)
    ])

    logger.info("building VisA training datasets for classes: {}".format(cfg.classes))
    dataset = VisADataset_train(
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


def build_visa_test_dataloader(cfg):
    cfg.update(cfg.get("test", {}))
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.normalize.mean, cfg.normalize.std)
    ])
    gt_transforms = transforms.Compose([transforms.ToTensor()])
    logger.info("building VisA test datasets for classes: {}".format(cfg.classes))
    dataset = VisADataset_test(
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
    )

    return data_loader

def build_visa_vis_train_dataloader(cfg):
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
    dataset = VisADataset_test(
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

class VisADataset_train(torch.utils.data.Dataset):
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
        # image = cv2.resize(image / 255., (self.input_size[0], self.input_size[1]))
        image = cv2.resize(image, (self.input_size[0], self.input_size[1]))
        image = PIL.Image.fromarray(image, "RGB")

        return image

    def load_dataset(self):
        image_paths = []
        image_classes = []
        for cls in self.classes:
            paths = sorted(glob.glob(os.path.join(self.image_dir, cls + '/train/good/*.JPG')))
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
        # image_path = self.image_paths[idx]
        # image_cls = self.image_classes[idx]
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


class VisADataset_test(torch.utils.data.Dataset):
    def __init__(self, image_dir, classes, input_size, data_transform=None, gt_transform=None, use_cache=True):
        self.input_size = input_size
        self.classes = classes
        self.image_dir = image_dir
        # load datasets
        # self.image_paths, self.mask_paths, self.labels, self.image_types, self.image_classes = self.load_dataset()  # self.labels => good : 0, anomaly : 1
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
            mask = np.zeros((self.input_size[0], self.input_size[1])).astype(np.uint8)
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
                paths = sorted(glob.glob(os.path.join(cls_dir, defect_type + "/*.JPG")))
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
