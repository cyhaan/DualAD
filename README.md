
# DualAD

This repository contains the source code for the official implementation in PyTorch of "DualAD: Exploring Coupled Dual-Branch Networks for Multi-class
Unsupervised Anomaly Detection" . 
![Image text](docs/DualAD.png)


## 1. Dependencies

```setup
torch==1.9.0
torchvision==0.10.0
PyYAML==6.0.1
easydict==1.10
tensorboardX==2.6.2.2
opencv-python==3.4.2.16
numpy==1.19.5
einops==0.4.1
tabulate==0.8.10
scikit-learn==0.24.2
Pillow==8.4.0
```
## 2. Data preparation
### 2.1 MVTec-AD
- Download the MVTec-AD dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad). Unzip the file to `./data/mvtec_anomaly_detection/`. If you wish to adjust the location where the file is stored, you can customize this by editing the `dataset.image_dir` field in  `./experiments/MVTec-AD/config.yaml`. Make sure that the dataset directory follow the data tree:
```
|-- mvtec_anomaly_detection
    |-- bottle
        |-- ground_truth
            |-- broken_large
            |-- ...
        |-- test
            |-- broken_large
            |-- ...
            |-- good
        |-- train
            |-- good
    |-- cable
        |-- ...
    |-- ...
```
### 2.2 VisA
- Download the VisA dataset from [here](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar). Unzip the file to `./data/VisA_20220922/`. If you wish to adjust the location where the file is stored, you can customize this by editing the `dataset.download_dir` field in  `./experiments/VisA/config.yaml`. Make sure that the dataset directory follow the data tree:
```
|-- VisA_20220922
    |-- candle
        |-- Data
            |-- Images
                |-- Anomaly
                |-- Normal
            |-- Masks
                |-- Anomaly
        |-- image_anno.csv
    |-- capsules
        |-- ...
    |-- ...
```
- The first time you run the **training** or **evaluation** command, it will take a few minutes to generate the target dataset directory `./data/VisA_pytorch/`, which is as follows:
```
|-- VisA_pytorch
    |-- candle
        |-- ground_truth
            |-- bad
        |-- test
            |-- bad
            |-- good
        |-- train
            |-- good
    |-- capsules
        |-- ...
    |-- ...
```

## 3. Training

- To train the DualAD model on **MVTec-AD**, please run:
```train
cd ./experiments/MVTec-AD/
sh train.sh
```
- To train the DualAD model on **VisA**, please run:
```train
cd ./experiments/VisA/
sh train.sh
```

## 4. Evaluation
- To evaluate the DualAD model on **MVTec-AD**, please firstly set `saver.load_path` field in `./experiments/MVTec-AD/config.yaml` to load the checkpoints, then run:
```eval
cd ./experiments/MVTec-AD/
sh eval.sh
```
- To evaluate the DualAD model on **VisA**, please firstly set `saver.load_path` field in `./experiments/VisA/config.yaml` to load the checkpoints, then run:
```eval
cd ./experiments/VisA/
sh eval.sh
```

## 5. Pre-trained Models

You can download pretrained DualAD models here:
- [DualAD checkpoint](https://drive.google.com/file/d/1Le7mzFqhKKVpPLqweuSo_nO1Urm-7BrQ/view?usp=drive_link) trained on all categories of **MVTec-AD** dataset. 
- [DualAD checkpoint](https://drive.google.com/file/d/1oUcUT1qM8ScR91-zIQjXxYuNlR7miDuw/view?usp=drive_link) trained on all categories of **VisA** dataset. 



## 6. Results

Our model achieves the following performance on:


### 6.1 MVTec-AD
| Model name | I-AUROC | I-AP | I-F1_max | P-AUROC | P-AP | P-F1_max | P-AUPRO |
|------------|---------|------|----------|---------|------|----------|---------|
| DualAD     | 99.0    | 99.7 | 98.3     | 98.2    | 51.6 | 56.8     | 91.6    |
### 6.2 VisA
| Model name | I-AUROC | I-AP | I-F1_max | P-AUROC | P-AP | P-F1_max | P-AUPRO |
|------------|---------|------|----------|---------|------|----------|---------|
| DualAD     | 94.7    | 95.9 | 91.2     | 98.9    | 37.6 | 41.4     | 87.9    |




