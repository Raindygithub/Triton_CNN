# 将 CIFAR-10 转换为 LMDB 格式
import lmdb
import pickle
import torchvision.datasets as datasets
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import time
from datetime import datetime
import subprocess  # 用于获取GPU利用率
from model import Net
from torchvision.transforms import Compose
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from dali_pipeline import cifar10_pipeline


#数据集格式转换
def convert_to_lmdb(dataset, path):
    env = lmdb.open(path, map_size=1099511627776)  # 1TB空间
    with env.begin(write=True) as txn:
        for idx, (img, label) in enumerate(dataset):
            key = f"{idx:08}".encode()
            value = pickle.dumps((img.numpy(), label))
            txn.put(key, value)

# 数据集配置
def load_dataset():
    transform = Compose([
    torchvision.transforms.RandomCrop(32, padding=4), # 数据增强
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )])
    
    train_data = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    # 转换为 LMDB 格式
    convert_to_lmdb(train_data, "./data/cifar10_train.lmdb")
    convert_to_lmdb(test_data, "./data/cifar10_test.lmdb")
    print("成功\n")
    return train_data, test_data

# 主程序
def main():

    train_data, test_data = load_dataset()
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True,
                              num_workers=8, pin_memory=True, 
                              persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=True,
                              num_workers=8, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)




if __name__ == "__main__":
    main()