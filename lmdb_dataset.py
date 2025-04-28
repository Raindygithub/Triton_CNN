# lmdb_dataset.py
import pickle
import torch
import lmdb
import numpy as np
from torch.utils.data import Dataset
import torchvision

# class CIFAR10LMDB(Dataset):
#     def __init__(self, lmdb_path, transform=None):
#         """
#         :param lmdb_path: LMDB数据库路径（如./data/cifar10_train.lmdb）
#         :param transform: 数据增强变换
#         """
#         self.env = lmdb.open(
#             lmdb_path,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False
#         )
#         self.txn = self.env.begin()
#         self.length = self.env.stat()["entries"]  # 获取总样本数
#         self.transform = transform

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         # 生成固定长度的键（8位补零）
#         key = f"{idx:08}".encode('ascii')
#         value = self.txn.get(key)
        
#         # 反序列化存储的二进制数据
#         img, label = np.frombuffer(value, dtype=np.uint8).reshape(3073,)[:3072], int(value[-1])
#         img = img.reshape(3, 32, 32).transpose(1, 2, 0)  # CHW -> HWC

#         # 转换为Tensor并应用变换
#         img = torch.from_numpy(img).float() / 255.0
#         if self.transform:
#             img = self.transform(img)
            
#         return img, label

# 在train.py中添加以下代码

class CIFAR10LMDB(Dataset):
    def __init__(self, lmdb_path, transform=None):
        """
        LMDB数据集加载器
        :param lmdb_path: LMDB数据库路径（如"./data/cifar10_train.lmdb"）
        :param transform: 数据增强变换
        """
        self.env = lmdb.open(lmdb_path, 
                           readonly=True, 
                           lock=False, 
                           readahead=False, 
                           meminit=False)
        self.txn = self.env.begin()
        self.length = self.env.stat()['entries']
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 使用二进制8位数字作为键（00000000~00049999）
        key = f"{idx:08d}".encode('ascii')
        img, label = pickle.loads(self.txn.get(key))
        
        # 转换为PIL图像（原始CIFAR-10数据格式为numpy数组）
        img = torchvision.transforms.ToPILImage()(img)
        
        if self.transform:
            img = self.transform(img)
        return img, label