# train.py
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
from model_v1 import Net
from torchvision.transforms import Compose
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from dali_pipeline import cifar10_lmdb_pipeline

from torchvision import transforms
import lmdb
import pickle

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start

def get_gpu_utilization():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
    )
    return int(result.decode().strip())


# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 在train.py中添加以下代码
import lmdb
import pickle
from torch.utils.data import Dataset

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

# #数据集配置
# def load_dataset():
#     transform = Compose([
#     torchvision.transforms.RandomCrop(32, padding=4), # 数据增强
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(
#         mean=[0.4914, 0.4822, 0.4465],
#         std=[0.2023, 0.1994, 0.2010]
#     )])
    
#     # 初始化数据集
#     train_dataset = CIFAR10LMDB("./data/cifar10_train.lmdb", transform=train_transform)
#     test_dataset = CIFAR10LMDB("./data/cifar10_test.lmdb", transform=test_transform)
#     return train_data, test_data



# 数据集配置

# 新增：使用 DALI 加载数据
# 定义训练和验证管道
# train_pipe = lmdb_cifar10_pipeline(
#     data_dir="./data/cifar10_train.lmdb",
#     is_training=True,
# )
# train_loader = DALIGenericIterator(
#     train_pipe,
#     output_map=["images", "labels"],
#     reader_name="Reader",
#     auto_reset=True,
# )

# valid_pipe = lmdb_cifar10_pipeline(
#     data_dir="./data/cifar10_test.lmdb",
#     is_training=False,
# )
# valid_pipe.disable_cpu_operations()  # 完全使用GPU处理
# valid_loader = DALIGenericIterator(
#     valid_pipe,
#     output_map=["images", "labels"],
#     reader_name="Reader",
#     auto_reset=True,
# )
# train_pipe = cifar10_pipeline("./data/cifar-10-batches-py", is_training=True)
# train_loader = DALIGenericIterator(
#     train_pipe,
#     output_map=["images", "labels"],
#     reader_name="Reader",
#     auto_reset=True
# )

# valid_pipe = cifar10_pipeline("./data/cifar-10-batches-py", is_training=False)
# valid_pipe.disable_cpu_operations()  # 完全使用GPU处理
# valid_loader = DALIGenericIterator(
#     valid_pipe,
#     output_map=["images", "labels"],
#     reader_name="Reader",
#     auto_reset=True
# )



# 模型训练组件
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, writer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.train_losses = []
        self.test_accuracies = []

    def train_epoch(self, train_loader, first:bool):
        self.model.train()
        epoch_loss = 0.0
        total_samples = 0
        start_time = time.time()

        # 新增：各阶段耗时统计
        data_time = 0.0
        forward_time = 0.0
        backward_time = 0.0


        for data in train_loader:

            # 数据加载到GPU的时间
            with Timer() as t:
                inputs = data["images"].to(device)
                labels = data["labels"].squeeze().to(device)
                #inputs, labels = inputs.to(device), labels.to(device)
            data_time += t.duration

            # 前向传播
            with Timer() as t:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            forward_time += t.duration

            # 反向传播
            with Timer() as t:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            backward_time += t.duration

            epoch_loss += loss.item()
            total_samples += inputs.size(0)

        # 计算吞吐量和时间占比
        epoch_duration = time.time() - start_time
        samples_per_sec = total_samples / epoch_duration
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        gpu_util = get_gpu_utilization()

        if(first):
            print(
                
                f"Duration   单个Epoch总耗时:    {epoch_duration:.2f}s\n"
                f"Samples/s  数据吞吐量:         {samples_per_sec:.1f}\n"
                f"Data Load  数据加载耗时:       {data_time:.2f}s ({data_time/epoch_duration*100:.1f}%)\n"
                f"Forward    前向传播耗时:       {forward_time:.2f}s ({forward_time/epoch_duration*100:.1f}%)\n"
                f"Backward   反向传播耗时:       {backward_time:.2f}s ({backward_time/epoch_duration*100:.1f}%)\n"
                f"GPU Mem    显存占用:           {gpu_mem:.2f}MB\n"
                f"GPU Util   GPU利用率:          {gpu_util}%\n"
                f"{'-'*50}"
            )
        

        torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计
        return epoch_loss / len(train_loader)

    def test_epoch(self, test_loader):
        #self.model.eval()
        total_correct = 0
        total_samples = 0
        epoch_loss = 0.0

        with torch.no_grad():
            for data in test_loader:
                inputs = data["images"].to(device)
                labels = data["labels"].squeeze().to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                epoch_loss += loss.item()

        accuracy = total_correct / total_samples
        return accuracy, epoch_loss / len(test_loader)


# 主程序
def main():

    # train_data, test_data = load_dataset()
    # train_loader = DataLoader(train_data, batch_size=256, shuffle=True,
    #                           num_workers=8, pin_memory=True, 
    #                           persistent_workers=True, prefetch_factor=4)
    # test_loader = DataLoader(test_data, batch_size=256, shuffle=True,
    #                           num_workers=8, pin_memory=True,
    #                           persistent_workers=True, prefetch_factor=4)
    
    # 初始化DALI流水线
    train_pipe = cifar10_lmdb_pipeline(
        "./data/cifar10_train.lmdb", 
        batch_size=256, 
        num_threads=4,
        device_id=0
    )
    test_pipe = cifar10_lmdb_pipeline(
        "./data/cifar10_test.lmdb",
        batch_size=256,
        num_threads=4,
        device_id=0
    )


    # # 关闭测试集的数据增强
    # test_pipe.disable_ops('random_resized_crop')
    # test_pipe.disable_ops('flip')

    # 创建DALI迭代器
    train_loader = DALIGenericIterator(
        train_pipe,
        output_map=["images", "labels"],
        reader_name="Reader",
        auto_reset=True,  # 每个epoch后自动重置
        dynamic_shape=True
    )
    
    valid_loader = DALIGenericIterator(
        test_pipe,
        output_map=["images", "labels"],
        reader_name="Reader",
        auto_reset=True,
        dynamic_shape=True
    )


    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    writer = SummaryWriter("./log/cifar10")

    trainer = Trainer(model, criterion, optimizer, scheduler, writer)

    train_loss = trainer.train_epoch(train_loader,False)
    

    for epoch in range(5):
        # 训练阶段
        print(f"[Epoch Stats][{epoch + 1}/5]\n")
        train_loss = trainer.train_epoch(train_loader,True)
        trainer.train_losses.append(train_loss)
        #print(f"Epoch [{epoch + 1}/5] | Train Loss: {train_loss:.4f}")

        # 测试阶段
        accuracy, test_loss = trainer.test_epoch(valid_loader)
        trainer.test_accuracies.append(accuracy)
        #print(f"Epoch [{epoch + 1}/5] | Test Loss: {test_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")
        print(f"Acuuracy:   {accuracy*100:.2f}%\n")

        # 学习率调整
        trainer.scheduler.step()
        writer.add_scalars("Loss", {"train": train_loss, "test": test_loss}, epoch)
        writer.add_scalar("Accuracy", accuracy, epoch)

    
    # 保存与可视化
    writer.close()
    torch.save(model.state_dict(), "./model/cifar10.pth")


if __name__ == "__main__":
    main()