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
scaler = torch.amp.GradScaler('cuda',enabled=True)  # 初始化梯度缩放器


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
    return train_data, test_data


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


        for inputs, labels in train_loader:

            # 数据加载到GPU的时间
            with Timer() as t:
                inputs, labels = inputs.to(device), labels.to(device)
            data_time += t.duration

            # 前向传播
            with Timer() as t:
                with torch.amp.autocast('cuda',enabled=True):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            forward_time += t.duration

            # 反向传播
            with Timer() as t:
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                #loss.backward()
                scaler.step(self.optimizer)
                scaler.update()
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
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
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

    train_data, test_data = load_dataset()
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True,
                              num_workers=8, pin_memory=True, 
                              persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=True,
                              num_workers=8, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)

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
        accuracy, test_loss = trainer.test_epoch(test_loader)
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