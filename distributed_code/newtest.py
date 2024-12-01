import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO)

class Partition1:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.01)
        self.activations = None
    def forward(self, data):
        self.activations = data.clone().detach().required_grad(True)
        return self.layers(self.activations)

class Partition2:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.01)

    def forward(self, data):
        return self.layers(data)

class Partition3:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU()
        )
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.01)

    def forward(self, data):
        return self.layers(data)

class Partition4:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Linear(512, 10)  # Output logits
        )
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.01)

    def forward(self, data):
        return self.layers(data)

class FinalPartition:
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, predictions, labels):
        loss = self.loss_fn(predictions, labels)
        return loss

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

def train_model():
    partition1 = Partition1()
    partition2 = Partition2()
    partition3 = Partition3()
    partition4 = Partition4()
    final_partition = FinalPartition()

    for epoch in range(5):  # اجرای 5 اپوک
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.requires_grad_(), labels

            # فوروارد (پرس و جوی داده‌ها از پارتیشن‌ها)
            out1 = partition1.forward(data)
            out2 = partition2.forward(out1)
            out3 = partition3.forward(out2)
            out4 = partition4.forward(out3)
            loss = final_partition.forward(out4, labels)

            # بکوارد (محاسبه گرادیان‌ها)
            loss.backward()  # استفاده از loss.backward() برای بک‌پروپاگیشن

            # به‌روزرسانی وزن‌ها
            partition1.optimizer.step()
            partition2.optimizer.step()
            partition3.optimizer.step()
            partition4.optimizer.step()

            # صفر کردن گرادیان‌ها برای هر مرحله
            partition1.optimizer.zero_grad()
            partition2.optimizer.zero_grad()
            partition3.optimizer.zero_grad()
            partition4.optimizer.zero_grad()

            # چاپ مقدار loss برای هر بچ
            print(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}")

# شروع آموزش مدل
train_model()
