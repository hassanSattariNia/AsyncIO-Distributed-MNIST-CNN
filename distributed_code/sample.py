import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

# Set up logging
logging.basicConfig(level=logging.CRITICAL)

# Partition Classes
class Partition1:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.01)

    def forward(self, data):
        # Ensure data requires gradient
        self.activations = data.detach().requires_grad_(True)
        output = self.layers(self.activations)
        return output

    def backward(self, grad_output):
        self.optimizer.zero_grad()
        output = self.layers(self.activations)
        output.backward(grad_output)
        self.optimizer.step()
        # Get gradient to pass to previous partition (if needed)
        grad_input = self.activations.grad
        return grad_input

class Partition2:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.01)

    def forward(self, data):
        # Ensure data requires gradient
        self.activations = data.detach().requires_grad_(True)
        output = self.layers(self.activations)
        return output

    def backward(self, grad_output):
        self.optimizer.zero_grad()
        output = self.layers(self.activations)
        output.backward(grad_output)
        self.optimizer.step()
        grad_input = self.activations.grad
        return grad_input

class Partition3:
    def __init__(self):
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU()
        )
        self.optimizer = optim.SGD(self.linear.parameters(), lr=0.01)

    def forward(self, data):
        # Ensure data requires gradient
        self.activations = data.detach().requires_grad_(True)
        x = self.flatten(self.activations)
        output = self.linear(x)
        return output

    def backward(self, grad_output):
        self.optimizer.zero_grad()
        x = self.flatten(self.activations)
        output = self.linear(x)
        output.backward(grad_output)
        self.optimizer.step()
        grad_input = self.activations.grad
        return grad_input

class Partition4:
    def __init__(self):
        self.layers = nn.Linear(512, 10)
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.01)

    def forward(self, data):
        # Ensure data requires gradient
        self.activations = data.detach().requires_grad_(True)
        output = self.layers(self.activations)
        return output

    def backward(self, grad_output):
        self.optimizer.zero_grad()
        output = self.layers(self.activations)
        output.backward(grad_output)
        self.optimizer.step()
        grad_input = self.activations.grad
        return grad_input

class FinalPartition:
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss_and_grad(self, predictions, labels):
        # Ensure predictions require gradient
        predictions = predictions.detach().requires_grad_(True)
        loss = self.loss_fn(predictions, labels)
        logging.critical(f"Loss value is: {loss.item()}")
        # Compute gradient of loss w.r.t. predictions
        loss.backward()
        grad_output = predictions.grad
        return loss.item(), grad_output

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Training function
def train_model():
    # Initialize partitions
    partition1 = Partition1()
    partition2 = Partition2()
    partition3 = Partition3()
    partition4 = Partition4()
    final_partition = FinalPartition()

    # Optionally, assign devices for distributed setup
    # device1 = torch.device('cuda:0')
    # device2 = torch.device('cuda:1')
    # device3 = torch.device('cuda:2')
    # device4 = torch.device('cuda:3')
    # partition1.layers.to(device1)
    # partition2.layers.to(device2)
    # partition3.linear.to(device3)
    # partition4.layers.to(device4)
    # final_partition.loss_fn.to(device4)

    for epoch in range(5):
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Move data to device1 if using distributed setup
            # data = data.to(device1)

            # Forward pass through each partition
            out1 = partition1.forward(data)
            # out1 = out1.to(device2)  # Move to device2 if using distributed setup

            out2 = partition2.forward(out1)
            # out2 = out2.to(device3)

            out3 = partition3.forward(out2)
            # out3 = out3.to(device4)

            out4 = partition4.forward(out3)

            # Move labels to device where loss is computed
            # labels = labels.to(device4)

            # Compute the loss and get gradient w.r.t. predictions
            loss_value, grad4 = final_partition.compute_loss_and_grad(out4, labels)

            # Backward pass through each partition
            grad3 = partition4.backward(grad4)
            # grad3 = grad3.to(device3)

            grad2 = partition3.backward(grad3)
            # grad2 = grad2.to(device2)

            grad1 = partition2.backward(grad2)
            # grad1 = grad1.to(device1)

            _ = partition1.backward(grad1)

            # Logging the loss
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss_value}")

    print("Training completed.")

train_model()
