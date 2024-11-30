import asyncio
import logging
from queue import Queue
from distributed_code.dataloader import DataLoader , DataManager
import uuid
import time
import torch.nn as nn 
import torch.optim as optim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        # logging.StreamHandler()
    ]
)


class Partition1:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.01)

    async def forward(self, data):
        # Ensure data requires gradient
        self.activations = data.detach().requires_grad_(True)
        output = self.layers(self.activations)
        return output

    async def backward(self, grad_output):
        logging.error(f"backward on partition 1")
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

    async def forward(self, data):
        # Ensure data requires gradient
        self.activations = data.detach().requires_grad_(True)
        output = self.layers(self.activations)
        return output

    async def backward(self, grad_output):
        logging.error(f"backward on partition 2")
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

    async def forward(self, data):
        # Ensure data requires gradient
        self.activations = data.detach().requires_grad_(True)
        x = self.flatten(self.activations)
        output = self.linear(x)
        return output

    async def backward(self, grad_output):
        logging.error(f"backward on partition 3")
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

    async def forward(self, data):
        # Ensure data requires gradient
        self.activations = data.detach().requires_grad_(True)
        output = self.layers(self.activations)
        return output

    async def backward(self, grad_output):
        logging.error(f"backward on partition 4")
        self.optimizer.zero_grad()
        output = self.layers(self.activations)
        output.backward(grad_output)
        self.optimizer.step()
        grad_input = self.activations.grad
        return grad_input


class FinalPartition:
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss()

    async def compute_loss_and_grad(self, predictions, labels):
        # Ensure predictions require gradient
       
        predictions = predictions.detach().requires_grad_(True)
        loss = self.loss_fn(predictions, labels)
        print(f"loss function:{loss.item()}")
        # Compute gradient of loss w.r.t. predictions
        loss.backward()
        grad_output = predictions.grad
        return loss.item(), grad_output
   