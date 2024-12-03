import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from logger import create_logger , write_to_file
# Set up logging
logger = create_logger("modelSplitted",True)

# DataStore class to store intermediate activations and gradients
class DataStore:
    def __init__(self):
        self.data = {}

    def save(self, name, value):
        self.data[name] = value

    def get(self, name):
        return self.data.get(name, None)

# Partition Classes with DataStore integration
class Partition1:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.001)
        self.data_store = DataStore()  # DataStore for this partition

    def forward(self, data):
        # Ensure data requires gradient
        self.activations = data.detach().requires_grad_(True)
        output = self.layers(self.activations)
        self.data_store.save('activations', self.activations)  # Save activations for backward pass
        return output

    def backward(self, grad_output):
        self.optimizer.zero_grad()
        activations = self.data_store.get('activations')
        output = self.layers(activations)
        output.backward(grad_output)
        self.optimizer.step()
        grad_input = activations.grad
        return grad_input

class Partition2:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.001)
        self.data_store = DataStore()

    def forward(self, data):
        # Ensure data requires gradient
        self.activations = data.detach().requires_grad_(True)
        output = self.layers(self.activations)
        self.data_store.save('activations', self.activations)
        return output

    def backward(self, grad_output):
        self.optimizer.zero_grad()
        activations = self.data_store.get('activations')
        output = self.layers(activations)
        output.backward(grad_output)
        self.optimizer.step()
        grad_input = activations.grad
        return grad_input

class Partition3:
    def __init__(self):
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU()
        )
        self.optimizer = optim.SGD(self.linear.parameters(), lr=0.01)
        self.data_store = DataStore()

    def forward(self, data):
        # Ensure data requires gradient
        self.activations = data.detach().requires_grad_(True)
        x = self.flatten(self.activations)
        output = self.linear(x)
        self.data_store.save('activations', self.activations)
        return output

    def backward(self, grad_output):
        self.optimizer.zero_grad()
        activations = self.data_store.get('activations')
        x = self.flatten(activations)
        output = self.linear(x)
        output.backward(grad_output)
        self.optimizer.step()
        grad_input = activations.grad
        return grad_input

class Partition4:
    def __init__(self):
        self.layers = nn.Linear(512, 10)
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.001)
        self.data_store = DataStore()

    def forward(self, data):
        # Ensure data requires gradient
        self.activations = data.detach().requires_grad_(True)
        output = self.layers(self.activations)
        self.data_store.save('activations', self.activations)
        return output

    def backward(self, grad_output):
        self.optimizer.zero_grad()
        activations = self.data_store.get('activations')
        output = self.layers(activations)
        output.backward(grad_output)
        self.optimizer.step()
        grad_input = activations.grad
        return grad_input

class FinalPartition:
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss_and_grad(self, predictions, labels):
        # Ensure predictions require gradient
        predictions = predictions.detach().requires_grad_(True)
        loss = self.loss_fn(predictions, labels)
        logger.critical(f"Loss value is: {loss.item()}")
        # Compute gradient of loss w.r.t. predictions
        loss.backward()
        grad_output = predictions.grad
        return loss.item(), grad_output

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalizing MNIST images
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

write_to_file("splitted_loss.text","loss values ",True)

# Training function
def train_model():
    # Initialize partitions
    partition1 = Partition1()
    partition2 = Partition2()
    partition3 = Partition3()
    partition4 = Partition4()
    final_partition = FinalPartition()

    # Training loop
    for epoch in range(1):
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Forward pass through each partition
            out1 = partition1.forward(data)
            out2 = partition2.forward(out1)
            out3 = partition3.forward(out2)
            out4 = partition4.forward(out3)

          
            loss_value, grad4 = final_partition.compute_loss_and_grad(out4, labels)

            # Backward pass through each partition
            grad3 = partition4.backward(grad4)
            grad2 = partition3.backward(grad3)
            grad1 = partition2.backward(grad2)
            _ = partition1.backward(grad1)

            # Logging the loss
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss_value}")
                data_loss = f'{loss_value}'
                write_to_file("splitted_loss.text",f'{batch_idx + 1} {data_loss}',False)
            
    print("Training completed.")

train_model()
