import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# DataStore class to store required data for forward and backward passes
class DataStore:
    def __init__(self):
        self.data = {}
    
    def save(self, name, value):
        self.data[name] = value
    
    def get(self, name):
        return self.data.get(name, None)

# Partition Class1 (First part of the network)
class Partition1(nn.Module):
    def __init__(self):
        super(Partition1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.data_store = DataStore()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        self.data_store.save('x', x)  # Save data for backward
        return x

    def backward(self, grad_output):
        # For backward, get saved data and compute gradient manually
        x = self.data_store.get('x')
        grad_input = grad_output * (x > 0).float()  # Backprop through ReLU
        return grad_input

# Partition Class2 (Second part of the network)
class Partition2(nn.Module):
    def __init__(self):
        super(Partition2, self).__init__()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.data_store = DataStore()

    def forward(self, x):
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        self.data_store.save('x', x)
        return x

    def backward(self, grad_output):
        x = self.data_store.get('x')
        grad_input = grad_output * (x > 0).float()  # Backprop through ReLU
        return grad_input

# Partition Class3 (Fully connected layer 1)
class Partition3(nn.Module):
    def __init__(self):
        super(Partition3, self).__init__()
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.relu = nn.ReLU()
        self.data_store = DataStore()

    def forward(self, x):
        x = x.view(-1, 64 * 4 * 4)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        self.data_store.save('x', x)
        return x

    def backward(self, grad_output):
        x = self.data_store.get('x')
        grad_input = grad_output * (x > 0).float()  # Backprop through ReLU
        return grad_input

# Partition Class4 (Output layer)
class Partition4(nn.Module):
    def __init__(self):
        super(Partition4, self).__init__()
        self.fc2 = nn.Linear(512, 10)  # Output layer (logits for 10 classes)
        self.data_store = DataStore()

    def forward(self, x):
        x = self.fc2(x)
        self.data_store.save('x', x)
        return x

    def backward(self, grad_output):
        # No activation in the output layer, just return the gradients as is
        return grad_output

# FinalClass for loss computation
class FinalClass(nn.Module):
    def __init__(self):
        super(FinalClass, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def calculate_loss(self, output, target):
        return self.criterion(output, target)

# Simulate forward and backward flow
def forward_and_backward(model, data, target):
    # Forward pass through each partition
    out1 = model['partition1'].forward(data)
    out2 = model['partition2'].forward(out1)
    out3 = model['partition3'].forward(out2)
    out4 = model['partition4'].forward(out3)

    # Loss calculation
    final_class = FinalClass()
    loss = final_class.calculate_loss(out4, target)

    # Backward pass through each partition
    grad_output = torch.autograd.grad(loss, out4, create_graph=True)[0]  # Gradients for output layer
    grad3 = model['partition4'].backward(grad_output)  # Backpropagate through output layer
    grad3 = grad3.view(-1, 512)  # Reshape the gradient to match input size of Partition3
    grad2 = model['partition3'].backward(grad3)  # Backpropagate through Partition3
    grad2 = grad2.view(-1, 64, 4, 4)  # Reshape the gradient to match input size of Partition2
    grad1 = model['partition2'].backward(grad2)  # Backpropagate through Partition2
    grad1 = grad1.view(-1, 1, 28, 28)  # Reshape the gradient to match input size of Partition1
    grad0 = model['partition1'].backward(grad1)  # Backpropagate through Partition1

    return loss

# Example usage with MNIST dataset
def main():
    # Define data transform (normalize images)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize MNIST images
    ])
    
    # Load MNIST training data
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Simulate the partitions and network
    model = {
        'partition1': Partition1(),
        'partition2': Partition2(),
        'partition3': Partition3(),
        'partition4': Partition4()
    }

    # Get a batch of real MNIST data
    data, target = next(iter(train_loader))

    # Simulate forward and backward pass
    loss = forward_and_backward(model, data, target)
    print(f"Loss: {loss.item()}")

if __name__ == "__main__":
    main()
