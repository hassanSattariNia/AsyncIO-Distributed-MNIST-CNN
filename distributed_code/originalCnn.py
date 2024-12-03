import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from logger import write_to_file
# original model
def mnist_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5),  
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=5),  
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 512),   
        nn.ReLU(),
        nn.Linear(512, 10)   
    )

 
batch_size = 32
epochs = 1
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# normalize data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))   
])


train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                               transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                              transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False)


model = mnist_cnn().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


write_to_file("original_loss.text","loss values ",True)
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() 
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
          print(f'Epoch [{epoch}] Batch [{batch_idx+1}/{len(train_loader)}] '
                f'Loss: {loss.item():.4f}')
          data_loss = f'{batch_idx + 1} {loss.item():.4f}'
          write_to_file("original_loss.text",data_loss,False)
          
    avg_loss = total_loss / len(train_loader)
    print(f'==> Epoch [{epoch}] Average training loss: {avg_loss:.4f}')


def test(model, device, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f'==> Test set: Average loss: {avg_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)')


for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)
    test(model, device, test_loader, criterion)
