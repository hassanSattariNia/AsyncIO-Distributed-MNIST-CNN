import asyncio
import logging
from queue import Queue
from dataloader import DataLoader , DataManager
import uuid
import time
import torch.nn as nn 
import torch.optim as optim


class DataStore:
    def __init__(self):
        self.data = {}
    
    def save (self,name,value):
        self.data[name] = value
    
    def get (self , name):
        return self.data.get(name , None)
    
    def delete(self , name):
        del self.data[name]


class Partition1:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.01)
        self.dataStore = DataStore()

    async def forward(self, data , batch_id):
        self.activations = data.detach().requires_grad_(True)
        output = self.layers(self.activations)
        self.dataStore.save(batch_id , self.activations)  # Save activations for backward pass
        return output

    async def backward(self, grad_output,batch_id):
        self.optimizer.zero_grad()
        activations = self.dataStore.get(batch_id)
        self.dataStore.delete(batch_id)
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
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.01)
        self.dataStore = DataStore()
    
    async def forward(self, data,batch_id):
        # Ensure data requires gradient        
        self.activations = data.detach().requires_grad_(True)
        output = self.layers(self.activations)
        self.dataStore.save(batch_id, self.activations)
        return output

    async def backward(self, grad_output,batch_id):
        self.optimizer.zero_grad()
        activations = self.dataStore.get(batch_id)
        self.dataStore.delete(batch_id)
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
        self.dataStore = DataStore()

    async def forward(self, data ,batch_id):
        # Ensure data requires gradient

        self.activations = data.detach().requires_grad_(True)
        x = self.flatten(self.activations)
        output = self.linear(x)
        self.dataStore.save(batch_id, self.activations)
        return output

    async def backward(self, grad_output ,batch_id):

        self.optimizer.zero_grad()
        activations = self.dataStore.get(batch_id)
        self.dataStore.delete(batch_id)
        x = self.flatten(activations)
        output = self.linear(x)
        output.backward(grad_output)
        self.optimizer.step()
        grad_input = activations.grad
        return grad_input
    
class Partition4:
    def __init__(self):
        self.layers = nn.Linear(512, 10)
        self.optimizer = optim.SGD(self.layers.parameters(), lr=0.01)
        self.dataStore = DataStore()

    async def forward(self, data, batch_id):
        # Ensure data requires gradient
        self.activations = data.detach().requires_grad_(True)
        output = self.layers(self.activations)
        self.dataStore.save(batch_id, self.activations)
        return output

    async def backward(self, grad_output ,batch_id):
        self.optimizer.zero_grad()
        activations = self.dataStore.get(batch_id)
        self.dataStore.delete(batch_id)
        output = self.layers(activations)
        output.backward(grad_output)
        self.optimizer.step()
        grad_input = activations.grad
        return grad_input


class FinalPartition:
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss()

    async def compute_loss_and_grad(self, predictions, labels):
        # Ensure predictions require gradient
        predictions = predictions.detach().requires_grad_(True)
        loss = self.loss_fn(predictions, labels)
        logging.critical(f"Loss value is: {loss.item()}")
        # Compute gradient of loss w.r.t. predictions
        loss.backward()
        grad_output = predictions.grad
        return loss.item(), grad_output
   