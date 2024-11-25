import asyncio
import logging
from queue import Queue
from distributed_code.dataloader import DataLoader , DataManager
import uuid
import time
import torch.nn as nn 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)


class Partition1:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    async def process(self, data):
        logging.info("Partition 1 processing")
        return self.layers(data)


class Partition2:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    async def process(self, data):
        logging.info("Partition 2 processing")
        return self.layers(data)

class Partition3:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU()
        )

    async def process(self, data):
        logging.info("Partition 3 processing")
        return self.layers(data)


class Partition4:
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Linear(512, 10)  # Output logits
        )

    async def process(self, data):
        logging.info("Partition 4 processing")
        return self.layers(data)


class FinalPartition:
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss()

    async def process(self, predictions, labels):
        logging.info("Final partition processing")
        loss = self.loss_fn(predictions, labels)
        logging.info(f"Loss: {loss.item()}")
        return loss
