import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as TorchDataLoader

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, train_ratio=0.8):
        """
        Initialize the DataLoader with specific configurations.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_ratio = train_ratio

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # For single-channel images
        ])

        self.dataset_train, self.dataset_test = self.loadDataSet()

        # Create train and test dataset
        self.train_loader = TorchDataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)
        self.test_loader = TorchDataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False)

        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)
        self.mode = "train"

    def set_mode(self, mode):
        """
        Set the mode to 'train' or 'test'.
        """
        if mode not in ["train", "test"]:
            raise ValueError("Mode must be either 'train' or 'test'")
        self.mode = mode
        if self.mode == "train":
            self.train_iter = iter(self.train_loader)
        else:
            self.test_iter = iter(self.test_loader)

    def loadDataSet(self):
        """
        Load MNIST dataset.
        """
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        return trainset, testset

    def getLoader(self):
        """
        Return the appropriate DataLoader based on the mode.
        """
        return self.train_loader if self.mode == "train" else self.test_loader


class DataManager:
    def __init__(self, data_loader):
        """
        Initialize the DataManager with a DataLoader instance.
        """
        self.data_loader = data_loader
        self.epoch = 0  # Track the current epoch
        self.current_iter = iter(self.data_loader.getLoader())  # Initialize iterator
        self.batch_count = 0 
    def next_batch(self):
        """
        Get the next batch of data (features and labels).
        If the iterator is exhausted, increase the epoch and restart the iterator.
        """
        try:
            # Try to get the next batch
            batch = next(self.current_iter)
            self.batch_count += 1
        except StopIteration:
            # If StopIteration is raised, reset the iterator and increment epoch
            self.epoch += 1
            self.batch_count = 1
            print(f"Epoch {self.epoch} completed. Restarting...")
            self.current_iter = iter(self.data_loader.getLoader())
            batch = next(self.current_iter)  # Get the first batch of the new epoch
        
        return batch  # Return features and labels


# Usage example
if __name__ == "__main__":
    # Import and initialize the DataLoader for MNIST
    data_loader = DataLoader(dataset='mnist', batch_size=64)

    # Set mode to 'train'
    data_loader.set_mode("train")

    # Initialize DataManager
    data_manager = DataManager(data_loader)

    # Set the maximum number of epochs
    max_epochs = 3

    # Loop over the dataset for the specified number of epochs
    while data_manager.epoch < max_epochs:
        features, labels = data_manager.next_batch()
        print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
        print(f"Current Epoch: {data_manager.epoch} current batch[{data_manager.batch_count}]")
        
    print("Finished processing dataset for 3 epochs.")

