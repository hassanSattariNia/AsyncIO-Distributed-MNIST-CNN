import asyncio
import logging
from queue import Queue
from distributed_code.dataloader import DataLoader , DataManager
import uuid
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Partition Classes
class Partition1:
    async def process(self, data):
        logging.info("Partition 1")
        return 1

class Partition2:
    async def process(self, x):
        logging.info("Partition 2")
        return x + 2

class Partition3:
    async def process(self, x):
        logging.info("Partition 3")
        return x * 4

class Partition4:
    async def process(self, x):
        logging.info("Partition 4")
        return x - 1

class FinalPartition:
    async def process(self, x, label):
        logging.info(f"Computed Value: {x}, Expected Label: {label}")
        if x == label:
            logging.info("Success!")
        else:
            logging.info("Failed!")

# Labels
labels = {1: 11, 2: 15, 3: 19, 4: 23}

# Async Client Class
class Client:
    def __init__(self, client_id, partition1, partition, final_partition, input_queue):
        self.client_id = client_id
        self.partition1 = partition1
        self.partition = partition
        self.final_partition = final_partition
        self.input_queue = input_queue
        self.dataManager = self.dataLoaderMnist()
        self.dataStore = {}

    def dataLoaderMnist(self):
        data_loader = DataLoader(dataset='mnist', batch_size=64)
        data_loader.set_mode("train")
        data_manager = DataManager(data_loader)
        return data_manager

    async def run(self):
        print(f"start run function in client {self.client_id}")
        while True:
            if not self.input_queue[self.client_id].empty():
                data = self.input_queue[self.client_id].get()
                x = data["output"]
                stage = data["stage"]
                epoch = data["epoch"]
                batch_id = data["batch_id"]

                if stage == 1 :
                    x = await self.partition1.process(x)
                    stage +=1 
                    self.input_queue[stage].put({
                        "output":x  ,
                        "stage": stage,
                        "source": self.client_id,
                        "destination": stage,
                        "client_id": self.client_id,
                        "batch_id":batch_id ,
                        "epoch": epoch })


                elif stage != 5:
                    x = await self.partition.process(x)
                    # logging.info(f"Epoch [{current_epoch}]: x value in client[{client_id}] stage:[{stage}] = {x}")
                    stage += 1
                    self.input_queue[stage].put({
                        "output": x,
                        "stage": stage,
                        "source": self.client_id,
                        "destination": stage,
                        "client_id": self.client_id,
                        "batch_id":batch_id ,
                        "epoch": epoch
                    })
                else:
                    logging.info("Final partition")
                    await self.final_partition.process(x, self.dataStore[batch_id])
            
            if self.dataManager.epoch < 1:
                logging.info(f"reading at client {self.client_id}")
                random_id = str(uuid.uuid4())
                features, labels = self.dataManager.next_batch()
                self.input_queue[self.client_id].put({
                    "output":features ,
                    "stage": 1,
                    "source": self.client_id,
                    "destination": self.client_id,
                    "client_id": self.client_id,
                    "batch_id":random_id ,
                    "epoch":self.dataManager.epoch
                })

                self.dataStore[random_id] = labels            

            else:
                logging.info(f'all data read and data Store for client{self.client_id} is:{self.dataStore}')
            
            await asyncio.sleep(2)
# Initialize Queues and Partitions
queues = {i: Queue() for i in range(1, 6)}
partition1 = Partition1()
partition2 = Partition2()
partition3 = Partition3()
partition4 = Partition4()
final_partition = FinalPartition()

# Create Clients
client1 = Client(1, partition1, None, final_partition, queues)
client2 = Client(2, partition1, partition2, final_partition, queues)
client3 = Client(3, partition1, partition3, final_partition, queues)
client4 = Client(4, partition1, partition4, final_partition, queues)

# Run Clients in Asyncio
async def main():
    tasks = [
        client1.run(),
        client2.run(),
        client3.run(),
        client4.run()
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
