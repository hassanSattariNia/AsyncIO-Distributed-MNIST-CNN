import asyncio
import logging
from queue import Queue
from distributed_code.dataloader import DataLoader , DataManager
import uuid
import time
import torch
from distributed_code.partitions import Partition1 , Partition2 , Partition3 , Partition4 , FinalPartition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        # logging.StreamHandler()  # Remove or comment this line
    ]
)


# Async Client Class
class Client:
    def __init__(self, client_id, partition1, partition, final_partition, input_queue):
        self.client_id = client_id
        self.partition1 = partition1
        self.partition = partition
        self.final_partition = final_partition
        self.input_queue = input_queue
        self.dataManager = self.dataLoaderMnist()
        self.dataStoreLabels = {}
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
                client_id = data["client_id"]
                message_type = data["message_type"]
                                        
                if message_type == "backward":
                    oldGradient = data.get("gradient")
                    newGradient = await self.partition.backward(oldGradient)
                    logging.critical(f"calculate backward ${stage} on client{self.client_id}")
                    logging.critical(f"new gradient:{newGradient} \n old gradient:{oldGradient}")
                    stage -= 1

                    if stage !=0:
                        self.input_queue[stage].put({
                            "output":x ,
                            "gradient":newGradient,
                            "stage":stage,
                            "source":client_id,
                            "batch_id":batch_id,
                            "client_id":client_id,
                            "epoch":epoch,
                            "message_type":"backward"
                        })


                else:
                    if stage == 1 :
                        output = await self.partition1.forward(x)
                        stage +=1 

                        # save for backward handling
                        if batch_id not in self.dataStore:
                            self.dataStore[batch_id] = {"1": {}}
                        self.dataStore[batch_id]["1"]["input_data"] = x
                        self.dataStore[batch_id]["1"]["output_data"] = output    
                        


                        self.input_queue[stage].put({
                            "output":output  ,
                            "stage": stage,
                            "source": self.client_id,
                            "destination": stage,
                            "client_id": self.client_id,
                            "batch_id":batch_id ,
                            "epoch": epoch,
                            "message_type":"forward" })


                    elif stage != 5:
                        output = await self.partition.forward(x)
                        # logging.info(f"Epoch [{current_epoch}]: x value in client[{client_id}] stage:[{stage}] = {x}")

                           # save for backward handling
                        if batch_id not in self.dataStore:
                            self.dataStore[batch_id] = {f"{stage}": {}}  # اگر batch_id وجود ندارد، به همراه "1" آن را ایجاد می‌کنیم
                        elif f"{stage}" not in self.dataStore[batch_id]:
                            self.dataStore[batch_id][f"{stage}"] = {}  
                        self.dataStore[batch_id][f"{stage}"]["input_data"] = x
                        self.dataStore[batch_id][f"{stage}"]["output_data"] = output    
                        
                        
                        if stage == 4:
                            # if stage 4 => send data to client that is owner of data
                            self.input_queue[client_id].put({
                                "output": output,
                                "stage": 5,
                                "source": self.client_id,
                                "destination": client_id,
                                "client_id": client_id,
                                "batch_id":batch_id ,
                                "epoch": epoch,
                                "message_type":"forward"
                            })
                        else:
                            stage += 1
                            self.input_queue[stage].put({
                                "output": output,
                                "stage": stage,
                                "source": self.client_id,
                                "destination": stage,
                                "client_id": client_id,
                                "batch_id":batch_id ,
                                "epoch": epoch,
                                "message_type":"forward"
                            })
                    else:
                        logging.info("Final partition")
                        # idBackwardMessage = str(uuid.uuid4())
                        loss , gradFinal = await self.final_partition.compute_loss_and_grad(x, self.dataStoreLabels[batch_id])
                        self.input_queue[4].put({
                            "output":loss ,
                            "gradient":gradFinal,
                            "stage":4,
                            "source":client_id,
                            "batch_id":batch_id,
                            "client_id":client_id,
                            "epoch":epoch,
                            "message_type":"backward"
                        })


                                 
            if self.dataManager.epoch < 1:
                logging.info(f"reading batch[{self.dataManager.batch_count}] client {self.client_id}")
                random_id = str(uuid.uuid4())
                features, labels = self.dataManager.next_batch()
                x = features.clone().detach().requires_grad_(True)
                self.input_queue[self.client_id].put({
                    "output":x ,
                    "stage": 1,
                    "source": self.client_id,
                    "destination": self.client_id,
                    "client_id": self.client_id,
                    "batch_id":random_id ,
                    "epoch":self.dataManager.epoch,
                    "message_type":"forward"
                })
                self.dataStoreLabels[random_id]= labels

                  

            else:
                logging.info(f'all data read and data Store for client ')
            
            await asyncio.sleep(0)




# Initialize Queues and Partitions
queues = {i: Queue() for i in range(1, 6)}
partition1 = Partition1()
partition2 = Partition2()
partition3 = Partition3()
partition4 = Partition4()
final_partition = FinalPartition()

# Create Clients
client1 = Client(1, partition1, partition1, final_partition, queues)
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
