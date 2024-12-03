import asyncio
import logging
from queue import Queue
from dataloader import DataLoader , DataManager
import uuid
import time
import torch
from partitions import Partition1 , Partition2 , Partition3 , Partition4 , FinalPartition

from logger import create_logger

logger1 = create_logger("client1", clear_log=True)
logger2 = create_logger("client2", clear_log=True)
logger3 = create_logger("client3", clear_log=True)
logger4 = create_logger("client4", clear_log=True)
# write message with client logger
def writeLog(client_id,message):
    if client_id == 1:
        logger1.info(message)
    elif client_id ==2:
        logger2.info(message)
    elif client_id==3:
        logger3.info(message)
    else:
        logger4.info(message)
    

class Client:
    def __init__(self, client_id, partition1, partition, final_partition, input_queue):
        self.client_id = client_id
        self.partition1 = partition1
        self.partition = partition
        self.final_partition = final_partition
        self.input_queue = input_queue
        self.dataManager = self.dataLoaderMnist()
        self.dataStoreLabels = {}
        self.dataStore = []

    def dataLoaderMnist(self):
        data_loader = DataLoader(dataset='mnist', batch_size=64)
        data_loader.set_mode("train")
        data_manager = DataManager(data_loader)
        return data_manager

    async def run(self):
        print(f"client is running... {self.client_id}")
        while True:
            if not self.input_queue[self.client_id].empty():
                data = self.input_queue[self.client_id].get()
                x = data["output"]
                stage = data["stage"]
                batch_id = data["batch_id"]
                client_id = data["client_id"]
                message_type = data["message_type"]
                trace = data["trace"]
                                        
                if message_type == "backward":
                    # completed backward
                    if stage ==0:
                        self.dataStore.remove(batch_id)
                        message = f"client id:{client_id} in client:{self.client_id},trace is:{trace}"
                        writeLog(client_id, message=message)
                    else :
                        oldGradient = data.get("gradient")
                        newGradient = None
                        destination_client = None
                        if stage == 1 :
                            newGradient = await self.partition1.backward(oldGradient,batch_id)
                            destination_client = client_id
                        else:
                            newGradient = await self.partition.backward(oldGradient,batch_id)
                            destination_client = stage - 1
                        trace += f"-backward{stage}_{stage-1}-"
                        stage -= 1

                        
                        self.input_queue[destination_client].put({
                            "output":x ,
                            "gradient":newGradient,
                            "stage":stage,
                            "source":client_id,
                            "batch_id":batch_id,
                            "client_id":client_id,
                            "message_type":"backward",
                            "trace":trace
                        })
                  


                else:
                    if stage == 1 :
                        output = await self.partition1.forward(x,batch_id)
                        stage += 1 
                        trace += "-forward1_2-"
                    
                        # forward to second stage
                        self.input_queue[stage].put({
                            "output":output  ,
                            "stage": stage,
                            "client_id": client_id,
                            "batch_id":batch_id ,
                            "trace":trace ,
                            "message_type":"forward" })


                    elif stage != 5:
                        output = await self.partition.forward(x,batch_id)
                        # logger1.info(f"Epoch [{current_epoch}]: x value in client[{client_id}] stage:[{stage}] = {x}")
                        trace += f"-forward{stage}_{stage+1}-"
                        if stage == 4:
                            # if stage 4 => send data to client that is owner of data
                            self.input_queue[client_id].put({
                                "output": output,
                                "stage": 5,
                                "client_id": client_id,
                                "batch_id":batch_id ,
                                "message_type":"forward",
                                "trace":trace
                            })
                        else:
                            stage += 1
                            self.input_queue[stage].put({
                                "output": output,
                                "stage": stage,
                                "client_id": client_id,
                                "batch_id":batch_id ,
                                "message_type":"forward",
                                "trace":trace
                            })
                    # stage = 5
                    else:
                        # idBackwardMessage = str(uuid.uuid4())
                        loss , gradFinal = await self.final_partition.compute_loss_and_grad(x, self.dataStoreLabels[batch_id])
                        trace += f"-loss:${loss}-backward5_4"
                        self.input_queue[4].put({
                            "output":loss ,
                            "gradient":gradFinal,
                            "stage":4,
                            "source":client_id,
                            "batch_id":batch_id,
                            "client_id":client_id,
                            "message_type":"backward",
                            "trace":trace
                        })
                        print(f"loss value is :{loss}")
                        writeLog(client_id , f"loss :{loss}")


            
            
                                 
            # read new data => (have to change and read data after complete backward) 
            if self.dataManager.epoch < 1:
                if not self.dataStore:
                    random_id = str(uuid.uuid4())
                    features, labels = self.dataManager.next_batch()
                    x = features.clone().detach().requires_grad_(True)
                    self.input_queue[self.client_id].put({
                        "output": x ,
                        "stage": 1,
                        "client_id": self.client_id,
                        "batch_id":random_id ,
                        "message_type":"forward",
                        "trace":f"{self.dataManager.batch_count}"
                    })
                    self.dataStoreLabels[random_id]= labels
                    self.dataStore.append(random_id)
                  

            else:
                writeLog(self.client_id, "all data read ..")
            
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
