import queue
import time
import threading
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format of log messages
    handlers=[
        logging.FileHandler("app.log"),  # Logs written to a file
        logging.StreamHandler()          # Logs printed to console
    ]
)


class Partition1:
    def process(self, client_id):
        logging.info("partition 1")
        return client_id 

class Partition2:
    def process(self, x):
        logging.info("partition 2")
        return x + 2

class Partition3:
    def process(self, x):
        logging.info("partition 3")
        return x * 4

class Partition4:
    def process(self, x):
        logging.info("partition 4")
        return x - 1

class FinalPartition:
    def process(self, x, label):
        logging.info(f"Computed Value: {x}, Expected Label: {label}")
        if x == label:
            logging.info("Success!")
        else:
            logging.info("Failed!")



# Label‌ها
labels = {1: 11, 2: 15, 3: 19, 4: 23}

class Client(threading.Thread):
    def __init__(self, client_id, partition1, partition, final_partition  , input_queue ):
        threading.Thread.__init__(self)
        self.client_id = client_id
        self.partition1 = partition1
        self.partition = partition
        self.final_partition = final_partition
        self.input_queue = input_queue
        self.currentEpoch = 1
    def run(self):
      while True:
          # check input list
        if not self.input_queue[self.client_id].empty():
          data = self.input_queue[self.client_id].get()
          x = data["output"]
          stage = data["stage"]
          source = data["source"]
          destination = data["destination"]
          client_id = data["client_id"]
          currentEpoch = data["currentEpoch"]
          if self.client_id==1 and (stage !=1 and stage !=4):
            logging.info(f'fuck................{stage}')
            break

          # data come from stage4
          # if stage == 4 and client_id==self.client_id:
          #   logging.info(f"final partition for client ${self.client_id}")
          #   self.final_partition.process(x)
          if stage == 1:
            x = self.partition1.process(x)
            stage += 1
            self.input_queue[stage].put({
                "output":x ,
                "stage":stage,
                "source":self.client_id,
                "destination":stage,
                "client_id":client_id,
                "currentEpoch":currentEpoch
            })
          elif stage!=5:
            x = self.partition.process(x)
            logging.info(f"epoch [{currentEpoch}]:x value in client[{client_id}]stage:[{stage}]={x} ")
            stage +=1
            self.input_queue[stage].put({
                "output":x ,
                "stage":stage,
                "source":self.client_id,
                "destination":stage,
                "client_id":client_id ,
                "currentEpoch":currentEpoch
            })
          else:
            logging.info("final partition")
            self.final_partition.process(x , labels[self.client_id])
        else:
          self.input_queue[self.client_id].put({
                "output":self.client_id ,
                "stage":1,
                "source":self.client_id,
                "destination":self.client_id,
                "client_id":self.client_id ,
                "currentEpoch":self.currentEpoch})
          self.currentEpoch +=1 
        time.sleep(5)
      # read from input





def print_queues(queues):
    for key, q in queues.items():
        items = list(q.queue)  # Accessing internal queue list (not thread-safe)
        logging.info(f"Queue {key}: {items}")



import queue

queues = {
    1: queue.Queue(),
    2: queue.Queue(),
    3: queue.Queue(),
    4: queue.Queue(),
    5: queue.Queue()
}
partition1 = Partition1()
partition2 = Partition2()
partition3 = Partition3()
partition4 = Partition4()
final_partition = FinalPartition()


client1 = Client(1, partition1, None, final_partition, queues)
client2 = Client(2, partition1, partition2, final_partition, queues)
client3 = Client(3, partition1, partition3, final_partition, queues)
client4 = Client(4, partition1, partition4, final_partition, queues)

# runinng client parallel
clients = [client1, client2, client3, client4]
for client in clients:
    client.start()

for client in clients:
    client.join()
