import logging
import os

# create logger
def create_logger(client_id, clear_log=True):
    print("create logger ")
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"client{client_id}.log")
    
   
    if clear_log and os.path.exists(log_file):
        with open(log_file, 'w'): 
            pass
    
   
    logger = logging.getLogger(f"client{client_id}")
    logger.setLevel(logging.DEBUG)
    
   
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger

logger1 = create_logger(1, clear_log=True)
logger2 = create_logger(2, clear_log=True)
logger3 = create_logger(3, clear_log=True)
logger4 = create_logger(4, clear_log=True)
logger1.info("logger for client 1")
logger2.info("logger for client 2")
logger3.info("logger for client 3")
logger4.info("logger for client 4")