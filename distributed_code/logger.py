import logging
import os

# create logger
def create_logger(filename, clear_log=True):
    print("create logger ")
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{filename}.log")   
    if clear_log and os.path.exists(log_file):
        with open(log_file, 'w'): 
            pass
    logger = logging.getLogger(f"{filename}")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger


def write_to_file(file_name, data, overwrite=False):
    # Open file in 'w' mode if overwrite is True, else 'a' for appending
    mode = 'w' if overwrite else 'a'
    
    with open(file_name, mode) as file:
          file.write(f"{data}\n")
