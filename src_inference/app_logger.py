import logging
import os

log_dir = 'logs'
os.makedirs(log_dir,exist_ok = True)
log_file_path = 'logs/scicap_logs.log'


class Logging:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(self.log_file_path)
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
        self.logger = logging.getLogger(__name__)

# Create a log instance
log = Logging(log_file_path).logger