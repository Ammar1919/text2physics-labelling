import shutil
import logging 

# Checking disk space test
def check_space():
    usage = shutil.disk_usage("/")
    free_gb = usage.free / 1e9
    total_gb = usage.total / 1e9
    print(f"Disk: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    if free_gb < 5:
        print("Low disk space! Stopping.")
        return False
    return True

# Logging message filter
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Filter out BlockCache and HTTP Request messages
        message = record.getMessage()
        unwanted_patterns = [
            'BlockCache',
            'HTTP Request:',
            'HTTP/1.1',
            'fetching block'
        ]
        return not any(pattern in message for pattern in unwanted_patterns)

# Logging setup function
def setup_logging():
    """Configure logging with custom formatting and filters."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Suppress HTTP and download messages
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.WARNING)
    logging.getLogger('filelock').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('hf_transfer').setLevel(logging.WARNING)
    logging.getLogger('fsspec').setLevel(logging.WARNING)
    
    # Apply the filter to the root logger
    logging.getLogger().addFilter(MessageFilter())

import numpy as np

# Load the .npy file
data = np.load('datasets/tcf_trajectory.npy')

# Print shape and other info
print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Number of dimensions: {data.ndim}")
print(f"Total elements: {data.size}")
print(f"Memory size: {data.nbytes / (1024**2):.2f} MB")

# If it's small enough, show min/max values
if data.size < 1e8:  # Only if less than 100M elements
    print(f"Min value: {np.min(data):.6f}")
    print(f"Max value: {np.max(data):.6f}")
    print(f"Mean value: {np.mean(data):.6f}")