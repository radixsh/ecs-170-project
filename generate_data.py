import os
from custom_functions import make_dataset
from env import CONFIG

# If running this file directly as a script, then generate some training
# examples and save them to a file for later use
if __name__ == "__main__":
    data_directory = 'data'
    os.makedirs(data_directory, exist_ok=True)
    make_dataset(CONFIG,mode='TRAIN')
    make_dataset(CONFIG,mode='TEST')
