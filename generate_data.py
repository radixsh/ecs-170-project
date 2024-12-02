from data_handling import get_dataset
from env import CONFIG

# Uses settings in env.py to generate the necessary amount of data in the
# appropriate and saves to the data/ directory.

if __name__ == "__main__":
    get_dataset(CONFIG, mode="TRAIN")
    get_dataset(CONFIG, mode="TEST")
