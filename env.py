# How many (data points, labels) pairs to have for training/testing
TRAINING_SIZE = 1000 #500
TEST_SIZE = 50

# How many data points should be sampled from each distribution
SAMPLE_SIZE = 50

# How many times to generate new data and train model on it
RUNS = 10 #3

# How many times to repeat the training process per generated dataset
EPOCHS = 10 #10

# Define a canonical ordering (from generate_data.py on main branch)
DISTRIBUTION_TYPES = ["exponential", "normal"]

# Old values, from train.py on samples branch:
# TRAINING_SIZE = 10000
# TEST_SIZE = 1000
# SAMPLE_SIZE = 10
# EPOCHS = 10
