import torch
import torch.nn as nn
from torch.utils.data import Dataset
from env import CONFIG, NUM_DIMENSIONS
import re
import math
import scipy.stats as sps
import numpy as np
import mpmath
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Formats the filename for a model weight file
# Format: "weights_train_$TRAIN_SIZE$_sample_$SAMPLE_SIZE$_dims_$NUM_DIMS$"
# Example: "weights_train_1000_sample_30_dims_2"
def make_weights_filename(train_size, sample_size, num_dims, batch_size,learning_rate):
    return (f"models/"
            f"weights_train_{train_size}_"
            f"sample_{sample_size}_"
            f"dims_{num_dims}_"
            f"batch_{batch_size}_"
            f"lrate_{learning_rate}"
            f".pth")

# Checks the filename for a model weight file
# returns a dict of the train size, sample size, and num_dims
def parse_weights_filename(filename):
    # Define the regex pattern
    pattern = r"(models/)?weights_train_(\d+)_sample_(\d+)_dims_(\d+)_batch_(\d+)_lrate_(\d.\d+).pth"

    # Match the pattern with the filename
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match the expected "
                         f"format. (Try removing `models` and "
                         f"trying again.)")

    # Extract the values and convert to appropriate types
    _,train_size, sample_size, num_dims, batch_size, learning_rate = match.groups()
    return {
        "TRAIN_SIZE": int(train_size),
        "SAMPLE_SIZE": int(sample_size),
        "NUM_DIMS": int(num_dims),
        "BATCH_SIZE": int(batch_size),
        "LEARNING_RATE": float(learning_rate)
    }

# Formats the filename for a dataset
# Format: "dataset_$TYPE$_len_$TRAIN/TEST_SIZE$_sample_$SAMPLE_SIZE$_dims_$NUM_DIMS$"
# Example: "dataset_train_len_1000_sample_30_dims_2"
def make_data_filename(mode, length, sample_size, num_dims):
    return (f"data/"
            f"dataset_{mode}_"
            f"len_{length}_"
            f"sample_{sample_size}_"
            f"dims_{num_dims}"
            f".pth")

# Checks the filename for a dataset
# returns a dict of the type, length, sample size, and num_dims
def parse_data_filename(filename):
    # Define the regex pattern
    pattern = r"(data/)?dataset_(TRAIN|TEST)_len_(\d+)_sample_(\d+)_dims_(\d+).pth"

    # Match the pattern with the filename
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match the expected "
                         f"format. (Try removing `data` and "
                         f"trying again.)")

    # Extract the values and convert to appropriate types
    _, mode, length, sample_size, num_dims = match.groups()
    return {
        "MODE": mode,
        "LENGTH": int(length),
        "SAMPLE_SIZE": int(sample_size),
        "NUM_DIMS": int(num_dims)
    }

# Generates the indices for use in our vector format:
# dists=True gets the onehot portion(s) of the vector
# mean=True gets the mean(s), stddev=True gets the stddev(s)
# dims can either be an int or list of ints,
#   indicates which dimension (distribution-wise) the indices are for
#   gets every dimension available by default
# mode indicates whether different sets of indices
#   should be in one array, or in a multidimensional array
# mode='join' -> 1d array (default)
# mode='split' -> 2d array

# Example:
# x = [1,0,0,0,0,0,0,0,0,3.1,0.6,0,1,0,0,0,0,0,0,0,4.5,1.2]
# x[get_indices(dists=True,dims=1)] -> [1,0,0,0,0,0,0,0,0]
# x[get_indices(mean=True,stddev=True,dims=2)] -> [4.5,1.2]
# x[get_indices(mean=True,dims=[1,2])] -> [3.1,4.5]
# x[get_indices(dists=True,dims=[1,2],mode='split')]
#   -> [[1,0,0,0,0,0,0,0,0],
#       [0,1,0,0,0,0,0,0,0]]

def get_indices(dists=False, mean=False, stddev=False,
                dims=range(1,NUM_DIMENSIONS+1)):
    num_dists = len(DISTRIBUTIONS)
    if isinstance(dims,int):
        dims = [dims]
    out = []
    for dim in dims:
        if dists:
            onehot_range = range((dim - 1) * (num_dists + 2),
                                 (dim - 1) * (num_dists + 2) + num_dists)
            out += onehot_range
        if mean:
            mean_idx = dim * (num_dists + 2) - 2
            out.append(mean_idx)
        if stddev:
            stddev_idx = dim * (num_dists + 2) - 1
            out.append(stddev_idx)
    return out

class Multitask(nn.Module):
    def __init__(self, temp=1.0):
        super(Multitask, self).__init__()
        self.temp = temp

    def forward(self, x):
        out = x.clone()
        for n in range(1, NUM_DIMENSIONS+1):
            idxs = get_indices(dists=True, dims=n)
            # How does torch possibly support this
            # softmaxes the slice representing the dimension vector
            out[0][idxs] = torch.nn.functional.softmax(self.temp * out[0][idxs], dim=0)
        return out

class CustomLoss(nn.Module):
    def __init__(self, use_mean=True, use_stddev=True, use_dists=True, alpha=1):
        super(CustomLoss, self).__init__()

        self.use_mean = use_mean
        self.use_stddev = use_stddev
        self.use_dists = use_dists
        # Alpha controls how much the network gets punished
        # on the classification task relative to the regression task
        self.alpha = alpha

        # Number of distribution functions
        self.num_dists = len(DISTRIBUTIONS)

    def forward(self, pred, y):
        # Absolute difference of vectors normalized by number of dimensions
        diff = torch.abs(pred - y) / NUM_DIMENSIONS

        # Calculate MAE on means per dimension
        mean_idxs = get_indices(mean=True)
        mean_loss = torch.sum(diff[mean_idxs])

        # Calculate MAE on stddevs per dimension
        stddev_idxs = get_indices(stddev=True)
        stddev_loss = torch.sum(diff[stddev_idxs])

        # Approximate total variation distance between distributions
        # Need to look up weights in dist_var_matrix
        dist_idxs = get_indices(dists=True)
        #weights = self.get_weights(y[0][dist_idxs])
        #weights = 1 - y[dist_idxs]
        
        #classification_loss_normal = torch.dot(diff[dist_idxs], weights) / 2
            # Normalize from the abs diff call above
            # Need weird constant to keep the denormalization in check and cap it
            # 10/11 = 0.909090...
            # -> Currently softcapped at 10, doesn't really matter
        classification_loss_normal = 0.9091 * torch.sum(diff[dist_idxs]) / 2


        # Standard denormalization: [0,1] -> [0,oo), we can now compare
        # We can now compare classification loss to the others
        classification_loss = self.alpha * ((1 / (1 - classification_loss_normal)) - 1)


        # Make various losses trivial if not in use
        if not self.use_mean:
            mean_loss = 0
        if not self.use_stddev:
            stddev_loss = 0
        if not self.use_dists:
            classification_loss = 0

        params_in_use = int(self.use_mean + self.use_stddev + self.use_dists)

        # Sum the losses, normalize by number of params being used
        return (mean_loss + stddev_loss + classification_loss) / params_in_use

### SEED THE RNG
rng = np.random.default_rng()

## CLASS DEFINITIONS

# Parent class
class Distribution:
    def __init__(self, mean, stddev, support, name):
        self.name = name
        self.support = support
        self.mean = mean if not isinstance(mean, str) \
                else self.generate_mean()
        self.stddev = stddev if not isinstance(stddev, str) \
                else self.generate_stddev()
        self.onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def __str__(self):
        string = f"{self.name} (support: {self.support}), "
        string += f"mean={self.mean:.3f}, "
        string += f"stddev={self.stddev:.3f}, "
        string += f"label={self.onehot}"
        return string

    def get_label(self):
        label = self.onehot + [self.mean, self.stddev]
        return label

    # Returns a random positive real according to the lognormal distribution
    # with mean and stddev 1. Useful for generating stddevs and positive means.
    def random_pos(self):
        return rng.lognormal(mean=-math.log(2) / 2, sigma=math.sqrt(math.log(2)))

    def generate_mean(self):
        # Support is all of R
        if self.support == 'R':
            return rng.normal(0, 1)
        # Support is positive
        elif self.support == 'R+':
            return self.random_pos()
        # Otherwise, it's the beta distribution
        # random val in (0, 1)
        elif self.support == 'I':
            sign = rng.choice([-1, 1])
            random_in_open_interval = rng.uniform() * sign
            random_in_open_interval = (random_in_open_interval + 1) / 2
            return random_in_open_interval

    def generate_stddev(self):
        # Special behavior for some dists
        # Default case
        if self.name not in ['Beta', 'Rayleigh']:
            return self.random_pos()
        elif self.name == 'Beta':

            # Make a random value in (0,1)
            sign = rng.choice([-1, 1])
            random_in_open_interval = rng.uniform() * sign
            random_in_open_interval = (random_in_open_interval + 1) / 2

            # We want a random value in (0, mean - mean^2)
            upper_bound = math.sqrt(self.mean * (1 - self.mean))
            return random_in_open_interval * upper_bound

        # Rayleigh
        else:
            weird_constant = math.sqrt((4 / math.pi)  - 1)
            return self.mean * weird_constant

# Child classes
class Beta(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='I', name='Beta')
        self.onehot = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        nonsense_constant = (self.mean * (1 - self.mean) / (self.stddev ** 2)) - 1
        #nonsense_constant = ((self.mean * (1 - self.mean)) - (self.stddev ** 2)) / (self.stddev ** 2)
        self.alpha = self.mean * nonsense_constant
        self.beta = (1 - self.mean) * nonsense_constant

    def rng(self, sample_size):
        return rng.beta(self.alpha, self.beta, sample_size)

    def pdf(self, x):
        return sps.beta.pdf(x, self.alpha, self.beta)

class Gamma(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R+', name='Gamma')
        self.onehot = [0, 1, 0, 0, 0, 0, 0, 0, 0]
        self.shape = (self.mean / self.stddev) ** 2
        self.scale = (self.stddev ** 2) / self.mean

    def rng(self, sample_size):
        return rng.gamma(self.shape, self.scale, sample_size)

    def pdf(self, x):
        return sps.gamma.pdf(x, self.shape, scale=self.scale)

class Gumbel(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R', name='Gumbel')
        self.onehot = [0, 0, 1, 0, 0, 0, 0, 0, 0]
        self.scale = self.stddev * math.sqrt(6) / math.pi
        self.loc = self.mean - self.scale * float(mpmath.euler)

    def rng(self, sample_size):
        return rng.gumbel(self.loc, self.scale, sample_size)

    def pdf(self, x):
        return sps.gumbel_r.pdf(x, loc=self.loc, scale=self.scale)

class Laplace(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R', name='Laplace')
        self.onehot = [0, 0, 0, 1, 0, 0, 0, 0, 0]
        self.scale = self.stddev / math.sqrt(2)

    def rng(self, sample_size):
        return rng.laplace(self.mean, self.scale, sample_size)

    def pdf(self, x):
        return sps.laplace.pdf(x, loc=self.mean, scale=self.scale)

class Logistic(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R', name='Logistic')
        self.onehot = [0, 0, 0, 0, 1, 0, 0, 0, 0]
        self.scale = self.stddev * math.sqrt(3) / math.pi

    def rng(self, sample_size):
        return rng.logistic(self.mean, self.scale, sample_size)

    def pdf(self, x):
        return sps.logistic.pdf(x, loc=self.mean, scale=self.scale)

class Lognormal(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R+', name='Lognormal')
        self.onehot = [0, 0, 0, 0, 0, 1, 0, 0, 0]
        self.shape = math.sqrt(math.log(1 + (self.stddev / self.mean) ** 2))
        self.loc = math.log((self.mean ** 2) / math.sqrt((self.mean ** 2) + (self.stddev ** 2)))

    def rng(self, sample_size):
        return rng.lognormal(self.loc, self.shape, sample_size)

    def pdf(self, x):
        return sps.lognorm.pdf(x, self.shape, loc=self.loc)

class Normal(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R', name='Normal')
        self.onehot = [0, 0, 0, 0, 0, 0, 1, 0, 0]

    def rng(self, sample_size):
        return rng.normal(self.mean, self.stddev, sample_size)

    def pdf(self, x):
        return sps.norm.pdf(x, loc=self.mean, scale=self.stddev)

class Rayleigh(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R+', name='Rayleigh')
        self.onehot = [0, 0, 0, 0, 0, 0, 0, 1, 0]
        self.scale = self.mean * math.sqrt(2 / math.pi)
        # self.stddev = self.scale * math.sqrt((4 / math.pi)  - 1)

    def rng(self, sample_size):
        return rng.rayleigh(self.scale, sample_size)

    def pdf(self, x):
        return sps.rayleigh.pdf(x, scale=self.scale)

class Wald(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R+', name='Wald')
        self.onehot = [0, 0, 0, 0, 0, 0, 0, 0, 1]
        self.lam = (self.mean ** 3) / (self.stddev ** 2)
        self.mu = self.mean / self.lam

    def rng(self, sample_size):
        return rng.wald(self.mean, self.lam, sample_size)

    def pdf(self, x):
        return sps.invgauss.pdf(x, self.mu, scale=self.lam)

# Prob move inside of class
DISTRIBUTIONS = {
    "beta": Beta,
    "gamma": Gamma,
    "gumbel": Gumbel,
    "laplace": Laplace,
    "logistic": Logistic,
    "lognormal": Lognormal,
    "normal": Normal,
    "rayleigh": Rayleigh,
    "wald": Wald,
}

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label
