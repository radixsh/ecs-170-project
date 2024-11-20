import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
import math
import mpmath
import scipy.stats as sps

from env import CONFIG, NUM_DIMENSIONS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

### SEED THE RNG
rng = np.random.default_rng()

### DISTRIBUTION CLASSES

## CLASS HEADERS
# # Parent Class
# class Distribution(): pass
#
# # Child Classes
# class Beta(Distribution): pass
# class Gamma(Distribution): pass
# class Gumbel(Distribution): pass
# class Laplace(Distribution): pass
# class Logistic(Distribution): pass
# class Lognormal(Distribution): pass
# class Normal(Distribution): pass
# class Rayleigh(Distribution): pass
# class Wald(Distribution): pass

## CLASS DEFINITIONS

# Parent class
class Distribution:
    def __init__(self,mean,stddev,support,name):
        self.name = name
        self.support = support
        self.mean = mean if isinstance(mean, float) else self.generate_mean()
        self.stddev = stddev if isinstance(stddev, float) else self.generate_stddev()
        self.onehot = [0,0,0,0,0,0,0,0,0]

    def __str__(self):
        string = f"self.name: {self.name}, "
        string += f"self.support: {self.support}, "
        string += f"self.function: {self.onehot}, "
        string += f"self.mean: {self.mean}, "
        string += f"self.stddev: {self.stddev}"
        return string

    def get_label(self):
        label = self.onehot + [self.mean, self.stddev]
        return label

    # Returns a random positive real according to the lognormal distribution
    # with mean and stddev 1. Useful for generating stddevs and positive means.
    def random_pos(self):
        return rng.lognormal(mean = -math.log(2) / math.sqrt(2), sigma=math.sqrt(math.log(2)))

    def generate_mean(self):
        # Support is all of R
        if self.support == 'R':
            return rng.normal(0, 1)
        # Support is positive
        elif self.support == 'R+':
            return self.random_pos()
        # Otherwise, it's the beta distribution
        # random val in (0,1)
        elif self.support == 'I':
            sign = rng.choice([-1, 1])
            open_interval = rng.uniform() * sign
            return (open_interval + 1) / 2

    def generate_stddev(self):
        # Special behavior for some dists
        # Default case
        if self.name not in ['Beta','Rayleigh']:
            return self.random_pos()
        # Beta's "mean" function is strange.
        elif self.name == 'Beta':
            open_interval = self.generate_mean()
            upper_bound = (self.mean - (self.mean ** 2))
            return open_interval * upper_bound
        # Rayleigh
        else:
            weird_constant = math.sqrt((4 / math.pi)  - 1)
            return self.mean * weird_constant

# Child classes
class Beta(Distribution):
    def __init__(self,mean="not set",stddev="not set"):
        super().__init__(mean, stddev, support='I',name='Beta')
        self.onehot = [1,0,0,0,0,0,0,0,0]
        self.alpha = math.sqrt((((self.mean ** 2) - (self.mean ** 3)) / self.stddev) - self.mean)
        self.beta = (self.alpha / self.mean) - self.alpha

    def rng(self,sample_size):
        return rng.beta(self.alpha,self.beta,sample_size)

    def pdf(self,x):
        return sps.beta(x,self.alpha,self.beta)

class Gamma(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R+',name='Gamma')
        self.onehot = [0,1,0,0,0,0,0,0,0]
        self.shape = (self.mean / self.stddev) ** 2
        self.scale = (self.stddev ** 2) / self.mean

    def rng(self,sample_size):
        return rng.gamma(self.shape,self.scale,sample_size)

    def pdf(self,x):
        return sps.gamma(x,self.shape,scale=self.scale)

class Gumbel(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R',name='Gumbel')
        self.onehot = [0,0,1,0,0,0,0,0,0]
        self.scale = self.stddev * math.sqrt(6) / math.pi
        self.loc = self.mean - self.scale * float(mpmath.euler)

    def rng(self,sample_size):
        return rng.gumbel(self.loc,self.scale,sample_size)

    def pdf(self,x):
        return sps.gumbel_r(x,loc=self.loc,scale=self.scale)

class Laplace(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R',name='Laplace')
        self.onehot = [0,0,0,1,0,0,0,0,0]
        self.scale = self.stddev * math.sqrt(2)

    def rng(self,sample_size):
        return rng.laplace(self.mean,self.scale,sample_size)

    def pdf(self,x):
        return sps.laplace(x,loc=self.mean,scale=self.scale)

class Logistic(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R',name='Logistic')
        self.onehot = [0,0,0,0,1,0,0,0,0]
        self.scale = self.stddev * math.sqrt(3) / math.pi

    def rng(self,sample_size):
        return rng.logistic(self.mean,self.scale,sample_size)

    def pdf(self,x):
        return sps.logistic(x,loc=self.mean,scale=self.scale)

class Lognormal(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R+',name='Lognormal')
        self.onehot = [0,0,0,0,0,1,0,0,0]
        # TODO: CHECK THIS MATH
        self.shape = math.sqrt(math.log(1 + (self.stddev / self.mean) ** 2))
        self.loc = math.log((self.mean ** 2) / math.sqrt((self.mean ** 2) + (self.stddev ** 2)))

    def rng(self,sample_size):
        return rng.lognormal(self.loc,self.shape,sample_size)

    def pdf(self,x):
        return sps.lognorm(x,self.shape, loc=self.loc)

class Normal(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R',name='Normal')
        self.onehot = [0,0,0,0,0,0,1,0,0]

    def rng(self,sample_size):
        return rng.normal(self.mean,self.stddev,sample_size)

    def pdf(self,x):
        return sps.norm(x,loc=self.mean,scale=self.stddev)

class Rayleigh(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R+',name='Rayleigh')
        self.onehot = [0,0,0,0,0,0,0,1,0]
        self.scale = self.mean * math.sqrt(2 / math.pi)
        self.stddev = self.mean * math.sqrt((4 / math.pi)  - 1)

    def rng(self,sample_size):
        # TODO: fix
        # return rng.rayleigh(self.mean,self.scale,sample_size)
        return rng.rayleigh(self.mean,sample_size)

    def pdf(self,x):
        return sps.rayleigh(x,loc=self.mean,scale=self.scale)

class Wald(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support='R+',name='Wald')
        self.onehot = [0,0,0,0,0,0,0,0,1]
        self.lam = (self.mean ** 3) / (self.stddev ** 2)
        self.mu = self.mean / self.lam

    def rng(self,sample_size):
        return rng.wald(self.mean,self.lam,sample_size)

    def pdf(self,x):
        return sps.invgauss(x,self.mu, scale = self.lam)

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

### Other stuff

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

def generate_data(count, sample_size):
    data = []
    # Generate 'count' examples uniformly at random (before, it was 9 * 'count')
    for _ in range(count):
        dist_class = np.random.choice(DISTRIBUTIONS.items())
        dist_object = dist_class()
        points = dist_object.rng(sample_size)
        label = dist_object.get_label()
        data.append((points, label))
        # items = list(DISTRIBUTIONS.items())
        # choice = np.random.choice(len(items))
        # _, dist_func = items[choice]
        # points, labels = dist_func(sample_size)
        # data.append((points, labels))
    return data

'''
# TODO: multidimensional generation
it should randomly generate N distribution types, where N is NUM_DIMENSIONS
the function should call the randomly chosen distribution functions from distributions.py, and make an array of N-tuples out of the result
it should also make an array of labels, which is the concatenation of the 1hot, mean and stddev of each distribution
return a pair of those arrays

e.g., if NUM_DIMENSIONS is 2 and SAMPLE_SIZE is 3, we randomly choose 3 dimensions (say normal and gamma)
the function should return something that looks like:
[
   [n1, n2, n3, g1, g2, g3],
   [0,0,0,0,0,0,1,0,0,normal_mean,normal_stddev,0,1,0,0,0,0,0,0,0,gamma_mean,gamma_stddev]
]
where n1,g1 are the first points generated by the normal dist and gamma dist, etc

count is the number of examples to generate
sample_size is the number of points to generate for each distribution
'''

def generate_multidim_data(dimensions, count, sample_size):
    data = []

    for _ in range(count):
        points = []
        labels = []
        items = list(DISTRIBUTIONS.items())
        for _ in range(dimensions):
            dist_name = np.random.choice(list(DISTRIBUTIONS.keys()))
            logger.debug(f'dist_name: {dist_name}')

            # dist_class = np.random.choice(DISTRIBUTIONS.items())
            dist_class = DISTRIBUTIONS[dist_name]
            # Instantiate the class!!! 
            dist_object = dist_class()

            dist_points = dist_object.rng(sample_size)
            logger.debug(f'dist_points from this dist: {dist_points}')
            dist_points = list(dist_points)
            logger.debug(f'dist_points but cast to list: {dist_points}')
            points.append(dist_points)

            labels = dist_object.get_label()
            logger.debug(f'labels from this dist: {labels}')
            labels.extend(labels)

        # List of each dimension's points: [[1,3,5], [2,4,6]]
        logger.debug(f"Each dimension's points: {points}")

        # Flattened: [1,3,5, 2,4,6]
        # The actual points in the 2D distribution will be (1,2), (3,4), (5,6)
        points = [item for dim in points for item in dim]
        logger.debug(f"Flattened: {points}")

        data.append((points, labels))

    return data

def make_dataset(filename, examples_count):
    start = time.time()

    # raw_data = generate_data(count=examples_count,
    #                          sample_size=CONFIG['SAMPLE_SIZE'])

    raw_data = generate_multidim_data(dimensions=NUM_DIMENSIONS,
                                      count=examples_count,
                                      sample_size=CONFIG['SAMPLE_SIZE'])

    samples = np.array([elem[0] for elem in raw_data])
    labels = np.array([elem[1] for elem in raw_data])
    dataset = MyDataset(samples, labels)
    torch.save(dataset, filename)

    end = time.time()
    logger.info(f"Generated and saved {examples_count} examples out "
                f"to {filename} in {end - start:.2f} seconds "
                f"(BATCH_SIZE={CONFIG['BATCH_SIZE']})")

    return dataset

def get_dataloader(config, filename=None, examples_count=-1):
    try:
        dataset = torch.load(filename)

        # If examples_count is passed in, then adhere to it. Otherwise,
        # examples_count is default value, indicating dataset can have any size
        acceptable_count = (len(dataset) == examples_count \
                            or examples_count == -1)

        dataset_sample_size = len(dataset.__getitem__(0)[0])
        acceptable_sample_size = (config['SAMPLE_SIZE'] == dataset_sample_size)

        if not isinstance(dataset, Dataset):
            raise Exception(f'Could not read dataset from {filename}')
        if not acceptable_count:
            raise Exception(f"Incorrect dataset size: {len(dataset)}")
        if not acceptable_sample_size:
            raise Exception(f"Incorrect sample size: model expects "
                            f"{config['SAMPLE_SIZE']} points as input, but "
                            f"this dataset would give {dataset_sample_size}")

        logger.info(f"Using dataset from {filename} (size: {len(dataset)})")

    except Exception as e:
        logger.info(e)
        logger.info(f'Generating fresh data...')
        dataset = make_dataset(filename, examples_count)

    # If no filename is passed in, the file does not exist, or the file's
    # contents do not represent a DataLoader as expected, then generate some
    # new data, and write it out to the given filename
    dataloader = DataLoader(dataset, batch_size=config['BATCH_SIZE'])
    return dataloader

# If running this file directly as a script, then generate some training
# examples and save them to a file for later use
if __name__ == "__main__":
    data_directory = 'data'
    os.makedirs(data_directory, exist_ok=True)

    make_dataset('data/train_dataset', examples_count=CONFIG['TRAINING_SIZE'])
    make_dataset('data/test_dataset', examples_count=CONFIG['TEST_SIZE'])
