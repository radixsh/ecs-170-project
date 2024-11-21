import torch
import torch.nn as nn
from torch.utils.data import Dataset
from env import CONFIG, NUM_DIMENSIONS
import re
import math
import scipy.stats as sps
import numpy as np
import mpmath

# Formats the filename
# Format: "dataset_$TYPE$_len_$TRAIN/TEST_SIZE$_sample_$SAMPLE_SIZE$_dims_$NUM_DIMS$"
# Example: "dataset_train_len_1000_sample_30_dims_2"
def make_filename(type, len, sample_size, num_dims):
    return (f"dataset_{type}_"
            f"len_{len}_"
            f"sample_{sample_size}_"
            f"dims_{num_dims}")

# Checks the filename
# returns a dict of the type, length, sample size, and num_dims
def parse_filename(filename):
    # Define the regex pattern
    pattern = r"dataset_(TRAIN|TEST)_len_(\d+)_sample_(\d+)_dims_(\d+)"
    
    # Match the pattern with the filename
    match = re.match(pattern, filename)
    if not match:
        raise ValueError("Filename does not match the expected format.")
    
    # Extract the values and convert to appropriate types
    type, size, sample_size, num_dims = match.groups()
    return {
        "TYPE": type,
        "SIZE": int(size),
        "SAMPLE_SIZE": int(sample_size),
        "NUM_DIMS": int(num_dims)
    }

# Gets the indices specified of the input... goofy, but convenient
def get_indices(dists=False,mean=False,stddev=False, dims=[1]):
    num_dists = len(DISTRIBUTIONS)
    out = []
    for dim in dims:
        if dists:
            out += range((dim - 1) * (num_dists + 2),(dim - 1) * (num_dists + 2) + num_dists)
        if mean:
            out.append(dim * (num_dists + 2) - 2)
        if stddev:
            out.append(dim * (num_dists + 2) - 1)
    return out

# Grabs the specified feature(s) at the specified dimension(s)
def extract_features(tensor, dists=False,mean=False,stddev=False, dims=[1]):
    return tensor[get_indices(dists, mean, stddev, dims)]

class Multitask(nn.Module):
    def __init__(self, temp=1.0):
        super(Multitask, self).__init__()
        self.temp = temp

    def forward(self, x):
        out = x.clone()
        for n in range(1,NUM_DIMENSIONS+1):
            idxs = get_indices(dists=True,dims=[n])
            # How does torch possibly support this
            # softmaxes the slice representing the dimension vector
            out[0][idxs] = torch.nn.functional.softmax(self.temp * out[0][idxs], dim=0)
        return out

class CustomLoss(nn.Module):
    def __init__(self, use_mean=True, use_stddev=True, use_dists=True, alpha = 1):
        super(CustomLoss, self).__init__()
        
        self.use_mean = use_mean
        self.use_stddev = use_stddev
        self.use_dists = use_dists
        # Alpha controls how much the network gets punished
        # on the classification task relative to the regression task
        self.alpha = alpha

        # Manually calculated: (total variation distance) / sqrt(2) to normalize between 0 and 1
        # See: https://www.desmos.com/calculator/8h3zthas2q
        # Indexed by DISTRIBUTION_TYPES
        # Each entry is the non-similarity of two distributions
        # If the distributions have different support, use SFP
        # Symmetric about the diagonal (which is all 0s)
        self.dist_var_matrix = torch.tensor([
            [0.1111111111111111, 0.057930075279555564, 0.08681502120533333, 0.08253449168377777, 0.09561208326577779, 0.050799639289777786, 0.10406997045022222, 0.05557577632355556, 0.048745651330888894], 
            [0.057930075279555564, 0.1111111111111111, 0.07864087734200001, 0.060734356420222214, 0.06321621206155556, 0.085887385818, 0.05470651109888888, 0.06273752771355556, 0.08599199803222222], 
            [0.08681502120533333, 0.07864087734200001, 0.1111111111111111, 0.08389804967644444, 0.0903702521041111, 0.070451370148, 0.0895872215451111, 0.06587602966288889, 0.06817723370288889], 
            [0.08253449168377777, 0.060734356420222214, 0.08389804967644444, 0.1111111111111111, 0.09654356442355555, 0.05961180821755555, 0.08890760666711112, 0.077198461922, 0.05590634795000001], 
            [0.09561208326577779, 0.06321621206155556, 0.0903702521041111, 0.09654356442355555, 0.1111111111111111, 0.058801521104888885, 0.10257146074811112, 0.06754045752044445, 0.05590461354133333], 
            [0.050799639289777786, 0.085887385818, 0.070451370148, 0.05961180821755555, 0.058801521104888885, 0.1111111111111111, 0.05470651109888888, 0.07293306735022223, 0.10447294336744445], 
            [0.10406997045022222, 0.05470651109888888, 0.0895872215451111, 0.08890760666711112, 0.10257146074811112, 0.05470651109888888, 0.1111111111111111, 0.06061134021111111, 0.052337773638222215], 
            [0.05557577632355556, 0.06273752771355556, 0.06587602966288889, 0.077198461922, 0.06754045752044445, 0.07293306735022223, 0.06061134021111111, 0.1111111111111111, 0.06749338132444443], 
            [0.048745651330888894, 0.08599199803222222, 0.06817723370288889, 0.05590634795000001, 0.05590461354133333, 0.10447294336744445, 0.052337773638222215, 0.06749338132444443, 0.1111111111111111]
        ])

        # Number of distribution functions
        self.num_dists = len(DISTRIBUTIONS)

    def get_weights(self,catted_1hot):
        weights = torch.empty(0)
        # Loop through each dimension
        for n in range(1,NUM_DIMENSIONS+1):
            # Slice up the input vector to just look at the current dimension
            curr = catted_1hot[(n-1) * self.num_dists:n * self.num_dists]
            # Look up the appropriate row in the dist var matrix
            #curr_weights = self.dist_var_matrix[torch.argmax(curr)]
            curr_weights = self.dist_var_matrix[(curr == 1).nonzero(as_tuple=True)[0]]
            # append
            weights = torch.cat((weights,curr_weights),1)
        return weights[0]

    def forward(self, pred, y):
        # Useful constant
        dim_range = range(1,NUM_DIMENSIONS+1)
        
        # Absolute difference of vectors normalized by number of dimensions
        diff = torch.abs(pred[0] - y[0]) / NUM_DIMENSIONS
        
        # Calculate MAE on means per dimension
        mean_idxs = get_indices(mean=True, dims=dim_range)
        mean_loss = torch.sum(diff[mean_idxs])

        # Calculate MAE on stddevs per dimension
        stddev_idxs = get_indices(stddev=True, dims=dim_range)
        stddev_loss = torch.sum(diff[stddev_idxs])

        # Approximate total variation distance between distributions
        # Need to look up weights in dist_var_matrix
        dist_idxs = get_indices(dists=True,dims=dim_range)
        weights = self.get_weights(y[0][dist_idxs])
        classification_loss = torch.dot(diff[dist_idxs],weights)   
        
        # Make various losses trivial if not in use
        if not self.use_mean:
            mean_loss = 0
            stddev_loss *= 2
        if not self.use_stddev:
            stddev_loss = 0
            mean_loss *= 2
        if not self.use_dists:
            classification_loss = 0

        # Average numerical loss of mean and stddev loss
        regression_loss = (mean_loss + stddev_loss) / 2

        # classification loss of 0 means loss = regression loss
        # classification loss of 1 means loss -> infinity
        return (regression_loss + self.alpha * classification_loss) / (1 - classification_loss)

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
        return rng.rayleigh(self.scale,sample_size)

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