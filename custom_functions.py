import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, Subset
import time
import os
import re
import math
import scipy.stats as sps
import numpy as np
import mpmath
import logging
from sklearn.metrics import *
from collections import defaultdict
# NO IMPORTING ENV

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

def generate_multidim_data(config, mode):
    data = []
    for _ in range(config[f'{mode}_SIZE']): # NO LOGGING INSIDE THIS LOOP
        points, labels = [], []
        for _ in range(config['NUM_DIMENSIONS']):
            # Get a random distribution class and instantiate
            dist = np.random.choice(list(DISTRIBUTIONS.values()))()
            # Generate some data using the distribution
            dist_points = list(dist.rng(config['SAMPLE_SIZE']))
            points.append(dist_points)
            labels += dist.get_label() 
        points = np.ravel(points) # Identical to flatten here, but faster
        data.append((points, labels))
    return data

def get_good_filename(config, mode):
    for filename in os.listdir('data'):
        if 'dataset' not in filename: # Catches weird stuff (like .DS_store)
            continue
        # Check the type, that there's enough data,
        # the sample size is right, and that num_dimensions matches
        file_info = parse_data_filename(filename)
        if file_info['MODE'] == mode \
            and file_info[f'{mode}_SIZE'] >= config[f'{mode}_SIZE'] \
            and file_info['SAMPLE_SIZE'] == config['SAMPLE_SIZE'] \
            and file_info['NUM_DIMENSIONS'] == config['NUM_DIMENSIONS']:
                good_filename = os.path.join("data", filename)
                return good_filename
    return None

def make_dataset(config, mode):
    # Check for a preexisting good file
    good_filename = get_good_filename(config,mode)
    if good_filename:
        logger.info(f"Sufficient data already exists at {good_filename}!")
        dataset = torch.load(good_filename)
        return dataset
    start = time.time()

    examples_count = config[f'{mode}_SIZE'] # Get the correct size
    
    filename = make_data_filename(mode, config)
    logger.info(f"Generating {examples_count} pieces of {config['NUM_DIMENSIONS']}"
                f"-dimensional {mode.lower()} data, each with {config['SAMPLE_SIZE']} samples.")

    raw_data = generate_multidim_data(config, mode)

    samples = np.array([elem[0] for elem in raw_data])
    labels = np.array([elem[1] for elem in raw_data])
    dataset = MyDataset(samples, labels)

    torch.save(dataset, filename)

    end = time.time()
    logger.info(f"Generated and saved {examples_count} examples out to "
                f"{filename} in {end - start:.3f} seconds.")
    return dataset

def get_dataloader(config, mode='TRAIN'): #mode should be 'TRAIN' or 'TEST'
    good_filename = get_good_filename(config,mode)

    if good_filename:
        logger.info(f"Using data from {good_filename}")
        dataset = torch.load(good_filename, weights_only=False)
        if parse_data_filename(good_filename)[f'{mode}_SIZE'] != config[f'{mode}_SIZE']:
            logger.debug(f"Taking the first {config[f'{mode}_SIZE']} entries")
            dataset = Subset(dataset, indices=range(config[f'{mode}_SIZE']))

    else:
        logger.info(f'No valid data found, generating fresh data...')
        dataset = make_dataset(config, mode)

    if mode == 'TRAIN':
        dataloader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    elif mode == 'TEST': # batch size should be the entire thing for testing.
        dataloader = DataLoader(dataset, batch_size=config['TRAIN_SIZE'], shuffle=True)

    return dataloader

def run_model(model, dataloader, loss_function, optimizer, device, num_dimensions, mode):
    model.train()

    # Keep these separate for now
    losses = []
    metrics = defaultdict(list)

    for X, y in dataloader:
        X, y = X.to(device).float(), y.to(device).float()

        pred = model(X)                     # Forward pass
        loss = loss_function(pred, y)       # Compute loss (prediction error)
        losses.append(loss.item())     # Save for metrics

        optimizer.zero_grad()
        loss.backward()                     # Backpropagation
        optimizer.step()

        ## Validation
        # Use 10% of the data to generate validation metrics
        # Doesn't add much training time since we're not calling .backward()
        validation_size = X.size(0) // 10
        
        with torch.no_grad(): # Already trained on these
            # Data is randomized, so we can just take the first slice
            X_subset = X[:validation_size]
            y_subset = y[:validation_size]
            pred_subset = model(X_subset)
            batch_metrics = calculate_metrics(pred_subset, y_subset, num_dimensions, mode=mode)

            # Aggregate batch metrics
            # This is why we kept losses separate
            for key, value in batch_metrics.items():
                metrics[key].append(value)
            
    # Average metrics over all batches
    # Remember to add in the loss
    metrics['loss'] = losses
    metrics = {key: np.mean(value) for key, value in metrics.items()}

    return metrics

#######################
###  LOSS FUNCTION  ###
#######################

# Generates the indices for use in our vector format:
# dists=True gets the onehot portion(s) of the vector
# mean=True gets the mean(s), stddev=True gets the stddev(s)
# dim indicates the dimension of info to get

# Example:
# x = [1,0,0,0,0,0,0,0,0,3.1,0.6,0,1,0,0,0,0,0,0,0,4.5,1.2]
# x[get_indices(dists=True,dim=1)] -> [1,0,0,0,0,0,0,0,0]
# x[get_indices(mean=True,stddev=True,dims=2)] -> [4.5,1.2]

def get_indices(dists=False, mean=False, stddev=False, dim=-1):
    num_dists = len(DISTRIBUTIONS)
    out = []
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

class CustomLoss(nn.Module):
    def __init__(self, use_mean=False, use_stddev=False, use_dists=True, num_dimensions = -1):
        super(CustomLoss, self).__init__()
        self.use_mean = use_mean
        self.use_stddev = use_stddev
        self.use_dists = use_dists
        self.num_dimensions = num_dimensions
        if self.num_dimensions == -1:
            raise Exception("CustomLoss needs to be called with kwarg num_dimensions = ...")

    def forward(self, pred, y):
        mean_loss = 0
        stddev_loss = 0
        classification_loss = 0

        for dim in range(self.num_dimensions):
            # Extract targets for this dimension
            dists_idx = get_indices(dists=True, dim=dim + 1)
            mean_idx = get_indices(mean=True, dim=dim + 1)
            stddev_idx = get_indices(stddev=True, dim=dim + 1)

            # Extract targets for this dimension
            class_targets = y[:, dists_idx]  # Shape [batch_size, num_classes]
            mean_targets = y[:, mean_idx]  # Shape [batch_size]
            stddev_targets = y[:, stddev_idx]  # Shape [batch_size]

            # Compute losses
            if self.use_dists:
                classification_loss += nn.CrossEntropyLoss()(
                    pred['classification'][:, dim, :],  # Raw logits
                    torch.argmax(class_targets, dim=1)  # Class indices
                )

            #sqrt(MSE) -> RMSE loss, converges to mean
            # whereas MAE converges to the median
            # don't ask why, statistics is crazy 
            if self.use_mean:
                mean_loss += torch.sqrt(nn.MSELoss()(pred['mean'][:, dim].unsqueeze(1), mean_targets))

            if self.use_stddev:
                stddev_loss += torch.sqrt(nn.MSELoss()(pred['stddev'][:, dim].unsqueeze(1), stddev_targets))
            


        # Adjust weights of different losses if necessary
        params_in_use = int(self.use_mean + self.use_stddev + self.use_dists)
        total_loss = (mean_loss + stddev_loss + classification_loss) / params_in_use

        return total_loss
    
def calculate_metrics(pred, y, num_dimensions, mode):

    metrics = defaultdict(list)

    for dim in range(num_dimensions):
        # Extract ground-truth indices for this dimension
        dists_idx = get_indices(dists=True, dim=dim+1)
        mean_idx = get_indices(mean=True, dim=dim+1)
        stddev_idx = get_indices(stddev=True, dim=dim+1)

        # Classification
        class_targets = y[:, dists_idx]  # One-hot class vector
        class_targets = torch.argmax(class_targets, dim=1).numpy()  # Convert to class indices
        class_preds = torch.argmax(pred['classification'][:, dim, :], dim=1).numpy()

        # Mean regression
        mean_targets = y[:, mean_idx].numpy()
        mean_preds = pred['mean'][:, dim].detach().numpy()

        # Stddev regression
        stddev_targets = y[:, stddev_idx].numpy()
        stddev_preds = pred['stddev'][:, dim].detach().numpy()

        # Calculate standard metrics
        metrics['accuracy'].append(accuracy_score(class_targets, class_preds))
        metrics['mean_r2'].append(r2_score(mean_targets, mean_preds))
        metrics['stddev_r2'].append(r2_score(stddev_targets, stddev_preds))

        # Additional metrics for testing
        if mode == 'TEST':
            metrics['mean_mae'].append(mean_absolute_error(mean_targets, mean_preds))
            metrics['mean_mape'].append(mean_absolute_percentage_error(stddev_targets, stddev_preds))
            metrics['mean_rmse'].append(np.sqrt(mean_squared_error(mean_targets, mean_preds)))
            metrics['stddev_mae'].append(mean_absolute_error(stddev_targets, stddev_preds))
            metrics['stddev_mape'].append(mean_absolute_percentage_error(mean_targets, mean_preds))
            metrics['stddev_rmse'].append(np.sqrt(mean_squared_error(stddev_targets, stddev_preds)))
            # The complicated ones...
            metrics['avg_precision'].append(precision_score(class_targets,
                                                        class_preds,
                                                        labels=range(len(DISTRIBUTIONS)),
                                                        average=None,
                                                        zero_division=0.0))
            metrics['avg_recall'].append(recall_score(class_targets,
                                                        class_preds,
                                                        labels=range(len(DISTRIBUTIONS)),
                                                        average=None,
                                                        zero_division=0.0))
            metrics['avg_f1'].append(f1_score(class_targets,
                                                        class_preds,
                                                        labels=range(len(DISTRIBUTIONS)),
                                                        average=None,
                                                        zero_division=0.0))
    if mode == 'TEST':
        # need to take mean across different axis since these are arrays        
        precision = np.mean(metrics['avg_precision'],axis=0)
        recall = np.mean(metrics['avg_recall'],axis=0)
        f1 = np.mean(metrics['avg_f1'],axis=0)

    # Return average the metrics over all dimensions
    metrics = {key: np.mean(value) for key, value in metrics.items()}

    if mode == 'TEST':
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1

    return metrics

    
#######################
### FILE FORMATTING ###
#######################

# Formats the filename for a model weight file
# Format: "weights_train_$TRAIN_SIZE$_sample_$SAMPLE_SIZE$_dims_$NUM_DIMS$"
# Example: "weights_train_1000_sample_30_dims_2"
def make_weights_filename(config):
    return (f"models/"
            f"weights_train_{config['TRAIN_SIZE']}_"
            f"sample_{config['SAMPLE_SIZE']}_"
            f"dims_{config['NUM_DIMENSIONS']}_"
            f"batch_{config['BATCH_SIZE']}_"
            f"lrate_{config['LEARNING_RATE']}"
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
        "NUM_DIMENSIONS": int(num_dims),
        "BATCH_SIZE": int(batch_size),
        "LEARNING_RATE": float(learning_rate)
    }

# Formats the filename for a dataset
# Format: "dataset_$TYPE$_len_$TRAIN/TEST_SIZE$_sample_$SAMPLE_SIZE$_dims_$NUM_DIMS$"
# Example: "dataset_train_len_1000_sample_30_dims_2"
def make_data_filename(mode, config):
    return (f"data/"
            f"dataset_{mode}_"
            f"len_{config[f'{mode}_SIZE']}_"
            f"sample_{config['SAMPLE_SIZE']}_"
            f"dims_{config['NUM_DIMENSIONS']}"
            f".pth")

# Checks the filename for a dataset
# returns a dict of the type, length, sample size, and num_dims
def parse_data_filename(filename):
    # Define the regex pattern
    pattern = r"(data\\|data/)?dataset_(TRAIN|TEST)_len_(\d+)_sample_(\d+)_dims_(\d+).pth"

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
        f"{mode}_SIZE": int(length),
        "SAMPLE_SIZE": int(sample_size),
        "NUM_DIMENSIONS": int(num_dims)
    }

#######################
### DATA GENERATION ###
#######################

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

##########################
### MODEL ARCHITECTURE ###
##########################

class StddevActivation(nn.Module):
    def __init__(self):
        super(StddevActivation, self).__init__()

    def forward(self, x):
        # derivative of sqrt at 0 -> oo, need to min/max the inputs
        return torch.sqrt(torch.clamp(torch.abs(x), 1e-6, 1e6))

class Head(nn.Module):
    def __init__(self, input_dim, layer_sizes = [], output_size = 1, activation = nn.ReLU, final_activation = None):
        super(Head, self).__init__()

        # List of layers
        self.layers = nn.ModuleList()

        # Check that we didn't give a list of layer sizes
        #   This is the case when we're making a mean head
        if len(layer_sizes) == 0:
            self.layers.append(nn.Linear(input_dim, output_size, bias=False))
        else:
            # Input layer
            self.layers.append(nn.Linear(input_dim, layer_sizes[0]))
            self.layers.append(activation())

            # "Hidden" layers
            # Need to slice off last layer size because we lookahead
            for n in range(len(layer_sizes)-1):
                self.layers.append(nn.Linear(layer_sizes[n],
                                            layer_sizes[n+1]))
                self.layers.append(activation())

            # Output layer
            self.layers.append(nn.Linear(layer_sizes[-1], output_size, bias=False))
        if final_activation:
            self.layers.append(final_activation())

    # Beautiful
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MultiTaskModel(nn.Module):
    def __init__(self, config, architecture, num_classes, activation=nn.ReLU):
        super(MultiTaskModel, self).__init__()
        self.num_dimensions = config['NUM_DIMENSIONS']
        self.num_classes = num_classes
        # Useful value
        input_dim = config['SAMPLE_SIZE'] * self.num_dimensions

        ### HYPERPARAMETERS

        # first shared dim is always fixed to be input_dim
        # secondary backbone has one more layer than there are items in list
        # input size of each layer in secondary backbone
        self.shared_layer_sizes = architecture['SHARED_LAYER_SIZES']

        # each has one more layer than there are items in list
        # input sizes of each layer in the heads
        self.stddev_head_layer_sizes = architecture['STDDEV_LAYER_SIZES']
        self.class_head_layer_sizes = architecture['CLASS_LAYER_SIZES']

        ## LAYERS

        # First backbone, shared by everything, stays fixed since 
        #   mean only ever needs one layer
        # Bias = False theoretically helps mean performance
        # Linear activation function here, don't add anything else
        self.input = nn.Linear(input_dim, input_dim, bias=False)
        
        # Second backbone, shared only by stddev and classification
        self.shared_layers = nn.ModuleList()
        # Match up with the input layer
        self.shared_layers.append(nn.Linear(input_dim, self.shared_layer_sizes[0]))
        self.shared_layers.append(activation())
        # Add hidden layers
        # Need to slice off last layer size because we look ahead
        for n in range(len(self.shared_layer_sizes)-1):
            self.shared_layers.append(nn.Linear(self.shared_layer_sizes[n], self.shared_layer_sizes[n+1]))
            self.shared_layers.append(activation())

        # Make a head of each metric for each dimension
        self.mean_head_array = nn.ModuleList([
            Head(input_dim) for _ in range(self.num_dimensions)
        ])

        self.stddev_head_array = nn.ModuleList([
            Head(self.shared_layer_sizes[-1], 
                layer_sizes=self.stddev_head_layer_sizes,
                activation=activation,
                final_activation = StddevActivation,
                ) for _ in range(self.num_dimensions)
        ])
        
        self.class_head_array = nn.ModuleList([
            Head(self.shared_layer_sizes[-1],
                layer_sizes=self.class_head_layer_sizes,
                output_size=num_classes,
                activation=activation) for _ in range(self.num_dimensions)
        ])
            
        # Done contructing stuff, initialize weights
        # Assumes relu activation
        def initializer(module): 
               if isinstance(module, nn.Linear):  # For linear layers
                    init.kaiming_normal_(module.weight, nonlinearity='relu')  # He initialization
                    if module.bias is not None:
                        init.constant_(module.bias, 0)  # Initialize biases to 0

        self.apply(initializer)
    
    def forward(self, x):
        batch_size = x.size(0)
        outputs = {
            "classification": torch.zeros(batch_size, self.num_dimensions, self.num_classes, device=x.device),
            "mean": torch.zeros(batch_size, self.num_dimensions, device=x.device),
            "stddev": torch.zeros(batch_size, self.num_dimensions, device=x.device),
        }
        
        # First shared -> mean, second shared
        input = self.input(x)

        # Second shared -> stddev, classification
        shared = input
        for layer in self.shared_layers:
            shared = layer(shared)

        for n in range(self.num_dimensions):
            # Classification
            # This has the shape [batch_size, num_dimensions, num_distributions]
            # For example, [1000, 2, 9] means:
            #   1000 "rows" (one per item in batch), within each row:
            #       a vector of length 2 (one for each dimension) where each entry is:
            #           a onehot vector of length 9 (one for each distribution)
            outputs["classification"][:, n, :] = self.class_head_array[n](shared)

            # Mean
            # This has the shape [batch_size, num_dimensions]
            # For example, [1000, 2] means:
            #   1000 "rows" (one per item in batch), within each row:
            #       a vector of length 2 (one for each dimension) where each entry is the mean
            # One less tensor-dimension than class, since means are just represented as numbers instead of onehot vectors
            outputs["mean"][:, n] = self.mean_head_array[n](input).squeeze(-1)

            # Stddev
            # Exactly the same as mean, see above
            outputs["stddev"][:, n] = self.stddev_head_array[n](shared).squeeze(-1)

        return outputs
