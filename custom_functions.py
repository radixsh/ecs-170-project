import torch
import torch.nn as nn
from env import NUM_DIMENSIONS, CONFIG
from generate_data import DISTRIBUTION_FUNCTIONS

# Gets the indices specified of the input... goofy, but convenient
def get_indices(dists=False,mean=False,stddev=False, dims=[1]):
    num_dists = len(DISTRIBUTION_FUNCTIONS)
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
    def __init__(self, use_mean=True,use_stddev=True,use_dists=True):
        super(CustomLoss, self).__init__()
        
        self.use_mean = use_mean
        self.use_stddev = use_stddev
        self.use_dists = use_dists

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
        self.num_dists = len(DISTRIBUTION_FUNCTIONS)

    def get_weights(self,catted_1hot):
        weights = torch.empty(0)
        for n in range(NUM_DIMENSIONS):
            # Slice up the input vector
            curr = catted_1hot[n * self.num_dists:(n+1) * self.num_dists]
            # ??? torch pls
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
        return (regression_loss + CONFIG['ALPHA'] * classification_loss) / (1 - classification_loss)
