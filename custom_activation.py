import torch
from env import NUM_DIMENSIONS, CONFIG
from distributions import DISTRIBUTION_FUNCTIONS

def get_indices(dists=False,mean=False,stddev=False, dims=[1]):
    

# Grabs the specified feature(s) at the specified dimension(s)
# Returns a 1D tensor (i.e. a vector)
def extract_features(tensor, dists=False,mean=False,stddev=False, dims=[1]):

    out = torch.empty(0, dtype=tensor.dtype)
    num_dists = len(DISTRIBUTION_FUNCTIONS)
    dim_count = 1
    
    while dim_count <= NUM_DIMENSIONS:
        # Index of start of NEXT dimensional vector
        dim_index = dim_count*(num_dists+2)

        
        curr_mean = ...
        out.append(curr_vals)

        # slice actual and guess to get the appropriate "1hot" vectors
        # softmax the guess to put entries between 0 and 1
        curr_dist = tensor[dim_index-(num_dists+2):dim_index-2]

        out = torch.cat((out,curr_dist,1))
        dim_count += 1

    return tensor
    pass

def multitask(tensor):
    out = torch.tensor(())
    num_dists = len(DISTRIBUTION_FUNCTIONS)
    dim_count = 1
    Softmax = torch.nn.Softmax(dim=0)
    
    while dim_count <= NUM_DIMENSIONS:
        # Index of start of NEXT dimensional vector
        dim_index = dim_count*(num_dists+2)

        curr_vals = ...
        out.append(curr_vals)

        # slice actual and guess to get the appropriate "1hot" vectors
        # softmax the guess to put entries between 0 and 1
        curr_dist = Softmax(tensor[dim_index-(num_dists+2):dim_index-2])

        out = torch.cat((out,curr_dist,1))
        dim_count += 1

    return tensor