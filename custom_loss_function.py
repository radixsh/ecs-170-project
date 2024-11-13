import numpy as np
import torch
import torch.nn as nn

from env import setup, NUM_DIMENSIONS
from distributions import DISTRIBUTION_FUNCTIONS

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

        # Manually calculated: (total variation distance) / sqrt(2) to normalize between 0 and 1
        # See: https://www.desmos.com/calculator/8h3zthas2q
        # Indexed by DISTRIBUTION_TYPES
        # Each entry is the non-similarity of two distributions
        # If the distributions have different support, use SFP
        # Symmetric about the diagonal (which is all 0s)
        self.dist_var_matrix = [
            [0.1111111111111111, 0.057930075279555564, 0.08681502120533333, 0.08253449168377777, 0.09561208326577779, 0.050799639289777786, 0.10406997045022222, 0.05557577632355556, 0.048745651330888894], 
            [0.057930075279555564, 0.1111111111111111, 0.07864087734200001, 0.060734356420222214, 0.06321621206155556, 0.085887385818, 0.05470651109888888, 0.06273752771355556, 0.08599199803222222], 
            [0.08681502120533333, 0.07864087734200001, 0.1111111111111111, 0.08389804967644444, 0.0903702521041111, 0.070451370148, 0.0895872215451111, 0.06587602966288889, 0.06817723370288889], 
            [0.08253449168377777, 0.060734356420222214, 0.08389804967644444, 0.1111111111111111, 0.09654356442355555, 0.05961180821755555, 0.08890760666711112, 0.077198461922, 0.05590634795000001], 
            [0.09561208326577779, 0.06321621206155556, 0.0903702521041111, 0.09654356442355555, 0.1111111111111111, 0.058801521104888885, 0.10257146074811112, 0.06754045752044445, 0.05590461354133333], 
            [0.050799639289777786, 0.085887385818, 0.070451370148, 0.05961180821755555, 0.058801521104888885, 0.1111111111111111, 0.05470651109888888, 0.07293306735022223, 0.10447294336744445], 
            [0.10406997045022222, 0.05470651109888888, 0.0895872215451111, 0.08890760666711112, 0.10257146074811112, 0.05470651109888888, 0.1111111111111111, 0.06061134021111111, 0.052337773638222215], 
            [0.05557577632355556, 0.06273752771355556, 0.06587602966288889, 0.077198461922, 0.06754045752044445, 0.07293306735022223, 0.06061134021111111, 0.1111111111111111, 0.06749338132444443], 
            [0.048745651330888894, 0.08599199803222222, 0.06817723370288889, 0.05590634795000001, 0.05590461354133333, 0.10447294336744445, 0.052337773638222215, 0.06749338132444443, 0.1111111111111111]
        ]

        # Number of distribution functions
        self.num_dists = len(DISTRIBUTION_FUNCTIONS)

    
    # Calculate the distribution loss:
    # Average the difference between each
    # Currently not in use, will be eventually, possibly
    def calc_dist_loss(self,actual,guess):
        # Get a vector of weights according to the approx total variation
        # pytorch magic, who knows how this actually works
        weights = self.dist_var_matrix[(actual == 1).nonzero(as_tuple=True)[0]]
        # Calculate the error in each component
        # Take the dot product of mean average error and appropriate weights
        error_vec = torch.abs(actual - guess)
        return torch.dot(error_vec, weights) / self.num_dists
        
    
    def forward(self, pred, y):
        # Approximates total variation quickly:
        # 1. lookup dist_loss between different fundamental distributions
        # 2. calculate mean_loss by MAE
        # 3. calculate stddev_loss by MAE
        # 4. average mean_loss and stddev_loss
        # 5. scale previous value by 1/(1 - dist_loss)
        # Final result has same units as mean/stddev
        actual = y[0]
        guess = pred[0]
        dist_loss = 0
        mean_loss = 0
        stddev_loss = 0
        dim_count = 1
        curr_actual = torch.tensor(())
        curr_guess = torch.tensor(())
        Softmax = torch.nn.Softmax(dim=0)
        # # The length of each vector is NUM_DIMENSIONS*(self.num_dists+2)
        # for idx in range(env.NUM_DIMENSIONS*(self.num_dists+2)):
        #     # If the current index % NUM_DIMENSIONS = self.num_dists, then we must be at a mean
        #     if (idx % env.NUM_DIMENSIONS == self.num_dists):
        #         mean_loss += torch.abs(actual[idx] - guess[idx])
        #     # If the current index % NUM_DIMENSIONS = self.num_dists+1, then we must be at a stddev
        #     elif (idx % env.NUM_DIMENSIONS == self.num_dists + 1):
        #         stddev_loss += torch.abs(actual[idx] - guess[idx])
        #     # At the last index in the 1-hot section
        #     elif (idx % env.NUM_DIMENSIONS == self.num_dists -1):
        #         # Tensor equivalent of appending on actual[idx]
        #         curr_actual = torch.cat((curr_actual,actual[idx]), 0)
        #         curr_guess = torch.cat((curr_guess,guess[idx]),0)
        #         dist_loss += self.calc_dist_loss(curr_actual,curr_guess)
        #     # Otherwise, at the non-last index in 1-hot section
        #     else:
        #         curr_actual = torch.cat((curr_actual,actual[idx]), 0)
        #         curr_guess = torch.cat((curr_guess,guess[idx]),0)
        
        while dim_count <= NUM_DIMENSIONS:
            # Index of start of NEXT dimensional vector
            dim_index = dim_count*(self.num_dists+2)
            mean_loss += torch.abs(
                # subtract 2 to get the index of mean
                actual[dim_index - 2] 
                - guess[dim_index - 2]
            )
            stddev_loss += torch.abs(
                # same deal, just subtract 1 to get stddev instead
                actual[dim_index - 1] 
                - guess[dim_index - 1]
            )
            # slice actual and guess to get the appropriate "1hot" vectors
            # softmax the guess to put entries between 0 and 1
            curr_actual = actual[(dim_count-1)*(self.num_dists+2):dim_count*(self.num_dists+2)-2]
            curr_guess = Softmax(guess[(dim_count-1)*(self.num_dists+2):dim_count*(self.num_dists+2)-2])
            # get the appropriate set of weights using torch magic
            weights = torch.tensor(self.dist_var_matrix[(curr_actual == 1).nonzero(as_tuple=True)[0]])
            curr_diff = torch.abs(curr_actual-curr_guess)
            dist_loss += torch.dot(weights,curr_diff)
            dim_count += 1
        
        # Average numerical loss of mean and stddev loss per dimension
        avg_num_loss = (mean_loss + stddev_loss) / (2 * NUM_DIMENSIONS)

        # Average distribution-based loss per dimension
        dist_loss /= NUM_DIMENSIONS
        # average of mean_loss and stddev_loss , times  1/(1-dist loss/dimensions)
        # dist_loss of 0 means loss just average of mean_loss and stddev_loss
        # dist_loss of 1 means loss -> infinity
        #return (mean_loss + stddev_loss) / (2 * (env.NUM_DIMENSIONS - dist_loss))
        return (avg_num_loss + setup['ALPHA'] * dist_loss) / (1 - dist_loss)
