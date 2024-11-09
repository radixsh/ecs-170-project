import numpy as np
import mpmath
import env
import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        # "Support fail penalty": the amount to add to loss when the network
        # TOTALLY guessues wrong for distribution type. This is going away
        # This will eventually go away
        SFP = 0.99  

        # Manually calculated: (total variation distance) / sqrt(2) to normalize between 0 and 1
        # See: https://www.desmos.com/calculator/8h3zthas2q
        # Indexed by DISTRIBUTION_TYPES
        # Each entry is the non-similarity of two distributions
        # If the distributions have different support, use SFP
        # Symmetric about the diagonal (which is all 0s)
        self.dist_var_matrix = [
            [0  ,SFP,SFP,SFP,SFP,SFP,SFP,SFP,SFP], # beta [0,1]
            [SFP,0  ,SFP,SFP,SFP,0.113506763819,0.353553390593,0.217681125289,0.113036008855], # gamma R+
            [SFP,SFP,0  ,0.122458776456,0.0933338655318,SFP,0.0968575030474,SFP,SFP], # gumbel R
            [SFP,SFP,0.122458776456,0  ,0.065553960094,SFP,0.0999157699984,SFP,SFP], # laplace R
            [SFP,SFP,0.0933338655318,0.065553960094,0  ,SFP,0.0384284266337,SFP,SFP], # logistic R
            [SFP,0.113506763819,SFP,SFP,SFP,0  ,SFP,0.171801196924,0.0298717548468], # lognormal R+
            [SFP,0.353553390593,0.0968575030474,0.0999157699984,0.0384284266337,SFP,0  ,SFP,SFP], # normal R
            [SFP,0.217681125289,SFP,SFP,SFP,0.171801196924,SFP,0  ,0.19627978404], # rayleigh R+
            [SFP,0.113036008855,SFP,SFP,SFP,0.0298717548468,SFP,0.19627978404,0  ]  # wald R+
        ]

    # Returns a dist type as an index in the DISTRIBUTION_TYPES
    def dist_type(self, vec, dim_count):
        # slice the input to get the appropriate 1hot vector
        curr_vec = vec[(dim_count-1)*(env.NUM_DISTS+2):dim_count*(env.NUM_DISTS+2)-2]
        # find the index of "1"
        return (vec == 1).nonzero(as_tuple=True)[0]
    
    """
    def forward_multi(self, dist_vec_actual, dist_vec_guess):
        # Approximates total variation quickly:
        # 1. precalculate total variation between different fundamental distributions
        # 2. calculate mean-loss by MAE
        # 3. calculate stddev-loss by MAE
        # 4. average across all dimensions
        dist_loss = 0
        mean_loss = 0
        stddev_loss = 0
        dim_count = 1
        while dim_count <= env.NUM_DIMENSIONS:
            mean_loss += torch.abs(
                # dim_count*(env.NUM_DISTS+2) counts through the concatted dist vectors
                # then subtract 2 to get the mean
                dist_vec_actual[dim_count*(env.NUM_DISTS+2) - 2] 
                - 
                dist_vec_guess[dim_count*(env.NUM_DISTS+2) - 2]
            )
            stddev_loss += torch.abs(
                # same deal, just subtract 1 to get stddev instead
                dist_vec_actual[dim_count*(env.NUM_DISTS+2) - 1] 
                - 
                dist_vec_guess[dim_count*(env.NUM_DISTS+2) - 1]
            )
            # Search the lookup table, get the number of the dists in the list first
            dist_loss += env.DIST_VAR_MATRIX[self.dist_type(dist_vec_actual,dim_count)][self.dist_type(dist_vec_actual,dim_count)]
            dim_count += 1

        return (dist_loss + mean_loss + stddev_loss) / (3 * env.NUM_DIMENSIONS)"""
    
    def forward(self, pred, y):
        # Approximates total variation quickly:
        # 1. precalculate total variation between different fundamental distributions
        # 2. calculate mean-loss by MAE
        # 3. calculate stddev-loss by MAE
        # 4. average across all dimensions
        dist_vec_actual = y[0]
        dist_vec_guess = pred[0]
        dist_loss = 0
        mean_loss = 0
        stddev_loss = 0
        dim_count = 1
        while dim_count <= env.NUM_DIMENSIONS:
            mean_loss += torch.abs(
                # dim_count*(env.NUM_DISTS+2) counts through the concatted dist vectors
                # then subtract 2 to get the mean
                dist_vec_actual[dim_count*(env.NUM_DISTS+2) - 2] 
                - 
                dist_vec_guess[dim_count*(env.NUM_DISTS+2) - 2]
            )
            stddev_loss += torch.abs(
                # same deal, just subtract 1 to get stddev instead
                dist_vec_actual[dim_count*(env.NUM_DISTS+2) - 1] 
                - 
                dist_vec_guess[dim_count*(env.NUM_DISTS+2) - 1]
            )
            # Search the lookup table, get the number of the dists in the list first
            dist_loss += self.dist_var_matrix[self.dist_type(dist_vec_actual,dim_count)][self.dist_type(dist_vec_actual,dim_count)]
            dim_count += 1
        # average of mean_loss and stddev_loss, times  1/(1-dist loss/dimensions)
        # dist_loss of 0 means loss just average of mean_loss and stddev_loss
        # dist_loss of 1 means loss -> infinity
        return (mean_loss + stddev_loss) / (2 * env.NUM_DIMENSIONS * (1- (dist_loss/env.NUM_DIMENSIONS)))

    # def forward(self, inputs, targets):
    #     loss = -1 * (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
    #     return loss.mean()





    