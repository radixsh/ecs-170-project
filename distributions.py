import numpy as np
import env

### TODO:
# IMPLEMENT 1HOT SYSTEM
# IMPLEMENT REMAINING DISTS

### SEED THE RNG
rng = np.random.default_rng()

### HELPERS
# Returns a uniformly random value in (-1,1)
def generate_mean():
    sign = rng.choice([-1, 1])
    return rng.uniform() * sign

# Returns a random positive value according to a normal Rayleigh distribution
# Mean return is 1
def generate_stddev():
    return rng.rayleigh(scale=1)

# Returns a uniformly random value in (0,1]
def generate_pos_mean():
    return 1-rng.uniform()

### CONTINUOUS DISTRIBUTIONS

## SUPPORT = R

def normal_dist():
    #TODO FIXULATE
    mean = generate_mean()
    stddev = generate_stddev()

    labels = [0, 0, 1, mean, stddev]

    # Get `` points from the normal distribution with given specs
    sample_data = rng.normal(mean, stddev, env.SAMPLE_SIZE)

    return (sample_data, labels)

def gumbel_dist():
    pass

def laplace_dist():
    pass

def logistic_dist():
    pass

def lognormal_dist():
    pass

## SUPPORT >= 0 

def exponential_dist():
    pass

def pareto_dist():
    pass

def noncentralchisquared_dist():
    pass

def rayleigh_dist():
    pass

## SUPPORT > 0

def beta_dist():
    # nonpositive mean is illegal
    mean = generate_pos_mean()
    stddev = generate_stddev()

    labels = [0, 0, 1, mean, stddev]
    # Convert mean and stddev to parameters
    # terrible, just awful
    a = - mean(mean ** 2 - mean + stddev)/stddev
    b = (mean - 1)(mean ** 2 - mean + stddev)/stddev

    sample_data = rng.beta(a, b, env.SAMPLE_SIZE)
    return (sample_data, labels)

def chisquared_dist():
    pass

def f_dist():
    pass

def gamma_dist():
    # nonpositive mean is illegal
    mean = generate_pos_mean()
    stddev = generate_stddev()

    labels = [0, 0, 1, mean, stddev]
    # Convert mean and stddev to parameters
    k = (mean / stddev) ** 2
    theta = (stddev ** 2) / mean

    sample_data = rng.gamma(k, theta, )
    return (sample_data, labels)

def wald_dist():
    pass