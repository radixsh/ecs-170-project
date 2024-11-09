import numpy as np
import math
import mpmath
import env 

### SEED THE RNG
rng = np.random.default_rng()

### HELPERS
# Returns a uniformly random value in (-1,1)
def generate_mean():
    sign = rng.choice([-1, 1])
    return rng.uniform() * sign * env.MEAN_SCALE

# Returns a random positive value according to the function x*exp(-x)
# Mean return is 1
def generate_stddev():
    return rng.gamma(2,scale=1)
# Returns a uniformly random value in (0,1]
def generate_pos_mean():
    return (1 - rng.uniform()) * env.MEAN_SCALE

def names_to_vector(dist_names):
    # Get the list of distribution names to determine vector size and index mapping
    dist_types = list(DISTRIBUTION_FUNCTIONS.keys())
    out = []
    
    # Loop through the names given, creating a one-hot vector for each
    for name in dist_names:
        vec = [0] * len(dist_types)
        
        # Check if the name exists in distribution functions
        if name in dist_types:
            vec[dist_types.index(name)] = 1
        else:
            # Handle cases where `name` is not a valid distribution name
            raise ValueError(f"Distribution name '{name}' not found in DISTRIBUTION_FUNCTIONS")
        
        out.append(vec)
    
    return out

## SUPPORT = R

def normal_dist():
    mean = generate_mean()
    stddev = generate_stddev()
    labels = [0,0,0,0,0,0,1,0,0,mean,stddev]
    sample_data = rng.normal(mean, stddev, env.SAMPLE_SIZE)

    return (sample_data, labels)

def gumbel_dist():
    mean = generate_mean()
    stddev = generate_stddev()
    labels = [0,0,1,0,0,0,0,0,0,mean,stddev]
    # stddev^2 = pi^2 beta^2 * 1/6
    # beta = sqrt(6)/pi * stddev 
    beta = stddev * math.sqrt(6) / math.pi
    # mean = mu + beta * euler (not e)
    # mu = mean - beta * euler
    mu = mean - beta * float(mpmath.euler)
    sample_data = rng.gumbel(mu, beta, env.SAMPLE_SIZE)
    return (sample_data, labels)

def laplace_dist():
    mean = generate_mean()
    stddev = generate_stddev()
    labels = [0,0,0,1,0,0,0,0,0,mean,stddev]
    # stddev^2 = 2b^2
    # b = sqrt(2)*stddev
    b = stddev * math.sqrt(2)
    sample_data = rng.laplace(mean, b, env.SAMPLE_SIZE)
    return (sample_data, labels)

def logistic_dist():
    mean = generate_mean()
    stddev = generate_stddev()
    labels = [0,0,0,0,1,0,0,0,0,mean,stddev]
    # stddev^2 = pi^2 s^2 * 1/3
    # s = sqrt(3)/pi * stddev 
    s = stddev * math.sqrt(3) / math.pi
    sample_data = rng.logistic(mean, s, env.SAMPLE_SIZE)
    return (sample_data, labels)


## SUPPORT >= 0 

def lognormal_dist():
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = [0,0,0,0,0,1,0,0,0,mean,stddev]
    # mean = exp(mu + sigma^2/2)
    # stddev^2 = (exp(sigma^2) - 1) exp(2 mu - sigma^2)
    # sigma = sqrt(ln(1 + (mean/stddev)^2))
    # mu = ln(mean / sqrt(1 + (mean/stddev)^2))
    sigma = math.sqrt(math.log(1 + (mean / stddev) ** 2))
    mu = math.log(mean / math.sqrt(1 + (mean / stddev) ** 2))
    sample_data = rng.lognormal(mu, sigma, env.SAMPLE_SIZE)
    return (sample_data, labels)

def rayleigh_dist():
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = [0,0,0,0,0,0,0,1,0,mean,stddev]
    # mean = sigma sqrt(pi/2)
    # sigma = mean * sqrt(2/pi)
    # one param, must be positive
    sigma = mean * math.sqrt(2 / math.pi)
    sample_data = rng.rayleigh(sigma, env.SAMPLE_SIZE)
    return (sample_data, labels)

## SUPPORT > 0

def beta_dist():
    # need a mean in (0,1) and a stddev in (0, mean - mean^2)
    # Possibly write this differently
    mean = (generate_mean() / env.MEAN_SCALE + 1) / 2
    stddev = ((generate_mean() / env.MEAN_SCALE + 1) / 2) * (mean - mean ** 2)
    labels = [1,0,0,0,0,0,0,0,0,mean,stddev]
    # Convert mean and stddev to parameters
    a = math.sqrt((mean ** 2 - mean**3) / stddev - mean)
    b = a / mean - a

    sample_data = rng.beta(a, b, env.SAMPLE_SIZE)
    return (sample_data, labels)

def gamma_dist():
    # nonpositive mean is illegal
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = [0,1,0,0,0,0,0,0,0,mean,stddev]
    # Convert mean and stddev to parameters
    k = (mean / stddev) ** 2
    theta = (stddev ** 2) / mean

    sample_data = rng.gamma(k, theta, env.SAMPLE_SIZE)
    return (sample_data, labels)

def wald_dist():
    # nonpositive mean is illegal
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = [0,0,0,0,0,0,0,0,1,mean,stddev]
    # stddev^2 = mean^3/lam
    # lam = mean^3/stddev^2 
    lam = (mean ** 3) / (stddev ** 2)

    sample_data = rng.wald(mean, lam, env.SAMPLE_SIZE)
    return (sample_data, labels)

# Define a canonical ordering (from generate_data.py on main branch)
DISTRIBUTION_FUNCTIONS = {
    "beta": beta_dist,
    "gamma": gamma_dist,
    "gumbel": gumbel_dist,
    "laplace": laplace_dist,
    "logistic": logistic_dist,
    "lognormal": lognormal_dist,
    "normal": normal_dist,
    "rayleigh": rayleigh_dist,
    "wald": wald_dist,
}
