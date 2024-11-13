import numpy as np
import math
import mpmath
from env import MEAN_SCALE, SAMPLE_SIZE

### SEED THE RNG
rng = np.random.default_rng()

### HELPERS
# Returns a uniformly random value in (-1, 1)
def generate_mean():
    sign = rng.choice([-1, 1])
    return rng.uniform() * sign * MEAN_SCALE
# Returns a random positive value according to the function x*exp(-x) (mean=1)
def generate_stddev():
    return rng.gamma(2, scale=1)
# Returns a uniformly random value in (0, 1]
def generate_pos_mean():
    return (1 - rng.uniform()) * MEAN_SCALE

## SUPPORT = R

def normal():
    mean = generate_mean()
    stddev = generate_stddev()
    labels = [0,0,0,0,0,0,1,0,0,mean,stddev]
    sample_data = rng.normal(mean, stddev, SAMPLE_SIZE)
    return (sample_data, labels)

def gumbel():
    mean = generate_mean()
    stddev = generate_stddev()
    labels = [0,0,1,0,0,0,0,0,0,mean,stddev]

    # stddev^2 = (pi^2 beta^2) / 6
    # beta = stddev * sqrt(6) / pi
    beta = stddev * math.sqrt(6) / math.pi

    # euler is the Euler-Mascheroni constant
    # mean = mu + beta * euler
    # mu = mean - beta * euler
    mu = mean - beta * float(mpmath.euler)

    sample_data = rng.gumbel(mu, beta, SAMPLE_SIZE)
    return (sample_data, labels)

def laplace():
    mean = generate_mean()
    stddev = generate_stddev()
    labels = [0,0,0,1,0,0,0,0,0,mean,stddev]

    # stddev^2 = 2b^2
    # b = sqrt(2)*stddev
    b = stddev * math.sqrt(2)
    sample_data = rng.laplace(mean, b, SAMPLE_SIZE)
    return (sample_data, labels)

def logistic():
    mean = generate_mean()
    stddev = generate_stddev()
    labels = [0,0,0,0,1,0,0,0,0,mean,stddev]

    # stddev^2 = pi^2 s^2 * 1/3
    # s = sqrt(3)/pi * stddev
    s = stddev * math.sqrt(3) / math.pi
    sample_data = rng.logistic(mean, s, SAMPLE_SIZE)
    return (sample_data, labels)


## SUPPORT >= 0

def lognormal():
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = [0,0,0,0,0,1,0,0,0,mean,stddev]

    # mean = exp(mu + sigma^2/2)
    # stddev^2 = (exp(sigma^2) - 1) exp(2 mu - sigma^2)
    # sigma = sqrt(ln(1 + (mean/stddev)^2))
    # mu = ln(mean / sqrt(1 + (mean/stddev)^2))
    sigma = math.sqrt(math.log(1 + (mean / stddev) ** 2))
    mu = math.log(mean / math.sqrt(1 + (mean / stddev) ** 2))
    sample_data = rng.lognormal(mu, sigma, SAMPLE_SIZE)
    return (sample_data, labels)

def rayleigh():
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = [0,0,0,0,0,0,0,1,0,mean,stddev]

    # mean = sigma sqrt(pi/2)
    # sigma = mean * sqrt(2/pi)
    # One parameter, which must be positive
    sigma = mean * math.sqrt(2 / math.pi)
    sample_data = rng.rayleigh(sigma, SAMPLE_SIZE)
    return (sample_data, labels)

## SUPPORT > 0
# Non-positive mean is illegal for these distributions

def beta():
    # Need a mean in (0, 1) and a stddev in (0, mean - mean^2)
    # Possibly write this differently
    mean = (generate_mean() / MEAN_SCALE + 1) / 2
    stddev = ((generate_mean() / MEAN_SCALE + 1) / 2) * (mean - mean ** 2)
    labels = [1,0,0,0,0,0,0,0,0,mean,stddev]

    a = math.sqrt((mean ** 2 - mean ** 3) / stddev - mean)
    b = (a / mean) - a
    sample_data = rng.beta(a, b, SAMPLE_SIZE)
    return (sample_data, labels)

def gamma():
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = [0,1,0,0,0,0,0,0,0,mean,stddev]

    k = (mean / stddev) ** 2
    theta = (stddev ** 2) / mean
    sample_data = rng.gamma(k, theta, SAMPLE_SIZE)
    return (sample_data, labels)

def wald():
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = [0,0,0,0,0,0,0,0,1,mean,stddev]

    # stddev^2 = mean^3 / lam
    # lam = mean^3 / stddev^2
    lam = (mean ** 3) / (stddev ** 2)
    sample_data = rng.wald(mean, lam, SAMPLE_SIZE)
    return (sample_data, labels)

DISTRIBUTION_FUNCTIONS = {
    "beta": beta,
    "gamma": gamma,
    "gumbel": gumbel,
    "laplace": laplace,
    "logistic": logistic,
    "lognormal": lognormal,
    "normal": normal,
    "rayleigh": rayleigh,
    "wald": wald,
}
