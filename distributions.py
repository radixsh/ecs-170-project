import numpy as np
import mpmath
import env

### TODO:
# TEST STUFF
# MAKE GENERAL DIST

### SEED THE RNG
rng = np.random.default_rng()

### HELPERS
# Returns a uniformly random value in (-1,1)
def generate_mean():
    sign = rng.choice([-1, 1])
    return rng.uniform() * sign

# Returns a random positive value according to the function x*exp(-x)
# Mean return is 1
def generate_stddev():
    return rng.gamma(2,scale=1)
    #return rng.rayleigh(scale=1)

# Returns a uniformly random value in (0,1]
def generate_pos_mean():
    return 1 - rng.uniform()


# Returns a vector corresponding to the types of distributions
def names_to_vector(dist_names):
    out = []
    # Loop through the names given, increment the index of the output vector 
    # according to the index of DISTRIBUTION_TYPES
    # then shove each one together
    for name in dist_names:
        vec = [0]*len(env.DISTRIBUTION_TYPES)
        vec[env.DISTRIBUTION_TYPES.index(name)] = 1
        out.append(vec)
    return out

# Returns a vector of names of distributions from 1-hot vector (possibly of multiple dists)
def vector_to_names(vec):
    out = []
    for n in vec:
        # cursed
        out.append(env.DISTRIBUTION_TYPES[vec.index(n) % len(env.DISTRIBUTION_TYPES)])
    return out

# general distribution from a vector of names
def names_to_dist(vec):
    sample_data = []
    labels = []
    for name in vec:
        # terrible
        if True:
            pass
        sample_data.append()
    return (sample_data,labels)
        
    
def vec_to_dist(vec):
    sample_data = []
    labels = []
    ... #loop???
    return (sample_data,labels)

## SUPPORT = R

def normal_dist():
    mean = generate_mean()
    stddev = generate_stddev()
    labels = names_to_vector("normal").append([mean, stddev])
    sample_data = rng.normal(mean, stddev, env.SAMPLE_SIZE)
    return (sample_data, labels)

def gumbel_dist():
    mean = generate_mean()
    stddev = generate_stddev()
    labels = names_to_vector("gumbel").append([mean, stddev])
    # stddev^2 = pi^2 beta^2 * 1/6
    # beta = sqrt(6)/pi * stddev 
    beta = stddev * mpmath.sqrt(6)/mpmath.pi
    # mean = mu + beta * euler (not e)
    # mu = mean - beta * euler
    mu = mean - beta * mpmath.euler
    sample_data = rng.gumbel(mu, beta, env.SAMPLE_SIZE)
    return (sample_data, labels)

def laplace_dist():
    mean = generate_mean()
    stddev = generate_stddev()
    labels = names_to_vector("laplace").append([mean, stddev])
    # stddev^2 = 2b^2
    # b = sqrt(2)*stddev
    b = stddev * mpmath.sqrt(2)
    sample_data = rng.laplace(mean, b, env.SAMPLE_SIZE)
    return (sample_data, labels)

def logistic_dist():
    mean = generate_mean()
    stddev = generate_stddev()
    labels = names_to_vector("logistic").append([mean, stddev])
    # stddev^2 = pi^2 s^2 * 1/3
    # s = sqrt(3)/pi * stddev 
    s = stddev * mpmath.sqrt(3)/mpmath.pi
    sample_data = rng.logistic(mean, s, env.SAMPLE_SIZE)
    return (sample_data, labels)

def lognormal_dist():
    mean = generate_mean()
    stddev = generate_stddev()
    labels = names_to_vector("lognormal").append([mean, stddev])
    # grotesque conversions ahead
    # mean = exp(mu + sigma^2/2)
    # stddev^2 = (exp(sigma^2) - 1) exp(2 mu - sigma^2)
    # sigma = sqrt(ln(1 + (mean/stddev)^2))
    # mu = ln(mean / sqrt(1 + (mean/stddev)^2))
    sigma = mpmath.sqrt(mpmath.ln(1 + (mean / stddev) ** 2))
    mu = mpmath.ln(mean / mpmath.sqrt(1 + (mean / stddev) ** 2))
    sample_data = rng.lognormal(mu, sigma, env.SAMPLE_SIZE)
    return (sample_data, labels)

## SUPPORT >= 0 

"""# Cut due to being a special case of gamma dist.
def exponential_dist():
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = names_to_vector("exponential").append([mean, stddev])
    # one parameter, "lambda" not allowed
    lam = 1 / mean
    sample_data = rng.exponential(lam, env.SAMPLE_SIZE)
    return (sample_data, labels)"""

# Pareto dist is cut for bad documentation and sus numpy implementation

# noncentral_chisquare is cut for having intractable Bessel functions in PDF/CDF
# also for generally not being well behaved
"""def noncentral_chisquare_dist():
    # TODO: double check this for illegal parameter range
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = names_to_vector("noncentral_chisquare").append([mean, stddev])
    # mean = k + lam
    # stddev^2 = 2k + 4lam
    # lam = stddev^2/2 - mean
    # k = 2*mean - stddev^2/2
    lam = (stddev ** 2)/2 - mean #noncentrality
    k = 2*mean - (stddev ** 2)/2 #degrees of freedom
    sample_data = rng.noncentral_chisquare(k, lam, env.SAMPLE_SIZE)
    return (sample_data, labels)"""

def rayleigh_dist():
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = names_to_vector("rayleigh").append([mean, stddev])
    # mean = sigma sqrt(pi/2)
    # sigma = mean * sqrt(2/pi)
    # one param, must be positive
    sigma = mean * mpmath.sqrt(2 / mpmath.pi)
    sample_data = rng.rayleigh(sigma, env.SAMPLE_SIZE)
    return (sample_data, labels)

## SUPPORT = [0,1]

def beta_dist():
    # need a mean in (0,1)
    mean = (generate_mean() + 1)/2
    stddev = generate_stddev()

    labels = names_to_vector("beta").append([mean, stddev])
    # Convert mean and stddev to parameters
    a = - mean(mean ** 2 - mean + stddev)/stddev
    b = (mean - 1)(mean ** 2 - mean + stddev)/stddev

    sample_data = rng.beta(a, b, env.SAMPLE_SIZE)
    return (sample_data, labels)

## SUPPORT > 0

# Chisquared is cut for being a special case of the gamma distribution
"""def chisquared_dist():
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = names_to_vector("chisquare").append([mean, stddev])
    # mean = k
    sample_data = rng.chisquare(mean, env.SAMPLE_SIZE)
    return (sample_data, labels)"""

# f is cut for having intractable Bessel functions in the PDF/CDF
"""def f_dist():
    # need a mean in (0,1)
    mean = (generate_mean() + 1)/2
    stddev = generate_stddev()
    labels = names_to_vector("f").append([mean, stddev])
    # lord help me
    # mean = d2/(d2 - 2)
    # stddev^2 = 2 mean^2/d1 * (d1+d2-2)/(d2 - 4)
    # d2 = 2*mean/(mean - 1)
    # d1 = 2/(stddev/mean^2 *(2-mean) - mean + 1)
    d1 = 2/((stddev * (mean ** -2) (2 - mean)) - mean + 1)
    d2 = 2 * mean / (mean - 1)
    sample_data = rng.f(d1, d2, env.SAMPLE_SIZE)
    return (sample_data, labels)"""

def gamma_dist():
    # nonpositive mean is illegal
    mean = generate_pos_mean()
    stddev = generate_stddev()

    labels = names_to_vector("gamma").append([mean, stddev])
    # Convert mean and stddev to parameters
    k = (mean / stddev) ** 2
    theta = (stddev ** 2) / mean

    sample_data = rng.gamma(k, theta, env.SAMPLE_SIZE)
    return (sample_data, labels)

def wald_dist():
    # nonpositive mean is illegal
    mean = generate_pos_mean()
    stddev = generate_stddev()

    labels = names_to_vector("wald").append([mean, stddev])
    # stddev^2 = mean^3/lam
    # lam = mean^3/stddev^2 
    lam = (mean ** 3) / (stddev ** 2)

    sample_data = rng.wald(mean, lam, env.SAMPLE_SIZE)
    return (sample_data, labels)