import math
import scipy.stats as sps
import numpy as np

### No importing any custom files.

rng = np.random.default_rng()


class Distribution:
    """
    Constructor to be inherited. Sets the name and support, then generates an
    appropriate mean and standard deviation for the specified distribution.

    Default methods:
        __init__: Constructor.
        __str__: String representation.
        get_label: Returns a list containing a onehot encoding, mean, and stddev.
        random_pos: Randomly generates a positive real.
        generate_mean: Generates an appropriate mean for the given distribution.
        generate_stddev: Generates an appropriate stddev for the given distribution.

    Child methods:
        rng: Returns a randomly sampled a set of points from the distribution.
        pdf: Returns the probability density function for the distribution.
    """

    def __init__(self, mean, stddev, support, name):
        """
        Constructor to be inherited. Sets the name and support, then generates an
        appropriate mean and standard deviation for the specified distribution.

        Args:
            mean (float or *): The mean of the distribution, or generate a new one.
            stddev (float or *): The stddev of the distribution, or generate a new one.
            support (str): The set of real numbers on which the distribution's
                probability density function is positive.
            name (str): The name of the distribution.
        """
        self.name = name
        self.support = support
        self.mean = mean if not isinstance(mean, str) else self.generate_mean()
        self.stddev = stddev if not isinstance(stddev, str) else self.generate_stddev()
        self.onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def __str__(self):
        string = f"{self.name} (support: {self.support}), "
        string += f"mean={self.mean:.3f}, "
        string += f"stddev={self.stddev:.3f}, "
        string += f"label={self.onehot}"
        return string

    def get_label(self):
        return self.onehot + [self.mean, self.stddev]

    # Returns a random positive real according to the lognormal distribution
    # with mean and stddev 1. Useful for generating stddevs and positive means.
    def random_pos(self):
        return rng.lognormal(mean=-math.log(2) / 2, sigma=math.sqrt(math.log(2)))

    # Uses the support to randomly generate an appropriate mean.
    def generate_mean(self):
        if self.support == "R":
            return rng.normal(0, 1)  # Support = R
        elif self.support == "R+":
            return self.random_pos()  # Support > 0
        elif self.support == "I":  # Otherwise [0,1]
            sign = rng.choice([-1, 1])
            random_in_open_interval = rng.uniform() * sign
            random_in_open_interval = (random_in_open_interval + 1) / 2
            return random_in_open_interval

    # Uses the name to generate an appropriate stddev.
    def generate_stddev(self):
        if self.name not in ["Beta", "Rayleigh"]:
            return self.random_pos()
        # Stddev should be a random value in (0, mean-mean^2)
        elif self.name == "Beta":
            sign = rng.choice([-1, 1])
            random_in_open_interval = rng.uniform() * sign
            random_in_open_interval = (random_in_open_interval + 1) / 2
            upper_bound = math.sqrt(self.mean * (1 - self.mean))
            return random_in_open_interval * upper_bound
        else:
            return self.mean * math.sqrt((4 / math.pi) - 1)


class Beta(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support="I", name="Beta")
        self.onehot = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        nonsense_constant = (self.mean * (1 - self.mean) / (self.stddev**2)) - 1
        self.alpha = self.mean * nonsense_constant
        self.beta = (1 - self.mean) * nonsense_constant

    def rng(self, sample_size):
        return rng.beta(self.alpha, self.beta, sample_size)

    def pdf(self, x):
        return sps.beta.pdf(x, self.alpha, self.beta)


class Gamma(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support="R+", name="Gamma")
        self.onehot = [0, 1, 0, 0, 0, 0, 0, 0, 0]
        self.shape = (self.mean / self.stddev) ** 2
        self.scale = (self.stddev**2) / self.mean

    def rng(self, sample_size):
        return rng.gamma(self.shape, self.scale, sample_size)

    def pdf(self, x):
        return sps.gamma.pdf(x, self.shape, scale=self.scale)


class Gumbel(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support="R", name="Gumbel")
        self.onehot = [0, 0, 1, 0, 0, 0, 0, 0, 0]
        self.scale = self.stddev * math.sqrt(6) / math.pi
        self.loc = self.mean - self.scale * float(np.euler_gamma)

    def rng(self, sample_size):
        return rng.gumbel(self.loc, self.scale, sample_size)

    def pdf(self, x):
        return sps.gumbel_r.pdf(x, loc=self.loc, scale=self.scale)


class Laplace(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support="R", name="Laplace")
        self.onehot = [0, 0, 0, 1, 0, 0, 0, 0, 0]
        self.scale = self.stddev / math.sqrt(2)

    def rng(self, sample_size):
        return rng.laplace(self.mean, self.scale, sample_size)

    def pdf(self, x):
        return sps.laplace.pdf(x, loc=self.mean, scale=self.scale)


class Logistic(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support="R", name="Logistic")
        self.onehot = [0, 0, 0, 0, 1, 0, 0, 0, 0]
        self.scale = self.stddev * math.sqrt(3) / math.pi

    def rng(self, sample_size):
        return rng.logistic(self.mean, self.scale, sample_size)

    def pdf(self, x):
        return sps.logistic.pdf(x, loc=self.mean, scale=self.scale)


class Lognormal(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support="R+", name="Lognormal")
        self.onehot = [0, 0, 0, 0, 0, 1, 0, 0, 0]
        self.shape = math.sqrt(math.log(1 + (self.stddev / self.mean) ** 2))
        self.loc = math.log(
            (self.mean**2) / math.sqrt((self.mean**2) + (self.stddev**2))
        )

    def rng(self, sample_size):
        return rng.lognormal(self.loc, self.shape, sample_size)

    def pdf(self, x):
        return sps.lognorm.pdf(x, self.shape, loc=self.loc)


class Normal(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support="R", name="Normal")
        self.onehot = [0, 0, 0, 0, 0, 0, 1, 0, 0]

    def rng(self, sample_size):
        return rng.normal(self.mean, self.stddev, sample_size)

    def pdf(self, x):
        return sps.norm.pdf(x, loc=self.mean, scale=self.stddev)


class Rayleigh(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support="R+", name="Rayleigh")
        self.onehot = [0, 0, 0, 0, 0, 0, 0, 1, 0]
        self.scale = self.mean * math.sqrt(2 / math.pi)

    def rng(self, sample_size):
        return rng.rayleigh(self.scale, sample_size)

    def pdf(self, x):
        return sps.rayleigh.pdf(x, scale=self.scale)


class Wald(Distribution):
    def __init__(self, mean="not set", stddev="not set"):
        super().__init__(mean, stddev, support="R+", name="Wald")
        self.onehot = [0, 0, 0, 0, 0, 0, 0, 0, 1]
        self.lam = (self.mean**3) / (self.stddev**2)
        self.mu = self.mean / self.lam

    def rng(self, sample_size):
        return rng.wald(self.mean, self.lam, sample_size)

    def pdf(self, x):
        return sps.invgauss.pdf(x, self.mu, scale=self.lam)


# Distributions in use for data generator.
# The length is a helpful constant to reference.
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

# Extremely useful constant
NUM_DISTS = len(DISTRIBUTIONS)
