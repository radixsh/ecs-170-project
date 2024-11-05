# ECS 170 AI Project
Current status: We have a neural net with 1 hidden layer of size 32 that learns
to distinguish between normal and exponential distributions, as well as guess
the mean and standard deviation of those distributions.

The architecture is as follows:
* 50 units of input (`SAMPLE_SIZE=50` in `env.py`)
* 32-unit hidden layer, post-processed by an activation function `ReLU()`
* 4-unit output layer (a loose one-hot vector indicating confidence that it's
  Normal or Exponential, mean, and standard deviation)

## train.py
`train.py` creates a new model which consists of a linear layer and a ReLU
layer. Then it generates training data and trains the model. It trains the model
on the same data for several epochs before generating new data and doing it all
over again. It does this for several runs. Finally, it writes the finished
model's connection weights into `model_weights.pth`.

## analyze_performance.py
`analyze_performance.py` accesses the model saved in `model_weights.pth` and
runs it on newly generated data.

## generate_data.py
`generate_data.py` provides the functions for generating training data.

Each entry in the training set is an ordered tuple whose first entry is a list
of data points and whose second entry is a list of label data. The label data is
organized such that the last 2 entries are mean and standard deviation, and the
other entries are a "one-hot" array representing the distribution family (in
arbitrary alphabetical order ["binomial", "exponential", "normal"]).

For instance, this dataset has three entries, and the last entry represents that
the points in [-2.55, 1.13, 0.94, 1.09, 0.94] are drawn from a Normal
distribution with mean=-0.10 and stddev=0.96:

```
[([67, 72, 84, 75, 75],
	[1, 0, 0, 77.37, 6.99]),
([0.21, 0.04, 0.98, 3.99, 0.01],
	 [0, 1, 0, 0.91, 0.91]),
([-2.55,  1.13,  0.94,  1.09,  0.94],
	 [0, 0, 1, -0.10, 0.96])]
```
# Performance
## Hyperparameters held fixed for now
```python
TRAINING_SIZE = 1000
TEST_SIZE = 200
SAMPLE_SIZE = 50

RUNS = 10
EPOCHS = 10

DISTRIBUTION_TYPES = ["exponential", "normal"]

MODEL = nn.Sequential(
        nn.Linear(in_features=SAMPLE_SIZE, out_features=SAMPLE_SIZE),
        nn.ReLU(),
        nn.Linear(in_features=SAMPLE_SIZE, out_features=len(DISTRIBUTION_TYPES)+2))

LOSS_FN = nn.L1Loss()

OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=1e-3)
```

## Stats
50x50x4: 1 hidden layer of 50 units, with ReLU()
* Training time: INFO:main:Finished overall in 209.98 seconds
* Loss: INFO:analyze_performance:Avg loss over 20 tests: 0.2968025193359936

50x20x4: 1 hidden layer of 20 units, with ReLU()
* Training time: INFO:main:Finished overall in 251.66 seconds
* Loss: INFO:analyze_performance:Avg loss over 20 tests: 0.3921664776731049

50x50x10x4: 2 hidden layers, of 50 and 10 units each, both ReLU()'d
* Training time: INFO:main:Finished overall in 250.18 seconds
* Loss: INFO:analyze_performance:Avg loss over 20 tests: 0.214894094788935
Conclusion: 2 hidden layers is better than 1

50x50x10x4: 2 hidden layers, of 50 and 10 units each, no ReLU()
* Training time: INFO:main:Finished overall in 245.14 seconds
* Loss: INFO:analyze_performance:Avg loss over 20 tests: 0.4624589592027477
Conclusion: ReLU helps


## Another run, with different hyperparameters
```python
TRAINING_SIZE = 1000
TEST_SIZE = 100
SAMPLE_SIZE = 20

RUNS = 10
EPOCHS = 10

DISTRIBUTION_TYPES = ["exponential", "normal"]

MODEL = nn.Sequential(
        nn.Linear(in_features=SAMPLE_SIZE, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=len(DISTRIBUTION_TYPES)+2))

LOSS_FN = nn.L1Loss()
OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=1e-3)
```
20x64x32x4: 2 hidden layers, 64 and 32 units, both ReLU()'d:
* Training time: INFO:main:Finished overall in 250.36 seconds
* Loss: INFO:analyze_performance:Avg loss over 20 tests: 0.23489890832826504


# Resources
* https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
* https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
* https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
