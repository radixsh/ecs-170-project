# ECS 170 AI Project
Currently, this repo creates a crappy little thing that doesn't learn very well.
But it's a proof of concept. The model accepts newly generated (one-dimensional)
data from any of three different distributions, trains on it, and tests its own
performance.

## train.py
`train.py` creates a new model which consists of a linear layer and a ReLU
layer. Then it generates training data and trains the model. It trains the model
on the same data for several epochs before generating new data and doing it all
over again. It does this for several runs. Finally, it writes the finished
model's connection weights into `model_weights.pth`. I've taken the liberty of
uploading this weights file to GitHub along with everything else, but it will
undoubtedly change as our methods get better.

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

# Resources
* https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
* https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
* https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
