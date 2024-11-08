# ECS 170 AI Project
We have a prototype that identifies normal and exponential distributions and
guesses their mean and standard deviations. Trained on hyperparameters saved in
`env.py`, the model's weights are uploaded here as `model_weights.pth`.

The architecture is as follows:
* About 30 input units (`SAMPLE_SIZE=30` in `env.py`)
* 64-unit hidden layer, with `ReLU` activation function
* 32-unit hidden layer, with `ReLU` activation function
* 4-unit output layer (a loose one-hot vector indicating confidence that it's
  Normal or Exponential, mean, and standard deviation)

Next steps: take stock of current performance, generate some nice pretty graphs,
move on to multivariate data analysis, move on to more types of distributions. 

## Main useful files
### analyze_performance.py
`analyze_performance.py` runs the model saved in `model_weights.pth` on newly
generated data 20 times. It outputs the average loss over all 20 runs at the
end. Run this file to test the current model's performance.

### train.py
`train.py` trains a new model based on the hyperparameters set in `env.py`. It
generates fresh training data and trains the model using cross-validation. The
finished model's connection weights are saved in `model_weights.pth`. Run this
file to train a new model.

## Documentation for supporting files
### loss_fn.py
Provides a custom loss function that appropriately deals with both categorical
data (distribution family) and numerical data (mean, standard deviation). The
custom loss function is set in `env.py`.

### generate_data.py
`generate_data.py` provides the function for generating data from normal and
exponential distributions; it will soon be supplanted with `distributions.py`.

Each entry in the training set is an ordered tuple whose first entry is a list
of data points and whose second entry is a list of label data. The label data is
organized such that the first `NUM_DISTS` entries are a "loose" one-hot vector
(expressing confidence rather than making a strict one hot choice) representing
the distribution family (`['exponential', 'normal']`), while the last 2 entries
are mean and standard deviation.

For instance, this dataset has two entries, and the second entry represents that
the points in `[-2.55, 1.13, 0.94, 1.09, 0.94]` are drawn from a Normal
distribution with `mean=-0.10` and `stddev=0.96`:

```
[([0.21, 0.04, 0.98, 3.99, 0.01],
	 [1, 0, 0.91, 0.91]),
([-2.55,  1.13,  0.94,  1.09,  0.94],
	 [0, 1, -0.10, 0.96])]
```

### performance_plot.py
Generates a graph of hardcoded data from the performance analysis section below.

# Performance
### Some hyperparameters
```python
TRAINING_SIZE = 1000
TEST_SIZE = 200
SAMPLE_SIZE = 50
RUNS = 10
EPOCHS = 10

LOSS_FN = nn.L1Loss()
```

50x50x4: 1 hidden layer of 50 units, with ReLU()
* Training time: INFO:main:Finished overall in 209.98 seconds
* Loss: INFO:analyze_performance:Avg loss over 20 tests: 0.2968025193359936

50x20x4: 1 hidden layer of 20 units, with ReLU()
* Training time: INFO:main:Finished overall in 251.66 seconds
* Loss: INFO:analyze_performance:Avg loss over 20 tests: 0.3921664776731049

50x50x10x4: 2 hidden layers, of 50 and 10 units each, both ReLU()'d
* Training time: INFO:main:Finished overall in 250.18 seconds
* Loss: INFO:analyze_performance:Avg loss over 20 tests: 0.214894094788935

_Conclusion: 2 hidden layers is better than 1_

50x50x10x4: 2 hidden layers, of 50 and 10 units each, no ReLU()
* Training time: INFO:main:Finished overall in 245.14 seconds
* Loss: INFO:analyze_performance:Avg loss over 20 tests: 0.4624589592027477

_Conclusion: ReLU helps_

### Another run, with different hyperparameters
```python
TRAINING_SIZE = 1000
TEST_SIZE = 100
SAMPLE_SIZE = 20
RUNS = 10
EPOCHS = 10

LOSS_FN = nn.L1Loss()
```
20x64x32x4: 2 hidden layers, 64 and 32 units, both ReLU()'d:
* Training time: INFO:main:Finished overall in 250.36 seconds
* Loss: INFO:analyze_performance:Avg loss over 20 tests: 0.23489890832826504

### Another run, with different hyperparameters and also a new loss function (sorry)
```python
TRAINING_SIZE = 1000
TEST_SIZE = int(TRAINING_SIZE * 0.2 / 0.8) # Fixed at ~20% of training + test
SAMPLE_SIZE = 30
RUNS = 3
EPOCHS = 5

# Also using James' new loss function rather than MSE or MAE, so loss is higher
LOSS_FN = CustomLoss()
```
30x64x32x4: 2 hidden layers, 64 and 32 units, both ReLU()'d:
* Training time: Finished 3 runs in 231.50 seconds
* Loss: Avg loss over 20 tests: 0.31893037010598924

# Resources
* https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
* https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
* https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
