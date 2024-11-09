# ECS 170 AI Project
Our current neural net is a prototype that identifies the distribution family,
mean, and standard deviation from which some samples were drawn. Trained with
hyperparameters saved in `env.py`, the model's weights are uploaded here as
`model_weights.pth`. (Except the model weights are super outdated right now.)

The architecture is as follows:
* Input layer of `SAMPLE_SIZE=30` units
* Hidden layer of `SAMPLE_SIZE=30` units
* Hidden layer of 30 units, using `ReLU` activation function
* Hidden layer of 30 units, using `ReLU` activation function
* Output layer of 11 * `NUM_DIMENSIONS=1` units (each of `NUM_DIMENSIONS`
  dimensions has a probability vector indicating confidence that it's each of 9
  distribution families, concatenated with the network's guesses for that
  dimension's mean and standard deviation)

Next steps: take stock of current performance, generate some nice pretty graphs,
move on to multivariate data analysis, move on to more types of distributions. 

## Main useful files
### analyze_performance.py
`analyze_performance.py` runs the model saved in `model_weights.pth` on newly
generated data 20 times. It outputs the average loss over all 20 runs at the
end. Run this file to test the current model's performance.

### train.py
`train.py` trains a new model based on the hyperparameters set in `env.py`. It
generates fresh training data and trains the model using cross-validation (we
might get rid of cross-validation later, though, because we have theoretically
infinite quantities of fresh data). The finished model's connection weights are
saved in `model_weights.pth`. Run this file to train a new model.

## Documentation for supporting files
### loss_fn.py
Provides a custom loss function that appropriately deals with both categorical
data (distribution family) and numerical data (mean, standard deviation). The
custom loss function is set in `env.py` and called during training/testing.

In the loss function, we call `softmax()` on each one-hot guess individually to
convert it into what looks like a probability measure. This gets the loss
function working. However, since this post-processing is happening only in the
loss function and not in the model itself, if we ever want to directly read the
outputs of the model, then we have to also call `softmax()` there too.
We tried doing `softmax()` in `train.py`, but
PyTorch said it couldn't figure out what the loss function's `backward()` should
be. (For context, the gradient function `backward()` (which is the partial
derivative wrt output), is usually calculated automatically.)
We may try to put this `softmax()` thing at the end of the model itself, perhaps
as some sort of pseudo–activation function processing the model's output layer. 


### generate_data.py
`generate_data.py` provides a wrapper around functions from `distributions.py`
for generating distributions from various families. We might remove
`generate_data`.

Each entry in the returned list is an ordered tuple whose first entry is a list
of data points and whose second entry is a list of label data. The label data is
organized in the same way as the output layer of the neural network: the first
`NUM_DISTRIBUTIONS=9` entries are a one-hot vector indicating which distribution
family it came from, and the last two entries are mean and standard deviation.

For instance, this dataset has 9 entries, one example per distribution family.
The first entry in the dataset indicates that the points `[1.29251874e-05,
1.67349174e-56, 1.21143669e-12, 6.01269753e-23, 1.00459315e-02]` were drawn from
a beta distribution with `mean=0.09957` and `stddev=0.0873`.
```
[(array([1.29251874e-05, 1.67349174e-56, 1.21143669e-12, 6.01269753e-23,
       1.00459315e-02]),
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0.09957346966779268, 0.08731748466386675]),
 (array([3.99146125, 9.9012487 , 1.97723887, 3.49259786, 1.16247772]),
  [0, 1, 0, 0, 0, 0, 0, 0, 0, 4.673371077920807, 3.7975365401682444]),
 (array([ 8.08308912,  0.47715482,  2.41761488,  4.64226987, -0.71095202]),
  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0.6764766143216339, 3.6020288006537338]),
 (array([ -3.22296815, -14.72603958, -20.17003093,   0.57810446,
        -4.31809143]),
  [0, 0, 0, 1, 0, 0, 0, 0, 0, -6.039077657069879, 3.4819606824119886]),
 (array([2.01111439, 0.76294418, 2.01394174, 2.09013958, 1.23978003]),
  [0, 0, 0, 0, 1, 0, 0, 0, 0, 1.725394342447606, 0.335232733648692]),
 (array([ 3.51561654,  3.78890991, 16.36308546,  2.7305376 ,  2.89570909]),
  [0, 0, 0, 0, 0, 1, 0, 0, 0, 4.499069251794211, 2.7070011250709993]),
 (array([1.64160123, 1.18052268, 9.8111385 , 8.82923718, 9.32143734]),
  [0, 0, 0, 0, 0, 0, 1, 0, 0, 8.114733752479781, 4.048134521030961]),
 (array([0.44524952, 0.78114124, 0.64880018, 0.82160327, 0.54251034]),
  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0.5981106185754592, 0.6473892756123549]),
 (array([ 5.87227715, 11.89298907,  5.25918688, 10.20213871, 10.24144629]),
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 8.531292456133006, 3.6040852489596937])]
```

### performance_plot.py
Generates a graph of some hardcoded data from the performance analysis section
below. However, this needs to be standardized and updated.

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
