# ECS 170 AI Project: Deep Learning for Meta-Statistical Inference
## Project Report can be found at [ECS 170 Project Report.pdf](ECS%20170%20Project%20Report.pdf)
The Project Report contains background information and motivations for the project.

This README contains only documentation for running the code, with little information about the project as a whole.

## Quickstart
* Set hyperparameters in `env.py`. HYPERPARAMETER and VALUES indicate a variable
  hyperparameter which will be altered with each training run.
* Run `train_multiple.py` to train models for each HYPERPARAMETER value you set.
  The model weights will be saved into a new `models` subdirectory.
* Run `performance.py` to measure classification and regression performance.
* _Optionally, choose a model and run `sanity_check.py
  models/your_model_here.pth` to visualize the specified model's classification
  performance._
This repository includes a pretrained model.

## Background
The goal of this code is to train a multitask deep neural network
which can identify the probability distribution underlying a given dataset.
It attempts to extract the mean, standard deviation, and family of distribution.
The general architecture is as follows:
* Input layer of `SAMPLE_SIZE * NUM_DIMENSIONS` units. Feeds to the mean head and
  main shared layers.
* A one-layer head for the mean-regression task for each dimension. Accepts input
  from the input layer and outputs to the loss function.
* A series of hidden layers which are shared by the stddev and classification
  tasks. Accepts input from the input layer and outputs to the stddev and
  classification heads.
* A multilayer head for the stddev-regression task for each dimension. Accepts
  input from the main shared layers and outputs to the loss function.
* A multilayer head for the classification task for each dimension. Accepts
  input from the main shared layers and outputs to the loss function.
  The outputs are packaged back together into a vector which matches that of the
  dataset labels - see below.

## Documentation for runnable scripts
### `train_multiple.py`
Trains multiple models, and saves each one's weights into a new `models`
subdirectory. Each model has hyperparameters set in `env.py`.

This script generates new data if needed: for instance, if a file in the `data`
directory has 1000 training examples, but our current `TRAIN_SIZE` value is 500,
then it will use only the first 500 entries. Conversely, if there is no suitable
dataset in the `data` directory that is big enough (and has matching
dimensionality and `SAMPLE_SIZE`), then this script will generate new data.

Imported by: None.
Imports from: `env.py`, `data_handling.py`, `core.py`, `model.py`, `distributions.py`.
Illegal imports: None.

### `performance.py`
Creates scatter plots of performance of each model which aligns with the
current `CONFIG` in `env.py`

- Classification: accuracy, precision, recall, F1-score
- Regression: mean average error, mean average percentage error, root mean
  squared error, and r2 (coefficient of determination, not correlation coefficient).

The images are saved in the `results` subdirectory.

Imported by: None.
Imports from: `env.py`, `data_handling.py`, `core.py`, `model.py`, `distributions.py`,
`visualizations.py`.
Illegal imports: None.

### `sanity_check.py`
Plots the model's predicted distribution and the ground truth distribution on
the same plot so you can visualize its classification performance.

Takes one command-line argument: the name of the model weights file.

If `env.py`'s `NUM_DIMENSIONS` is 1, then `sanity_check.py` tests the model on
each of the 9 distribution families. If `env.py`'s NUM_DIMENSIONS is 2, then
`sanity_check.py` tests the model on all 81 possible permutation pairs.

### `generate_data.py`
Generates train AND test data based on `CONFIG` in `env.py`,
saves it to `data`.

Imported by: None.
Imports from: `env.py`, `data_handling.py`.
Illegal imports: None.

## Documentation for supporting files
### `core.py`
Where all training and testing for the model occurs.
Learning rate schedule and optimizer live here.

*run_model*: Main call to train or test the model. If in training mode,
sets up an optimizer, creates a loss function object, then loops through
epochs and adjusts model weights, displaying metrics after each epoch.
If in testing mode, runs one epoch and prints more detailed metrics.

Imported by: `train_multiple.py`, `performance.py`.
Imports from: `model.py`, `metrics.py`, `data_handling.py`.
Illegal imports: `env.py`, `train_multiple.py`, `performance.py`,
  `generate_data.py`, `sanity_check.py`.


### `data_handling.py`
Data is generated, packaged into `Dataset` objects, and saved here.
See the section `distributions.py` below for data formatting.

*get_dataset*: Searches the `data` directory for an appropriate dataset
and returns it if found. Otherwise, generates a new set of data.

*make_weights_filename*: Formats a `CONFIG`-style dict into a string
with the necessary data to check compatability with another `CONFIG`-style
dict. This becomes the filename to which the model's weights are saved.

*MyDataset*: Minimalist dataset for use with torch's `Dataloader`.

Imported by: `core.py`, `generate_data.py`.
Imports from: `distributions.py`.
Illegal imports: `env.py`, `core.py`, `metrics.py`, `train_multiple.py`,
  `performance.py`, `generate_data.py`, `sanity_check.py`.

### `metrics.py`
Handles calculation and print-display of the model's performance metrics.

*calculate_metrics*: Calculates the model's performance on a battery of
tests and averages them over the dimensionality of the data.
For classification tasks, these are accuracy, recall, precision, and f1.
For regression tasks, these are r2, MAE, MAPE, and RMSE.
No loss calculations are performed.

*display_metrics*: Displays the loss, classification metrics,
and regression metrics, averaged over the model's full run.

Imported by: `core.py`.
Imports from: `model.py`, `distributions.py`.
Illegal imports: `env.py`, `core.py`, `data_handling.py`, `train_multiple.py`,
`performance.py`, `generate_data.py`, `sanity_check.py`.

### `model.py`
Contains the model architecture and loss function.

*CustomLoss*: How the model calculates prediction error. Uses an average
of cross entropy loss and RMSE (root mean squared error) over each dimension
and classification/regression task, respectively.

*Head*: Class template for the task-specific heads.

*MultiTaskModel*: Constructs the model. There are two sets of shared layers,
the first of which is only the input layer, fixed at size `SAMPLE_SIZE`.
The input layer feeds into both the head for mean regression and
the second set of shared layers. If the network displays evidence of shared
representations, that representation is probably going on in the second
set of shared layers. These layers feed into both the stddev and
classification heads, which are controlled by their respective lists.
Generally, `SHARED_LAYER_SIZES` ought to be short and wide to encourage
more sparse representations, and individual class heads ought to be thinner
and longer to extract features from those representations.

*get_indices*: Helper, calculates the indices for use in our label format.

*stddev_activation*: Helper, implements a sqrt activation for stddev heads.

Imported by: `core.py`, `metrics.py`.
Imports from: None.
Illegal imports: Everything.

### `visualizations.py`
Creates and saves plots of the model's performance over a series of runs.
Also displays heatmaps for activations and weights, and a confusion matrix.

Imported by: `core.py`, `metrics.py`.
Imports from: None.
Illegal imports: Everything.

### `distributions.py`
Contains the parent `Distribution` class and all of its nine children,
as well as a dict which maps strings to class constructor functions.
When constructed, each child class generates its own mean and
standard deviation, and uses those to calculate its standard parameters.
These parameters are used in each of the child classes two main methods,
`rng` and `pdf`. The former returns a random sample of `SAMPLE_SIZE`
many points from the distribution. The latter returns the distribution's
probability density function, for use in visualization.

Each child class also has a `get_label` method for data generation.
The label data is organized in the same way as the output layer of the
neural network: the first `NUM_DISTS` entries are a one-hot vector
indicating which distribution family it came from, and the last two
entries aremean and standard deviation.

Each piece of training or test data is a tuple, where the first entry
is an output from `rng`, and the second entry is an output from `get_label`.

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
Imported by: `data_handling.py`, `metrics.py`, `performance.py`,
  `train_multiple.py`.
Imports from: None.
Illegal imports: Everything.

# Resources
* https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
* https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
* https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
