# ECS 170 AI Project: Deep Learning for Meta-Statistical Inference
**Quickstart**:
* _(Remove the `data` and `models` directories in case there are outdated
  files)_
* In `env.py`, set HYPERPARAMETER to whatever hyperparameter you want to change,
  and populate VALUES with the values you want HYPERPARAMETER to take on.
* Run `train_multiple.py` to train models for each HYPERPARAMETER value you set.
  The model weights will be saved into a new `models` subdirectory.
* Then run `performance.py` to measure classification and regression performance
  of each model. The scatter-plots will be saved as pngs into the `results`
  subdirectory.
* _Optionally, choose a model and run `sanity_check.py
  models/your_model_here.pth` to visualize the specified model's classification
  performance._

**Background**: These neural nets identify the distribution family, mean, and
standard deviation from which data points are drawn. The general architecture is
as follows:
* Input layer of `SAMPLE_SIZE` units
* Hidden layer of `SAMPLE_SIZE` units
* 3 hidden layers of 30 units each, using `ReLU` activation function
* Output layer of 11 * `NUM_DIMENSIONS=1` units (each of `NUM_DIMENSIONS`
  dimensions has a probability vector for distribution family, concatenated with
  estimated mean and standard deviation)

Next steps:
- [x] Distinguish between 9 different distribution families
- [x] Plot single-variable regression performance (MAE, MAPE, R^2)
- [x] Sanity-check (visualize) the model's classification performance on
  single-variable data: Plot the sample data points, the model's predicted
  distribution, and the ground truth distribution (2D visualizations)
- [x] Generate multi-dimensional data, and add multi-dimensional training
  support to neural network code
- [x] Sanity-check (visualize) the model's classification performance on
  2-dimensional distributions (3D visualizations)
- [x] Plot multi-variable classification and regression performance (accuracy,
  precision, recall, F1, MAE, MAPE, R^2)
- [x] Multidimensional hyperparameter tuning: Find best SAMPLE_SIZE (30, 33, 36,
  39, 42)
- [ ] Multidimensional hyperparameter tuning: Find best BATCH_SIZE (in progress)
- [ ] Multidimensional hyperparameter tuning: Find best EPOCHS
- [ ] Multidimensional hyperparameter tuning: Find best TRAIN_SIZE (less
  priority because, presumably, biggest is best)
- [ ] Train the model separately on different runs, and save only the best one?
  (probably not implementing this)

## Main useful files
### `train_multiple.py`
Trains multiple models, and saves each one's weights into a new `models`
subdirectory. Each model has hyperparameters set in `env.py`.

This script generates new data if needed: for instance, if a file in the `data`
directory has 1000 training examples, but our current TRAIN_SIZE value is 500,
then it will use only the first 500 entries. Conversely, if there is no suitable
dataset in the `data` directory that is big enough (and has matching
dimensionality and SAMPLE_SIZE), then this script will generate new data.

### `performance.py`
Creates scatter plots of performance of each model in `models`.

- Classification: accuracy, precision, recall, F1-score
- Regression: mean average error, mean average percentage error, and R^2
  correlation coefficient

The images are saved in the `results` subdirectory.

### `sanity_check.py`
Plots the model's predicted distribution and the ground truth distribution on
the same plot so you can visualize its classification performance.

Takes one command-line argument: the name of the model weights file.

If `env.py`'s NUM_DIMENSIONS is 1, then `sanity_check.py` tests the model on
each of the 9 distribution families. If `env.py`'s NUM_DIMENSIONS is 2, then
`sanity_check.py` tests the model on all 81 possible permutation pairs.

## Documentation for supporting files
### `custom_functions.py`
Provides a custom loss function, custom multitask layer, and distribution
classes for each of the 9 distributions.

*Multitask*: Custom multitask layer for our neural net ("multitask" because it
performs both classification and regression).

*CustomLossFunction()*: Deals with both categorical data (distribution family)
and numerical data (mean and standard deviation).

### `build_model.py`
Defines neural network architecture. Uses custom multitask layer defined in
`custom_functions.py`.

### `generate_data.py`
`generate_data.py` provides a wrapper around functions from `distributions.py`
for generating distributions from various families. When run as a standalone
script, it generates training and test data and puts it into the `data`
directory.

Each entry in the list returned by `generate_data()` is an ordered tuple whose
first entry is a list of data points and whose second entry is a list of label
data. The label data is organized in the same way as the output layer of the
neural network: the first `NUM_DISTRIBUTIONS=9` entries are a one-hot vector
indicating which distribution family it came from, and the last two entries are
mean and standard deviation.

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

# Resources
* https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
* https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
* https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
