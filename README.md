# ECS 170 AI Project: Deep Learning for Meta-Statistical Inference
**Quickstart**: Run `generate_data.py` first to generate data and save it into the `data`
subdirectory. Then run `train_multiple.py` to train multiple models on different
values of the hyperparameter set in `env.py` and save their weights into the
`models` subdirectory. Then run `regression_performance.py` to measure the
performance for guessing mean and standard deviation; the scatter-plots will be
saved into the `results` subdirectory.

**Background**: These neural nets identify the distribution family, mean, and standard deviation
from which data points are drawn. Trained with hyperparameters saved in
`env.py`, each model's weights are uploaded here in `models/` subdirectory.

The general architecture is as follows:
* Input layer of `SAMPLE_SIZE` units
* Hidden layer of `SAMPLE_SIZE` units
* 3 hidden layers of 30 units each, using `ReLU` activation function
* Output layer of 11 * `NUM_DIMENSIONS=1` units (each of `NUM_DIMENSIONS`
  dimensions has a probability vector for distribution family, concatenated with
  estimated mean and standard deviation)

Next steps:
- [x] Move on to more types of distributions
- [x] Measure current performance (using some standardized method), generate
  some nice pretty graphs
- [x] Hyperparameter tuning: Find best SAMPLE_SIZE (10)
- [x] Hyperparameter tuning: Find best TRAINING_SIZE (100)
- [ ] **Hyperparameter tuning: Find best BATCH_SIZE**
- [ ] Hyperparameter tuning: Find best EPOCHS
- [ ] Train the model separately on different runs, and save only the best one
  to `model_weights.pth` (currently, it's saving the last run's weights, not the
  best one)
- [ ] Sanity-check the model's performance: For a few different queries, graph
  the samples provided to the model, and overlay both the model's guesses and
  the true distribution family, mean, and standard deviation
- [ ] Move on to multi-dimensional data

## Main useful files
### `train_multiple.py`
Trains multiple models and saves each one's weights into `models` subdirectory.
Each model has hyperparameters set in `env.py`.

### `regression_performance.py`
Measures regression performance of each model in `models` directory wrt mean and
stddev guesses (hence regression only, not classification). Performance is
measured in terms of MAE, MAPE, and R^2. The performance is graphed as scatter
plots in `results/` subdirectory.

## Documentation for supporting files
### `pipeline.py`
Trains a new model using hyperparameters passed in via the `config` parameter.
The model is trained on either data from the `data` directory or on newly
generated data (which it then saves out to the `data` directory to save time
later). The finished model's connection weights are returned, and then
`train_multiple.py` is able to catch this and save it to the `models` directory.

### `custom_loss_fn.py`
Provides a custom loss function that appropriately deals with both categorical
data (distribution family) and numerical data (mean and standard deviation).

In the loss function, we call `softmax()` on each one-hot guess individually to
convert it into what looks like a probability measure. This gets the loss
function working. However, since this post-processing is happening only in the
loss function and not in the model itself, if we ever want to directly read the
outputs of the model, then we have to also call `softmax()` there too.

We tried doing `softmax()` in `train.py`, but PyTorch said it couldn't figure
out what the loss function's `backward()` should be. (For context, the gradient
function `backward()` (which is the partial derivative wrt output), is usually
calculated automatically.) We may try to put this `softmax()` thing at the end
of the model itself, perhaps as some sort of pseudo–activation function
processing the model's output layer.

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
