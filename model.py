import torch
from torch.nn import Module, ModuleList, Linear, ReLU, CrossEntropyLoss, MSELoss
from torch.nn.init import constant_, kaiming_normal_

### No importing any custom files.


def get_indices(dim, num_dists, dists=False, mean=False, stddev=False, support=False):
    """
    Calculates the indices for use in our label format. Ex:
    x = [1,0,0,0,0,0,0,0,0,3.1,0.6,0,1,0,0,0,0,0,0,0,4.5,1.2]
    x[get_indices(dists=True,dim=1)] -> [1,0,0,0,0,0,0,0,0]
    x[get_indices(mean=True,stddev=True,dims=2)] -> [4.5,1.2]

    Args:
        dists (bool): Whether to get the indices for the onehot portion.
        mean (bool): Whether to get the indices for the mean.
        stddev (bool): Whether to get the indices for the standard deviation.
        dim (int): Which dimension to check at.

    Returns:
        list: A list of ints to be used as indices for a label vector.
    """
    out = []
    # length of 1 dimension = num_dists + 5
    if dists:
        dists_range = range(
            (dim * (num_dists + 5)) - (num_dists + 5), (dim * (num_dists + 5)) - 5
        )
        out += dists_range
    if mean:
        out.append(dim * (num_dists + 5) - 5)
    if stddev:
        out.append(dim * (num_dists + 5) - 4)
    if support:
        out += range((dim * (num_dists + 5)) - 3, (dim * (num_dists + 5)))
    return out


class CustomLoss(Module):
    """
    Loss function for the multi-headed model.
    """

    def __init__(
        self,
        use_mean=True,
        use_stddev=True,
        use_dists=True,
        use_support=True,
        num_dims=-1,
        num_dists=9,
    ):
        """
        Constructor for loss function class.

        Args:
            use_mean (bool): Whether to use the predicted mean in loss calculations.
            use_stddev (bool): Whether to use the predicted stddev in loss calculations.
            use_dists (bool): Whether to use the predicted class in loss calculations.
            num_dimensions (int): The dimensionality of the data in use.

        Returns:
            CustomLoss object.
        """
        super(CustomLoss, self).__init__()
        self.use_mean = use_mean
        self.use_stddev = use_stddev
        self.use_dists = use_dists
        self.use_support = use_support
        self.num_dims = num_dims
        self.num_dists = num_dists
        self.num_params_in_use = int(use_mean + use_stddev + use_dists + use_support)
        self.dist_loss = CrossEntropyLoss(
            label_smoothing=0.15,
        )
        self.support_loss = CrossEntropyLoss(
            # Use normalized inverse F1 scores to approximate how hard
            # different classes are to identify.
            weight=torch.tensor([0.351, 0.321, 0.328]),
        )
        self.regression_loss = MSELoss()

    def forward(self, pred, y):
        """
        Calculates the model's prediction error. Assuming all bools in init were true,
        uses CrossEntropyLoss for the classification task(s), RMSE for the regression
        tasks, and returns the average over all dimensions.

        Args:
            pred (tensor of floats): The model's prediction.
            y (tensor): Ground truth values.

        Returns:
            float: A measure of 'distance' between pred and y.
        """
        loss = 0
        for dim in range(self.num_dims):
            dists_idx = get_indices(dim + 1, self.num_dists, dists=True)
            mean_idx = get_indices(dim + 1, self.num_dists, mean=True)
            stddev_idx = get_indices(dim + 1, self.num_dists, stddev=True)
            support_idx = get_indices(dim + 1, self.num_dists, support=True)

            # Class_targets has shape [batch_size, num_classes], others have shape
            # [batch_size], so need to be sliced differently later.
            class_targets = y[:, dists_idx]
            mean_targets = y[:, mean_idx]
            stddev_targets = y[:, stddev_idx]
            support_targets = y[:, support_idx]

            if self.use_dists:

                loss += self.dist_loss(
                    pred["classification"][:, dim, :],
                    torch.argmax(class_targets, dim=1),
                )
            if self.use_mean:
                loss += torch.sqrt(
                    self.regression_loss(
                        pred["mean"][:, dim].unsqueeze(1), mean_targets
                    )
                )
            if self.use_stddev:
                loss += torch.sqrt(
                    self.regression_loss(
                        pred["stddev"][:, dim].unsqueeze(1), stddev_targets
                    )
                )
            if self.use_support:
                loss += self.support_loss(
                    pred["support"][:, dim, :],
                    torch.argmax(support_targets, dim=1),
                )

        # Average by the number of losses actually used
        return loss / (self.num_params_in_use * self.num_dims)


class StddevActivation(Module):
    """
    Sqrt activation for the stddev to align with real life calculations.
    The derivative of sqrt explodes at 0, so clamp the inputs.
    """

    def __init__(self):
        super(StddevActivation, self).__init__()

    def forward(self, x):
        return torch.sqrt(torch.clamp(torch.abs(x), 1e-6, 1e6))


class Head(Module):
    """
    Arbitrary head module for the multitask network. Receives input from a shared layer
    and outputs directly to the loss function. There is one for each pair of tasks
    and dimensions. There is always one more layer than the length of layer_sizes.
    """

    def __init__(
        self,
        input_dim,
        layer_sizes=[],
        output_size=1,
        activation=ReLU,
        final_activation=None,
    ):
        super(Head, self).__init__()
        self.layers = ModuleList()
        if len(layer_sizes) == 0:
            self.layers.append(Linear(input_dim, output_size))
        else:
            self.layers.append(Linear(input_dim, layer_sizes[0]))
            self.layers.append(activation())
            # "Hidden" layers generated here
            for n in range(len(layer_sizes) - 1):
                self.layers.append(Linear(layer_sizes[n], layer_sizes[n + 1]))
                self.layers.append(activation())

            # Output layer
            self.layers.append(Linear(layer_sizes[-1], output_size))

        if final_activation:
            self.layers.append(final_activation())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MultiTaskModel(Module):
    """
    Network architecture is determined here. There are two sets of shared layers,
    the first of which is only the input layer, fixed at size SAMPLE_SIZE.
    The input layer feeds directly into the head for mean regression and
    the second set of shared layers, which is controlled by SHARED_LAYER_SIZES.
    If the network displays evidence of shared representations, that representation
    is probably going on here. These layers feed into both the stddev and
    classification heads, which are controlled by their respective lists.
    Generally speaking, SHARED_LAYER_SIZES ought to be short and wide to encourage
    more sparse representations, and individual class heads ought to be thinner
    and longer to extract features from those representations.
    """

    def __init__(self, config, architecture, num_classes, activation=ReLU):
        super(MultiTaskModel, self).__init__()
        self.num_dimensions = config["NUM_DIMENSIONS"]
        self.num_classes = num_classes
        self.architecture = architecture
        self.shared_layer_sizes = architecture["SHARED_LAYER_SIZES"]
        self.stddev_head_layer_sizes = architecture["STDDEV_LAYER_SIZES"]
        self.class_head_layer_sizes = architecture["CLASS_LAYER_SIZES"]

        # Shared layers, the bulk of the networks
        self.backbone = ModuleList()
        self.backbone.append(
            Linear(
                config["SAMPLE_SIZE"] * self.num_dimensions, self.shared_layer_sizes[0]
            )
        )
        self.backbone.append(activation())
        for n in range(len(self.shared_layer_sizes) - 1):
            self.backbone.append(
                Linear(self.shared_layer_sizes[n], self.shared_layer_sizes[n + 1])
            )
            self.backbone.append(activation())

        # Make a head of each metric for each dimension.
        self.mean_head_array = ModuleList(
            [Head(self.shared_layer_sizes[-1]) for _ in range(self.num_dimensions)]
        )

        self.stddev_head_array = ModuleList(
            [
                Head(
                    self.shared_layer_sizes[-1],
                    layer_sizes=self.stddev_head_layer_sizes,
                    activation=activation,
                    final_activation=StddevActivation,
                )
                for _ in range(self.num_dimensions)
            ]
        )

        self.class_head_array = ModuleList(
            [
                Head(
                    self.shared_layer_sizes[-1],
                    layer_sizes=self.class_head_layer_sizes,
                    output_size=num_classes,
                    activation=activation,
                )
                for _ in range(self.num_dimensions)
            ]
        )

        self.support_head_array = ModuleList(
            [
                Head(
                    self.shared_layer_sizes[-1],
                    output_size=3,
                    activation=activation,
                )
                for _ in range(self.num_dimensions)
            ]
        )

        # Uses He initialization for optimality with ReLU activations.
        def initializer(module):
            if isinstance(module, Linear):
                kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(initializer)

    def forward(self, x):
        batch_size = x.size(0)

        # This needs to be really fast, so pre-initialize everything possible.
        outputs = {
            "classification": torch.zeros(
                batch_size, self.num_dimensions, self.num_classes
            ),
            "mean": torch.zeros(batch_size, self.num_dimensions),
            "stddev": torch.zeros(batch_size, self.num_dimensions),
            "support": torch.zeros(batch_size, self.num_dimensions, 3),
        }

        # Call through the shared layers
        for layer in self.backbone:
            x = layer(x)

        for n in range(self.num_dimensions):
            # Shape: [batch_size, num_dimensions, num_distributions]
            # For example, [1000, 2, 9] means:
            #   1000 "rows" (one per item in batch), each containing:
            #       a vector of length 2 (one for each dim) containing:
            #           a onehot vector of length 9 (one for each distribution)
            outputs["classification"][:, n, :] = self.class_head_array[n](x)
            outputs["support"][:, n, :] = self.support_head_array[n](x)

            # Shape: [batch_size, num_dimensions]
            # For example, [1000, 2] means:
            #   1000 "rows" (one per item in batch), within each row:
            #       a vector of length 2 (one for each dim) containing means
            # One less tensor-dimension than class since
            #   means are numbers instead of onehot vectors
            outputs["mean"][:, n] = self.mean_head_array[n](x).squeeze(-1)
            outputs["stddev"][:, n] = self.stddev_head_array[n](x).squeeze(-1)

        return outputs
