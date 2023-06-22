import torch
import torch.nn as nn
from torch.nn.utils import prune


class TwoLayerAdditiveModel(nn.Module):
    """
    A simple two-layer additive model with sigmoid activation functions.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        """
        Initialize the model with the input, output, and hidden dimensions.
        :param input_dim: Input dimension
        :param output_dim: Output dimension
        :param hidden_dim: Hidden dimension
        """
        super(TwoLayerAdditiveModel, self).__init__()

        # Two linear layers
        self.layer_1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.layer_2 = nn.Linear(hidden_dim, output_dim, bias=True)

        # initialize weights to be random
        self.layer_1.weight.data = torch.randn(hidden_dim, input_dim)
        self.layer_2.weight.data = torch.randn(output_dim, hidden_dim)

        # initialize biases to be zero
        self.layer_1.bias.data = torch.zeros(hidden_dim)
        self.layer_2.bias.data = torch.randn(output_dim)

    def prune_weights(self, mask: torch.Tensor):
        """
        Fix the weights in the mask to 0 for the first layer.
        :param mask: Tensor of 0s and 1s of size (hidden_dim, input_dim)
        """
        prune.custom_from_mask(self.layer_1, name='weight', mask=mask)

    def sep_train_layer(self, layer: int = 0):
        """
        Option to train only one layer at a time.
        :param layer: The layer to train (1 or 2)
        """
        if layer == 0:
            for param in self.layer_1.parameters():
                param.requires_grad = True
            for param in self.layer_2.parameters():
                param.requires_grad = True
        elif layer == 1:
            for param in self.layer_1.parameters():
                param.requires_grad = True
            for param in self.layer_2.parameters():
                param.requires_grad = False
        elif layer == 2:
            for param in self.layer_1.parameters():
                param.requires_grad = False
            for param in self.layer_2.parameters():
                param.requires_grad = True
        else:
            raise ValueError("layer must be 1 or 2")

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the model.
        :param x: Input of the model
        :return: Predicted output of the model in [0,1]
        """
        x = self.layer_1(x)
        x = torch.sigmoid(x)
        x = self.layer_2(x)
        return torch.sigmoid(x)


def train_model(model, input: torch.Tensor, labels: torch.Tensor, epochs: int, optimiser, loss_func, sep_layers: bool = False, verbose: bool = False):
    """
    Train the model.
    :param model: the model to be trained
    :param input: input data
    :param labels: labels
    :param epochs: number of epochs to train for
    :param optimiser: optimiser to train with
    :param loss_func: loss function
    :param sep_layers: whether to train one layer at a time
    :param verbose: whether to print training information
    :return: trained model
    """

    # Intervals to print training information
    interval = epochs / 10
    interval = int(interval)
    interval = int(round(interval, -2))

    # Separate training of layers setup
    if sep_layers:
        layers = [1, 2]
    else:
        layers = [0]

    # Loop through each layer
    for layer in layers:
        model.sep_train_layer(layer)
        # Loop through each epoch
        for epoch in range(epochs):
            optimiser.zero_grad()
            output = model(input)
            loss = loss_func(output, labels)
            loss.backward()
            optimiser.step()
            if verbose and sep_layers and (epoch % interval == 0):
                print("Layer: {} | Epoch: {} | Loss: {}".format(layer, epoch, loss.item()))
            elif verbose and (epoch % interval == 0):
                print("Epoch: {} | Loss: {}".format(epoch, loss.item()))

    return model
