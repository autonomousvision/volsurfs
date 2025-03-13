import torch
import numpy as np
from copy import deepcopy
from volsurfs_py.utils.common import leaky_relu_init
from volsurfs_py.utils.common import apply_weight_init_fn


class MLP(torch.nn.Module):
    """basic MLP with GELU activation"""

    def __init__(
        self, in_channels, nr_out_channels_per_layer, last_layer_linear, bias=True
    ):
        super().__init__()

        self.last_layer_linear = last_layer_linear
        self.in_channels = in_channels
        self.nr_out_channels_per_layer = nr_out_channels_per_layer
        self.bias = bias

        in_channels_ = self.in_channels

        modules = []
        for i, _ in enumerate(self.nr_out_channels_per_layer):
            cur_out_channels = self.nr_out_channels_per_layer[i]
            # linear layer
            modules.append(
                torch.nn.Linear(in_channels_, cur_out_channels, bias=self.bias)
            )

            # gelu activation
            is_last_layer = i == (len(self.nr_out_channels_per_layer) - 1)
            if is_last_layer and self.last_layer_linear:
                continue

            modules.append(torch.nn.GELU())

            in_channels_ = cur_out_channels

        self.layers = torch.nn.Sequential(*modules)

        # # initialize weights
        # apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        # if self.last_layer_linear:
        #     leaky_relu_init(self.layers[-1], negative_slope=1.0)

        self.reset()

    def forward(self, x):
        """forward pass"""
        y = self.layers(x)
        return y

    def reset(self):
        """reset the network"""
        # # TODO: try reset the weights
        # layers_copies = []
        # for layer in self.layers:
        #     layers_copies.append(deepcopy(layer))
        # self.layers = torch.nn.Sequential(*layers_copies)

        # for params in self.layers.parameters():
        #    print(params.shape)

        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                in_channels = layer.in_features
                out_channels = layer.out_features
                layer.__init__(in_channels, out_channels, bias=self.bias)
