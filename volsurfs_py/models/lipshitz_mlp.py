import torch

from volsurfs_py.utils.common import leaky_relu_init


# from https://arxiv.org/pdf/2202.08345.pdf
class LipshitzMLP(torch.nn.Module):
    def __init__(self, in_channels, nr_out_channels_per_layer, last_layer_linear):
        super().__init__()

        self.last_layer_linear = last_layer_linear

        self.layers = torch.nn.ModuleList()
        for i in range(len(nr_out_channels_per_layer)):
            cur_out_channels = nr_out_channels_per_layer[i]
            self.layers.append(torch.nn.Linear(in_channels, cur_out_channels))
            in_channels = cur_out_channels

        self.reset()

    def normalization(self, w, softplus_ci):
        absrowsum = torch.sum(torch.abs(w), dim=1)
        # scale = torch.minimum(torch.tensor(1.0), softplus_ci/absrowsum)
        # this is faster than the previous line since we don't constantly recreate a torch.tensor(1.0)
        scale = softplus_ci / absrowsum
        scale = torch.clamp(scale, max=1.0)
        return w * scale[:, None]

    def lipshitz_bound_full(self):
        lipshitz_full = 1
        for i in range(len(self.layers)):
            lipshitz_full = lipshitz_full * torch.nn.functional.softplus(
                self.lipshitz_bound_per_layer[i]
            )

        return lipshitz_full

    def forward(self, x):
        for i in range(len(self.layers)):
            weight = self.weights_per_layer[i]
            bias = self.biases_per_layer[i]

            weight = self.normalization(
                weight, torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i])
            )

            x = torch.nn.functional.linear(x, weight, bias)

            is_last_layer = i == (len(self.layers) - 1)

            if is_last_layer and self.last_layer_linear:
                pass
            else:
                x = torch.nn.functional.gelu(x)

        return x

    def reset(self):
        """reset the network"""
        # TODO: test reset
        for i, layer in enumerate(self.layers):
            negative_slope = 0.0
            if i == len(self.layers) - 1 and self.last_layer_linear:
                negative_slope = 1.0

            # initialize weights
            leaky_relu_init(layer, negative_slope=negative_slope)

        # we make each weight separately because we want to add the normalize to it
        self.weights_per_layer = torch.nn.ParameterList()
        self.biases_per_layer = torch.nn.ParameterList()
        for i in range(len(self.layers)):
            self.weights_per_layer.append(self.layers[i].weight)
            self.biases_per_layer.append(self.layers[i].bias)

        self.lipshitz_bound_per_layer = torch.nn.ParameterList()
        for i in range(len(self.layers)):
            max_w = torch.max(torch.sum(torch.abs(self.weights_per_layer[i]), dim=1))
            # we actually make the initial value quite large because we don't want at the beggining to hinder the rgb model in any way. A large c means that the scale will be 1
            c = torch.nn.Parameter(torch.ones((1)) * max_w * 2)
            self.lipshitz_bound_per_layer.append(c)
