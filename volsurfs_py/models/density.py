import torch
import os
import copy
import numpy as np

from volsurfs_py.activations.truncated_exp import TruncatedExp
from volsurfs_py.models.mlp import MLP
from volsurfs_py.utils.encoder import get_encoder


class Density(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        mlp_layers_dims,
        encoding_type,
        out_channels=1,
        geom_feat_size=32,
        nr_iters_for_c2f=0,
        bb_sides=2.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mlp_layers_dims = copy.deepcopy(mlp_layers_dims)
        self.out_channels = out_channels + geom_feat_size
        self.encoding_type = encoding_type
        # self.out_of_bounds_value = 0.0

        # bounding box
        self.bb_sides = bb_sides
        if isinstance(self.bb_sides, float):
            self.bb_sides = np.array([self.bb_sides] * in_channels)
        if isinstance(self.bb_sides, np.ndarray):
            self.bb_sides = torch.tensor(self.bb_sides, dtype=torch.float32)
        self.bb_sides = self.bb_sides.to("cuda")

        # create encoding
        nr_levels = 24
        self.pos_encoder = get_encoder(
            self.encoding_type,
            input_dim=in_channels,
            nr_levels=nr_levels,
            nr_iters_for_c2f=nr_iters_for_c2f,
            multires=6,  # used in frequency encoding
            bb_sides=self.bb_sides,
        )
        self.encoding_output_dims = self.pos_encoder.output_dim

        # create mlp
        self.mlp = MLP(
            in_channels=self.encoding_output_dims,
            nr_out_channels_per_layer=self.mlp_layers_dims + [self.out_channels],
            last_layer_linear=True,
        )

        self.softplus = torch.nn.Softplus()
        # self.truncated_exp = TruncatedExp(threshold=10.0)
        # self.trunc_exp = TruncExp()

    def forward(self, points, iter_nr=None):
        assert points.shape[1] == self.in_channels, "points should be N x in_channels"
        # assert (torch.abs(points) <= 1.0).any(), "points should be in range [-1,1]"

        point_features, points_out_of_bounds = self.pos_encoder(points, iter_nr=iter_nr)

        density_and_feat = self.mlp(point_features)

        # set out of bounds values (points outside bounding box)
        # density_and_feat[points_out_of_bounds, 0] = self.out_of_bounds_value
        # density_and_feat[points_out_of_bounds, 1:] = 0.0

        if self.out_channels != 1:
            # separate density and geom_feat
            density = density_and_feat[:, 0:1]
            geom_feat = density_and_feat[:, 1:]
        else:
            # only density
            density = density_and_feat
            geom_feat = None

        # activate
        # density = self.truncated_exp(density)  # from instant-ngp
        density = self.softplus(density)  # similar to mipnerf

        return density, geom_feat

    # TODO: get gradient

    def save(self, save_path, override_name=None):
        """save checkpoint"""
        model_name = self.__class__.__name__.lower()
        if override_name is not None:
            model_name = override_name
        torch.save(
            self.state_dict(),
            os.path.join(save_path, f"{model_name}.pt"),
        )
        return save_path
