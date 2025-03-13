import os
import math
import copy
import torch
import numpy as np

from volsurfs_py.models.mlp import MLP
from volsurfs_py.utils.encoder import get_encoder


class SDF(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        mlp_layers_dims,
        encoding_type,
        geom_feat_size=32,
        nr_iters_for_c2f=0,
        pred_adaptive_kernel_size=False,
        bb_sides=2.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mlp_layers_dims = copy.deepcopy(mlp_layers_dims)
        # self.pred_adaptive_kernel_size = pred_adaptive_kernel_size
        self.geom_feat_size = geom_feat_size
        self.encoding_type = encoding_type
        self.out_channels = 1
        # self.out_channels += geom_feat_size
        # if self.pred_adaptive_kernel_size:
        #    self.out_channels += 1
        self.is_training_main_surf = True

        # bounding box
        self.bb_sides = bb_sides
        if isinstance(self.bb_sides, float):
            self.bb_sides = np.array([self.bb_sides] * in_channels)
        if isinstance(self.bb_sides, np.ndarray):
            self.bb_sides = torch.tensor(self.bb_sides, dtype=torch.float32)
        self.bb_sides = self.bb_sides.to("cuda")

        # self.out_of_bounds_value = torch.linalg.norm(self.bb_sides/2)
        # self.out_of_bounds_value = torch.max(self.bb_sides/2)
        # print("out_of_bounds_value", self.out_of_bounds_value)

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
        self.mlp_sdf = MLP(
            self.encoding_output_dims,
            self.mlp_layers_dims + [1 + self.geom_feat_size],
            last_layer_linear=True,
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, iter_nr=None):
        # preconditions
        assert points.shape[1] == self.in_channels, "points should be N x in_channels"
        point_features, points_out_of_bounds = self.pos_encoder(points, iter_nr=iter_nr)
        prediction = self.mlp_sdf(point_features)

        #
        if self.geom_feat_size > 0:
            # separate sdf and geom_feats
            sdf = prediction[:, 0:1]
            geom_feats = prediction[:, 1:]
        else:
            # only sdf
            sdf = prediction
            geom_feats = None

        # no non-linearity

        return sdf, geom_feats

    def main_sdf(self, points, iter_nr=None):
        sdf, geom_feats = self.forward(points, iter_nr)
        return sdf, geom_feats

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
