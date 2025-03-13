import os
import math
import torch
import copy
import numpy as np
import torch.nn.functional as F

from volsurfs_py.models.mlp import MLP
from volsurfs_py.models.lipshitz_mlp import LipshitzMLP
from volsurfs_py.utils.encoder import get_encoder


class RGB(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        mlp_layers_dims,
        pos_encoder_type,
        dir_encoder_type,
        out_channels=3,
        pos_dep=True,
        view_dep=True,
        geom_feat_dep=False,
        normal_dep=False,
        sh_deg=5,
        in_geom_feat_size=32,
        nr_iters_for_c2f=0,
        use_lipshitz_mlp=False,
        bb_sides=2.0,
    ):
        super().__init__()

        assert (pos_dep and in_channels > 0) or (
            not pos_dep and in_channels == 0
        ), "pos_dep and in_channels must be consistent"

        self.in_channels = in_channels
        self.mlp_layers_dims = copy.deepcopy(mlp_layers_dims)
        self.out_channels = out_channels
        self.pos_encoder_type = pos_encoder_type
        self.dir_encoder_type = dir_encoder_type
        self.sh_deg = sh_deg
        self.pos_dep = pos_dep
        self.view_dep = view_dep
        self.normal_dep = normal_dep
        self.geom_feat_dep = geom_feat_dep
        self.in_geom_feat_size = in_geom_feat_size
        self.use_lipshitz_mlp = use_lipshitz_mlp

        # bounding box
        self.bb_sides = bb_sides
        if isinstance(self.bb_sides, float):
            self.bb_sides = np.array([self.bb_sides] * in_channels)
        if isinstance(self.bb_sides, np.ndarray):
            self.bb_sides = torch.tensor(self.bb_sides, dtype=torch.float32)
        self.bb_sides = self.bb_sides.to("cuda")

        # create encodings

        if self.pos_dep:
            nr_levels = 24
            self.pos_encoder = get_encoder(
                self.pos_encoder_type,
                input_dim=in_channels,
                nr_levels=nr_levels,
                nr_iters_for_c2f=nr_iters_for_c2f,
                multires=6,  # used in frequency encoding
                bb_sides=self.bb_sides,
            )
            self.pos_encoder_output_dims = self.pos_encoder.output_dim

        if view_dep:
            self.dir_encoder = get_encoder(
                self.dir_encoder_type, input_dim=3, degree=self.sh_deg
            )
            self.dir_encoder_output_dims = self.dir_encoder.output_dim

        # calculate input channels for MLP
        mlp_in_channels = 0
        if self.pos_dep:
            mlp_in_channels += self.pos_encoder_output_dims
        if self.view_dep:
            mlp_in_channels += self.dir_encoder_output_dims
        if self.normal_dep:
            mlp_in_channels += 3
        if self.geom_feat_dep:
            mlp_in_channels += in_geom_feat_size

        if self.use_lipshitz_mlp:
            self.mlp = LipshitzMLP(
                mlp_in_channels,
                self.mlp_layers_dims + [self.out_channels],
                last_layer_linear=True,
            )
        else:
            self.mlp = MLP(
                mlp_in_channels,
                self.mlp_layers_dims + [self.out_channels],
                last_layer_linear=True,
            )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self,
        points=None,
        samples_dirs=None,
        normals=None,  # make sure they are normalized first
        iter_nr=None,
        geom_feat=None,
    ):
        """forward pass"""

        # concat additional inputs
        data = torch.empty(0)

        if self.pos_dep:
            point_features, _ = self.pos_encoder(points, iter_nr=iter_nr)
            data = torch.cat([data, point_features], 1)

        if self.view_dep:
            # dirs encoding
            with torch.set_grad_enabled(False):
                samples_dirs_enc = self.dir_encoder(samples_dirs, iter_nr=iter_nr)
            data = torch.cat([data, samples_dirs_enc], 1)

        if self.normal_dep:
            # normals
            data = torch.cat([data, normals], 1)

        if self.geom_feat_dep and self.in_geom_feat_size > 0:
            # geometric features
            if geom_feat is None:
                print("\n[bold red]ERROR[/bold red]: geom_feat is required")
                exit(1)
            data = torch.cat([data, geom_feat], 1)

        # run inference
        result = self.mlp(data)

        # TODO: color calibration module
        # if model_colorcal is not None:
        #     x = model_colorcal.calib_RGB_samples_packed(
        #         x, img_indices, ray_start_end_idx
        #     )

        result = self.sigmoid(result)

        return result

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

    def reset(self):
        """reset models"""
        self.mlp.reset()
        if self.pos_dep:
            self.pos_encoder.reset()
        if self.view_dep:
            self.dir_encoder.reset()
