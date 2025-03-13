import os
import math
from rich import print
import copy
import torch
import numpy as np
import torch.nn.functional as F

from volsurfs_py.models.mlp import MLP
from volsurfs_py.models.lipshitz_mlp import LipshitzMLP
from volsurfs_py.utils.encoder import get_encoder
from volsurfs_py.encodings.sphericalharmonics import SHEncoder


class ColorSH(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        mlp_layers_dims,
        pos_encoder_type,
        out_channels=3,
        sh_deg=3,
        geom_feat_dep=False,
        normal_dep=False,
        in_geom_feat_size=0,
        nr_iters_for_c2f=0,
        bb_sides=2.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.sh_deg = sh_deg
        self.mlp_layers_dims = copy.deepcopy(mlp_layers_dims)
        self.nr_coeffs = (self.sh_deg + 1) ** 2
        self.color_channels = out_channels
        self.out_channels = self.nr_coeffs * out_channels
        self.pos_encoder_type = pos_encoder_type
        self.pos_dep = True
        self.normal_dep = normal_dep
        self.geom_feat_dep = geom_feat_dep
        self.in_geom_feat_size = in_geom_feat_size

        # bounding box
        self.bb_sides = bb_sides
        if isinstance(self.bb_sides, float):
            self.bb_sides = np.array([self.bb_sides] * in_channels)
        if isinstance(self.bb_sides, np.ndarray):
            self.bb_sides = torch.tensor(self.bb_sides, dtype=torch.float32)
        self.bb_sides = self.bb_sides.to("cuda")

        # create input encodings

        nr_levels = 24
        self.pos_encoder = get_encoder(
            self.pos_encoder_type,
            input_dim=in_channels,
            nr_levels=nr_levels,
            nr_iters_for_c2f=nr_iters_for_c2f,
            multires=6,  # used in frequency encoding
            points_scaling=self.bb_sides,
        )
        self.pos_encoder_output_dims = self.pos_encoder.output_dim

        # calculate input channels for MLP
        mlp_in_channels = 0
        if self.pos_dep:
            mlp_in_channels += self.pos_encoder_output_dims
        if self.normal_dep:
            mlp_in_channels += 3
        if self.geom_feat_dep:
            mlp_in_channels += in_geom_feat_size

        self.mlp = MLP(
            mlp_in_channels,
            self.mlp_layers_dims + [self.out_channels],
            last_layer_linear=True,
        )

        # final output activation
        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self,
        points,
        samples_dirs=None,
        normals=None,  # make sure they are normalized first
        geom_feat=None,
        iter_nr=None,
    ):
        """forward pass"""

        # network is position dependent, and normal dependent

        point_features, _ = self.pos_encoder(points, iter_nr=iter_nr)

        data = point_features

        if self.normal_dep:
            if normals is None:
                print(
                    "\n[bold red]ERROR[/bold red]: normals are required for normal dependent model"
                )
                exit(1)
            data = torch.cat([data, normals], 1)

        if self.geom_feat_dep and self.in_geom_feat_size > 0:
            # geometric features
            if geom_feat is None:
                print("\n[bold red]ERROR[/bold red]: geom_feat is required")
                exit(1)
            data = torch.cat([data, geom_feat], 1)

        # run inference (returns sh coeffs)
        pred = self.mlp(data)

        # something like this (?)
        # pred = torch.sign(pred) * self.sigmoid(pred)
        # pred *= 10.0

        # sh_coeffs = pred.reshape(-1, self.color_channels, self.nr_coeffs)
        # raw_output = SHEncoder.eval(sh_coeffs, samples_dirs, degree=self.sh_deg)
        # output = self.sigmoid(raw_output)

        # exit()

        if samples_dirs is None:
            # return pred sh coeffs
            return pred
        else:
            # return pred color

            sh_coeffs = pred.reshape(-1, self.color_channels, self.nr_coeffs)
            raw_output = SHEncoder.eval(sh_coeffs, samples_dirs, degree=self.sh_deg)

            # TODO: color calibration module
            # if model_colorcal is not None:
            #     x = model_colorcal.calib_RGB_samples_packed(
            #         x, img_indices, ray_start_end_idx
            #     )

            output = self.sigmoid(raw_output)

        return output

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
