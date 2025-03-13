import os
import torch

from volsurfs_py.models.mlp import MLP
from volsurfs_py.utils.encoder import get_encoder

# from volsurfs_py.utils.common_utils import map_range_val
import permutohedral_encoding as permuto_enc


class NerfHash(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        pos_encoder_type,
        dir_encoder_type,
        nr_iters_for_c2f=0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.pos_encoder_type = pos_encoder_type
        self.dir_encoder_type = dir_encoder_type

        # create encoding
        nr_levels = 24
        self.pos_encoder = get_encoder(
            self.pos_encoder_type,
            input_dim=in_channels,
            nr_levels=nr_levels,
            nr_iters_for_c2f=nr_iters_for_c2f,
            multires=6,  # used in frequency encoding,
            bb_sides=2.0,
        )
        self.pos_encoder_output_dims = self.pos_encoder.output_dim

        self.dir_encoder = get_encoder(self.dir_encoder_type, input_dim=3, degree=3)
        self.dir_encoder_output_dims = self.dir_encoder.output_dim

        self.nr_feat_for_rgb = 64

        self.mlp_feat_and_density = MLP(
            self.pos_encoder_output_dims,
            [64, 64, 64, self.nr_feat_for_rgb + 1],
            last_layer_linear=True,
        )

        self.mlp_rgb = MLP(
            self.nr_feat_for_rgb + self.dir_encoder_output_dims,
            [64, 64, 3],
            last_layer_linear=True,
        )

        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()
        self.gelu = torch.nn.GELU()

    def forward(self, samples_3d, samples_dirs, iter_nr):
        # make sure everything is ok
        assert (
            samples_3d.shape[1] == self.in_channels
        ), "points should be N x in_channels"

        points = samples_3d
        point_features, _ = self.pos_encoder(points, iter_nr=iter_nr)

        # dirs encoded with spherical harmonics
        with torch.set_grad_enabled(False):
            samples_dirs_enc = self.dir_encoder(samples_dirs)

        # predict density without using directions
        feat_and_density = self.mlp_feat_and_density(point_features)
        density = feat_and_density[:, 0:1]
        feat_rgb = feat_and_density[:, 1 : self.nr_feat_for_rgb + 1]

        # predict rgb using directions of ray
        feat_rgb_with_dirs = torch.cat([self.gelu(feat_rgb), samples_dirs_enc], 1)
        rgb = self.mlp_rgb(feat_rgb_with_dirs)

        # activate
        density = self.softplus(density)  # similar to mipnerf

        # TODO: color calibration module
        # if model_colorcal is not None:
        #     rgb = model_colorcal.calib_RGB_samples_packed(
        #         rgb, img_indices, ray_start_end_idx
        #     )

        rgb = self.sigmoid(rgb)

        return rgb, density

    def get_only_density(self, ray_samples, iter_nr):
        # given the rays, create points
        points = ray_samples.view(-1, ray_samples.shape[-1])
        point_features, _ = self.pos_encoder(points, iter_nr=iter_nr)

        # predict density without using directions
        feat_and_density = self.mlp_feat_and_density(point_features)
        density = feat_and_density[:, 0:1]
        density = self.softplus(density)  # similar to mipnerf

        return density

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
