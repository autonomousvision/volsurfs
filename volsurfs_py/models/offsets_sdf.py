import os
import copy
import torch
import numpy as np
import torch.nn.functional as F

from volsurfs_py.models.mlp import MLP
from volsurfs_py.utils.encoder import get_encoder
from volsurfs_py.utils.fields_utils import get_field_gradients


class OffsetsSDF(torch.nn.Module):

    def __init__(
        self,
        in_channels=2,
        mlp_layers_dims=[32, 32, 32],
        encoding_type="gridhash",
        nr_inner_surfs=1,
        nr_outer_surfs=1,
        geom_feat_size=32,
        min_offset=1e-4,
        nr_iters_for_c2f=0,
        bb_sides=2.0,
        use_per_offset_mlp=True,
        # main_surf_shift=0.0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mlp_layers_dims = copy.deepcopy(mlp_layers_dims)
        self.nr_inner_surfs = nr_inner_surfs
        self.nr_outer_surfs = nr_outer_surfs
        self.nr_surfs = nr_inner_surfs + nr_outer_surfs + 1
        self.geom_feat_size = geom_feat_size
        self.out_channels = self.nr_surfs
        self.encoding_type = encoding_type
        self.is_training_main_surf = True
        self.is_training_offsets = True
        self.use_per_offset_mlp = use_per_offset_mlp
        self.min_offset = min_offset
        self.main_surf_idx = self.nr_inner_surfs
        # self.main_surf_shift = main_surf_shift

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

        # create mlps

        # main surf head
        self.mlp_sdf = MLP(
            self.encoding_output_dims,
            self.mlp_layers_dims + [1 + self.geom_feat_size],
            last_layer_linear=True,
        )

        # one head for each support surface
        if self.use_per_offset_mlp:
            self.mlps_eps = []
            for i in range(self.nr_surfs - 1):
                self.mlps_eps.append(
                    MLP(
                        self.geom_feat_size,
                        [32] + [1],
                        last_layer_linear=True,
                    )
                )
        else:
            self.mlp_eps = MLP(
                self.geom_feat_size,
                [32, 32] + [self.nr_surfs - 1],
                last_layer_linear=True,
            )

    def forward(self, points, iter_nr=None):

        sdf, geom_feats = self.main_sdf(points, iter_nr=iter_nr)

        if not self.is_training_main_surf:
            sdf = sdf.detach()
            geom_feats = geom_feats.detach()

        if self.nr_surfs == 1:
            offsets = None
            sdfs = sdf.unsqueeze(-1)  # [N, 1, 1]
        else:
            cum_inner_eps, cum_outer_eps, inner_eps, outer_eps = self.get_offsets(
                geom_feats
            )

            # apply espilons to the main sdf to get support surfaces sdfs
            inner_sdfs = sdf + cum_inner_eps
            outer_sdfs = sdf + cum_outer_eps
            sdfs = torch.cat([inner_sdfs, sdf, outer_sdfs], dim=1)  # [N, nr_surfs]
            sdfs = sdfs.unsqueeze(-1)  # [N, nr_surfs, 1]

            offsets = torch.cat(
                [inner_eps, torch.zeros_like(sdf), outer_eps], dim=1
            )  # [N, nr_surfs]
            offsets = offsets.unsqueeze(-1)  # [N, nr_surfs, 1]

        return sdfs, offsets, geom_feats

    def main_sdf(self, points, iter_nr=None):
        # preconditions
        assert points.shape[1] == self.in_channels, "points should be N x in_channels"

        point_features, _ = self.pos_encoder(
            points, iter_nr=iter_nr
        )  # [N, encoding_output_dims]
        # geom_feats = self.mlp_shared(point_features)  # [N, 32]
        prediction = self.mlp_sdf(point_features)  # [N, 1 + geom_feat_size]

        #
        if self.geom_feat_size > 0:
            # separate sdf and geom_feat
            sdf = prediction[:, 0:1]
            geom_feats = prediction[:, 1:]
        else:
            # only sdf
            sdf = prediction
            geom_feats = None

        # sdf = sdf + self.main_surf_shift

        return sdf, geom_feats

    def get_offsets(self, geom_feats):
        # concat sdf to the point features
        mlp_eps_input = geom_feats  # [N, encoding_output_dims + 1]

        if self.use_per_offset_mlp:
            # eps = []
            # for i in range(self.nr_surfs - 1):
            #     eps.append(self.mlps_eps[i](mlp_eps_input))  # [N, 1]
            # eps = torch.cat(eps, dim=1)  # [N, nr_surfs-1]
            eps = torch.stack(
                [self.mlps_eps[i](mlp_eps_input) for i in range(self.nr_surfs - 1)],
                dim=1,
            ).squeeze(dim=-1)
        else:
            eps = self.mlp_eps(mlp_eps_input)  # [N, nr_surfs-1]

        # first nr_surfs/2 epsilons are forced to be positive (inner surfs)
        inner_eps = torch.nn.Softplus()(eps[:, self.nr_outer_surfs :])
        # last nr_surfs/2 epsilons are forced to be negative (outer surfs)
        outer_eps = -1 * torch.nn.Softplus()(eps[:, : self.nr_outer_surfs])

        # cumulative sum of the offsets
        # this way they are ordered from the smallest to the largest
        cum_outer_eps = torch.cumsum(outer_eps, dim=1) - self.min_offset
        cum_inner_eps = torch.cumsum(inner_eps, dim=1) + self.min_offset

        # flip to get the largest offset first
        cum_inner_eps = torch.flip(cum_inner_eps, dims=[1])

        return cum_inner_eps, cum_outer_eps, inner_eps, outer_eps

    def freeze_main_surf(self):
        # freeze main surface during offsets init
        if self.is_training_main_surf:
            # stop training main sdf, pos encoder
            for param in self.mlp_sdf.parameters():
                param.requires_grad = False
            for param in self.pos_encoder.parameters():
                param.requires_grad = False
            self.is_training_main_surf = False

    def unfreeze_main_surf(self):
        # reactivate training main surf
        if not self.is_training_main_surf:
            # stop training main sdf, pos encoder and mlp_shared
            for param in self.mlp_sdf.parameters():
                param.requires_grad = True
            for param in self.pos_encoder.parameters():
                param.requires_grad = True
            self.is_training_main_surf = True

    def freeze_offsets(self):
        # freeze offsets during color init
        if self.is_training_offsets:
            if self.use_per_offset_mlp:
                for eps_mlp in self.mlps_eps:
                    for param in eps_mlp.parameters():
                        param.requires_grad = False
            else:
                for param in self.mlp_eps.parameters():
                    param.requires_grad = False
            self.is_training_offsets = False

    def unfreeze_offsets(self):
        if not self.is_training_offsets:
            if self.use_per_offset_mlp:
                for eps_mlp in self.mlps_eps:
                    for param in eps_mlp.parameters():
                        param.requires_grad = True
            else:
                for param in self.mlp_eps.parameters():
                    param.requires_grad = True
            self.is_training_offsets = True

    def load_main_sdf_ckpt(self, ckpt_path):
        sdf_ckpt = torch.load(ckpt_path)

        # Keys for the first dictionary
        encoder_ckpt_keys = [
            key if "pos_encoder" in key else None for key in sdf_ckpt.keys()
        ]
        mlp_ckpt_keys = [key if "mlp_sdf" in key else None for key in sdf_ckpt.keys()]
        # Removing the None values
        encoder_ckpt_keys = [key for key in encoder_ckpt_keys if key is not None]
        mlp_ckpt_keys = [key for key in mlp_ckpt_keys if key is not None]
        # Creating the two dictionaries
        encoder_ckpt = {
            key.replace("pos_encoder.", ""): sdf_ckpt[key] for key in encoder_ckpt_keys
        }
        mlp_ckpt = {key.replace("mlp_sdf.", ""): sdf_ckpt[key] for key in mlp_ckpt_keys}
        # Loading the weights
        self.pos_encoder.load_state_dict(encoder_ckpt)
        self.mlp_sdf.load_state_dict(mlp_ckpt)

    def __getitem__(self, i):
        def get_surface_sdf_wrapper(points, iter_nr=None):
            sdfs, offsets, geom_feats = self.forward(points, iter_nr)
            return sdfs[:, i]

        return get_surface_sdf_wrapper

    def save(self, save_path, override_name=None):
        """save checkpoint"""
        model_name = self.__class__.__name__.lower()
        if override_name is not None:
            model_name = override_name
        torch.save(
            self.state_dict(),
            os.path.join(save_path, f"{model_name}.pt"),
        )
        if self.mlps_eps is not None:
            for i, mlp in enumerate(self.mlps_eps):
                torch.save(
                    mlp.state_dict(),
                    os.path.join(save_path, f"{model_name}_eps_{i}.pt"),
                )
        return save_path
