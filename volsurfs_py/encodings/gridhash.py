import torch
import numpy as np
from copy import deepcopy
import permutohedral_encoding as permuto_enc

from volsurfs_py.encodings.encoder import Encoder
from volsurfs_py.utils.common import map_range_val

import tinycudann as tcnn


class GridHashEncoder(Encoder):
    def __init__(
        self,
        input_dim=3,
        nr_levels=24,
        log2_hashmap_size=18,
        nr_feat_per_level=2,
        base_resolution=16,
        growth_factor=2,
        nr_iters_for_c2f=0,
        concat_points=True,
        bb_sides=2.0,
    ):
        self.config = {
            "otype": "Grid",  # Component type
            "type": "Hash",  # Type of backing storage of the grids. Can be "Hash", "Tiled" or "Dense".
            "n_levels": nr_levels,  # Number of levels (resolutions)
            "n_features_per_level": nr_feat_per_level,  # Dimensionality of feature vector stored in each level's entries.
            "log2_hashmap_size": log2_hashmap_size,  # If type is "Hash", is the base-2 logarithm of the number of elements in each backing hash table.
            "base_resolution": base_resolution,  # The resolution of the coarsest level is base_resolution^input_dims.
            "per_level_scale": growth_factor,  # The geometric growth factor, i.e. the factor by which the resolution of each grid is larger (per axis) than that of the preceding level.
            "interpolation": "Linear",  # How to interpolate nearby grid lookups. Can be "Nearest", "Linear", or "Smoothstep" (for smooth derivatives).
        }

        # print("config", config)
        encoder = tcnn.Encoding(input_dim, self.config, dtype=torch.float32)
        output_dim = encoder.n_output_dims + input_dim

        super().__init__(input_dim, output_dim)

        self.encoder = encoder
        self.concat_points = concat_points

        # bounding box
        self.bb_sides = bb_sides
        # self.bb_sided can be None in NerfHash
        if self.bb_sides is not None:
            if isinstance(self.bb_sides, float):
                self.bb_sides = np.array([self.bb_sides] * input_dim)
            if isinstance(self.bb_sides, np.ndarray):
                self.bb_sides = torch.tensor(self.bb_sides, dtype=torch.float32)
            self.bb_sides = self.bb_sides.to("cuda")

        self.c2f = permuto_enc.Coarse2Fine(nr_levels)
        self.nr_iters_for_c2f = nr_iters_for_c2f

    def __call__(self, points, iter_nr=None, **kwargs):

        # c2f window
        if iter_nr is None or iter_nr < 0:
            t = 1.0
        else:
            t = map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.3, 1.0)
        window = self.c2f(t).to(points.device)
        window = window.repeat_interleave(self.config["n_features_per_level"])

        if self.bb_sides is not None:
            points_out_of_bounds = torch.logical_or(
                (points <= -self.bb_sides / 2).any(dim=1),
                (points >= self.bb_sides / 2).any(dim=1),
            )

            points_scaling = 1 / (self.bb_sides / 2)
            points = points * points_scaling  # scaling in [-1, 1]
            points = (points + 1) / 2  # map in range [0, 1]
        else:
            points_out_of_bounds = None

        enc_points = self.encoder(points)
        enc_points = enc_points * window
        # concat input to enc_x
        if self.concat_points:
            enc_points = torch.cat([enc_points, points], dim=1)

        return enc_points, points_out_of_bounds

    def reset(self):
        pass
        # TODO: reset c2f
        # TODO: test reset
        # self.encoder = tcnn.Encoding(input_dim, self.config, dtype=torch.float32)
