import numpy as np
import torch
from copy import deepcopy
import permutohedral_encoding as permuto_enc

from volsurfs_py.encodings.encoder import Encoder
from volsurfs_py.utils.common import map_range_val


class PermutoHashEncoder(Encoder):
    def __init__(
        self,
        input_dim=3,
        nr_levels=24,
        log2_hashmap_size=18,
        nr_feat_per_level=2,
        coarsest_scale=1.0,
        finest_scale=0.0001,
        nr_iters_for_c2f=0,
        appply_random_shift_per_level=True,
        concat_points=True,
        concat_points_scaling=1.0,
        remove_last_element=True,  # set to false to load last checkpoints
        bb_sides=2.0,
    ):
        capacity = pow(2, log2_hashmap_size)
        scale_list = np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
        encoder = permuto_enc.PermutoEncoding(
            input_dim,
            capacity,
            nr_levels,
            nr_feat_per_level,
            scale_list,
            appply_random_shift_per_level=appply_random_shift_per_level,
            concat_points=concat_points,
            concat_points_scaling=concat_points_scaling,
        )
        if remove_last_element:
            output_dim = encoder.output_dims() - 1
        else:
            output_dim = encoder.output_dims()
        self.remove_last_element = remove_last_element

        super().__init__(input_dim, output_dim)

        self.encoder = encoder
        self.capacity = capacity
        self.nr_levels = nr_levels
        self.nr_feat_per_level = nr_feat_per_level
        self.scale_list = scale_list
        self.appply_random_shift_per_level = appply_random_shift_per_level
        self.concat_points = concat_points
        self.concat_points_scaling = concat_points_scaling

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
        window = self.c2f(t)

        if self.bb_sides is not None:
            points_out_of_bounds = torch.logical_or(
                (points <= -self.bb_sides / 2).any(dim=1),
                (points >= self.bb_sides / 2).any(dim=1),
            )

            points_scaling = 1 / (self.bb_sides / 2)
            # print("points_scaling", points_scaling)
            points = points * points_scaling  # scaling in [-1, 1]
            points = (points + 1) / 2  # map in range [0, 1]
            # print("scaled points", points[:5])
        else:
            points_out_of_bounds = None

        if self.remove_last_element:
            enc_points = self.encoder(points, window.view(-1))[:, :-1]
        else:
            enc_points = self.encoder(points, window.view(-1))

        return enc_points, points_out_of_bounds

    def reset(self):
        self.encoder.reset()
