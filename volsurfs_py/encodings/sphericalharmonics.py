# Adapted from: https://github.com/sxyu/plenoctree/blob/master/nerf_sh/nerf/sh.py

#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np
from scipy.special import sph_harm
import torch.nn.functional as F
import permutohedral_encoding as permuto_enc

from volsurfs_py.utils.common import map_range_val
from volsurfs_py.encodings.encoder import Encoder


class SHEncoder(Encoder):
    # hardcoded SH polynomials
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    ]
    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]

    def __init__(
        self,
        input_dim=3,
        degree=3,
        # nr_iters_for_c2f=0
    ):
        output_dim = (degree + 1) ** 2
        super().__init__(input_dim, output_dim)

        assert input_dim == 3, "SH encoding only supports 3D inputs"
        assert degree >= 0 and degree <= 4, "SH degree must be 0-4"

        self.degree = degree
        # self.c2f = permuto_enc.Coarse2Fine(degree)
        # self.nr_iters_for_c2f = nr_iters_for_c2f

    def __call__(self, dirs, **kwargs):
        """
        Express the dirs vectors as a linear combination of the precomputed harmonics.
        Each coefficient in the linear combination represents the amplitude of a specific harmonic function in the vector's representation.

        Returns spherical harmonics encoding of unit directions using hardcoded SH polynomials.

        Args:
            dirs: [:, 3]

        Returns:
            sh: [:, (self.degree + 1) ** 2]
        """

        # normalize dirs
        # dirs = F.normalize(dirs, dim=-1)

        x, y, z = dirs[:, 0:1].squeeze(), dirs[:, 1:2].squeeze(), dirs[:, 2:3].squeeze()

        # compute SH coeffs

        result = torch.zeros(
            (dirs.shape[0], self.output_dim), dtype=dirs.dtype, device=dirs.device
        )

        result[:, 0] = self.C0

        if self.degree > 0:
            result[:, 1] = -self.C1 * y
            result[:, 2] = self.C1 * z
            result[:, 3] = -self.C1 * x

            if self.degree > 1:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[:, 4] = self.C2[0] * xy
                result[:, 5] = self.C2[1] * yz
                result[:, 6] = self.C2[2] * (2.0 * zz - xx - yy)
                result[:, 7] = self.C2[3] * xz
                result[:, 8] = self.C2[4] * (xx - yy)
                # # c2f
                # result[:, 4:9] = result[:, 4:9] * window[1]

                if self.degree > 2:
                    result[:, 9] = self.C3[0] * y * (3 * xx - yy)
                    result[:, 10] = self.C3[1] * xy * z
                    result[:, 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[:, 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[:, 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[:, 14] = self.C3[5] * z * (xx - yy)
                    result[:, 15] = self.C3[6] * x * (xx - 3 * yy)
                    # # c2f
                    # result[:, 9:16] = result[:, 9:16] * window[2]

                    if self.degree > 3:
                        result[:, 16] = self.C4[0] * xy * (xx - yy)
                        result[:, 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[:, 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[:, 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[:, 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[:, 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[:, 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[:, 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[:, 24] = self.C4[8] * (
                            xx * (xx - 3 * yy) - yy * (3 * xx - yy)
                        )
                        # # c2f
                        # result[:, 16:25] = result[:, 16:25] * window[3]

        return result

    @staticmethod
    def eval(sh, dirs, degree, iter_nr=None):
        """
        Evaluate spherical harmonics at unit directions
        using hardcoded SH polynomials.

        Args:
            sh (torch.tensor): SH coeffs [..., channels, (self.degree + 1) ** 2]
            dirs (torch.tensor): unit directions [..., 3]
            deg (int): SH deg. Currently, 0-4 supported

        Returns:
            [..., channels]
        """

        # # c2f window
        # if iter_nr is None:
        #     t = 1.0
        # else:
        #     t = map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.0, 1.0)
        # window = self.c2f(t)
        # print("window", window)

        # # normalize dirs (just to be sure)
        # dirs = F.normalize(dirs, dim=-1)

        result = SHEncoder.C0 * sh[..., 0]

        if degree > 0:
            x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
            result = (
                result
                - SHEncoder.C1 * y * sh[..., 1]
                + SHEncoder.C1 * z * sh[..., 2]
                - SHEncoder.C1 * x * sh[..., 3]
            )
            if degree > 1:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result = (
                    result
                    + SHEncoder.C2[0] * xy * sh[..., 4]
                    + SHEncoder.C2[1] * yz * sh[..., 5]
                    + SHEncoder.C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                    + SHEncoder.C2[3] * xz * sh[..., 7]
                    + SHEncoder.C2[4] * (xx - yy) * sh[..., 8]
                )

                if degree > 2:
                    result = (
                        result
                        + SHEncoder.C3[0] * y * (3 * xx - yy) * sh[..., 9]
                        + SHEncoder.C3[1] * xy * z * sh[..., 10]
                        + SHEncoder.C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                        + SHEncoder.C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                        + SHEncoder.C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                        + SHEncoder.C3[5] * z * (xx - yy) * sh[..., 14]
                        + SHEncoder.C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                    )
                    if degree > 3:
                        result = (
                            result
                            + SHEncoder.C4[0] * xy * (xx - yy) * sh[..., 16]
                            + SHEncoder.C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                            + SHEncoder.C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                            + SHEncoder.C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                            + SHEncoder.C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                            + SHEncoder.C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                            + SHEncoder.C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                            + SHEncoder.C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                            + SHEncoder.C4[8]
                            * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                            * sh[..., 24]
                        )
        return result
