# https://therealmjp.github.io/posts/sg-series-part-2-spherical-gaussians-101/

# Adapted from: https://github.com/sxyu/plenoctree/blob/master/nerf_sh/nerf/sg.py

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


def spher2cart(r, theta, phi):
    """Convert spherical coordinates into Cartesian coordinates."""
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)


# a network predicts a SG basis for each color channel (N=3)


def eval_sg(sg_lambda, sg_mu, sg_coeffs, dirs):
    """
    Evaluate spherical gaussians at unit directions
    using learnable SG basis.
    Works with torch.
    ... Can be 0 or more batch dimensions.
    N is the number of SG basis we use.

    Output = \sigma_{i}{coeffs_i * \exp ^ {lambda_i * (\dot(mu_i, dirs) - 1)}}

    Args:
        sg_lambda: The sharpness of the SG lobes. [N] or [..., N]
        sg_mu: The directions of the SG lobes. [N, 3 or 2] or [..., N, 3 or 2]
        sg_coeffs: The coefficients of the SG (lob amplitude). [..., C, N]
        dirs: unit directions [..., 3]

    Returns:
        [..., C]
    """
    sg_lambda = torch.nn.Softplus(sg_lambda)  # force lambda > 0
    # spherical coordinates -> Cartesian coordinates
    if sg_mu.shape[-1] == 2:
        theta, phi = sg_mu[..., 0], sg_mu[..., 1]
        sg_mu = spher2cart(1.0, theta, phi)  # [..., N, 3]
    product = torch.einsum("...ij,...j->...i", sg_mu, dirs)  # [..., N]
    basis = torch.exp(
        torch.einsum("...i,...i->...i", sg_lambda, product - 1)
    )  # [..., N]
    output = torch.einsum("...ki,...i->...k", sg_coeffs, basis)  # [..., C]
    output /= sg_lambda.shape[-1]
    return output


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

    Args:
        angle: rotation angle along 3 dim (in radians). [..., 3]
    Returns:
        Rotation matrix corresponding to the euler angles. [..., 3, 3]
    """
    x, y, z = angle[..., 0], angle[..., 1], angle[..., 2]
    cosz = torch.cos(z)
    sinz = torch.sin(z)
    cosy = torch.cos(y)
    siny = torch.sin(y)
    cosx = torch.cos(x)
    sinx = torch.sin(x)
    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)
    zmat = torch.stack(
        [
            torch.stack([cosz, -sinz, zeros], dim=-1),
            torch.stack([sinz, cosz, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-1,
    )
    ymat = torch.stack(
        [
            torch.stack([cosy, zeros, siny], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-siny, zeros, cosy], dim=-1),
        ],
        dim=-1,
    )
    xmat = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cosx, -sinx], dim=-1),
            torch.stack([zeros, sinx, cosx], dim=-1),
        ],
        dim=-1,
    )
    rotMat = torch.einsum("...ij,...jk,...kq->...iq", xmat, ymat, zmat)
    return rotMat
