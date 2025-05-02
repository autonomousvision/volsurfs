import torch
import math
import torch.nn.functional as F


def get_field_gradients(
    field_fn, points, iter_nr=None, eps=1e-4, grad_method="finite-diff"
):

    if grad_method == "finite-diff":
        # finite difference
        with torch.no_grad():
            points_xplus = points.clone()
            points_yplus = points.clone()
            points_zplus = points.clone()
            points_xplus[:, 0] += eps
            points_yplus[:, 1] += eps
            points_zplus[:, 2] += eps
            points_full = torch.cat(
                [points, points_xplus, points_yplus, points_zplus], 0
            )

        if iter_nr is not None:
            points_res = field_fn(points_full, iter_nr)
        else:
            points_res = field_fn(points_full)

        if isinstance(points_res, tuple):
            sdfs_full = points_res[0]
        else:
            sdfs_full = points_res

        # if (nr_points,) then add a dim
        if sdfs_full.dim() < 2:
            sdfs_full = sdfs_full.unsqueeze(1)

        if sdfs_full.shape[-1] > 1:
            sdfs_full = sdfs_full[:, 0].unsqueeze(1)

        sdfs = sdfs_full.chunk(4, dim=0)
        sdf = sdfs[0]
        sdf_xplus = sdfs[1]
        sdf_yplus = sdfs[2]
        sdf_zplus = sdfs[3]

        grad_x = (sdf_xplus - sdf) / eps
        grad_y = (sdf_yplus - sdf) / eps
        grad_z = (sdf_zplus - sdf) / eps

        gradients = torch.cat([grad_x, grad_y, grad_z], dim=-1)

        # TODO: fix
        # elif self.grad_method == "autograd":
        #     # autograd
        #     with torch.set_grad_enabled(True):
        #         points.requires_grad_(True)
        #         sdf, _ = self.forward(points, iter_nr)
        #         d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        #         gradients = torch.autograd.grad(
        #             outputs=sdf,
        #             inputs=points,
        #             grad_outputs=d_output,
        #             create_graph=True,
        #             retain_graph=True,
        #             only_inputs=True,
        #         )[0]

        # # elif self.grad_method == "central-diff":
        # # TODO: implement

    else:
        raise ValueError(
            f"grad method {grad_method} not supported. \
                Use 'finite-diff'"
        )

    return gradients


def get_sdf_curvature(sdf_fn, points, sdf_gradients, iter_nr=None, eps=1e-4):
    # get the curvature along a certain random direction for each point

    # get the sdf_normals at the points
    sdfs_normals = F.normalize(sdf_gradients, dim=-1)

    if sdf_gradients.dim() > 2:
        nr_surfs = sdf_gradients.shape[1]
    else:
        nr_surfs = 1

    # get a random direction
    rand_directions = torch.randn_like(points)
    rand_directions = F.normalize(rand_directions, dim=-1)

    # calculate a random vector that is orthogonal to the normal and the random direction
    if sdfs_normals.dim() > 2:
        rand_directions = rand_directions.unsqueeze(1)

    # print("sdfs_normals", sdfs_normals.shape)
    # print("rand_directions", rand_directions.shape)
    tangent_directions = torch.cross(sdfs_normals, rand_directions, dim=-1)

    # flatten_tangent_directions = tangent_directions.view(-1, 3)
    # repeated_points = points.repeat(nr_surfs, 1)
    # print("flatten_tangent_directions", flatten_tangent_directions.shape)
    # print("repeated_points", repeated_points.shape)

    # repeated_points_shifted = repeated_points + flatten_tangent_directions * eps
    # print("repeated_points_shifted", repeated_points_shifted.shape)

    # sdfs_gradients_shifted = get_field_gradients(
    #     sdf_fn,
    #     repeated_points_shifted,
    #     iter_nr=iter_nr
    # )
    # print("sdfs_gradients_shifted", sdfs_gradients_shifted.shape)

    # iterate over surfaces and get the curvature
    sdfs_normals_shifted = torch.zeros_like(sdfs_normals)
    # sdfs_curvatures = torch.zeros(points.shape[0], nr_surfs, 1, device=points.device)
    for i in range(nr_surfs):

        if tangent_directions.dim() > 2:
            tangent_directions_ = tangent_directions[:, i]
        else:
            tangent_directions_ = tangent_directions

        # shift the points along the random direction
        points_shifted = points + tangent_directions_ * eps

        # get the gradient at the shifted point
        sdfs_gradients_shifted = get_field_gradients(
            sdf_fn, points_shifted, iter_nr=iter_nr
        )

        if sdfs_gradients_shifted.dim() > 2:
            sdf_gradients_shifted = sdfs_gradients_shifted[:, i]
        else:
            sdf_gradients_shifted = sdfs_gradients_shifted

        # get the sdf_normals at the shifted points
        sdf_normals_shifted = F.normalize(sdf_gradients_shifted, dim=-1)

        if sdfs_normals_shifted.dim() > 2:
            sdfs_normals_shifted[:, i] = sdf_normals_shifted
        else:
            sdfs_normals_shifted = sdf_normals_shifted

    # then computing a dot produt
    dot = torch.sum(torch.mul(sdfs_normals, sdfs_normals_shifted), dim=-1, keepdim=True)
    # dot = (sdf_normals * normals_shifted).sum(dim=-1, keepdim=True)

    # the dot would assign low weight importance to sdf_normals that are almost
    # the same, and increasing error the more they deviate

    # So it's something like and L2 loss. But we want a L1 loss
    # so we get the angle, and then we map it to range [0,1]
    angle = torch.acos(torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6))
    # goes to range 0 when the angle is the same and pi when is opposite

    sdfs_curvatures = angle / math.pi  # map to [0,1 range]

    if nr_surfs == 1:
        sdfs_curvatures = sdfs_curvatures.squeeze(1)

    return sdfs_curvatures
