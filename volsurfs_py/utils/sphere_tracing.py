import torch
from volsurfs import RaySampler
from volsurfs import RaySamplesPacked
from volsurfs import OccupancyGrid
import matplotlib.pyplot as plt
from volsurfs_py.utils.raycasting import intersect_bounding_primitive


@torch.no_grad()
def sphere_trace(
    sdf_fn,
    rays_o,
    rays_d,
    bounding_primitive,
    nr_sphere_traces=30,
    sdf_converged_tresh=1e-4,
    sdf_multiplier=1.0,
    occupancy_grid=None,
    iter_nr=None,
    surf_idx=None,
    unconverged_are_hits=False,
    profiler=None,
):
    if profiler is not None:
        profiler.start("sphere_tracing")

    # intersect bounding primitive
    raycast = intersect_bounding_primitive(bounding_primitive, rays_o, rays_d)
    # nr_rays = raycast["nr_rays"]
    t_near = raycast["t_near"]
    t_far = raycast["t_far"]
    p_near = raycast["points_near"]
    # is_hit = raycast["is_hit"]

    occupancy_grid = None

    # get the entry point of the ray to the aabb. if an occupancy grid is available, use that one instead
    if occupancy_grid is not None:

        # get first ray sample in an occupied region
        ray_samples_packed = (
            occupancy_grid.get_first_rays_sample_start_of_grid_occupied_regions(
                rays_o, rays_d, t_near, t_far
            )
        )

        # get only the rays that end up shooting through some occupied region
        ray_samples_packed = ray_samples_packed.compact_to_valid_samples()

        # extract data
        pos = ray_samples_packed.samples_3d
        dirs = ray_samples_packed.samples_dirs

        # move position slightyl inside the voxel
        voxel_size = 1.0 / occupancy_grid.get_nr_voxels_per_dim()
        pos = pos + dirs * voxel_size * 0.5

    else:

        # create one ray sample per ray
        ray_samples_packed = RaySampler.init_with_one_sample_per_ray(p_near, rays_d)

        pos = ray_samples_packed.samples_3d
        dirs = ray_samples_packed.samples_dirs

    pts = pos.clone()

    # all rays start as unconverged
    ray_hit_flag = torch.zeros(pos.shape[0], device=pts.device).bool()
    ray_converged_flag = torch.zeros(pos.shape[0], device=pts.device).bool()
    # print("ray_converged_flag", ray_converged_flag.shape)

    #
    if pos.shape[0] == 0:
        return ray_samples_packed, ray_hit_flag

    # sphere tracing iterations
    for i in range(nr_sphere_traces):

        # get the positions that are converged
        select_cur_iter = torch.logical_not(ray_converged_flag)
        # print("iter", i)
        # print("select_cur_iter", select_cur_iter.shape)
        # print("samples_3d", pts.shape)
        # print("dirs", dirs.shape)
        pos_unconverged = pts[select_cur_iter, :]
        dirs_unconverged = dirs[select_cur_iter, :]
        # print("pos_unconverged", pos_unconverged.shape)
        # print("dirs_unconverged", dirs_unconverged.shape)

        if pos_unconverged.shape[0] == 0:  # all points are converged
            # print(f"all rays converged after {i} sphere-tracing iterations")
            break

        # sphere trace
        if iter_nr is not None:
            pred = sdf_fn(pos_unconverged, iter_nr)
        else:
            pred = sdf_fn(pos_unconverged)

        # unpack if tuple
        if isinstance(pred, tuple):
            sdf = pred[0]
        else:
            sdf = pred

        # select output if index is given
        if surf_idx is not None:
            sdf = sdf[:, surf_idx]  # e.g. to select a specific sdf

        # advance the points
        pos_unconverged = pos_unconverged + dirs_unconverged * (sdf * sdf_multiplier)

        # check if points are now converged
        newly_converged_flag = (sdf.abs() < sdf_converged_tresh)[:, 0]
        # print("newly_converged_flag", newly_converged_flag.shape)

        # points are converged if they are newly converged or already converged
        ray_hit_flag[select_cur_iter] = torch.logical_or(
            ray_hit_flag[select_cur_iter], newly_converged_flag
        )
        ray_converged_flag[select_cur_iter] = torch.logical_or(
            ray_converged_flag[select_cur_iter], newly_converged_flag
        )

        # if occupancy_grid is not None:
        #     # check if the new positions are in unnocupied space and if they are move them towards the next occupied voxel
        #     (
        #         pos_unconverged,
        #         is_within_grid_bounds,
        #     ) = occupancy_grid.advance_ray_sample_to_next_occupied_voxel(dirs_unconverged, pos_unconverged)
        #     is_within_grid_bounds = is_within_grid_bounds[:, 0]
        #     # print("pos_unconverged", pos_unconverged.shape)
        #     # print("is_within_grid_bounds", is_within_grid_bounds.shape)
        # else:

        is_within_grid_bounds = bounding_primitive.check_points_inside(pos_unconverged)
        # print("pos_unconverged", pos_unconverged.shape)
        # print("is_within_grid_bounds", is_within_grid_bounds.shape)

        ray_converged_flag[select_cur_iter] = torch.logical_or(
            ray_converged_flag[select_cur_iter],
            torch.logical_not(is_within_grid_bounds),
        )
        # print("ray_converged_flag", ray_converged_flag.shape)

        # update the new points
        pts[select_cur_iter] = pos_unconverged
        pts_z = (pts - rays_o).norm(dim=-1, keepdim=True)

    # check if there are unconverged rays
    if unconverged_are_hits:
        ray_hit_flag[~ray_converged_flag] = True

    if profiler is not None:
        profiler.end("sphere_tracing")

    ray_samples_packed.samples_3d = pts
    ray_samples_packed.samples_z = pts_z

    return ray_samples_packed, ray_hit_flag


# @torch.no_grad()
# def sphere_trace(
#     model,
#     rays_o,
#     rays_d,  # (N_rays, 3), (N_rays, 3)
#     rays_t_near,
#     rays_t_far,  # (N_rays, 1), (N_rays, 1)
#     invalid_rays,  #
#     iter_nr=None,
#     max_steps=100,
#     eps=1e-4,
#     step_multiplier=1.0,
#     # TODO: use occupancy grid to skip empty space
# ):
#     r"""A vectorized implementation of sphere tracing of a batch of rays"""

#     dists = rays_t_near  # (N_rays, 1)
#     max_dists = rays_t_far  # (N_rays, 1)
#     n_rays = rays_o.shape[0]
#     pts = rays_o + rays_d * rays_t_near  # (N_rays, 3)
#     steps = torch.zeros_like(dists)  # (N_rays, 1)
#     converged_rays = torch.zeros(n_rays, dtype=torch.bool)  # (N_rays, )
#     out_of_bounds_rays = invalid_rays  # , (torch.abs(pts) > 1.0).any())  # (N_rays, )

#     # if there is at least one valid ray
#     valid_rays = torch.logical_not(
#         torch.logical_or(converged_rays, out_of_bounds_rays)
#     )
#     nr_valid_rays = valid_rays.sum()

#     if nr_valid_rays > 0:
#         # For each ray, trace it until either:
#         # - it hits a surface
#         # - it exits the bounding volume
#         # - we reach the maximum number of steps
#         for i in range(max_steps):
#             # print("sphere tracing iter", i)
#             # print("nr_converged_rays", converged_rays.sum())
#             # print("nr_out_of_bounds_rays", out_of_bounds_rays.sum())
#             # print("n valid rays", nr_valid_rays)

#             # Inference
#             sdf_valid_rays, _ = model(pts[valid_rays], iter_nr)  # (n_pts, 1)
#             sdf_valid_rays *= step_multiplier

#             sdf = torch.zeros_like(dists)  # (N_rays, 1)
#             sdf[valid_rays] = sdf_valid_rays

#             # Check if ray exited the bounding volume (only if valid ray)
#             new_out_of_bounds_rays = (dists + sdf > max_dists).squeeze()
#             new_out_of_bounds_rays[~valid_rays] = False
#             new_out_of_bounds_rays[out_of_bounds_rays] = False
#             new_out_of_bounds_rays[converged_rays] = False
#             # print("nr just oob rays", new_out_of_bounds_rays.sum())
#             steps[new_out_of_bounds_rays] = i
#             out_of_bounds_rays = torch.logical_or(
#                 out_of_bounds_rays, new_out_of_bounds_rays
#             )
#             sdf[out_of_bounds_rays] = 0.0

#             # Check if ray converged (only if it is a valid ray, a not new oob ray and a not already converged ray)
#             new_converged_rays = (sdf < eps).squeeze()
#             new_converged_rays[~valid_rays] = False
#             new_converged_rays[out_of_bounds_rays] = False
#             new_converged_rays[converged_rays] = False
#             # print("nr just converged rays", new_converged_rays.sum())
#             steps[new_converged_rays] = i
#             converged_rays = torch.logical_or(converged_rays, new_converged_rays)
#             sdf[converged_rays] = 0.0

#             # Terminate if all rays terminated
#             valid_rays = torch.logical_not(
#                 torch.logical_or(converged_rays, out_of_bounds_rays)
#             )
#             nr_valid_rays = valid_rays.sum()
#             if nr_valid_rays == 0:
#                 # print("All rays converged in {} steps".format(i))
#                 break

#             # Update traced pts
#             dists += sdf
#             pts = pts + (rays_d * sdf)

#     # check if there are unconverged rays
#     if nr_valid_rays > 0:
#         # mark them as converged
#         converged_rays[valid_rays] = True
#         steps[valid_rays] = max_steps

#     # out of bounds rays are not converged
#     pts[out_of_bounds_rays] = 0
#     dists[out_of_bounds_rays] = 0
#     steps[out_of_bounds_rays] = 0

#     assert (
#         converged_rays | out_of_bounds_rays
#     ).sum() == n_rays, "not all rays are converged or out of bounds"

#     res = {
#         "converged_rays": converged_rays,  # (n_rays, )
#         "samples_3d": pts,  # (n_rays, 3)
#         "dists": dists,  # (n_rays, 1)
#         "steps": steps,  # (n_rays, 1)
#     }

#     return res
