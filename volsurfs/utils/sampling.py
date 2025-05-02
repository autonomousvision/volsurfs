import torch

from volsurfs import RaySampler


@torch.no_grad()
def create_fg_samples(
    min_dist_between_samples,
    min_nr_samples_per_ray,
    max_nr_samples_per_ray,
    rays_o,
    rays_d,
    ray_t_entry,
    ray_t_exit,
    jitter_samples,
    occupancy_grid,
    values_dim,
    profiler=None,
):
    if profiler is not None:
        profiler.start("create_fg_samples")

    # foreground samples
    if occupancy_grid is not None:

        ray_samples_packed_fg = RaySampler.compute_samples_fg_in_grid_occupied_regions(
            rays_o,
            rays_d,
            ray_t_entry,
            ray_t_exit,
            min_dist_between_samples,
            min_nr_samples_per_ray,
            max_nr_samples_per_ray,
            jitter_samples,
            occupancy_grid.get_nr_voxels_per_dim(),
            occupancy_grid.get_grid_extent(),
            occupancy_grid.get_grid_occupancy(),
            occupancy_grid.get_grid_roi(),
            values_dim,
        )

    else:

        ray_samples_packed_fg = RaySampler.compute_samples_fg(
            rays_o,
            rays_d,
            ray_t_entry,
            ray_t_exit,
            min_dist_between_samples,
            min_nr_samples_per_ray,
            max_nr_samples_per_ray,
            jitter_samples,
            values_dim,
        )

    if profiler is not None:
        profiler.end("create_fg_samples")

    return ray_samples_packed_fg


@torch.no_grad()
def create_uncontracted_bg_samples(
    nr_samples_bg,
    rays_o,
    rays_d,
    t_min,
    t_max=100.0,
    jitter_samples=False,
    # bounding_primitive=None,
    # contract_3d_samples=False,
):
    values_dim = 0
    ray_samples_packed_bg = RaySampler.compute_samples_bg(
        rays_o,
        rays_d,
        t_min,
        t_max,
        nr_samples_bg,
        # bounding_primitive.get_radius(),
        # bounding_primitive.get_center(),
        jitter_samples,
        # contract_3d_samples,
        # values_dim
    )
    return ray_samples_packed_bg


# @torch.no_grad()
# def filter_samples_in_range(
#     ray_samples_packed_fg,
#     t_near,
#     t_far,
#     instance_bb
# ):
#     # fg_rays_samples_packed has to be compacted first

#     all_world_samples_3d = ray_samples_packed_fg.samples_3d
#     all_world_samples_dirs = ray_samples_packed_fg.samples_dirs

#     # to canonical space transformation
#     world_to_local_transform = instance_bb.get_pose()
#     all_local_samples_3d = apply_transformation_3d(
#         all_world_samples_3d,
#         world_to_local_transform
#     )
#     all_local_samples_dirs = apply_transformation_3d(
#         all_world_samples_dirs,
#         world_to_local_transform
#     )

#     # creates a new RaySamplesPacked object filtering the samples
#     # that are inside the instance bounding box
#     instance_ray_samples_packed = ray_samples_packed_fg.filter_samples_in_range(
#         all_local_samples_3d,
#         all_local_samples_dirs,
#         t_near,
#         t_far
#     )

#     #
#     print("I BEFORE COMPACTING")
#     print(instance_ray_samples_packed.samples_3d.shape)
#     print(instance_ray_samples_packed.samples_idx.shape)
#     print(instance_ray_samples_packed.ray_start_end_idx[0])
#     nr_empty_rays = (instance_ray_samples_packed.ray_start_end_idx[:, 0] == instance_ray_samples_packed.ray_start_end_idx[:, 1]).sum()
#     print("nr_empty_rays", nr_empty_rays)

#     # instance_ray_samples_packed = instance_ray_samples_packed.compact_to_valid_samples()

#     #
#     print("I AFTER COMPACTING")
#     print(instance_ray_samples_packed.samples_3d.shape)
#     print(instance_ray_samples_packed.samples_idx.shape)
#     print(instance_ray_samples_packed.ray_start_end_idx[0])
#     nr_empty_rays = (instance_ray_samples_packed.ray_start_end_idx[:, 0] == instance_ray_samples_packed.ray_start_end_idx[:, 1]).sum()
#     print("nr_empty_rays", nr_empty_rays)

#     # # if no samples are valid, return None
#     # if instance_ray_samples_packed.samples_idx.nelement() == 0:
#     #     return None

#     valid_samples_idx = instance_ray_samples_packed.samples_idx[:, 0]
#     world_samples_3d = all_world_samples_3d[valid_samples_idx]
#     world_samples_dir = all_world_samples_dirs[valid_samples_idx]

#     return instance_ray_samples_packed, world_samples_3d, world_samples_dir
