import torch

# import numpy as np
# from copy import deepcopy
from volsurfs import VolumeRendering
from volsurfs_py.utils.sampling import create_fg_samples

# from mvdatasets.geometry.contraction import contract_points


@torch.no_grad()
def importance_sampling_sdfs_iter(
    samples_sdfs,
    nr_surfs,
    ray_samples_packed,
    iter_nr,
    nr_imp_samples,
    logistic_beta_value,
    jitter_samples=False,
    profiler=None,
):
    ray_samples_packed.update_dt(False)
    logistic_beta = torch.ones_like(ray_samples_packed.samples_dt) * logistic_beta_value
    c2fs_aggregator = torch.zeros_like(ray_samples_packed.samples_dt)

    for surf_idx in range(nr_surfs):

        # get current surface sdf
        samples_sdf = samples_sdfs[:, surf_idx]
        # print(f"samples_sdf {surf_idx}", samples_sdf)

        alpha = VolumeRendering.sdf2alpha(
            ray_samples_packed,
            samples_sdf,
            logistic_beta,
        )
        # print(f"alpha {surf_idx}", alpha)

        # alpha = alpha.clip(0.0, 1.0)
        transmittance, _ = VolumeRendering.cumprod_one_minus_alpha_to_transmittance(
            ray_samples_packed, 1 - alpha + 1e-6
        )
        transmittance = transmittance.clip(0.0, 1.0)
        weights = alpha * transmittance

        # get normalized weights
        _, weight_sum_per_sample = VolumeRendering.sum_over_rays(
            ray_samples_packed, weights
        )
        weights /= torch.clip(weight_sum_per_sample, min=1e-6)

        # get distribution and importance sample
        cdf = VolumeRendering.compute_cdf(ray_samples_packed, weights)

        c2fs_aggregator += cdf
        # print(f"cdf {surf_idx}", cdf)

    # normalize

    cdf = c2fs_aggregator / nr_surfs
    # print("cdf", cdf)

    ray_samples_packed_imp = VolumeRendering.importance_sample(
        ray_samples_packed, cdf, nr_imp_samples, jitter_samples
    )
    return ray_samples_packed_imp


@torch.no_grad()
def importance_sampling_sdfs(
    sdfs_fn,
    nr_surfs,
    ray_samples_packed_uniform,
    iter_nr,
    nr_imp_samples,
    logistic_beta_value,
    min_dist_between_samples,
    jitter_samples=False,
    profiler=None,
):
    assert (
        not ray_samples_packed_uniform.is_empty()
    ), "ray_samples_packed_uniform should not be empty"

    if profiler is not None:
        profiler.start("importance_sampling_sdfs")

    # first iteration

    # get sdfs

    if iter_nr is None:
        res_samples = sdfs_fn(ray_samples_packed_uniform.samples_3d)
    else:
        res_samples = sdfs_fn(ray_samples_packed_uniform.samples_3d, iter_nr)
    if isinstance(res_samples, tuple):
        samples_sdfs = res_samples[0]
    else:
        samples_sdfs = res_samples
    assert samples_sdfs.dim() == 3, "sdfs_fn should return (N, nr_surfs, K) tensor"

    # remove extra channels
    if samples_sdfs.shape[2] > 1:
        samples_sdfs = samples_sdfs[..., 0:1]  # (N, nr_surfs, 1)

    ray_samples_packed_imp_1 = importance_sampling_sdfs_iter(
        samples_sdfs=samples_sdfs,
        nr_surfs=nr_surfs,
        ray_samples_packed=ray_samples_packed_uniform,
        nr_imp_samples=nr_imp_samples // 2,
        logistic_beta_value=logistic_beta_value / 2.0,
        jitter_samples=jitter_samples,
        iter_nr=iter_nr,
    )

    # second iteration (use importance samples to query the model sdf)

    if iter_nr is None:
        res_samples_imp_1 = sdfs_fn(ray_samples_packed_imp_1.samples_3d)
    else:
        res_samples_imp_1 = sdfs_fn(ray_samples_packed_imp_1.samples_3d, iter_nr)
    if isinstance(res_samples_imp_1, tuple):
        samples_sdfs_imp_1 = res_samples_imp_1[0]
    else:
        samples_sdfs_imp_1 = res_samples_imp_1
    assert (
        samples_sdfs_imp_1.dim() == 3
    ), "sdfs_fn should return (N, nr_surfs, 1) tensor"

    # remove extra channels
    if samples_sdfs_imp_1.shape[2] > 1:
        samples_sdfs_imp_1 = samples_sdfs_imp_1[..., 0:1]

    # print("samples_sdfs", samples_sdfs)
    # print("samples_sdfs_imp_1", samples_sdfs_imp_1)

    # set sdf for packets fusion
    ray_samples_packed_uniform.set_samples_values(samples_sdfs.view(-1, nr_surfs))
    ray_samples_packed_imp_1.set_samples_values(samples_sdfs_imp_1.view(-1, nr_surfs))

    # fuse the uniform and importance samples
    ray_samples_packed_combined = VolumeRendering.combine_ray_samples_packets(
        ray_samples_packed_uniform, ray_samples_packed_imp_1, min_dist_between_samples
    )

    samples_sdfs_combined = ray_samples_packed_combined.get_samples_values().view(
        -1, nr_surfs, 1
    )
    # print("samples_sdfs_combined", samples_sdfs_combined)

    ray_samples_packed_uniform.remove_samples_values()
    ray_samples_packed_imp_1.remove_samples_values()

    ray_samples_packed_imp_2 = importance_sampling_sdfs_iter(
        samples_sdfs=samples_sdfs_combined,
        nr_surfs=nr_surfs,
        ray_samples_packed=ray_samples_packed_combined,
        nr_imp_samples=nr_imp_samples // 2,
        logistic_beta_value=logistic_beta_value,
        jitter_samples=jitter_samples,
        iter_nr=iter_nr,
    )

    if profiler is not None:
        profiler.end("importance_sampling_sdfs")

    return ray_samples_packed_imp_1, ray_samples_packed_imp_2


# @torch.no_grad()
# def importance_sampling_sdfs(
#     sdfs_fn,
#     surf_idx,
#     ray_samples_packed_uniform,
#     iter_nr,
#     nr_samples,
#     logistic_beta_value,
#     min_dist_between_samples,
#     jitter_samples=False,
#     profiler=None
# ):
#     """importance sampling for surf (query the model sdf);
#     returns a RaySamplesPacked object combining the uniform and importance samples

#     ray_samples_packed: RaySamplesPacked object
#     """
#     assert not ray_samples_packed_uniform.is_empty(), "ray_samples_packed_uniform should not be empty"

#     if profiler is not None:
#         profiler.start("importance_sampling_sdfs")

#     # get sdf kernel size
#     logistic_beta_value = logistic_beta_value

#     # first iteration

#     if iter_nr is None:
#         res_samples = sdfs_fn(ray_samples_packed_uniform.samples_3d)
#     else:
#         res_samples = sdfs_fn(
#             ray_samples_packed_uniform.samples_3d, iter_nr
#         )
#     if isinstance(res_samples, tuple):
#         samples_sdf = res_samples[0]
#     else:
#         samples_sdf = res_samples
#     assert samples_sdf.dim() == 3, "sdfs_fn should return (N, nr_surfs, 1) tensor"

#     # remove extra channels
#     if samples_sdf.shape[2] > 1:
#         samples_sdf = samples_sdf[..., 0:1]

#     # get current surface sdf
#     samples_sdf = samples_sdf[:, surf_idx]

#     # TODO: use occupancy grid if available
#     ray_samples_packed_uniform.update_dt(False)

#     logistic_beta = torch.ones_like(ray_samples_packed_uniform.samples_dt) * logistic_beta_value
#     # print("logistic_beta", logistic_beta)
#     alpha = VolumeRendering.sdf2alpha(
#         ray_samples_packed_uniform,
#         samples_sdf,
#         logistic_beta / 2.0,
#     )
#     # alpha = alpha.clip(0.0, 1.0)
#     transmittance, _ = VolumeRendering.cumprod_one_minus_alpha_to_transmittance(
#         ray_samples_packed_uniform, 1 - alpha + 1e-6
#     )
#     transmittance = transmittance.clip(0.0, 1.0)
#     weights = alpha * transmittance

#     # get normalized weights
#     _, weight_sum_per_sample = VolumeRendering.sum_over_rays(
#         ray_samples_packed_uniform, weights
#     )
#     weight_sum_per_sample = torch.clip(weight_sum_per_sample, min=1e-6)
#     weights /= weight_sum_per_sample  # normalize so that cdf sums up to 1

#     # get distribution and importance sample
#     cdf = VolumeRendering.compute_cdf(ray_samples_packed_uniform, weights)
#     # print("cdf", cdf)

#     ray_samples_packed_imp_1 = VolumeRendering.importance_sample(
#         ray_samples_packed_uniform,
#         cdf,
#         nr_samples // 2,
#         jitter_samples
#     )

#     # second iteration (use importance samples to query the model sdf)

#     if iter_nr is None:
#         res_samples_imp_1 = sdfs_fn(ray_samples_packed_imp_1.samples_3d)
#     else:
#         res_samples_imp_1 = sdfs_fn(
#             ray_samples_packed_imp_1.samples_3d, iter_nr
#         )
#     if isinstance(res_samples_imp_1, tuple):
#         samples_sdf_imp_1 = res_samples_imp_1[0]
#     else:
#         samples_sdf_imp_1 = res_samples_imp_1
#     assert samples_sdf_imp_1.dim() == 3, "sdfs_fn should return (N, nr_surfs, 1) tensor"

#     # remove extra channels
#     if samples_sdf_imp_1.shape[2] > 1:
#         samples_sdf_imp_1 = samples_sdf_imp_1[..., 0:1]

#     # get current surface sdf
#     samples_sdf_imp_1 = samples_sdf_imp_1[:, surf_idx]

#     # set sdf for packets fusion
#     ray_samples_packed_uniform.set_samples_values(samples_sdf)
#     ray_samples_packed_imp_1.set_samples_values(samples_sdf_imp_1)

#     # fuse the uniform and importance samples
#     ray_samples_packed_combined_1 = VolumeRendering.combine_ray_samples_packets(
#         ray_samples_packed_uniform,
#         ray_samples_packed_imp_1,
#         min_dist_between_samples
#     )

#     samples_sdf_combined_1 = ray_samples_packed_combined_1.get_samples_values()

#     # remove samples vals from ray_samples_packed_uniform for next iteration
#     ray_samples_packed_uniform.remove_samples_values()

#     # TODO: use occupancy grid if available
#     ray_samples_packed_combined_1.update_dt(False)

#     logistic_beta = torch.ones_like(ray_samples_packed_combined_1.samples_dt) * logistic_beta_value
#     alpha = VolumeRendering.sdf2alpha(
#         ray_samples_packed_combined_1,
#         samples_sdf_combined_1,
#         logistic_beta,
#     )
#     # alpha = alpha.clip(0.0, 1.0)
#     # print("alpha", alpha)

#     transmittance, _ = VolumeRendering.cumprod_one_minus_alpha_to_transmittance(
#         ray_samples_packed_combined_1, 1 - alpha + 1e-6
#     )
#     # print("transmittance", transmittance)
#     transmittance = transmittance.clip(0.0, 1.0)
#     weights = alpha * transmittance
#     # print("weights", weights)

#     # get normalized weights
#     _, weight_sum_per_sample = VolumeRendering.sum_over_rays(
#         ray_samples_packed_combined_1, weights
#     )
#     weight_sum_per_sample = torch.clip(weight_sum_per_sample, min=1e-6)
#     weights /= weight_sum_per_sample  # normalize so that cdf sums up to 1

#     # get distribution and importance sample
#     cdf = VolumeRendering.compute_cdf(ray_samples_packed_combined_1, weights)
#     # print("cdf", cdf)

#     ray_samples_packed_imp_2 = VolumeRendering.importance_sample(
#         ray_samples_packed_combined_1,
#         cdf,
#         nr_samples // 2,
#         jitter_samples
#     )

#     # remove samples vals from ray_samples_packed_combined_1 for fusion
#     ray_samples_packed_combined_1.remove_samples_values()

#     # remove samples vals from ray_samples_packed_imp for next combination
#     ray_samples_packed_imp_1.remove_samples_values()

#     if profiler is not None:
#         profiler.end("importance_sampling_sdfs")

#     return ray_samples_packed_imp_1, ray_samples_packed_imp_2


# def get_rays_samples_packed_sdfs(
#     rays_o,
#     rays_d,
#     t_near,
#     t_far,
#     sdfs_fn,
#     nr_surfs,
#     logistic_beta_value,
#     occupancy_grid=None,
#     iter_nr=None,
#     min_dist_between_samples=1e-4,
#     min_nr_samples_per_ray=1,
#     max_nr_samples_per_ray=64,
#     max_nr_imp_samples_per_ray=32,
#     jitter_samples=False,
#     importace_sampling=True,
#     values_dim=1,
#     profiler=None,
# ):
#     """get rays samples packed for nerf

#     Args:
#         ...

#     Returns:
#         ray_samples_packed: RaySamplesPacked
#         sdfs_ray_samples_packed_imp: list of RaySamplesPacked objects
#     """

#     # run uniform-sampling per each surface
#     ray_samples_packed = create_fg_samples(
#         min_dist_between_samples=min_dist_between_samples,
#         min_nr_samples_per_ray=min_nr_samples_per_ray,
#         max_nr_samples_per_ray=max_nr_samples_per_ray,
#         rays_o=rays_o,
#         rays_d=rays_d,
#         ray_t_entry=t_near,
#         ray_t_exit=t_far,
#         jitter_samples=jitter_samples,
#         occupancy_grid=occupancy_grid,
#         values_dim=values_dim,
#         profiler=profiler
#     )

#     # per surface samples
#     sdfs_ray_samples_packed_imp = []

#     if not ray_samples_packed.is_empty():
#         if importace_sampling:
#             for surf_idx in range(nr_surfs):

#                 # importance sampling
#                 ray_samples_packed_imp_1, ray_samples_packed_imp_2 = importance_sampling_sdfs(
#                     sdfs_fn=sdfs_fn,
#                     surf_idx=surf_idx,
#                     ray_samples_packed_uniform=ray_samples_packed,
#                     logistic_beta_value=logistic_beta_value,
#                     iter_nr=iter_nr,
#                     nr_samples=max_nr_imp_samples_per_ray//nr_surfs,
#                     jitter_samples=jitter_samples,
#                     min_dist_between_samples=min_dist_between_samples,
#                     profiler=profiler
#                 )

#                 # combine uniform and importance samples
#                 # iter 1: uniform + imp_1
#                 # iter 2: prec_iter_samples + imp_2
#                 sdf_ray_samples_packed_imp = VolumeRendering.combine_ray_samples_packets(
#                     ray_samples_packed_imp_1,
#                     ray_samples_packed_imp_2,
#                     min_dist_between_samples,
#                 )
#                 sdfs_ray_samples_packed_imp.append(sdf_ray_samples_packed_imp)

#                 # combine surfaces samples
#                 ray_samples_packed = VolumeRendering.combine_ray_samples_packets(
#                     ray_samples_packed,
#                     sdf_ray_samples_packed_imp,
#                     min_dist_between_samples,
#                 )

#         # TODO: update dt with or without occupancy
#         ray_samples_packed.update_dt(False)

#     return (
#         ray_samples_packed,
#         sdfs_ray_samples_packed_imp
#     )


def get_rays_samples_packed_sdfs(
    rays_o,
    rays_d,
    t_near,
    t_far,
    sdfs_fn,
    nr_surfs,
    logistic_beta_value,
    occupancy_grid=None,
    iter_nr=None,
    min_dist_between_samples=1e-4,
    min_nr_samples_per_ray=1,
    max_nr_samples_per_ray=64,
    max_nr_imp_samples_per_ray=32,
    jitter_samples=False,
    importace_sampling=True,
    values_dim=1,
    profiler=None,
):
    """get rays samples packed for sdfs"""

    ray_samples_packed_imp = None

    # foreground samples
    ray_samples_packed = create_fg_samples(
        min_dist_between_samples=min_dist_between_samples,
        min_nr_samples_per_ray=min_nr_samples_per_ray,
        max_nr_samples_per_ray=max_nr_samples_per_ray,
        rays_o=rays_o,
        rays_d=rays_d,
        ray_t_entry=t_near,
        ray_t_exit=t_far,
        jitter_samples=jitter_samples,
        occupancy_grid=occupancy_grid,
        values_dim=values_dim,
        profiler=profiler,
    )

    if not ray_samples_packed.is_empty():
        if importace_sampling:

            # importance sampling
            ray_samples_packed_imp_1, ray_samples_packed_imp_2 = (
                importance_sampling_sdfs(
                    sdfs_fn=sdfs_fn,
                    nr_surfs=nr_surfs,
                    ray_samples_packed_uniform=ray_samples_packed,
                    logistic_beta_value=logistic_beta_value,
                    iter_nr=iter_nr,
                    nr_imp_samples=max_nr_imp_samples_per_ray,
                    jitter_samples=jitter_samples,
                    min_dist_between_samples=min_dist_between_samples,
                    profiler=profiler,
                )
            )

            # combine
            ray_samples_packed_imp = VolumeRendering.combine_ray_samples_packets(
                ray_samples_packed_imp_1,
                ray_samples_packed_imp_2,
                min_dist_between_samples,
            )

            ray_samples_packed = VolumeRendering.combine_ray_samples_packets(
                ray_samples_packed, ray_samples_packed_imp, min_dist_between_samples
            )

        # TODO: update dt with or without occupancy
        ray_samples_packed.update_dt(False)

    return ray_samples_packed, ray_samples_packed_imp
