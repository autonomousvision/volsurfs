import torch
import numpy as np
from volsurfs import VolumeRendering
from volsurfs_py.utils.sampling import create_fg_samples

# from mvdatasets.geometry.contraction import contract_points


def union(sdfs):
    return torch.min(sdfs, axis=1)[0]


def intersection(sdfs):
    return torch.max(sdfs, axis=1)[0]


def subtraction(sdf1, sdf2):
    return torch.max((sdf1, -sdf2), axis=0)[0]


def sdf_loss_sphere(
    points,
    points_sdf,
    points_sdf_gradients,
    scene_radius,
    sphere_center,
    distance_scale=1.0,
):
    points_in_sphere_coord = points - torch.as_tensor(sphere_center)
    point_dist_to_center = points_in_sphere_coord.norm(dim=-1, keepdim=True)
    dists = (point_dist_to_center - scene_radius) * distance_scale

    loss_dists = ((points_sdf - dists) ** 2).mean()
    eikonal_loss = (points_sdf_gradients.norm(dim=-1) - distance_scale) ** 2
    loss = loss_dists * 3e3 + eikonal_loss.mean() * 5e1

    # return also the loss sdf and loss eik
    loss_sdf = loss_dists
    loss_eik = eikonal_loss.mean()

    return loss, loss_sdf, loss_eik


@torch.no_grad()
def importance_sampling_sdf(
    sdf_fn,
    ray_samples_packed_uniform,
    iter_nr,
    nr_samples,
    logistic_beta_value,
    min_dist_between_samples,
    jitter_samples=False,
    profiler=None,
):
    """importance sampling for surf (query the model sdf);
    returns a RaySamplesPacked object combining the uniform and importance samples

    out:
    ray_samples_packed_imp_1: RaySamplesPacked object
    ray_samples_packed_imp_2: RaySamplesPacked object
    """

    assert (
        not ray_samples_packed_uniform.is_empty()
    ), "ray_samples_packed_uniform should not be empty"

    if profiler is not None:
        profiler.start("importance_sampling_sdf")

    # first iteration

    points = ray_samples_packed_uniform.samples_3d
    # if contract_points:
    #     points = contract_points(points)

    # compute sdf
    if iter_nr is None:
        res_sampled_packed = sdf_fn(points)
    else:
        res_sampled_packed = sdf_fn(points, iter_nr)
    if isinstance(res_sampled_packed, tuple):
        sdf_sampled_packed = res_sampled_packed[0]
    else:
        sdf_sampled_packed = res_sampled_packed
    assert sdf_sampled_packed.dim() == 2, "sdf_fn should return (N, 1) tensor"

    # remove extra channels
    if sdf_sampled_packed.shape[1] > 1:
        sdf_sampled_packed = sdf_sampled_packed[:, 0:1]

    # TODO: use occupancy grid if available
    ray_samples_packed_uniform.update_dt(False)

    logistic_beta = (
        torch.ones_like(ray_samples_packed_uniform.samples_dt) * logistic_beta_value
    )
    alpha = VolumeRendering.sdf2alpha(
        ray_samples_packed_uniform, sdf_sampled_packed, logistic_beta / 2.0
    )
    # alpha = alpha.clip(0.0, 1.0)
    transmittance, _ = VolumeRendering.cumprod_one_minus_alpha_to_transmittance(
        ray_samples_packed_uniform, 1 - alpha + 1e-6
    )
    weights = alpha * transmittance

    # get normalized weights
    _, weight_sum_per_sample = VolumeRendering.sum_over_rays(
        ray_samples_packed_uniform, weights
    )
    weight_sum_per_sample = torch.clip(weight_sum_per_sample, min=1e-6)
    weights /= weight_sum_per_sample  # normalize so that cdf sums up to 1

    # get distribution and importance sample
    cdf = VolumeRendering.compute_cdf(ray_samples_packed_uniform, weights)

    ray_samples_packed_imp_1 = VolumeRendering.importance_sample(
        ray_samples_packed_uniform, cdf, nr_samples // 2, jitter_samples
    )

    # second iteration (use importance samples to query the model sdf)

    points = ray_samples_packed_imp_1.samples_3d
    # if contract_points:
    #     points = contract_points(points)

    if iter_nr is None:
        res_sampled_packed_imp_1 = sdf_fn(points)
    else:
        res_sampled_packed_imp_1 = sdf_fn(points, iter_nr)
    if isinstance(res_sampled_packed_imp_1, tuple):
        sdf_sampled_packed_imp_1 = res_sampled_packed_imp_1[0]
    else:
        sdf_sampled_packed_imp_1 = res_sampled_packed_imp_1
    assert sdf_sampled_packed_imp_1.dim() == 2, "sdf_fn should return (N, 1) tensor"

    # remove extra channels
    if sdf_sampled_packed_imp_1.shape[1] > 1:
        sdf_sampled_packed_imp_1 = sdf_sampled_packed_imp_1[:, 0:1]

    # set sdf for fusion
    ray_samples_packed_uniform.set_samples_values(sdf_sampled_packed)
    ray_samples_packed_imp_1.set_samples_values(sdf_sampled_packed_imp_1)

    # fuse the uniform and importance samples
    ray_samples_combined_1 = VolumeRendering.combine_ray_samples_packets(
        ray_samples_packed_uniform, ray_samples_packed_imp_1, min_dist_between_samples
    )
    sdf_sampled_packed_combined_1 = ray_samples_combined_1.samples_values

    # remove samples values
    ray_samples_packed_uniform.remove_samples_values()
    ray_samples_packed_imp_1.remove_samples_values()

    # TODO: use occupancy grid if available
    ray_samples_combined_1.update_dt(False)

    logistic_beta = (
        torch.ones_like(ray_samples_combined_1.samples_dt) * logistic_beta_value
    )
    alpha = VolumeRendering.sdf2alpha(
        ray_samples_combined_1, sdf_sampled_packed_combined_1, logistic_beta
    )
    # alpha = alpha.clip(0.0, 1.0)
    transmittance, _ = VolumeRendering.cumprod_one_minus_alpha_to_transmittance(
        ray_samples_combined_1, 1 - alpha + 1e-6
    )
    weights = alpha * transmittance

    # get normalized weights
    _, weight_sum_per_sample = VolumeRendering.sum_over_rays(
        ray_samples_combined_1, weights
    )
    weight_sum_per_sample = torch.clip(weight_sum_per_sample, min=1e-6)
    weights /= weight_sum_per_sample  # normalize so that cdf sums up to 1

    # get distribution and importance sample
    cdf = VolumeRendering.compute_cdf(ray_samples_combined_1, weights)

    ray_samples_packed_imp_2 = VolumeRendering.importance_sample(
        ray_samples_combined_1, cdf, nr_samples // 2, jitter_samples
    )

    if profiler is not None:
        profiler.end("importance_sampling_sdf")

    return ray_samples_packed_imp_1, ray_samples_packed_imp_2


def get_rays_samples_packed_sdf(
    rays_o,
    rays_d,
    t_near,
    t_far,
    sdf_fn,
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
    """get rays samples packed for nerf

    Args:
        rays_o (_type_): _description_
        rays_d (_type_): _description_
        t_near (_type_): _description_
        t_far (_type_): _description_
        density_fn (_type_): _description_
        occupancy_grid (_type_): _description_
        iter_nr (_type_, optional): _description_. Defaults to None.
        min_dist_between_samples (_type_, optional): _description_. Defaults to 1e-3.
        min_nr_samples_per_ray (int, optional): _description_. Defaults to 1.
        max_nr_samples_per_ray (int, optional): _description_. Defaults to 64.
        max_nr_imp_samples_per_ray (int, optional): _description_. Defaults to 32.
        jitter_samples (bool, optional): _description_. Defaults to False.
        importace_sampling (bool, optional): _description_. Defaults to True.
        contract_points (bool, optional): _description_. Defaults to False.
        values_dim (int, optional): _description_. Defaults to 1.
        profiler (_type_, optional): _description_. Defaults to None.

    Returns:
        ray_samples_packed: _description_
        ray_samples_packed_imp: _description_
    """

    ray_samples_packed_imp = None

    # foreground samples
    ray_samples_packed = create_fg_samples(
        min_dist_between_samples,
        min_nr_samples_per_ray,
        max_nr_samples_per_ray,
        rays_o,
        rays_d,
        t_near,
        t_far,
        jitter_samples=jitter_samples,
        occupancy_grid=occupancy_grid,
        values_dim=values_dim,
        profiler=profiler,
    )

    if not ray_samples_packed.is_empty():
        if importace_sampling:

            # importance sampling
            ray_samples_packed_imp_1, ray_samples_packed_imp_2 = (
                importance_sampling_sdf(
                    sdf_fn=sdf_fn,
                    ray_samples_packed_uniform=ray_samples_packed,
                    logistic_beta_value=logistic_beta_value,
                    iter_nr=iter_nr,
                    nr_samples=max_nr_imp_samples_per_ray,
                    jitter_samples=jitter_samples,
                    min_dist_between_samples=min_dist_between_samples,
                    # contract_points=contract_points,
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
