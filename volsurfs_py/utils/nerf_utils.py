import torch

from volsurfs import VolumeRendering
from volsurfs_py.utils.sampling import create_fg_samples

# from mvdatasets.geometry.contraction import contract_points


@torch.no_grad()
def importance_sampling_nerf(
    density_fn,
    ray_samples_packed_uniform,
    rays_o,
    rays_d,
    iter_nr,
    nr_samples,
    jitter_samples=False,
    # contract_points=False,
    profiler=None,
):
    """importance sampling for nerf (query the model density);
    returns a RaySamplesPacked object combining the uniform and importance samples

    out:
    ray_samples_combined
    ray_samples_packed_imp
    """

    assert (
        not ray_samples_packed_uniform.is_empty()
    ), "ray_samples_packed_uniform should not be empty"

    if profiler is not None:
        profiler.start("importance_sampling_nerf")

    # compute density
    points = ray_samples_packed_uniform.samples_3d
    # if contract_points:
    #     points = contract_points(points)

    if iter_nr is None:
        samples_res_uniform = density_fn(points=points)
    else:
        samples_res_uniform = density_fn(
            points=points,
            iter_nr=iter_nr,
        )
    if isinstance(samples_res_uniform, tuple):
        samples_densities_uniform = samples_res_uniform[0]
    else:
        samples_densities_uniform = samples_res_uniform
    assert (
        samples_densities_uniform.dim() == 2
    ), "density_fn should return (N, 1) tensor"

    # remove extra channels
    if samples_densities_uniform.shape[1] > 1:
        samples_densities_uniform = samples_densities_uniform[:, 0:1]

    # TODO: use occupancy grid if available
    ray_samples_packed_uniform.update_dt(False)

    samples_dt = ray_samples_packed_uniform.samples_dt
    alpha = torch.clamp(
        1.0 - torch.exp(-samples_densities_uniform * samples_dt), min=0.0, max=1.0
    )
    transmittance, _ = VolumeRendering.cumprod_one_minus_alpha_to_transmittance(
        ray_samples_packed_uniform, 1 - alpha + 1e-6
    )
    weights = alpha * transmittance

    # get normalized weights
    _, weight_sum_per_sample = VolumeRendering.sum_over_rays(
        ray_samples_packed_uniform, weights
    )
    weight_sum_per_sample = torch.clamp(weight_sum_per_sample, min=1e-6)
    weights /= weight_sum_per_sample

    # compute cdf
    cdf = VolumeRendering.compute_cdf(ray_samples_packed_uniform, weights)

    ray_samples_packed_imp = VolumeRendering.importance_sample(
        ray_samples_packed_uniform,
        cdf,
        nr_samples,
        jitter_samples,
    )

    if profiler is not None:
        profiler.end("importance_sampling_nerf")

    return ray_samples_packed_imp


def get_rays_samples_packed_nerf(
    rays_o,
    rays_d,
    t_near,
    t_far,
    density_fn,
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
            ray_samples_packed_imp = importance_sampling_nerf(
                density_fn=density_fn,
                ray_samples_packed_uniform=ray_samples_packed,
                rays_o=rays_o,
                rays_d=rays_d,
                iter_nr=iter_nr,
                nr_samples=max_nr_imp_samples_per_ray,
                jitter_samples=jitter_samples,
                profiler=profiler,
            )

            ray_samples_packed = VolumeRendering.combine_ray_samples_packets(
                ray_samples_packed, ray_samples_packed_imp, min_dist_between_samples
            )

        # TODO: update dt with or without occupancy
        ray_samples_packed.update_dt(False)

    return ray_samples_packed, ray_samples_packed_imp
