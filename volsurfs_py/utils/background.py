import torch

from volsurfs import VolumeRendering
from volsurfs_py.utils.sampling import create_uncontracted_bg_samples
from volsurfs import RaySampler
from volsurfs_py.volume_rendering.volume_rendering_modules import VolumeRenderingNeRF


def get_bg_color(bg_color_str=None, device="cuda"):
    """Returns torch tensor of the constant background color.
    If bg_color_str is None, returns None."""
    if bg_color_str is None or bg_color_str == "trained":
        bg_color = None
        print("no constant background color specified, bg model will be used")
    elif bg_color_str == "random":
        bg_color = torch.rand(1, 3).to(device)
        print("using random constant background color")
    elif bg_color_str == "black":
        bg_color = torch.zeros(1, 3).to(device)
        print("using black constant background color")
    elif bg_color_str == "white":
        bg_color = torch.ones(1, 3).to(device)
        print("using white constant background color")
    else:
        raise ValueError(
            f"ERROR: {bg_color_str} is an invalid constant background color"
        )
    return bg_color


def render_contracted_bg(
    model_bg,
    raycast,
    nr_samples_bg,
    jitter_samples=False,
    iter_nr=None,
    override={},  # override parameters
    profiler=None,
    render_expected_depth=False,
    render_median_depth=True,
):
    """
    renders background
    """
    # background

    if profiler is not None:
        profiler.start("uniform_bg_sampling")

    # foreground samples
    ray_samples_packed_bg = create_uncontracted_bg_samples(
        nr_samples_bg,
        raycast["rays_o"],
        raycast["rays_d"],
        t_min=raycast["t_far"],
        t_max=100.0,
        jitter_samples=jitter_samples,
    )

    if profiler is not None:
        profiler.end("uniform_bg_sampling")

    if profiler is not None:
        profiler.start("render_contracted_bg")

    if ray_samples_packed_bg.is_empty():
        # no ray samples
        pred_rgb_bg = torch.zeros(raycast["nr_rays"], 3)
    else:

        # apply contraction
        c_ray_samples_packed_bg = RaySampler.contract_samples(ray_samples_packed_bg)

        # handle view dirs override
        override_view_dir = override.get("view_dir", None)
        if override_view_dir is not None:
            rays_d = torch.from_numpy(override_view_dir).float()[None, :]
            rays_d = rays_d.repeat(c_ray_samples_packed_bg.samples_dirs.shape[0], 1)
        else:
            rays_d = c_ray_samples_packed_bg.samples_dirs

        # print(f"rays_d: {rays_d.shape}")
        # print(f"samples_3d: {c_ray_samples_packed_bg.samples_3d.shape}")

        # compute rgb and density
        rgb_samples, density_samples = model_bg(
            c_ray_samples_packed_bg.samples_3d,  # contracted samples
            rays_d,
            iter_nr,
        )

        # compute weights
        alpha = 1.0 - torch.exp(
            -density_samples.view(-1, 1) * c_ray_samples_packed_bg.samples_dt
        )
        # alpha = alpha.clip(min=0.0, max=1.0)

        one_minus_alpha = 1 - alpha
        (
            transmittance,
            _,
        ) = VolumeRenderingNeRF().cumprod_one_minus_alpha_to_transmittance_module(
            c_ray_samples_packed_bg, one_minus_alpha + 1e-6
        )
        weights = alpha * transmittance
        # weights = weights.clip(min=0.0, max=1.0)

        # volumetric integration
        pred_rgb_bg = VolumeRenderingNeRF().integrate_3d(
            c_ray_samples_packed_bg, rgb_samples, weights
        )
        # pred_rgb_bg = pred_rgb_bg.clip(min=0.0, max=1.0)

        if render_expected_depth:
            # expected depth (no need to compute gradients)
            pred_depth_bg = VolumeRendering.integrate_with_weights_1d(
                ray_samples_packed_bg,
                ray_samples_packed_bg.samples_z,
                weights,  # uncontracted z
            )
        else:
            pred_depth_bg = None

        if render_median_depth:
            # median depth (distance where the accumulated weight reaches 0.5)
            median_depth_bg = VolumeRendering.median_depth_over_rays(
                ray_samples_packed_bg, weights, 0.5  # uncontracted z
            )
        else:
            median_depth_bg = None

    if profiler is not None:
        profiler.end("render_contracted_bg")

    res = {
        "pred_rgb": pred_rgb_bg,
        "expected_depth": pred_depth_bg,
        "median_depth": median_depth_bg,
    }

    return res
