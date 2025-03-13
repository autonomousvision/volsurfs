import numpy as np
import torch
from volsurfs_py.utils.points import get_planar_3d_points
from volsurfs_py.utils.plotting_2d import (
    plot_2d_sdfs_together,
    plot_2d_density,
    plot_2d_occupancy,
)
from volsurfs_py.utils.logistic_distribution import logistic_distribution
from volsurfs_py.utils.logistic_distribution import get_logistic_beta_from_variance


@torch.no_grad()
def visualize_fields_sections(method, iter_nr=None):
    # if training an sdf/sdfs, visualize iso-surface/s

    visualizations = {}

    if method.method_name in ["nerf", "surf", "offsets_surfs"]:

        occ_grid = None
        density_fn = None
        sdfs_fn = None
        main_sdf_idx = None
        nr_surfs = 0
        plot_res = 400
        planar_points = []

        if method.method_name == "nerf":
            density_fn = method.models["density"]
            occ_grid = method.occupancy_grid

        if method.method_name == "surf":
            sdfs_fn = method.models["sdf"]
            occ_grid = method.occupancy_grid
            nr_surfs = 1

        if method.method_name == "offsets_surfs":
            sdfs_fn = method.models["sdfs"]
            occ_grid = method.occupancy_grid
            main_sdf_idx = method.main_surf_idx
            nr_surfs = method.nr_surfs

        for i, axis in enumerate(["xy", "yz", "xz"]):

            aabb_min = -1 * method.bounding_primitive.get_radius()
            aabb_max = method.bounding_primitive.get_radius()
            pts = get_planar_3d_points(
                plot_res,
                plot_res,
                axis=axis,
                plane=0,
                min=aabb_min,
                max=aabb_max,
                device="cuda",
            )
            planar_points.append(pts)

        if sdfs_fn is not None:

            # sdfs
            plots_sdf_np = np.zeros((plot_res, plot_res * 3, 3))
            for i, planar_points_3d in enumerate(planar_points):
                planar_points_res = sdfs_fn(planar_points_3d, iter_nr=iter_nr)
                if isinstance(planar_points_res, tuple):
                    planar_points_sdfs = planar_points_res[0]
                else:
                    planar_points_sdfs = planar_points_res
                if planar_points_sdfs.dim() == 3:
                    planar_points_sdfs = planar_points_sdfs.squeeze(-1)
                split_tensors = torch.split(
                    planar_points_sdfs, split_size_or_sections=1, dim=1
                )
                plot_np = plot_2d_sdfs_together(
                    split_tensors, plot_res, plot_res, main_sdf_idx=main_sdf_idx
                )
                plots_sdf_np[:, i * 400 : (i + 1) * 400, :] = plot_np
            visualizations["sdfs"] = plots_sdf_np

            # densities
            if method.method_name == "offsets_surfs":
                plots_density_np = np.zeros((plot_res, plot_res * 3, 3))
                for i, planar_points_3d in enumerate(planar_points):
                    planar_points_res = sdfs_fn(planar_points_3d, iter_nr=iter_nr)
                    if isinstance(planar_points_res, tuple):
                        planar_points_sdfs = planar_points_res[0]
                    else:
                        planar_points_sdfs = planar_points_res
                    if planar_points_sdfs.dim() == 3:
                        planar_points_sdfs = planar_points_sdfs.squeeze(-1)
                    planar_points_density = torch.zeros_like(planar_points_sdfs)
                    s = get_logistic_beta_from_variance(method.variance)
                    for j in range(nr_surfs):
                        planar_points_density[:, j] = logistic_distribution(
                            planar_points_sdfs[:, j], beta=s
                        )
                    planar_points_density = planar_points_density.sum(dim=1)  # (N, 1)
                    # convert to log scale
                    planar_points_density = torch.log(planar_points_density + 1)
                    plot_np = plot_2d_density(planar_points_density, plot_res, plot_res)
                    plots_density_np[:, i * 400 : (i + 1) * 400] = plot_np
                visualizations["densities"] = plots_density_np

        if density_fn is not None:

            # densities
            plots_density_np = np.zeros((plot_res, plot_res * 3, 3))
            for i, planar_points_3d in enumerate(planar_points):
                planar_points_res = density_fn(planar_points_3d, iter_nr=iter_nr)
                if isinstance(planar_points_res, tuple):
                    planar_points_density = planar_points_res[0]
                else:
                    planar_points_density = planar_points_res
                if planar_points_density.dim() == 3:
                    planar_points_density = planar_points_density.squeeze(-1)
                # convert to log scale
                planar_points_density = torch.log(planar_points_density + 1)
                plot_np = plot_2d_density(planar_points_density, plot_res, plot_res)
                plots_density_np[:, i * 400 : (i + 1) * 400] = plot_np
            visualizations["densities"] = plots_density_np

        # occ grid
        if occ_grid is not None:

            if isinstance(occ_grid, list):
                plots_occupancy_np = np.zeros(
                    (plot_res * len(occ_grid), plot_res * 3, 3)
                )
                for k, occ in enumerate(occ_grid):
                    for i, planar_points_3d in enumerate(planar_points):
                        planar_points_occupancy, _ = occ.check_occupancy(
                            planar_points_3d
                        )
                        plot_np = plot_2d_occupancy(
                            planar_points_occupancy, plot_res, plot_res
                        )
                        plots_occupancy_np[
                            k * 400 : (k + 1) * 400, i * 400 : (i + 1) * 400
                        ] = plot_np
            else:
                plots_occupancy_np = np.zeros((plot_res, plot_res * 3, 3))
                for i, planar_points_3d in enumerate(planar_points):
                    planar_points_occupancy, _ = occ_grid.check_occupancy(
                        planar_points_3d
                    )
                    plot_np = plot_2d_occupancy(
                        planar_points_occupancy, plot_res, plot_res
                    )
                    plots_occupancy_np[:, i * 400 : (i + 1) * 400] = plot_np
            visualizations["occupancy"] = plots_occupancy_np

    return visualizations


def visualize_neural_textures(method, iter_nr=None):
    # if the method has neural textures, visualize them

    visualizations = {}

    if method.method_name in ["volsurfs"]:
        if method.hyper_params.using_neural_textures:
            for model_key, model in method.models.items():
                if model is not None:
                    if ("rgb" in model_key) or ("alpha" in model_key):
                        imgs_np = model.render(preview=True, bake=True, iter_nr=iter_nr)
                        img_row_np = np.zeros(
                            (imgs_np[0].shape[0], imgs_np[0].shape[1] * len(imgs_np), 3)
                        )
                        for i, img_np in enumerate(imgs_np):
                            img_row_np[
                                :, i * img_np.shape[1] : (i + 1) * img_np.shape[1]
                            ] = img_np[:, :, :, 0][:, :, :3]
                        visualizations[model_key] = img_row_np

    return visualizations
