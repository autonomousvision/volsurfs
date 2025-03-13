from rich import print
import torch
import torch.nn.functional as F

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from volsurfs import VolumeRendering
from volsurfs import OccupancyGrid
from volsurfs_py.schedulers.warmup import GradualWarmupScheduler
from volsurfs_py.volume_rendering.volume_rendering_modules import VolumeRenderingNeuS
from volsurfs_py.models.offsets_sdf import OffsetsSDF
from volsurfs_py.models.rgb import RGB
from volsurfs_py.models.color_sh import ColorSH
from volsurfs_py.models.nerfhash import NerfHash
from volsurfs_py.utils.sdfs_utils import get_rays_samples_packed_sdfs
from volsurfs_py.utils.debug import sanity_check
from volsurfs_py.utils.losses import loss_l1, eikonal_loss
from volsurfs_py.utils.raycasting import intersect_bounding_primitive
from volsurfs_py.utils.common import map_range_val
from volsurfs_py.methods.base_method import BaseMethod
from volsurfs_py.utils.background import render_contracted_bg
from volsurfs_py.utils.sphere_tracing import sphere_trace
from volsurfs_py.utils.fields_utils import get_field_gradients, get_sdf_curvature
from volsurfs_py.utils.offsets_utils import get_offsets_gt
from volsurfs_py.utils.logistic_distribution import logistic_distribution_stdev
from volsurfs_py.utils.logistic_distribution import get_logistic_beta_from_variance
from volsurfs_py.utils.occupancy_grid import init_occupancy_grid


class OffsetsSurfs(BaseMethod):

    method_name = "offsets_surfs"

    def __init__(
        self,
        train: bool,
        hyper_params,
        load_checkpoints_path,
        save_checkpoints_path,
        bounding_primitive,
        models_path,
        model_colorcal=None,
        bg_color=None,
        start_iter_nr=0,
        profiler=None,
    ):
        super().__init__(
            train,
            hyper_params,
            load_checkpoints_path,
            save_checkpoints_path,
            bounding_primitive=bounding_primitive,
            model_colorcal=model_colorcal,
            bg_color=bg_color,
            profiler=profiler,
            start_iter_nr=start_iter_nr,
        )

        stdev = logistic_distribution_stdev(
            get_logistic_beta_from_variance(
                self.hyper_params.first_phase_variance_start_value
            )
        )

        self.delta_surfs = stdev * self.hyper_params.delta_surfs_multiplier
        self.nr_inner_surfs = self.hyper_params.nr_inner_surfs
        self.nr_outer_surfs = self.hyper_params.nr_outer_surfs

        # offsets gt for initialization
        self.offsets_gt = get_offsets_gt(
            self.nr_outer_surfs, self.nr_inner_surfs, self.delta_surfs
        )

        # instantiate SDF model
        model_sdfs = OffsetsSDF(
            in_channels=3,
            mlp_layers_dims=self.hyper_params.sdf_mlp_layers_dims,
            encoding_type=self.hyper_params.sdf_encoding_type,
            nr_inner_surfs=self.nr_inner_surfs,
            nr_outer_surfs=self.nr_outer_surfs,
            geom_feat_size=self.hyper_params.geom_feat_size,
            nr_iters_for_c2f=0,
            bb_sides=bounding_primitive.get_radius() * 2.0,
        ).to("cuda")
        self.models["sdfs"] = model_sdfs

        if models_path is None and start_iter_nr == 0:
            print(
                "\n[bold yellow]ERROR[/bold yellow]: --models_path not specified, it has to be the path to a folder containing a sdf.pt checkpoint file"
            )
            exit(1)

        # main surf idx
        self.nr_surfs = model_sdfs.nr_surfs
        self.main_surf_idx = model_sdfs.main_surf_idx
        self.nr_inner_surfs = model_sdfs.nr_inner_surfs
        self.nr_outer_surfs = model_sdfs.nr_outer_surfs

        #
        self.in_offsets_init = False
        self.in_color_init = False
        self.in_first_phase = False
        self.in_second_phase = False

        #
        self.just_started_offsets_init = True
        self.just_started_color_init = True
        self.just_started_first_phase = True
        self.just_started_second_phase = True

        #
        self.variance = 1.0
        self.cos_anneal_ratio = 1.0
        #
        self.with_alpha_decay = self.hyper_params.with_alpha_decay
        self.alpha_decay_factor = 1000.0

        #
        self.render_sphere_traced = False
        self.update_occupancy_grid_every_nr_iters = 50

        # instantiate occupancy grid
        if hyper_params.use_occupancy_grid:
            self.occupancy_grid = init_occupancy_grid(bounding_primitive)
        else:
            self.occupancy_grid = None

        # instantiate color model(s)
        for i in range(self.nr_surfs):
            if self.hyper_params.appearance_predict_sh_coeffs:
                if not self.hyper_params.rgb_view_dep:
                    print(
                        "\n[bold yellow]WARNING[/bold yellow]: rgb_view_dep is False, but sh coeffs are used for color prediction; rgb_view_dep is ignored"
                    )
                model_rgb = ColorSH(
                    in_channels=3,
                    out_channels=3,
                    mlp_layers_dims=hyper_params.rgb_mlp_layers_dims,
                    pos_encoder_type=hyper_params.rgb_pos_encoder_type,
                    sh_deg=hyper_params.sh_degree,
                    normal_dep=self.hyper_params.rgb_normal_dep,
                    geom_feat_dep=hyper_params.rgb_geom_feat_dep,
                    in_geom_feat_size=hyper_params.geom_feat_size,
                    nr_iters_for_c2f=hyper_params.rgb_nr_iters_for_c2f,
                    bb_sides=bounding_primitive.get_radius() * 2.0,
                ).to("cuda")
            else:
                # direct color prediction
                model_rgb = RGB(
                    in_channels=3,
                    out_channels=3,
                    mlp_layers_dims=hyper_params.rgb_mlp_layers_dims,
                    pos_encoder_type=hyper_params.rgb_pos_encoder_type,
                    dir_encoder_type=hyper_params.rgb_dir_encoder_type,
                    sh_deg=hyper_params.sh_degree,
                    view_dep=hyper_params.rgb_view_dep,
                    normal_dep=self.hyper_params.rgb_normal_dep,
                    geom_feat_dep=hyper_params.rgb_geom_feat_dep,
                    in_geom_feat_size=hyper_params.geom_feat_size,
                    nr_iters_for_c2f=hyper_params.rgb_nr_iters_for_c2f,
                    use_lipshitz_mlp=False,
                    bb_sides=bounding_primitive.get_radius() * 2.0,
                ).to("cuda")

            if self.hyper_params.are_surfs_colors_indep:
                # one for each surf
                self.models[f"rgb_{i}"] = model_rgb
            else:
                # one for all surfs
                self.models["rgb"] = model_rgb
                break

        # instantiate transparency model(s)
        for i in range(self.nr_surfs):
            # if inner mesh is solid, do not predict transparency for it
            if self.hyper_params.is_inner_surf_solid and i == 0:
                model_transparency = None
            else:
                if self.hyper_params.appearance_predict_sh_coeffs:
                    if not self.hyper_params.transp_view_dep:
                        print(
                            "WARNING: transp_view_dep is False, but sh coeffs are used for transparency prediction; transp_view_dep is ignored"
                        )
                    model_transparency = ColorSH(
                        in_channels=3,
                        out_channels=1,
                        mlp_layers_dims=hyper_params.rgb_mlp_layers_dims,
                        pos_encoder_type=hyper_params.rgb_pos_encoder_type,
                        sh_deg=hyper_params.sh_degree,
                        normal_dep=self.hyper_params.transp_normal_dep,
                        geom_feat_dep=hyper_params.transp_geom_feat_dep,
                        in_geom_feat_size=hyper_params.geom_feat_size,
                        nr_iters_for_c2f=hyper_params.rgb_nr_iters_for_c2f,
                        bb_sides=bounding_primitive.get_radius() * 2.0,
                    ).to("cuda")
                else:
                    model_transparency = RGB(
                        in_channels=3,
                        out_channels=1,
                        mlp_layers_dims=hyper_params.rgb_mlp_layers_dims,
                        pos_encoder_type=hyper_params.rgb_pos_encoder_type,
                        dir_encoder_type=hyper_params.rgb_dir_encoder_type,
                        sh_deg=hyper_params.sh_degree,
                        view_dep=hyper_params.transp_view_dep,
                        normal_dep=self.hyper_params.transp_normal_dep,
                        geom_feat_dep=hyper_params.transp_geom_feat_dep,
                        in_geom_feat_size=hyper_params.geom_feat_size,
                        nr_iters_for_c2f=hyper_params.rgb_nr_iters_for_c2f,
                        use_lipshitz_mlp=False,
                        bb_sides=bounding_primitive.get_radius() * 2.0,
                    ).to("cuda")

            if self.hyper_params.are_surfs_transparency_indep:
                # one for each surf
                self.models[f"alpha_{i}"] = model_transparency
            else:
                # one for all surfs
                self.models["alpha"] = model_transparency
                break

        # instantiate bg model if needed
        if self.bg_color is None:
            model_bg = NerfHash(
                in_channels=3,
                pos_encoder_type=self.hyper_params.bg_pos_encoder_type,
                dir_encoder_type=self.hyper_params.bg_dir_encoder_type,
                nr_iters_for_c2f=self.hyper_params.bg_nr_iters_for_c2f,
            ).to("cuda")
        else:
            model_bg = None
        self.models["bg"] = model_bg

        self.print_models()

        # load checkpoint (continue training and/or evaluation)
        if start_iter_nr > 0:
            self.load(start_iter_nr)

        opt_params = self.collect_opt_params()

        if train:
            self.init_optim(opt_params)

        # if is first iteration, load main sdf checkpoint (and bg if needed)
        if models_path is not None and start_iter_nr == 0:
            ckpt_path = os.path.join(models_path, "sdf.pt")

            if os.path.exists(ckpt_path):
                self.models["sdfs"].load_main_sdf_ckpt(ckpt_path)
                print(
                    f"\n[bold green]SUCCESS[/bold green]: main surf loaded from {ckpt_path}"
                )
            else:
                print(
                    f"\n[bold red]ERROR[/bold red]: checkpoint {ckpt_path} does not exist"
                )
                exit(1)

            # # TODO: load main sdf rgb
            # ckpt_path = os.path.join(models_path, "rgb.pt")
            # if os.path.exists(ckpt_path):
            #     if self.hyper_params.are_surfs_colors_indep:
            #         model = self.models[f"rgb_{i}"]
            #     else:
            #         model = self.models["rgb"]
            #     model.load_state_dict(torch.load(ckpt_path))
            #     print(f"\n[bold green]SUCCESS[/bold green]: main surf loaded from {ckpt_path}")
            # else:
            #     print(f"\n[bold red]ERROR[/bold red]: checkpoint {ckpt_path} does not exist")
            #     exit(1)

            # if not self.hyper_params.are_surfs_colors_indep:
            #     # load rgb
            #     ckpt_path = os.path.join(models_path, "rgb.pt")
            #     if os.path.exists(ckpt_path):
            #         self.models["rgb"].load_state_dict(torch.load(ckpt_path))
            #         print(f"\n[bold green]SUCCESS[/bold green]: rgb loaded from {ckpt_path}")
            #     else:
            #         print(f"\n[bold red]ERROR[/bold red]: checkpoint {ckpt_path} does not exist")
            #         exit(1)

            if self.models["bg"] is not None:
                ckpt_path = os.path.join(models_path, "bg.pt")
                if os.path.exists(ckpt_path):
                    self.models["bg"].load_state_dict(torch.load(ckpt_path))
                    print(
                        f"\n[bold green]SUCCESS[/bold green]: bg loaded from {ckpt_path}"
                    )
                else:
                    print(
                        f"\n[bold red]ERROR[/bold red]: checkpoint {ckpt_path} does not exist"
                    )
                    exit(1)

        self.update_method_state(start_iter_nr)
        self.update_occupancy_grid(iter_nr=start_iter_nr)

    def collect_opt_params(self) -> list:

        # group parameters for optimization
        opt_params = []
        # SDFS
        opt_params.append(
            {
                "params": self.models["sdfs"].pos_encoder.parameters(),
                "weight_decay": 0.0,
                "lr": self.hyper_params.lr,
                "name": "sdfs_pos_encoder",
            }
        )
        opt_params.append(
            {
                "params": self.models["sdfs"].mlp_sdf.parameters(),
                "weight_decay": 0.0,
                "lr": self.hyper_params.lr,
                "name": "sdfs_mlp_sdf",
            }
        )
        # eps heads
        if self.models["sdfs"].use_per_offset_mlp:
            for i, eps_mlp in enumerate(self.models["sdfs"].mlps_eps):
                opt_params.append(
                    {
                        "params": eps_mlp.parameters(),
                        "weight_decay": 0.0,
                        "lr": self.hyper_params.lr,
                        "name": f"sdfs_mlp_eps_{i}",
                    }
                )
        else:
            opt_params.append(
                {
                    "params": self.models["sdfs"].mlp_eps.parameters(),
                    "weight_decay": 0.0,
                    "lr": self.hyper_params.lr,
                    "name": "sdfs_mlp_eps",
                }
            )
        # RGB, TRANSP, BG
        for key, model in self.models.items():
            if "rgb" in key or "alpha" in key or "bg" in key:
                if model is not None:
                    opt_params.append(
                        {
                            "params": model.parameters(),
                            "weight_decay": 0.0,
                            "lr": self.hyper_params.lr,
                            "name": key,
                        }
                    )

        return opt_params

    @torch.no_grad()
    def update_occupancy_grid(self, iter_nr=None, decay=0.0, **kwargs):
        """update occupancy grid with random sdf samples"""

        if self.occupancy_grid is not None:

            if self.profiler is not None:
                self.profiler.start("update_occupancy_grid")

            # sample points in grid voxels
            (
                grid_samples,
                grid_indices,
            ) = self.occupancy_grid.get_grid_samples(False)

            # print("grid_samples", grid_samples.shape)
            grid_samples_batches = torch.split(grid_samples, 256 * 256 * 100, dim=0)

            sdfs_grid = []
            pbar = tqdm(
                grid_samples_batches,
                desc="occupancy values",
                unit="batch",
                leave=False,
                # ncols=100,
                disable=True,
            )
            for grid_samples_batch in pbar:
                # compute sdf
                grid_res_batch = self.models["sdfs"](
                    grid_samples_batch, iter_nr=iter_nr
                )
                if isinstance(grid_res_batch, tuple):
                    sdfs_grid_batch = grid_res_batch[0]
                else:
                    sdfs_grid_batch = grid_res_batch
                sdfs_grid.append(sdfs_grid_batch)
            sdfs_grid = torch.cat(sdfs_grid, dim=0)

            # get min absolute distance value
            sdf_grid = torch.abs(sdfs_grid.squeeze(-1))
            sdf_grid = torch.min(sdf_grid, dim=-1, keepdim=True)[0]

            # do not allow kernel width to be too thin
            occ_grid_variance = min(0.8, self.variance)
            logistic_beta_value = get_logistic_beta_from_variance(occ_grid_variance)
            logistic_beta = torch.ones_like(sdf_grid) * logistic_beta_value

            # update grid values
            self.occupancy_grid.update_grid_values(grid_indices, sdf_grid, decay)

            # update grid occupancy
            occupancy_tresh = 1e-4
            check_neighbours = False
            self.occupancy_grid.update_grid_occupancy_with_sdf_values(
                grid_indices, logistic_beta, occupancy_tresh, check_neighbours
            )

            if self.profiler is not None:
                self.profiler.end("update_occupancy_grid")

    def render_fg_volumetric(
        self,
        nr_rays: int,
        ray_samples_packed,
        logistic_beta_value,
        cos_anneal_ratio=1.0,
        iter_nr=None,
        debug_ray_idx=None,
        override={},  # override parameters
    ):
        """
        renders foreground
        """

        samples_3d = None
        samples_sdfs = None
        samples_sdfs_grad = None

        # foreground

        nr_rays = ray_samples_packed.get_nr_rays()

        if debug_ray_idx is not None:
            debug_pixel_vis = torch.zeros(nr_rays, 1)
            debug_pixel_vis[debug_ray_idx] = 1.0
        else:
            debug_pixel_vis = None

        # final frame tensors
        surfs_rgb = torch.zeros(nr_rays, self.nr_surfs, 3)
        surfs_normals = torch.zeros(nr_rays, self.nr_surfs, 3)
        surfs_depths = torch.zeros(nr_rays, self.nr_surfs, 1)
        surfs_weight_sum = torch.zeros(nr_rays, self.nr_surfs, 1)
        surfs_alpha = torch.zeros(nr_rays, self.nr_surfs, 1)
        surfs_transmittance = torch.ones(nr_rays, self.nr_surfs, 1)
        surfs_blending_weights = torch.zeros(nr_rays, self.nr_surfs, 1)
        pred_rgb_fg = torch.zeros(nr_rays, 3)
        nr_samples = torch.zeros(nr_rays, 1)
        bg_transmittance = torch.ones(nr_rays, 1)

        # if it has samples
        if not ray_samples_packed.is_empty():

            if self.profiler is not None:
                self.profiler.start("render_fg_volumetric")

            # count number of samples per ray
            nr_samples = ray_samples_packed.get_nr_samples_per_ray()[:, None].int()

            samples_3d = ray_samples_packed.samples_3d  # uniform + is
            samples_dirs = ray_samples_packed.samples_dirs
            surf_samples_z = ray_samples_packed.samples_z

            # as in Ref-NeRF
            # reflected_dirs = reflect_rays(rays_dirs=samples_dirs, normals_dirs=normals_samples)

            # compute sdf and normals
            samples_res = self.models["sdfs"].forward(
                points=samples_3d, iter_nr=iter_nr
            )
            samples_sdfs = samples_res[0]  # (nr_samples, nr_surfs, 1)
            # samples_offsets = samples_res[1]  # (nr_samples, nr_surfs, 1)
            samples_geom_feat = samples_res[2]  # (nr_samples, geom_feat_size)

            # sdfs grad function
            samples_sdfs_grad = get_field_gradients(
                self.models["sdfs"].forward, points=samples_3d, iter_nr=iter_nr
            )  # (nr_samples, nr_surfs, 3)

            # normalize gradients
            samples_sdfs_normals = F.normalize(samples_sdfs_grad, dim=-1)

            # COLORS --------------------------------------------

            # iterate over surfaces and render each one independently with individual samples
            for surf_idx in range(self.nr_surfs):

                # color (per-surface or global)
                if self.hyper_params.are_surfs_colors_indep:
                    model_rgb = self.models[f"rgb_{surf_idx}"]
                else:
                    model_rgb = self.models["rgb"]

                surf_samples_rgb = model_rgb(
                    points=samples_3d,
                    samples_dirs=samples_dirs,  # reflected_dirs,
                    normals=samples_sdfs_normals[:, surf_idx],
                    iter_nr=iter_nr,
                    geom_feat=samples_geom_feat,
                )

                # TRANSPARENCY --------------------------------------------

                if self.hyper_params.are_surfs_transparency_indep:
                    model_transparency = self.models[f"alpha_{surf_idx}"]
                else:
                    model_transparency = self.models["alpha"]

                if model_transparency is None:
                    surf_samples_transparency = torch.ones(samples_3d.shape[0], 1)
                else:
                    surf_samples_transparency = model_transparency(
                        points=samples_3d,
                        samples_dirs=samples_dirs,  # reflected_dirs,
                        normals=samples_sdfs_normals[:, surf_idx],
                        iter_nr=iter_nr,
                        geom_feat=samples_geom_feat,
                    )

                # TODO: still testing
                threshold = self.alpha_decay_factor
                transparency_decay = torch.ones_like(surf_samples_transparency)
                if self.with_alpha_decay:
                    with torch.no_grad():
                        # modulate transparency with intersection angle
                        dot = torch.sum(
                            -samples_dirs * samples_sdfs_normals[:, surf_idx],
                            dim=1,
                            keepdim=True,
                        ).clamp(0.0, 1.0)
                        # transparency_decay[dot < threshold] = torch.exp(-10.0 * (threshold - dot[dot < threshold]) / threshold).clamp(0.0, 1.0)
                        # transparency_decay[dot < threshold] = ((1 / threshold) * dot[dot < threshold]).clamp(0.0, 1.0)
                        transparency_decay = torch.sigmoid(threshold * dot) * 2.0 - 1.0
                surf_samples_transparency = (
                    surf_samples_transparency * transparency_decay
                )

                # ALPHA, TRASMITTANCE, WEIGHTS ----------------------------------------

                # individual surface alpha (equivalent of 1 - exp(-sigma_i * dt_i) in NeRF)
                # alpha_beta_i, alpha_gamma_i, ...
                surf_samples_alpha = (
                    VolumeRenderingNeuS().compute_alphas_from_logistic_beta(
                        ray_samples_packed,
                        samples_sdfs[:, surf_idx],
                        samples_sdfs_grad[:, surf_idx],
                        cos_anneal_ratio,
                        logistic_beta_value,
                    )
                )

                surf_samples_transmittance = (
                    VolumeRenderingNeuS().compute_transmittance_from_alphas(
                        ray_samples_packed, surf_samples_alpha
                    )
                )

                surf_samples_weight = surf_samples_alpha * surf_samples_transmittance

                surf_rgb = VolumeRenderingNeuS().integrate_3d(
                    ray_samples_packed, surf_samples_rgb, surf_samples_weight
                )
                surfs_rgb[:, surf_idx] = surf_rgb

                surf_alpha = VolumeRenderingNeuS().integrate_1d(
                    ray_samples_packed, surf_samples_transparency, surf_samples_weight
                )
                surfs_alpha[:, surf_idx] = surf_alpha

                # PER SURFACE ADDITIONAL BUFFERS (no grad) ---------------

                with torch.no_grad():

                    # surfs_depths
                    surf_depths = VolumeRenderingNeuS().integrate_1d(
                        ray_samples_packed, surf_samples_z, surf_samples_weight
                    )
                    surfs_depths[:, surf_idx] = surf_depths

                    surf_weight_sum, _ = VolumeRenderingNeuS().sum_ray_module(
                        ray_samples_packed, surf_samples_weight
                    )
                    surfs_weight_sum[:, surf_idx] = surf_weight_sum

                    # surfs_normals
                    surf_normals = VolumeRendering.integrate_with_weights_3d(
                        ray_samples_packed,
                        samples_sdfs_normals[:, surf_idx],
                        surf_samples_weight,
                    )
                    surfs_normals[:, surf_idx] = surf_normals

            # outer to inner
            surfs_rgb = surfs_rgb.flip(1)
            surfs_alpha = surfs_alpha.flip(1)

            # SURFS + BG TRANSMITTANCES -------------------------------

            # surfs + bg transmittances (outer to inner)
            transmittances = torch.cumprod(1 - surfs_alpha, dim=1)

            if self.nr_surfs == 1:
                surfs_transmittance = torch.ones_like(transmittances)
                bg_transmittance = transmittances.squeeze(-1)
            else:
                surfs_transmittance = transmittances[:, :-1]
                surfs_transmittance = torch.cat(
                    [torch.ones_like(surfs_transmittance[:, -1:]), surfs_transmittance],
                    dim=1,
                )
                bg_transmittance = transmittances[:, -1:].squeeze(-1)

            if self.profiler is not None:
                self.profiler.end("render_fg_volumetric")

            if debug_ray_idx is not None:

                print(
                    "[bold yellow]DEBUG[/bold yellow]: ray debug visualization not implemented"
                )

                # TODO: figure
                # plt.show()

                # plt.savefig(
                #     os.path.join("plots", f"surf_debug_ray.png"),
                #     transparent=True,
                #     bbox_inches="tight",
                #     pad_inches=0,
                #     dpi=300
                # )
                # plt.close()
                exit(0)

            # ALPHA-BLENDING --------------------------------------------

            if self.profiler is not None:
                self.profiler.start("render_blending")

            # alpha_0 * 1.0
            # alpha_1 * (1 - alpha_0)
            # alpha_2 * (1 - alpha_1) * (1 - alpha_0)
            # ...
            # alpha_n-1 * (1 - alpha_n-2) * ... * (1 - alpha_0)

            # outer to inner
            surfs_blending_weights = surfs_transmittance * surfs_alpha

            pred_rgb_fg = (surfs_rgb * surfs_blending_weights).sum(dim=1)

            # inner to outer
            surfs_rgb = surfs_rgb.flip(1)
            surfs_blending_weights = surfs_blending_weights.flip(1)
            surfs_transmittance = surfs_transmittance.flip(1)
            surfs_alpha = surfs_alpha.flip(1)

            if self.profiler is not None:
                self.profiler.end("render_blending")

        renders = {
            # surfs
            "surfs_rgb": surfs_rgb,
            "surfs_normals": surfs_normals,
            "surfs_depths": surfs_depths,
            "surfs_weight_sum": surfs_weight_sum,
            "surfs_alpha": surfs_alpha,
            "surfs_transmittance": surfs_transmittance,
            "surfs_blending_weights": surfs_blending_weights,
            "nr_samples": nr_samples,
            # rays
            "rgb_fg": pred_rgb_fg,
            "bg_transmittance": bg_transmittance,
        }

        if debug_pixel_vis is not None:
            renders["debug_pixel_vis"] = debug_pixel_vis

        return renders, samples_3d, samples_sdfs, samples_sdfs_grad

    def render_fg_sphere_traced(
        self,
        raycast: dict,
        max_st_steps,
        converged_dist_tresh,
        iter_nr=None,
    ):
        """
        renders foreground with sphere tracing
        """

        nr_rays = raycast["nr_rays"]
        rays_o = raycast["rays_o"]
        rays_d = raycast["rays_d"]

        # final frame tensors
        surfs_hit = torch.zeros(nr_rays, self.nr_surfs, 1)
        surfs_normals = torch.zeros(nr_rays, self.nr_surfs, 3)
        surfs_depths = torch.zeros(nr_rays, self.nr_surfs, 1)
        surfs_rgb = torch.zeros(nr_rays, self.nr_surfs, 3)
        surfs_alpha = torch.zeros(nr_rays, self.nr_surfs, 1)
        surfs_transmittance = torch.ones(nr_rays, self.nr_surfs, 1)
        pred_rgb_fg = torch.zeros(nr_rays, 3)
        bg_transmittance = torch.ones(nr_rays, 1)

        surfs_samples_3d = []
        surfs_samples_sdf = []
        surfs_samples_sdf_grad = []

        for surf_idx in range(self.nr_surfs):

            with torch.no_grad():

                if self.profiler is not None:
                    self.profiler.start("sphere_tracing")

                #
                ray_samples_packed, ray_hit_flag = sphere_trace(
                    sdf_fn=self.models["sdfs"],
                    nr_sphere_traces=max_st_steps,
                    rays_o=rays_o,
                    rays_d=rays_d,
                    bounding_primitive=self.bounding_primitive,
                    sdf_multiplier=1.0,
                    sdf_converged_tresh=converged_dist_tresh,
                    occupancy_grid=self.occupancy_grid,
                    iter_nr=iter_nr,
                    surf_idx=surf_idx,
                )
                is_hit = ray_hit_flag  # (n_rays, )
                ray_end = ray_samples_packed.samples_3d  # (n_rays, 3)
                dists = ray_samples_packed.samples_z  # (n_rays, 1)

                if self.profiler is not None:
                    self.profiler.end("sphere_tracing")

            # foreground

            if self.profiler is not None:
                self.profiler.start("render_fg_sphere_traced")

            nr_hits = is_hit.sum()
            if nr_hits > 0:

                samples_3d = ray_end[is_hit]  # (nr_hits, 3)

                # compute sdf and normals
                res_ray_end_hit = self.models["sdfs"].forward(
                    samples_3d, iter_nr=iter_nr
                )
                if isinstance(res_ray_end_hit, tuple):
                    sdf_ray_end_hit = res_ray_end_hit[0]
                    samples_geom_feat = res_ray_end_hit[2]
                else:
                    sdf_ray_end_hit = res_ray_end_hit
                    samples_geom_feat = None
                sdf_ray_end_hit_grad = get_field_gradients(
                    self.models["sdfs"].forward, samples_3d, iter_nr=iter_nr
                )[:, surf_idx]

                surfs_samples_3d.append(samples_3d)
                surfs_samples_sdf.append(sdf_ray_end_hit)
                surfs_samples_sdf_grad.append(sdf_ray_end_hit_grad)

                surfs_hit[:, surf_idx] = is_hit.unsqueeze(-1)
                surfs_normals[is_hit, surf_idx] = F.normalize(
                    sdf_ray_end_hit_grad, dim=1
                )
                surfs_depths[is_hit, surf_idx] = dists[is_hit]

                # compute rgb
                if self.hyper_params.are_surfs_colors_indep:
                    model_rgb = self.models[f"rgb_{surf_idx}"]
                else:
                    model_rgb = self.models["rgb"]

                surfs_rgb_hit = model_rgb(
                    points=samples_3d,
                    samples_dirs=rays_d[is_hit],
                    normals=surfs_normals[is_hit, surf_idx],
                    iter_nr=iter_nr,
                    geom_feat=samples_geom_feat,
                )
                surfs_rgb[is_hit, surf_idx] = surfs_rgb_hit

                # compute alpha
                if self.hyper_params.are_surfs_transparency_indep:
                    model_transparency = self.models[f"alpha_{surf_idx}"]
                else:
                    model_transparency = self.models["alpha"]

                if model_transparency is None:
                    surfs_alpha_hit = torch.ones(samples_3d.shape[0], 1)
                else:
                    surfs_alpha_hit = model_transparency(
                        points=samples_3d,
                        samples_dirs=rays_d[is_hit],
                        normals=surfs_normals[is_hit, surf_idx],
                        iter_nr=iter_nr,
                        geom_feat=samples_geom_feat,
                    )
                surfs_alpha[is_hit, surf_idx] = surfs_alpha_hit

        # alpha-blending

        # flip from inner to outer to outer to inner
        surfs_rgb = surfs_rgb.flip(1)
        surfs_alpha = surfs_alpha.flip(1)

        # SURFS + BG TRANSMITTANCES -------------------------------

        # surfs + bg transmittances (outer to inner)
        transmittances = torch.cumprod(1 - surfs_alpha, dim=1)

        if self.nr_surfs == 1:
            surfs_transmittance = torch.ones_like(transmittances)
            bg_transmittance = transmittances.squeeze(-1)
        else:
            surfs_transmittance = transmittances[:, :-1]
            surfs_transmittance = torch.cat(
                [torch.ones_like(surfs_transmittance[:, -1:]), surfs_transmittance],
                dim=1,
            )
            bg_transmittance = transmittances[:, -1:].squeeze(-1)

        if self.profiler is not None:
            self.profiler.end("render_fg_sphere_traced")

        # ALPHA-BLENDING --------------------------------------------

        if self.profiler is not None:
            self.profiler.start("render_blending")

        # alpha_0 * 1.0
        # alpha_1 * (1 - alpha_0)
        # alpha_2 * (1 - alpha_1) * (1 - alpha_0)
        # ...
        # alpha_n-1 * (1 - alpha_n-2) * ... * (1 - alpha_0)

        # outer to inner
        surfs_blending_weights = surfs_transmittance * surfs_alpha

        pred_rgb_fg = (surfs_rgb * surfs_blending_weights).sum(dim=1)

        # inner to outer
        surfs_rgb = surfs_rgb.flip(1)
        surfs_blending_weights = surfs_blending_weights.flip(1)
        surfs_transmittance = surfs_transmittance.flip(1)
        surfs_alpha = surfs_alpha.flip(1)

        if self.profiler is not None:
            self.profiler.end("render_blending")

        # prepare output
        if len(surfs_samples_3d) > 0:
            samples_3d = torch.cat(surfs_samples_3d, dim=0)  # (nr_pts, 3)
            samples_sdf = torch.cat(surfs_samples_sdf, dim=0)  # (nr_pts, 1)
            samples_sdf_grad = torch.cat(surfs_samples_sdf_grad, dim=0)  # (nr_pts, 3)
        else:
            samples_3d = None
            samples_sdf = None
            samples_sdf_grad = None

        renders = {
            # surfs
            "surfs_rgb": surfs_rgb,
            "surfs_alpha": surfs_alpha,
            "surfs_depths": surfs_depths,
            "surfs_normals": surfs_normals,
            "surfs_transmittance": surfs_transmittance,
            "surfs_blending_weights": surfs_blending_weights,
            # rays
            "rgb_fg": pred_rgb_fg,
            "bg_transmittance": bg_transmittance,
        }

        return renders, samples_3d, samples_sdf, samples_sdf_grad

    def render_rays(
        self,
        rays_o,
        rays_d,
        iter_nr=None,
        override={},
        debug_ray_idx=None,
        **kwargs,
    ) -> dict:
        """render a batch of rays"""

        # intersect bounding primitive
        raycast = intersect_bounding_primitive(self.bounding_primitive, rays_o, rays_d)

        # FOREGROUND

        # prepare output
        st_renders = None
        vol_renders = None
        vol_samples_3d = None
        _ = None
        vol_samples_grad = None

        # handle variance override
        override_variance = override.get("variance", None)
        if override_variance is not None:
            variance = override_variance
        else:
            variance = self.variance
        logistic_beta_value = get_logistic_beta_from_variance(variance)

        # handle cos_anneal_ratio override
        override_cos_anneal_ratio = override.get("cos_anneal_ratio", None)
        if override_cos_anneal_ratio is not None:
            cos_anneal_ratio = override_cos_anneal_ratio
        else:
            cos_anneal_ratio = self.cos_anneal_ratio

        # foreground volume rendering
        res = get_rays_samples_packed_sdfs(
            rays_o,
            rays_d,
            t_near=raycast["t_near"],
            t_far=raycast["t_far"],
            sdfs_fn=self.models["sdfs"],
            nr_surfs=self.nr_surfs,
            occupancy_grid=self.occupancy_grid,
            iter_nr=iter_nr,
            logistic_beta_value=logistic_beta_value,
            min_dist_between_samples=self.hyper_params.min_dist_between_samples,
            min_nr_samples_per_ray=self.hyper_params.min_nr_samples_per_ray,
            max_nr_samples_per_ray=self.hyper_params.max_nr_samples_per_ray,
            max_nr_imp_samples_per_ray=self.hyper_params.max_nr_imp_samples_per_ray,
            jitter_samples=self.is_training,
            importace_sampling=self.hyper_params.do_importance_sampling,
            profiler=self.profiler,
        )
        ray_samples_packed = res[0]

        # volumetric rendering
        (vol_renders, vol_samples_3d, _, vol_samples_grad) = self.render_fg_volumetric(
            raycast["nr_rays"],
            ray_samples_packed=ray_samples_packed,
            logistic_beta_value=logistic_beta_value,
            cos_anneal_ratio=cos_anneal_ratio,
            iter_nr=iter_nr,
            debug_ray_idx=debug_ray_idx,
        )

        # foreground surface rendering (only at test time)
        if self.render_sphere_traced and not self.is_training:
            (st_renders, _, _, _) = self.render_fg_sphere_traced(
                raycast,
                max_st_steps=100,
                converged_dist_tresh=1e-3,
                iter_nr=iter_nr,
            )

        # BACKGROUND ---------------------------------------------

        if self.models["bg"] is None and self.bg_color is not None:
            # constant bg color
            rgb_bg = self.bg_color.expand(raycast["nr_rays"], 3)
            depth_bg = raycast["t_far"]
        else:
            bg_res = render_contracted_bg(
                self.models["bg"],
                raycast,
                nr_samples_bg=self.hyper_params.nr_samples_bg,
                jitter_samples=self.is_training,
                iter_nr=iter_nr,
                profiler=self.profiler,
            )
            rgb_bg = bg_res["pred_rgb"]
            # expected_depth_bg = bg_res["expected_depth"]
            median_depth_bg = bg_res["median_depth"]
            depth_bg = median_depth_bg

        # COMPOSITING ---------------------------------------------

        renders = {}

        # # combine sphere traced with bg
        # if st_renders is not None:
        #     st_renders["rgb"] = (
        #         st_renders["rgb_fg"]
        #         + st_renders["bg_transmittance"] * pred_rgb_bg
        #     )
        #     renders["sphere_traced"] = st_renders

        # # combine fine volumetric with bg
        # if vol_renders is not None:
        #     vol_renders["rgb"] = (
        #         vol_renders["rgb_fg"]
        #         + vol_renders["bg_transmittance"] * pred_rgb_bg
        #     )
        #     vol_renders["rgb_bg"] = pred_rgb_bg
        #     if expected_depth_bg is not None:
        #         vol_renders["expected_depth_bg"] = expected_depth_bg
        #     if median_depth_bg is not None:
        #         vol_renders["median_depth_bg"] = median_depth_bg

        #     renders["volumetric"] = vol_renders

        # combine volumetric with bg
        if vol_renders is not None:

            # blend rgb
            vol_renders["rgb_bg"] = rgb_bg
            rgb_fg = vol_renders["rgb_fg"]
            vol_renders["rgb"] = rgb_fg + rgb_bg * vol_renders["bg_transmittance"]

            # blend depth
            # vol_renders["depth_bg"] = depth_bg
            # depth_fg = vol_renders["depth_fg"]
            # vol_renders["depth"] = depth_fg * vol_renders["weights_sum"] + depth_bg * vol_renders["bg_transmittance"]

            renders["volumetric"] = vol_renders

        res = {
            "renders": renders,
            "samples_3d": vol_samples_3d,
            "samples_grad": vol_samples_grad,
        }

        return res

    def update_method_state(self, iter_nr):

        # init
        offsets_init_end_iter = self.hyper_params.init_phase_end_iter
        color_init_end_iter = self.hyper_params.color_init_phase_end_iter
        init_end_iter = color_init_end_iter
        self.in_offsets_init = iter_nr < offsets_init_end_iter
        self.in_color_init = (iter_nr >= offsets_init_end_iter) and (
            iter_nr < color_init_end_iter
        )

        # training
        first_phase_start_iter = init_end_iter
        first_phase_end_iter = self.hyper_params.first_phase_end_iter
        self.in_first_phase = (iter_nr >= first_phase_start_iter) and (
            iter_nr < first_phase_end_iter
        )
        self.in_second_phase = iter_nr >= first_phase_end_iter

        # check if should update occupancy grid (if method supports it)
        if self.is_training:
            if not self.in_color_init:
                if self.hyper_params.use_occupancy_grid is not None and (
                    iter_nr % self.update_occupancy_grid_every_nr_iters == 0
                ):
                    self.update_occupancy_grid(iter_nr=iter_nr)

        # offsets init
        if self.in_offsets_init:
            if self.is_training and self.just_started_offsets_init:
                print(f"\nstarting offsets init {self.offsets_gt}")
                self.models["sdfs"].freeze_main_surf()
                self.cos_anneal_ratio = 1.0
                self.alpha_decay_factor = 1000.0
                self.variance = self.hyper_params.first_phase_variance_start_value
                self.update_occupancy_grid(iter_nr)
                self.just_started_offsets_init = False

        # colors init
        if self.in_color_init:
            if self.is_training and self.just_started_color_init:
                print("\nstarting color init")
                # freeze offsets during color init
                self.models["sdfs"].freeze_main_surf()
                self.models["sdfs"].freeze_offsets()
                self.cos_anneal_ratio = 1.0
                self.alpha_decay_factor = 1000.0
                self.variance = self.hyper_params.first_phase_variance_start_value
                self.update_occupancy_grid(iter_nr)
                self.just_started_color_init = False

        if self.in_first_phase:

            # train support surfaces until they peak

            if self.is_training and self.just_started_first_phase:
                print("\nstarting first phase")
                self.models["sdfs"].unfreeze_main_surf()
                self.models["sdfs"].unfreeze_offsets()
                self.update_occupancy_grid(iter_nr)
                # init lr scheduler
                if self.lr_scheduler is None and self.scheduler_lr_decay is not None:
                    self.lr_scheduler = GradualWarmupScheduler(
                        self.optimizer,
                        multiplier=1,
                        total_epoch=self.hyper_params.nr_warmup_iters,
                        after_scheduler=self.scheduler_lr_decay,
                    )
                self.just_started_first_phase = False

            # self.cos_anneal_ratio = map_range_val(
            #     iter_nr,
            #     first_phase_start_iter,
            #     first_phase_end_iter,
            #     0.0,
            #     1.0,
            # )

            self.cos_anneal_ratio = 1.0
            self.variance = map_range_val(
                iter_nr,
                first_phase_start_iter,
                first_phase_end_iter,
                self.hyper_params.first_phase_variance_start_value,
                self.hyper_params.first_phase_variance_end_value,
            )

            # TODO: testing
            self.alpha_decay_factor = map_range_val(
                iter_nr,
                first_phase_start_iter,
                first_phase_end_iter,
                1000,
                10.0,
            )

        if self.in_second_phase:

            if self.is_training and self.just_started_second_phase:
                print("\nstarting second phase")
                self.models["sdfs"].unfreeze_main_surf()
                self.models["sdfs"].unfreeze_offsets()
                self.update_occupancy_grid(iter_nr)
                self.just_started_second_phase = False

                self.cos_anneal_ratio = 1.0
                self.variance = self.hyper_params.first_phase_variance_end_value
                self.alpha_decay_factor = 10.0

        if iter_nr % 100 == 0:
            print(f"\niter_nr: {iter_nr}")
            print(f"in_offsets_init: {self.in_offsets_init}")
            print(f"in_color_init: {self.in_color_init}")
            print(f"in_first_phase: {self.in_first_phase}")
            print(f"in_second_phase: {self.in_second_phase}")
            print(f"variance: {self.variance}")
            print(f"nr_surfs: {self.nr_surfs}")
            print(f"main_surf_idx: {self.main_surf_idx}")
            print(f"is_training_main_surf: {self.models['sdfs'].is_training_main_surf}")
            print(f"is_training_offsets: {self.models['sdfs'].is_training_offsets}")

    def forward(
        self,
        rays_o,
        rays_d,
        gt_rgb,
        gt_mask,
        iter_nr,
        is_first_iter=False,
    ):
        """forward pass, compute losses"""

        loss = 0.0
        curvature_loss = 0.0
        loss_eikonal_main = 0.0
        loss_eikonal_supp = 0.0
        loss_offsurface_high_sdf = 0.0
        loss_rgb = 0.0
        loss_mask = 0.0

        self.update_method_state(iter_nr)

        if self.in_offsets_init:

            if self.profiler is not None:
                self.profiler.start("offsets_init")

            if self.nr_surfs == 1:
                print(
                    "\n[bold yellow]WARNING[/bold yellow]: only one surface, offsets init not possible"
                )
                # skip forward
            elif not self.models["sdfs"].is_training_offsets:
                print(
                    "\n[bold yellow]WARNING[/bold yellow]: in offsets_init phase but offsets training is not activated"
                )
                # skip forward
            else:
                # sample random points
                with torch.set_grad_enabled(False):
                    samples_3d = self.bounding_primitive.get_random_points_inside(30000)

                # get geom feats
                _, geom_feats = self.models["sdfs"].main_sdf(
                    samples_3d, iter_nr=iter_nr
                )

                # eikonal loss on support surfaces
                points_sdfs_grad = get_field_gradients(
                    self.models["sdfs"].forward,
                    samples_3d,
                    iter_nr=iter_nr,
                )

                # get offsets predictions
                offsets_pos, offsets_neg, _, _ = self.models["sdfs"].get_offsets(
                    geom_feats
                )
                points_offsets = torch.cat((offsets_pos, offsets_neg), dim=1)

                offsets_gt = self.offsets_gt.unsqueeze(0).expand(
                    samples_3d.shape[0], self.nr_surfs - 1
                )

                offsets_loss = (torch.abs(points_offsets - offsets_gt)).mean()
                loss += offsets_loss

                if points_sdfs_grad.dim() > 2 and points_sdfs_grad.shape[1] > 1:
                    support_sdfs_grad = torch.cat(
                        (
                            points_sdfs_grad[:, : self.main_surf_idx],
                            points_sdfs_grad[:, (self.main_surf_idx + 1) :],
                        ),
                        dim=1,
                    )
                    loss_eikonal_supp = (
                        eikonal_loss(support_sdfs_grad)
                        * self.hyper_params.support_surfs_eikonal_weight
                    )
                    loss += loss_eikonal_supp

            if self.profiler is not None:
                self.profiler.end("offsets_init")

            s_points_3d = None

        else:

            # in_color_init or in_first_phase or in_second_phase

            # train from data

            if self.profiler is not None:
                self.profiler.start("forward_image")

            # forward through the network and get the prediction
            res = self.render_rays(
                rays_o=rays_o,
                rays_d=rays_d,
                iter_nr=iter_nr,
            )
            renders = res["renders"]
            s_points_3d = res["samples_3d"]
            s_points_sdfs_grad = res["samples_grad"]  # (nr_samples, nr_surfs, 3)

            pred_rgb = renders["volumetric"]["rgb"]
            # pred_mask = 1 - renders["volumetric"]["bg_transmittance"]

            # supersampling
            nr_training_rays_per_pixel = self.hyper_params.nr_training_rays_per_pixel
            if nr_training_rays_per_pixel > 1:
                # arithmetic mean of the subpixels values
                pred_rgb = pred_rgb.view(-1, nr_training_rays_per_pixel, 3).mean(dim=1)
                # pred_mask = pred_mask.view(-1, nr_training_rays_per_pixel, 1).mean(dim=1)

            if self.profiler is not None:
                self.profiler.end("forward_image")

            # rgb loss

            if self.profiler is not None:
                self.profiler.start("l1_loss")

            if self.hyper_params.is_training_masked:
                loss_rgb = loss_l1(gt_rgb, pred_rgb, mask=gt_mask)
            else:
                loss_rgb = loss_l1(gt_rgb, pred_rgb)
            loss += loss_rgb

            if self.profiler is not None:
                self.profiler.end("l1_loss")

            # vol rendering points losses

            # points losses

            if self.profiler is not None:
                self.profiler.start("forward_points")

            # sample random points in occupied space
            nr_points = 1024
            with torch.set_grad_enabled(False):
                r_points_3d = self.bounding_primitive.get_random_points_inside(
                    nr_points
                )

            # predict points 3d sdfs, offsets and grads
            (r_points_sdfs, _, _) = self.models["sdfs"](r_points_3d, iter_nr=iter_nr)
            r_points_sdfs_grad = get_field_gradients(
                self.models["sdfs"].forward,
                r_points_3d,
                iter_nr=iter_nr,
            )

            if self.profiler is not None:
                self.profiler.end("forward_points")

            # eikonal loss (main surf)
            if self.hyper_params.eikonal_weight > 0.0:

                if self.models["sdfs"].is_training_main_surf:

                    if self.profiler is not None:
                        self.profiler.start("main_eikonal_loss")

                    # on random points
                    if r_points_sdfs_grad.dim() > 2 and r_points_sdfs_grad.shape[1] > 1:
                        r_points_main_sdf_grad = r_points_sdfs_grad[
                            :, self.main_surf_idx
                        ]
                    else:
                        r_points_main_sdf_grad = r_points_sdfs_grad.squeeze(1)
                    loss_eikonal_main = (
                        eikonal_loss(r_points_main_sdf_grad)
                        * self.hyper_params.eikonal_weight
                    )
                    # on surface points
                    if s_points_3d is not None and s_points_3d.shape[0] > 0:
                        if (
                            s_points_sdfs_grad.dim() > 2
                            and s_points_sdfs_grad.shape[1] > 1
                        ):
                            s_points_sdf_grad = s_points_sdfs_grad[
                                :, self.main_surf_idx
                            ]
                        else:
                            s_points_sdf_grad = s_points_sdfs_grad.squeeze(1)
                        loss_eikonal_main += (
                            eikonal_loss(s_points_sdf_grad)
                            * self.hyper_params.eikonal_weight
                        )
                    loss += loss_eikonal_main

                    if self.profiler is not None:
                        self.profiler.end("main_eikonal_loss")

            # eikonal loss (support surfs)
            if (
                self.hyper_params.eikonal_weight > 0.0
                and self.hyper_params.support_surfs_eikonal_weight > 0.0
            ):

                if self.models["sdfs"].is_training_offsets:

                    if self.profiler is not None:
                        self.profiler.start("supp_eikonal_loss")

                    # on random points
                    if r_points_sdfs_grad.dim() > 2 and r_points_sdfs_grad.shape[1] > 1:
                        r_support_sdfs_grad = torch.cat(
                            (
                                r_points_sdfs_grad[:, : self.main_surf_idx],
                                r_points_sdfs_grad[:, (self.main_surf_idx + 1) :],
                            ),
                            dim=1,
                        )
                        loss_eikonal_supp = (
                            eikonal_loss(r_support_sdfs_grad)
                            * self.hyper_params.support_surfs_eikonal_weight
                        )
                        # on surface points
                        if s_points_3d is not None and s_points_3d.shape[0] > 0:
                            s_support_sdfs_grad = torch.cat(
                                (
                                    s_points_sdfs_grad[:, : self.main_surf_idx],
                                    s_points_sdfs_grad[:, (self.main_surf_idx + 1) :],
                                ),
                                dim=1,
                            )
                            loss_eikonal_supp += (
                                eikonal_loss(s_support_sdfs_grad)
                                * self.hyper_params.support_surfs_eikonal_weight
                            )
                        loss += loss_eikonal_supp
                    else:
                        print(
                            "\n[bold yellow]WARNING[/bold yellow]: support surfaces eikonal loss not possible"
                        )

                    if self.profiler is not None:
                        self.profiler.end("supp_eikonal_loss")

            # loss for empty space sdf
            if self.hyper_params.offsurface_weight > 0.0:

                if self.profiler is not None:
                    self.profiler.start("offsurface_loss")

                # enforce only on main sdf
                r_points_sdf = r_points_sdfs[:, self.main_surf_idx]

                # penalise low sdf values in empty space
                loss_offsurface_high_sdf = (
                    torch.exp(-1e2 * torch.abs(r_points_sdf)).mean()
                    * self.hyper_params.offsurface_weight
                )
                loss += loss_offsurface_high_sdf

                if self.profiler is not None:
                    self.profiler.end("offsurface_loss")

            if (
                self.hyper_params.curvature_weight > 0.0
                and s_points_3d is not None
                and s_points_3d.shape[0] > 0
            ):
                sdf_curvature = get_sdf_curvature(
                    self.models["sdfs"],
                    s_points_3d,
                    s_points_sdfs_grad,
                    iter_nr=iter_nr,
                )
                curvature_loss = (
                    sdf_curvature.mean() * self.hyper_params.curvature_weight
                )
                loss += curvature_loss

        losses = {
            "loss": loss,
            "curvature": curvature_loss,
            "eikonal_main": loss_eikonal_main,
            "eikonal_supp": loss_eikonal_supp,
            "loss_offsurface_high_sdf": loss_offsurface_high_sdf,
            "rgb": loss_rgb,
            "mask": loss_mask,
        }

        additional_info_to_log = {
            "stdev": logistic_distribution_stdev(
                get_logistic_beta_from_variance(self.variance)
            ),
        }
        if self.occupancy_grid is not None:
            additional_info_to_log["occupied_voxels_roi"] = (
                self.occupancy_grid.get_nr_occupied_voxels_in_roi()
            )

        return losses, additional_info_to_log, s_points_3d
