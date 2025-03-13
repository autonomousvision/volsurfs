import torch
import torch.nn.functional as F

import os
import numpy as np
from tqdm import tqdm

from volsurfs import VolumeRendering
from volsurfs import OccupancyGrid
import matplotlib.pyplot as plt
from volsurfs_py.schedulers.warmup import GradualWarmupScheduler
from volsurfs_py.volume_rendering.volume_rendering_modules import VolumeRenderingNeuS
from volsurfs_py.models.sdf import SDF

# from volsurfs_py.models.offsets_sdf import OffsetsSDF
from volsurfs_py.models.rgb import RGB
from volsurfs_py.models.color_sh import ColorSH
from volsurfs_py.models.nerfhash import NerfHash
from volsurfs_py.utils.sdf_utils import get_rays_samples_packed_sdf
from volsurfs_py.utils.debug import sanity_check
from volsurfs_py.utils.losses import loss_l1, eikonal_loss, sphere_init_loss
from volsurfs_py.utils.common import map_range_val
from volsurfs_py.methods.base_method import BaseMethod
from volsurfs_py.utils.raycasting import intersect_bounding_primitive, reflect_rays
from volsurfs_py.utils.background import render_contracted_bg
from volsurfs_py.utils.sphere_tracing import sphere_trace
from mvdatasets.geometry.primitives.bounding_sphere import BoundingSphere
from volsurfs_py.utils.fields_utils import get_field_gradients, get_sdf_curvature
from volsurfs_py.utils.logistic_distribution import logistic_distribution_stdev
from volsurfs_py.utils.logistic_distribution import get_logistic_beta_from_variance
from volsurfs_py.utils.occupancy_grid import init_occupancy_grid


class Surf(BaseMethod):

    method_name = "surf"

    def __init__(
        self,
        train: bool,
        hyper_params,
        load_checkpoints_path,
        save_checkpoints_path,
        bounding_primitive,
        model_colorcal=None,
        bg_color=None,
        start_iter_nr=0,
        init_sphere_radius=None,
        # contract_samples=False,
        profiler=None,
    ):
        self.train_appearance_only = False
        self.should_reset_appearance_model = False
        self.render_sphere_traced = False

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

        if start_iter_nr == 0 or start_iter_nr < self.hyper_params.init_phase_end_iter:
            if init_sphere_radius is None:
                print(
                    "[bold red]ERROR[/bold red]: init_sphere_radius must be provided for surf method when start_iter_nr == 0 or start_iter_nr < init_phase_end_iter"
                )
                exit(1)

        # instantiate occupancy grid
        if hyper_params.use_occupancy_grid:
            self.occupancy_grid = init_occupancy_grid(bounding_primitive)
        else:
            self.occupancy_grid = None

        #
        self.variance = self.hyper_params.first_phase_variance_end_value
        self.cos_anneal_ratio = 1.0
        self.in_process_of_sphere_init = False

        #
        # self.contract_samples = contract_samples

        #
        self.init_sphere_radius = init_sphere_radius

        # instantiate SDF model
        model_sdf = SDF(
            in_channels=3,
            geom_feat_size=hyper_params.geom_feat_size,
            mlp_layers_dims=hyper_params.sdf_mlp_layers_dims,
            encoding_type=hyper_params.sdf_encoding_type,
            nr_iters_for_c2f=hyper_params.sdf_nr_iters_for_c2f,
            bb_sides=bounding_primitive.get_radius() * 2.0,
        ).to("cuda")
        # model_sdf = OffsetsSDF(
        #     in_channels=3,
        #     nr_inner_surfs=0,
        #     nr_outer_surfs=0,
        #     geom_feat_size=hyper_params.geom_feat_size,
        #     mlp_layers_dims=hyper_params.sdf_mlp_layers_dims,
        #     encoding_type=hyper_params.sdf_encoding_type,
        #     nr_iters_for_c2f=hyper_params.sdf_nr_iters_for_c2f,
        #     bb_sides=bounding_primitive.get_radius()*2.0
        # )
        self.models["sdf"] = model_sdf

        # instantiate radiance model
        # instantiate radiance model
        model_rgb = None
        if hyper_params.appearance_predict_sh_coeffs:
            assert (
                hyper_params.rgb_view_dep
            ), "SH coeffs only implemented for view dependent color"
            model_rgb = ColorSH(
                in_channels=3,
                out_channels=3,
                mlp_layers_dims=hyper_params.rgb_mlp_layers_dims,
                pos_encoder_type=hyper_params.rgb_pos_encoder_type,
                sh_deg=hyper_params.sh_degree,
                normal_dep=hyper_params.rgb_normal_dep,
                geom_feat_dep=hyper_params.rgb_geom_feat_dep,
                in_geom_feat_size=hyper_params.geom_feat_size,
                nr_iters_for_c2f=hyper_params.rgb_nr_iters_for_c2f,
                bb_sides=bounding_primitive.get_radius() * 2.0,
            )
        else:
            model_rgb = RGB(
                in_channels=3,  # if hyper_params.rgb_pos_dep else 0,
                out_channels=3,
                mlp_layers_dims=hyper_params.rgb_mlp_layers_dims,
                pos_encoder_type=hyper_params.rgb_pos_encoder_type,
                dir_encoder_type=hyper_params.rgb_dir_encoder_type,
                sh_deg=hyper_params.sh_degree,
                pos_dep=True,  # hyper_params.rgb_pos_dep,
                view_dep=hyper_params.rgb_view_dep,
                normal_dep=hyper_params.rgb_normal_dep,
                geom_feat_dep=hyper_params.rgb_geom_feat_dep,
                in_geom_feat_size=hyper_params.geom_feat_size,
                nr_iters_for_c2f=hyper_params.rgb_nr_iters_for_c2f,
                use_lipshitz_mlp=hyper_params.rgb_use_lipshitz_mlp,
                bb_sides=bounding_primitive.get_radius() * 2.0,
            ).to("cuda")
        self.models["rgb"] = model_rgb

        # instantiate bg model if needed
        if self.bg_color is None:
            model_bg = NerfHash(
                in_channels=3,
                pos_encoder_type=hyper_params.bg_pos_encoder_type,
                dir_encoder_type=hyper_params.bg_dir_encoder_type,
                nr_iters_for_c2f=hyper_params.bg_nr_iters_for_c2f,
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

        self.update_method_state(iter_nr=start_iter_nr)
        self.update_occupancy_grid(iter_nr=start_iter_nr)

    def collect_opt_params(self) -> list:
        # group parameters for optimization
        opt_params = []
        # SDF
        opt_params.append(
            {
                "params": self.models["sdf"].pos_encoder.parameters(),
                "weight_decay": 0.0,
                "lr": self.hyper_params.lr,
                "name": "sdf_pos_encoder",
            }
        )
        opt_params.append(
            {
                "params": self.models["sdf"].mlp_sdf.parameters(),
                "weight_decay": 0.0,
                "lr": self.hyper_params.lr,
                "name": "sdf_mlp_sdf",
            }
        )
        # opt_params.append(
        #     {
        #         "params": self.models["sdf"].mlp_shared.parameters(),
        #         "weight_decay": 0.0,
        #         "lr": self.hyper_params.lr,
        #         "name": "sdf_mlp_shared",
        #     }
        # )
        # RGB
        opt_params.append(
            {
                "params": self.models["rgb"].pos_encoder.parameters(),
                "weight_decay": 0.0,
                "lr": self.hyper_params.lr,
                "name": "rgb_pos_encoder",
            }
        )
        opt_params.append(
            {
                "params": self.models["rgb"].mlp.parameters(),
                "weight_decay": 0.0,
                "lr": self.hyper_params.lr,
                "name": "rgb_mlp",
            }
        )
        # BG
        if self.models["bg"] is not None:
            opt_params.append(
                {
                    "params": self.models["bg"].parameters(),
                    "weight_decay": 0.0,
                    "lr": self.hyper_params.lr,
                    "name": "bg",
                }
            )
        # COLOR CAL
        # if model_colorcal is not None:
        #     params.append(
        #         {
        #             "params": model_colorcal.parameters(),
        #             "weight_decay": 1e-1,
        #             "lr": hyper_params.lr,
        #             "name": "colorcal",
        #         }
        #     )

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

            sdf_grid = []
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
                grid_res_batch = self.models["sdf"].main_sdf(
                    grid_samples_batch, iter_nr=iter_nr
                )
                if isinstance(grid_res_batch, tuple):
                    sdf_grid_batch = grid_res_batch[0]
                else:
                    sdf_grid_batch = grid_res_batch
                sdf_grid.append(sdf_grid_batch)
            sdf_grid = torch.cat(sdf_grid, dim=0)
            sdf_grid = torch.abs(sdf_grid)

            occ_grid_variance = min(0.8, self.variance)
            # occ_grid_variance = self.variance
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

    #
    def render_fg_volumetric(
        self,
        ray_samples_packed_fg,
        logistic_beta_value=2048,
        cos_anneal_ratio=1.0,
        iter_nr=None,
        debug_ray_idx=None,
        override={},
    ):
        """
        renders foreground
        """

        # foreground

        nr_rays = ray_samples_packed_fg.get_nr_rays()

        if debug_ray_idx is not None:
            debug_pixel_vis = torch.zeros(nr_rays, 1)
            debug_pixel_vis[debug_ray_idx] = 1.0
        else:
            debug_pixel_vis = None

        if ray_samples_packed_fg.is_empty():

            # no ray samples
            pred_rgb_fg = torch.zeros(nr_rays, 3)
            pred_normals = torch.zeros(nr_rays, 3)
            pred_depth = torch.zeros(nr_rays, 1)
            samples_sdf_grad = torch.zeros(nr_rays, 3)
            weights_sum = torch.zeros(nr_rays, 1)
            bg_transmittance = torch.ones(nr_rays, 1)
            nr_samples = torch.zeros(nr_rays, 1)
            samples_3d = None

        else:

            if self.profiler is not None:
                self.profiler.start("render_fg_volumetric")

            samples_3d = ray_samples_packed_fg.samples_3d

            # handle view dirs override
            override_view_dir = override.get("view_dir", None)
            if override_view_dir is not None:
                samples_dirs = torch.from_numpy(override_view_dir).float()[None, :]
                samples_dirs = samples_dirs.repeat(
                    ray_samples_packed_fg.samples_dirs.shape[0], 1
                )
            else:
                samples_dirs = ray_samples_packed_fg.samples_dirs

            # compute sdf and normals
            samples_sdf, geom_feat_samples = self.models["sdf"].main_sdf(
                samples_3d, iter_nr=iter_nr
            )
            samples_sdf_grad = get_field_gradients(
                self.models["sdf"].forward, samples_3d, iter_nr
            )
            if samples_sdf_grad.dim() == 3:
                samples_sdf_grad = samples_sdf_grad.squeeze(1)
            normals_samples = F.normalize(samples_sdf_grad, dim=1)

            # as in Ref-NeRF
            # reflected_dirs = reflect_rays(rays_dirs=samples_dirs, normals_dirs=normals_samples)

            # pred rgb
            rgb_samples = self.models["rgb"](
                points=samples_3d,
                samples_dirs=samples_dirs,  # reflected_dirs,
                normals=normals_samples,
                iter_nr=iter_nr,
                geom_feat=geom_feat_samples,
            )

            # volumetric integration

            # compute weights
            alpha = VolumeRenderingNeuS().compute_alphas_from_logistic_beta(
                ray_samples_packed_fg,
                samples_sdf,
                samples_sdf_grad,
                cos_anneal_ratio,
                logistic_beta_value,
                debug_ray_idx=debug_ray_idx,
            )

            # TODO: set alpha to zero for samples whose dt is > self.hyper_params.ray_max_dt
            #  TODO: ~maximum distance traversable in bounding primitive / max_nr_samples_per_ray
            # ray_max_dt = (self.bounding_primitive.get_radius() * 2) / self.hyper_params.max_nr_samples_per_ray
            # alpha[ray_samples_packed_fg.samples_dt > ray_max_dt] = 0.0
            # if last fg sample and sample_dt > ray_max_dt, do not set alpha to zero

            # TODO: only do this if it is not the last sample in the ray

            # # if samples_3d is in empty space, set alpha to zero
            # if self.occupancy_grid is not None:
            #     samples_occupied_space_mask, _ = self.occupancy_grid.check_occupancy(samples_3d)
            #     alpha[~samples_occupied_space_mask] = 0.0

            transmittance = VolumeRenderingNeuS().compute_transmittance_from_alphas(
                ray_samples_packed_fg, alpha
            )
            weights = alpha * transmittance
            # weights = weights.clip(min=0.0, max=1.0)
            weights_sum, _ = VolumeRenderingNeuS().sum_ray_module(
                ray_samples_packed_fg, weights
            )
            # weights_sum = weights_sum.clip(min=0.0, max=1.0)
            bg_transmittance = 1 - weights_sum

            # volumetric integration
            pred_rgb_fg = VolumeRenderingNeuS().integrate_3d(
                ray_samples_packed_fg, rgb_samples, weights
            )
            # pred_rgb_fg = pred_rgb_fg.clip(min=0.0, max=1.0)

            # pred depth (no need to compute gradients)
            pred_depth = VolumeRendering.integrate_with_weights_1d(
                ray_samples_packed_fg, ray_samples_packed_fg.samples_z, weights
            )

            # compute normals (no need to compute gradients)
            pred_normals = VolumeRendering.integrate_with_weights_3d(
                ray_samples_packed_fg, normals_samples, weights
            )

            # count number of samples per ray
            nr_samples = ray_samples_packed_fg.get_nr_samples_per_ray()[:, None].int()

            if self.profiler is not None:
                self.profiler.end("render_fg_volumetric")

            if debug_ray_idx is not None:
                debug_ray_t_near = (
                    ray_samples_packed_fg.get_ray_enter(debug_ray_idx)
                    .detach()
                    .cpu()
                    .numpy()
                )
                debug_ray_t_far = (
                    ray_samples_packed_fg.get_ray_exit(debug_ray_idx)
                    .detach()
                    .cpu()
                    .numpy()
                )
                debug_ray_samples_z = (
                    ray_samples_packed_fg.get_ray_samples_z(debug_ray_idx)
                    .detach()
                    .cpu()
                    .numpy()
                )
                debug_ray_start_end_idx = ray_samples_packed_fg.get_ray_start_end_idx(
                    debug_ray_idx
                )
                debug_ray_samples_sdf = (
                    samples_sdf[debug_ray_start_end_idx[0] : debug_ray_start_end_idx[1]]
                    .detach()
                    .cpu()
                    .numpy()
                )
                debug_ray_samples_alpha = (
                    alpha[debug_ray_start_end_idx[0] : debug_ray_start_end_idx[1]]
                    .detach()
                    .cpu()
                    .numpy()
                )
                debug_ray_samples_transmittance = (
                    transmittance[
                        debug_ray_start_end_idx[0] : debug_ray_start_end_idx[1]
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )
                debug_ray_samples_weight = (
                    weights[debug_ray_start_end_idx[0] : debug_ray_start_end_idx[1]]
                    .detach()
                    .cpu()
                    .numpy()
                )

                # plot
                plt.figure(figsize=(10, 5))
                plt.plot(
                    [debug_ray_t_near, debug_ray_t_far], [0, 0], "o-", color="black"
                )
                plt.plot(
                    debug_ray_samples_z,
                    debug_ray_samples_sdf * 10,
                    "o-",
                    label="sdf*10",
                    color="blue",
                )
                plt.plot(
                    debug_ray_samples_z,
                    debug_ray_samples_alpha,
                    "o-",
                    label="alpha",
                    color="red",
                )
                plt.plot(
                    debug_ray_samples_z,
                    debug_ray_samples_transmittance,
                    "o-",
                    label="transmittance",
                    color="orange",
                )
                plt.plot(
                    debug_ray_samples_z,
                    debug_ray_samples_weight,
                    "o-",
                    label="weight",
                    color="green",
                )
                plt.xlabel("z")
                plt.ylabel("alpha")
                plt.legend()
                plt.grid()

                plt.show()
                # plt.savefig(
                #     os.path.join("plots", f"surf_debug_ray.png"),
                #     transparent=True,
                #     bbox_inches="tight",
                #     pad_inches=0,
                #     dpi=300
                # )
                # plt.close()
                # exit(0)

        renders = {
            "rgb_fg": pred_rgb_fg,
            "depth_fg": pred_depth,
            "weights_sum": weights_sum,
            "bg_transmittance": bg_transmittance,
            "normals": pred_normals,
            "nr_samples": nr_samples,
        }

        if debug_pixel_vis is not None:
            renders["debug_pixel_vis"] = debug_pixel_vis

        return renders, samples_3d, samples_sdf_grad

    @torch.no_grad()
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
        t_near = raycast["t_near"]
        t_far = raycast["t_far"]
        is_hit = raycast["is_hit"]

        pred_normals = torch.zeros(nr_rays, 3)
        samples_sdf_grad = torch.zeros(nr_rays, 3)
        pred_depth = torch.zeros(nr_rays, 1)
        pred_rgb_fg = torch.zeros(nr_rays, 3)

        points = None

        if self.profiler is not None:
            self.profiler.start("sphere_tracing")

        #
        ray_samples_packed, ray_hit_flag = sphere_trace(
            sdf_fn=self.models["sdf"].main_sdf,
            nr_sphere_traces=max_st_steps,
            rays_o=rays_o,
            rays_d=rays_d,
            bounding_primitive=self.bounding_primitive,
            sdf_multiplier=1.0,
            sdf_converged_tresh=converged_dist_tresh,
            occupancy_grid=self.occupancy_grid,
            iter_nr=iter_nr,
        )
        is_hit = ray_hit_flag  # (n_rays, )
        ray_end = ray_samples_packed.samples_3d  # (n_rays, 3)
        dists = ray_samples_packed.samples_z  # (n_rays, 1)
        # nr_samples = res["steps"]  # (n_rays, 1)

        # foreground and background masks
        weights_sum = is_hit.float().unsqueeze(1)
        bg_transmittance = 1 - weights_sum
        points = ray_end[is_hit]  # (nr_hits, 3)

        if self.profiler is not None:
            self.profiler.end("sphere_tracing")

        # foreground

        if self.profiler is not None:
            self.profiler.start("render_fg_sphere_traced")

        nr_hits = is_hit.sum()
        if nr_hits > 0:
            # compute sdf and normals
            _, geom_feat_ray_end_hit = self.models["sdf"].main_sdf(
                points, iter_nr=iter_nr
            )
            sdf_ray_end_hit_grad = get_field_gradients(
                self.models["sdf"].main_sdf, points, iter_nr=iter_nr
            )
            if sdf_ray_end_hit_grad.dim() == 3:
                sdf_ray_end_hit_grad = sdf_ray_end_hit_grad.squeeze(1)

            samples_sdf_grad[is_hit] = sdf_ray_end_hit_grad
            pred_normals[is_hit] = F.normalize(sdf_ray_end_hit_grad, dim=1)
            pred_depth[is_hit] = dists[is_hit]

            # get also rgb
            pred_rgb_fg_hit = self.models["rgb"](
                points=points,
                samples_dirs=rays_d[is_hit],
                normals=pred_normals[is_hit],
                iter_nr=iter_nr,
                geom_feat=geom_feat_ray_end_hit,
            )
            pred_rgb_fg[is_hit] = pred_rgb_fg_hit

        if self.profiler is not None:
            self.profiler.end("render_fg_sphere_traced")

        renders = {
            "rgb_fg": pred_rgb_fg,
            "depth_fg": pred_depth,
            "weights_sum": weights_sum,
            "bg_transmittance": bg_transmittance,
            "normals": pred_normals,
            # "nr_samples": nr_samples,
        }

        return renders, points, samples_sdf_grad

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
        vol_samples_grad = None

        # handle variance override
        override_variance = override.get("variance", None)
        if override_variance is not None:
            variance = override_variance
        else:
            variance = self.variance
        logistic_beta_value = get_logistic_beta_from_variance(variance)

        # foreground volume rendering

        ray_samples_packed_fg, _ = get_rays_samples_packed_sdf(
            rays_o=rays_o,
            rays_d=rays_d,
            t_near=raycast["t_near"],
            t_far=raycast["t_far"],
            sdf_fn=self.models["sdf"],
            logistic_beta_value=logistic_beta_value,
            occupancy_grid=self.occupancy_grid,
            iter_nr=iter_nr,
            min_dist_between_samples=self.hyper_params.min_dist_between_samples,
            min_nr_samples_per_ray=self.hyper_params.min_nr_samples_per_ray,
            max_nr_samples_per_ray=self.hyper_params.max_nr_samples_per_ray,
            max_nr_imp_samples_per_ray=self.hyper_params.max_nr_imp_samples_per_ray,
            jitter_samples=self.is_training,
            importace_sampling=self.hyper_params.do_importance_sampling,
            profiler=self.profiler,
        )

        # handle cos_anneal_ratio override
        override_cos_anneal_ratio = override.get("cos_anneal_ratio", None)
        if override_cos_anneal_ratio is not None:
            cos_anneal_ratio = override_cos_anneal_ratio
        else:
            cos_anneal_ratio = self.cos_anneal_ratio

        # fine rendering
        (
            vol_renders,
            vol_samples_3d,
            vol_samples_grad,
        ) = self.render_fg_volumetric(
            ray_samples_packed_fg=ray_samples_packed_fg,
            logistic_beta_value=logistic_beta_value,
            cos_anneal_ratio=cos_anneal_ratio,
            iter_nr=iter_nr,
            debug_ray_idx=debug_ray_idx,
            override=override,
        )

        # foreground surface rendering (only at test time)
        if self.render_sphere_traced and not self.is_training:

            st_renders, _, _ = self.render_fg_sphere_traced(
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
        #         + st_renders["bg_transmittance"] * rgb_bg
        #     )

        #    renders["sphere_traced"] = st_renders

        # combine volumetric with bg
        if vol_renders is not None:

            # blend rgb
            vol_renders["rgb_bg"] = rgb_bg
            rgb_fg = vol_renders["rgb_fg"]
            vol_renders["rgb"] = rgb_fg + rgb_bg * vol_renders["bg_transmittance"]

            # blend depth
            vol_renders["depth_bg"] = depth_bg
            depth_fg = vol_renders["depth_fg"]
            vol_renders["depth"] = (
                depth_fg * vol_renders["weights_sum"]
                + depth_bg * vol_renders["bg_transmittance"]
            )

            renders["volumetric"] = vol_renders

        res = {
            "renders": renders,
            "samples_3d": vol_samples_3d,
            "samples_grad": vol_samples_grad,
        }

        return res

    def update_method_state(self, iter_nr):
        interpolation_start_iter = self.hyper_params.init_phase_end_iter
        first_phase_end_iter = self.hyper_params.first_phase_end_iter
        first_phase_variance_start_value = (
            self.hyper_params.first_phase_variance_start_value
        )
        first_phase_variance_end_value = (
            self.hyper_params.first_phase_variance_end_value
        )

        self.in_process_of_sphere_init = iter_nr < interpolation_start_iter
        self.just_started_first_phase = iter_nr == interpolation_start_iter

        if self.is_training:
            # check if should update occupancy grid (if method supports it)
            update_occupancy_grid_every_nr_iters = 50
            if self.hyper_params.use_occupancy_grid is not None and (
                iter_nr % update_occupancy_grid_every_nr_iters == 0
            ):
                self.update_occupancy_grid(iter_nr=iter_nr)

        if self.in_process_of_sphere_init:
            # sphere init
            pass
        else:
            # training from data
            self.cos_anneal_ratio = map_range_val(
                iter_nr,
                interpolation_start_iter,
                first_phase_end_iter,
                0.0,
                1.0,
            )
            self.variance = map_range_val(
                iter_nr,
                interpolation_start_iter,
                first_phase_end_iter,
                first_phase_variance_start_value,
                first_phase_variance_end_value,
            )

            if self.just_started_first_phase:
                self.update_occupancy_grid(iter_nr)

                if self.is_training:
                    print("\ntraining from data")

                    # lr scheduler
                    if (
                        self.lr_scheduler is None
                        and self.scheduler_lr_decay is not None
                    ):
                        self.lr_scheduler = GradualWarmupScheduler(
                            self.optimizer,
                            multiplier=1,
                            total_epoch=self.hyper_params.nr_warmup_iters,
                            after_scheduler=self.scheduler_lr_decay,
                        )

                    if self.train_appearance_only:
                        # # reset appearance
                        # if self.should_reset_appearance_model:
                        #     self.models["rgb"].reset()
                        #     opt_params = self.collect_opt_params()
                        #     self.init_optim(opt_params)
                        #     self.should_reset_appearance_model = False
                        # freeze main surface
                        if self.models["sdf"].is_training_main_surf:
                            # stop training main sdf, pos encoder and mlp_shared
                            for param in self.models["sdf"].mlp_sdf.parameters():
                                param.requires_grad = False
                            for param in self.models["sdf"].pos_encoder.parameters():
                                param.requires_grad = False
                            # for param in self.models["sdf"].mlp_shared.parameters():
                            #     param.requires_grad = False
                            self.models["sdf"].is_training_main_surf = False

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
        loss_sdf = 0.0
        loss_eikonal = 0.0
        loss_rgb = 0.0
        loss_curvature = 0.0
        loss_lipshitz = 0.0
        loss_offsurface_high_sdf = 0.0
        loss_mask = 0.0

        self.update_method_state(iter_nr)

        if self.in_process_of_sphere_init:
            # sphere init

            if is_first_iter:
                print(f"\ninitializing a sphere of radius {self.init_sphere_radius}")

            if self.profiler is not None:
                self.profiler.start("sphere_init")

            def sphere_3d_sdf(
                points_3d, center=torch.tensor([0.0, 0.0, 0.0]), radius=0.2
            ):
                center = center.to(points_3d.device)
                return ((points_3d - center).norm(dim=-1) - radius).unsqueeze(-1)

            # sample random points
            with torch.set_grad_enabled(False):
                points_3d = self.bounding_primitive.get_random_points_inside(30000)
                sdf_gt = sphere_3d_sdf(
                    points_3d,
                    radius=self.init_sphere_radius,
                    center=torch.tensor([0.0, 0.0, 0.0]),
                )

            res_pred = self.models["sdf"].main_sdf(points_3d)
            if isinstance(res_pred, tuple):
                sdf_pred = res_pred[0]
                if sdf_pred.dim() == 3:
                    sdf_pred = sdf_pred.squeeze(1)

            # get sdf gradients
            sdf_grad = get_field_gradients(self.models["sdf"].main_sdf, points_3d)
            if sdf_grad.dim() == 3:
                sdf_grad = sdf_grad.squeeze(1)

            loss_sdf = ((sdf_pred - sdf_gt) ** 2).mean()

            distance_scale = 1
            loss_eikonal = ((sdf_grad.norm(dim=-1) - distance_scale) ** 2).mean()
            loss = loss_sdf + loss_eikonal.mean() * 1e-3

            if self.profiler is not None:
                self.profiler.end("sphere_init")

            s_points_3d = None

        else:
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
            s_points_sdf_grad = res["samples_grad"]

            pred_rgb = renders["volumetric"]["rgb"]
            # pred_mask = renders["volumetric"]["weights_sum"]

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

            # sample random points
            nr_points = 1024
            with torch.set_grad_enabled(False):
                r_points_3d = self.bounding_primitive.get_random_points_inside(
                    nr_points
                )

            # predict random points 3d grads
            r_points_sdf, _ = self.models["sdf"].main_sdf(r_points_3d, iter_nr=iter_nr)
            r_points_sdf_grad = get_field_gradients(
                self.models["sdf"].main_sdf,
                r_points_3d,
                iter_nr=iter_nr,
            )

            if self.profiler is not None:
                self.profiler.end("forward_points")

            # eikonal loss
            if self.hyper_params.eikonal_weight > 0.0:

                if self.profiler is not None:
                    self.profiler.start("eikonal_loss")

                # on random points
                loss_eikonal = (
                    eikonal_loss(r_points_sdf_grad) * self.hyper_params.eikonal_weight
                )
                # on surface points
                if s_points_3d is not None and s_points_3d.shape[0] > 0:
                    loss_eikonal += (
                        eikonal_loss(s_points_sdf_grad)
                        * self.hyper_params.eikonal_weight
                    )
                loss += loss_eikonal

                if self.profiler is not None:
                    self.profiler.end("eikonal_loss")

            # loss for empty space sdf
            if self.hyper_params.offsurface_weight > 0.0:

                if self.profiler is not None:
                    self.profiler.start("offsurface_loss")

                # penalise low sdf values in empty space
                loss_offsurface_high_sdf = (
                    torch.exp(-1e2 * torch.abs(r_points_sdf)).mean()
                    * self.hyper_params.offsurface_weight
                )
                loss += loss_offsurface_high_sdf

                if self.profiler is not None:
                    self.profiler.end("offsurface_loss")

            # rendered points losses

            # curvature loss
            reduce_curv_start_iter = self.hyper_params.reduce_curv_start_iter
            reduce_curv_end_iter = self.hyper_params.reduce_curv_end_iter
            if reduce_curv_end_iter is not None and reduce_curv_start_iter is not None:
                if iter_nr < reduce_curv_end_iter:
                    # once we are converged onto good geometry we can safely descrease
                    # it's weight so we learn also high frequency detail geometry
                    global_weight_curvature = map_range_val(
                        iter_nr,
                        reduce_curv_start_iter,
                        reduce_curv_end_iter,
                        1.0,
                        0.0,
                    )
                else:
                    global_weight_curvature = 0.0
            else:
                # reduce_curv_end_iter and reduce_curv_start_iter are not set,
                # apply curvature loss all the time
                global_weight_curvature = 1.0

            if (
                self.hyper_params.curvature_weight > 0.0
                and global_weight_curvature > 0.0
                and s_points_3d is not None
                and s_points_3d.shape[0] > 0
            ):
                sdf_curvature = get_sdf_curvature(
                    self.models["sdf"].main_sdf,
                    s_points_3d,
                    s_points_sdf_grad,
                    iter_nr=iter_nr,
                )
                loss_curvature = (
                    sdf_curvature.mean()
                    * self.hyper_params.curvature_weight
                    * global_weight_curvature
                )
                loss += loss_curvature

            # loss on lipshitz
            if reduce_curv_start_iter is not None:
                if (
                    iter_nr >= reduce_curv_start_iter
                    and self.hyper_params.lipshitz_weight > 0
                ):
                    loss_lipshitz = (
                        self.models["rgb"].mlp.lipshitz_bound_full().mean()
                        * self.hyper_params.lipshitz_weight
                    )
                    loss += loss_lipshitz

            # # loss mask
            # if self.hyper_params.is_training_masked and self.hyper_params.mask_weight > 0.0:

            #     if self.profiler is not None:
            #         self.profiler.start("mask_loss")

            #     pred_mask = torch.clamp(pred_mask, min=0.0, max=1.0)
            #     loss_mask = loss_l1(pred_mask, gt_mask, mask=1-gt_mask)
            #     loss_mask = loss_mask * self.hyper_params.mask_weight
            #     loss += loss_mask

            #     if self.profiler is not None:
            #         self.profiler.end("mask_loss")

        losses = {
            "loss": loss,
            "sdf": loss_sdf,
            "eikonal": loss_eikonal,
            "rgb": loss_rgb,
            "curvature": loss_curvature,
            "lipshitz": loss_lipshitz,
            "offsurface_high_sdf": loss_offsurface_high_sdf,
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
