import torch
import sys
import os
import numpy as np
import time
import datetime
import argparse
import math
from tqdm import tqdm
from volsurfs import VolumeRendering
from volsurfs import OccupancyGrid
from volsurfs_py.schedulers.warmup import GradualWarmupScheduler
from volsurfs_py.volume_rendering.volume_rendering_modules import VolumeRenderingNeRF
from volsurfs_py.models.density import Density
from volsurfs_py.models.rgb import RGB
from volsurfs_py.models.color_sh import ColorSH
from volsurfs_py.models.nerfhash import NerfHash
from volsurfs_py.utils.nerf_utils import get_rays_samples_packed_nerf

# from volsurfs_py.utils.debug import sanity_check
from volsurfs_py.utils.losses import loss_l1, sparsity_loss
from volsurfs_py.methods.base_method import BaseMethod
from volsurfs_py.utils.raycasting import intersect_bounding_primitive
from volsurfs_py.utils.background import render_contracted_bg
from volsurfs_py.utils.occupancy_grid import init_occupancy_grid

# TODO: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf.py#L308
# adapt following original coarse-fine sampling


class NeRF(BaseMethod):

    method_name = "nerf"

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

        # instantiate occupancy grid
        if hyper_params.use_occupancy_grid:
            self.occupancy_grid = init_occupancy_grid(bounding_primitive)
        else:
            self.occupancy_grid = None

        # instantiate density model
        model_density = Density(
            in_channels=3,
            out_channels=1,
            geom_feat_size=hyper_params.geom_feat_size,
            mlp_layers_dims=hyper_params.density_mlp_layers_dims,
            encoding_type=hyper_params.density_encoding_type,
            nr_iters_for_c2f=hyper_params.density_nr_iters_for_c2f,
            bb_sides=bounding_primitive.get_radius() * 2.0,
        ).to("cuda")
        self.models["density"] = model_density

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
                use_lipshitz_mlp=False,
                bb_sides=bounding_primitive.get_radius() * 2.0,
            ).to("cuda")
        self.models["rgb"] = model_rgb

        # instantiate bg model (optional)
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
        self.update_occupancy_grid(
            iter_nr=start_iter_nr, decay=0.0, random_voxels=False, jitter_samples=False
        )

    def collect_opt_params(self) -> list:

        # group parameters for optimization
        opt_params = []
        opt_params.append(
            {
                "params": self.models["density"].parameters(),
                "weight_decay": 0.0,
                "lr": self.hyper_params.lr,
                "name": "model_density",
            }
        )
        opt_params.append(
            {
                "params": self.models["rgb"].parameters(),
                "weight_decay": 0.0,
                "lr": self.hyper_params.lr,
                "name": "model_rgb",
            }
        )
        if self.models["bg"] is not None:
            opt_params.append(
                {
                    "params": self.models["bg"].parameters(),
                    "weight_decay": 0.0,
                    "lr": self.hyper_params.lr,
                    "name": "model_bg",
                }
            )
        # if self.model_colorcal is not None:
        #     params.append(
        #         {
        #             "params": self.model_colorcal.parameters(),
        #             "weight_decay": 1e-1,
        #             "lr": hyper_params.lr,
        #             "name": "model_colorcal",
        #         }
        #     )
        return opt_params

    @torch.no_grad()
    def update_occupancy_grid(
        self, iter_nr, decay=0.8, random_voxels=True, jitter_samples=True, **kwargs
    ):
        """update occupancy grid with random density samples"""

        if self.occupancy_grid is not None:

            if self.profiler is not None:
                self.profiler.start("update_occupancy_grid")

            # sample points in grid voxels
            if random_voxels:
                (
                    grid_samples,
                    grid_indices,
                ) = self.occupancy_grid.get_random_grid_samples_in_roi(
                    256 * 256 * 4, jitter_samples
                )
            else:
                (
                    grid_samples,
                    grid_indices,
                ) = self.occupancy_grid.get_grid_samples(jitter_samples)

            # print("grid_samples", grid_samples.shape)
            grid_samples_batches = torch.split(grid_samples, 256 * 256 * 100, dim=0)

            density_grid = []
            pbar = tqdm(
                grid_samples_batches,
                desc="occupancy values",
                unit="batch",
                leave=False,
                # ncols=100,
                disable=True,
            )
            for grid_samples_batch in pbar:
                # compute density
                grid_res_batch = self.models["density"](
                    grid_samples_batch, iter_nr=iter_nr
                )
                if isinstance(grid_res_batch, tuple):
                    density_grid_batch = grid_res_batch[0]
                else:
                    density_grid_batch = grid_res_batch
                density_grid.append(density_grid_batch)
            density_grid = torch.cat(density_grid, dim=0)

            # update grid values
            self.occupancy_grid.update_grid_values(grid_indices, density_grid, decay)

            # update grid occupancy
            occupancy_tresh = 1e-4
            check_neighbours = False
            self.occupancy_grid.update_grid_occupancy_with_density_values(
                grid_indices, occupancy_tresh, check_neighbours
            )

            if self.profiler is not None:
                self.profiler.end("update_occupancy_grid")

    #
    def render_fg_volumetric(self, ray_samples_packed_fg, iter_nr=None, override={}):
        """
        renders foreground

        Args:
            nr_rays (int): number of rays in batch
            ray_samples_packed_bg (RaySamplesPacked): points and directions samples
            iter_nr (int, optional): current training iteration. Defaults to None.

        Returns:
            renders (dict): rgb (nr_rays, 3),
                        depth (nr_rays, 1),
                        weights_sum (nr_rays, 1),
                        bg_transmittance (nr_rays, 1)
        """

        # foreground

        if self.profiler is not None:
            self.profiler.start("render_fg_volumetric")

        nr_rays = ray_samples_packed_fg.get_nr_rays()

        if ray_samples_packed_fg.is_empty():

            # no ray samples
            pred_rgb_fg = torch.zeros(nr_rays, 3)
            pred_depth = torch.zeros(nr_rays, 1)
            weights_sum = torch.zeros(nr_rays, 1)
            bg_transmittance = torch.ones(nr_rays, 1)
            nr_samples = torch.zeros(nr_rays, 1)
            # t_near = torch.zeros(nr_rays, 1)
            # t_far = torch.zeros(nr_rays, 1)
            # dt_sum = torch.zeros(nr_rays, 1)
            samples_3d = None

        else:

            samples_3d = ray_samples_packed_fg.samples_3d

            # compute rgb and density
            samples_densities, geom_feat_samples = self.models["density"](
                points=samples_3d, iter_nr=iter_nr
            )

            # handle view dirs override
            override_view_dir = override.get("view_dir", None)
            if override_view_dir is not None:
                samples_dirs = torch.from_numpy(override_view_dir).float()[None, :]
                samples_dirs = samples_dirs.repeat(
                    ray_samples_packed_fg.samples_dirs.shape[0], 1
                )
            else:
                samples_dirs = ray_samples_packed_fg.samples_dirs

            # pred rgb
            rgb_samples = self.models["rgb"](
                points=samples_3d,
                samples_dirs=samples_dirs,
                iter_nr=iter_nr,
                geom_feat=geom_feat_samples,
            )

            dt = ray_samples_packed_fg.samples_dt
            alpha = 1.0 - torch.exp(-samples_densities * dt)
            # alpha = alpha.clip(0.0, 1.0)
            (
                transmittance,
                _,
            ) = VolumeRenderingNeRF().cumprod_one_minus_alpha_to_transmittance_module(
                ray_samples_packed_fg, 1 - alpha + 1e-6
            )
            # transmittance = transmittance.clip(0, 1)
            weights = alpha * transmittance

            weights_sum, _ = VolumeRenderingNeRF().sum_ray_module(
                ray_samples_packed_fg, weights
            )
            bg_transmittance = 1 - weights_sum

            # volumetric integration
            pred_rgb_fg = VolumeRenderingNeRF().integrate_3d(
                ray_samples_packed_fg, rgb_samples, weights
            )
            # pred_rgb_fg = pred_rgb_fg.clip(min=0.0, max=1.0)

            # pred depth (no need to compute gradients)
            pred_depth = VolumeRendering.integrate_with_weights_1d(
                ray_samples_packed_fg, ray_samples_packed_fg.samples_z, weights
            )

            # count number of samples per ray
            nr_samples = ray_samples_packed_fg.get_nr_samples_per_ray()[:, None].int()

            # t_near = ray_samples_packed_fg.ray_enter
            # t_far = ray_samples_packed_fg.ray_exit

            # dt_sum, _ = VolumeRenderingNeRF().sum_ray_module(
            #     ray_samples_packed_fg, ray_samples_packed_fg.samples_dt
            # )

        if self.profiler is not None:
            self.profiler.end("render_fg_volumetric")

        renders = {
            "rgb_fg": pred_rgb_fg,
            "depth": pred_depth,
            "weights_sum": weights_sum,
            "bg_transmittance": bg_transmittance,
            "nr_samples": nr_samples,
            # "t_near_depth": t_near,
            # "t_far_depth": t_far,
            # "dt_sum": dt_sum,
        }

        density_samples_grad = None

        return renders, samples_3d, density_samples_grad

    #
    def render_rays(self, rays_o, rays_d, iter_nr=None, override={}, **kwargs) -> dict:
        """render a batch of rays"""

        # intersect bounding primitive
        raycast = intersect_bounding_primitive(self.bounding_primitive, rays_o, rays_d)

        # FOREGROUND

        # prepare output
        vol_renders = {}
        fg_vol_samples = None
        fg_vol_samples_grad = None

        # foreground volume rendering

        ray_samples_packed_fg, _ = get_rays_samples_packed_nerf(
            rays_o,
            rays_d,
            t_near=raycast["t_near"],
            t_far=raycast["t_far"],
            density_fn=self.models["density"],
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

        # fine rendering
        (
            vol_renders,
            fg_vol_samples,
            fg_vol_samples_grad,
        ) = self.render_fg_volumetric(
            ray_samples_packed_fg, iter_nr=iter_nr, override=override
        )

        # BACKGROUND ---------------------------------------------

        if self.models["bg"] is None and self.bg_color is not None:
            # constant bg color
            pred_rgb_bg = self.bg_color.expand(raycast["nr_rays"], 3)
            expected_depth_bg = None
            median_depth_bg = None
        else:
            bg_res = render_contracted_bg(
                self.models["bg"],
                raycast,
                nr_samples_bg=self.hyper_params.nr_samples_bg,
                jitter_samples=self.is_training,
                iter_nr=iter_nr,
                profiler=self.profiler,
            )
            pred_rgb_bg = bg_res["pred_rgb"]
            expected_depth_bg = bg_res["expected_depth"]
            median_depth_bg = bg_res["median_depth"]

        # COMPOSITING ---------------------------------------------

        # combine fine volumetric with bg
        if vol_renders is not None:
            vol_renders["rgb"] = (
                vol_renders["rgb_fg"] + vol_renders["bg_transmittance"] * pred_rgb_bg
            )
            vol_renders["rgb_bg"] = pred_rgb_bg
            if expected_depth_bg is not None:
                vol_renders["expected_depth_bg"] = expected_depth_bg
            if median_depth_bg is not None:
                vol_renders["median_depth_bg"] = median_depth_bg

        res = {
            "renders": {"volumetric": vol_renders},
            "samples_3d": fg_vol_samples if fg_vol_samples is not None else None,
            "samples_grad": fg_vol_samples_grad,
        }

        return res

    def update_method_state(self, iter_nr):
        if self.is_training:
            # check if should update occupancy grid (if method supports it)
            update_occupancy_grid_every_nr_iters = 50
            if self.hyper_params.use_occupancy_grid is not None and (
                iter_nr % update_occupancy_grid_every_nr_iters == 0
            ):
                self.update_occupancy_grid(iter_nr=iter_nr)

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
        loss_sparsity = 0.0
        loss_mask = 0.0

        self.update_method_state(iter_nr)

        if is_first_iter:
            self.lr_scheduler = GradualWarmupScheduler(
                self.optimizer,
                multiplier=1,
                total_epoch=self.hyper_params.nr_warmup_iters,
                after_scheduler=self.scheduler_lr_decay,
            )

        if self.profiler is not None:
            self.profiler.start("forward")

        # forward through the network and get the prediction
        res = self.render_rays(rays_o=rays_o, rays_d=rays_d, iter_nr=iter_nr)
        renders, pts = res["renders"], res["samples_3d"]
        pred_rgb = renders["volumetric"]["rgb"]
        pred_mask = renders["volumetric"]["weights_sum"]

        # supersampling
        nr_training_rays_per_pixel = self.hyper_params.nr_training_rays_per_pixel
        if nr_training_rays_per_pixel > 1:
            # arithmetic mean of the subpixels values
            pred_rgb = pred_rgb.view(-1, nr_training_rays_per_pixel, 3).mean(dim=1)
            pred_mask = pred_mask.view(-1, nr_training_rays_per_pixel, 1).mean(dim=1)

        if self.profiler is not None:
            self.profiler.end("forward")

        if self.profiler is not None:
            self.profiler.start("losses")

        # rgb loss
        if self.hyper_params.is_training_masked:
            loss_rgb = loss_l1(gt_rgb, pred_rgb, mask=gt_mask)
        else:
            loss_rgb = loss_l1(gt_rgb, pred_rgb)
        loss += loss_rgb

        # sample random points
        nr_points = 1024
        with torch.set_grad_enabled(False):
            points_3d = self.bounding_primitive.get_random_points_inside(nr_points)

        # sparsity loss
        if iter_nr > 5000 and self.hyper_params.sparsity_weight > 0.0:
            points_density, _ = self.models["density"](points_3d, iter_nr)
            loss_sparsity = (
                sparsity_loss(points_density) * self.hyper_params.sparsity_weight
            )
            loss += loss_sparsity

        # loss mask
        if self.hyper_params.is_training_masked and self.hyper_params.mask_weight > 0.0:
            pred_mask = torch.clamp(pred_mask, min=0.0, max=1.0)
            loss_mask = loss_l1(pred_mask, gt_mask, mask=1 - gt_mask)
            loss_mask = loss_mask * self.hyper_params.mask_weight
            loss += loss_mask

        if self.profiler is not None:
            self.profiler.end("losses")

        losses = {
            "loss": loss,
            "rgb": loss_rgb,
            "sparsity": loss_sparsity,
            "mask": loss_mask,
        }

        additional_info_to_log = {}
        if self.occupancy_grid is not None:
            additional_info_to_log["occupied_voxels_roi"] = (
                self.occupancy_grid.get_nr_occupied_voxels_in_roi()
            )

        return losses, additional_info_to_log, pts
