from rich import print
import os
import torch
import numpy as np
import math
from tqdm import tqdm
from apex.optimizers import FusedAdam
from torch.cuda.amp.grad_scaler import GradScaler
from volsurfs_py.schedulers.multisteplr import MultiStepLR
from volsurfs_py.models.offsets_sdf import OffsetsSDF

from abc import ABC, abstractmethod
from mvdatasets.utils.raycasting import get_camera_rays


class BaseMethod(ABC):

    method_name = ""
    params = []

    def __init__(
        self,
        train: bool,
        hyper_params,
        load_checkpoints_path=None,
        save_checkpoints_path=None,
        bounding_primitive=None,
        model_colorcal=None,
        bg_color=None,
        occupancy_grid=None,
        profiler=None,
        start_iter_nr=0,
    ):
        print(f"\ninitializing method from iter {start_iter_nr}")

        self.hyper_params = hyper_params
        self.load_checkpoints_path = load_checkpoints_path
        self.save_checkpoints_path = save_checkpoints_path
        self.bounding_primitive = bounding_primitive
        self.bg_color = bg_color
        self.occupancy_grid = occupancy_grid
        self.profiler = profiler
        self.lr_scheduler = None
        self.models = {}
        self.renders_options = {}
        self.is_training = False
        self.opt_params = []
        self.optimizer = None
        self.grad_scaler = None
        self.scheduler_lr_decay = None

        if model_colorcal is not None:
            self.models["colorcal"] = model_colorcal

        if train:
            self.set_train_mode()
        else:
            self.set_eval_mode()

    def init_optim(self, opt_params):

        # store opt_params
        self.opt_params = opt_params

        # instantiate optimizer

        self.reset_optim(opt_params)

        # instantiate lr schedulers

        self.scheduler_lr_decay = MultiStepLR(
            self.optimizer,
            milestones=self.hyper_params.lr_milestones,
            gamma=0.3,
            verbose=False,
        )

        # gradient scaler

        if self.hyper_params.use_grad_scaler:
            self.grad_scaler = GradScaler(self.hyper_params.use_grad_scaler)
        else:
            self.grad_scaler = None

    def reset_optim(self, opt_params):
        # reset optimizer with new parameters
        self.optimizer = FusedAdam(
            opt_params,
            amsgrad=False,
            betas=(0.9, 0.99),
            eps=1e-15,
            weight_decay=0.0,
            lr=self.hyper_params.lr,
        )

    def print_models(self):
        print("\n[bold magenta]instanced models[/bold magenta]\n")
        for key, model in self.models.items():
            if model is not None:
                print(f"[bold black]{key}[/bold black]")
                # print("bb_sides", model.bb_sides)
                print(model)

    def set_train_mode(self):
        # iterate over all models and set them to train mode
        self.is_training = True
        for model in self.models.values():
            if model is not None:
                model.train()

    def set_eval_mode(self):
        # iterate over all models and set them to eval mode
        self.is_training = False
        for model in self.models.values():
            if model is not None:
                model.eval()

    def load(self, iter_nr):
        # load checkpoints
        if self.load_checkpoints_path is None:
            print("no load checkpoints path specified, skipping")
            return

        print("\nloading checkpoints")

        # load all method's models
        ckpt_path = os.path.join(
            self.load_checkpoints_path, format(iter_nr, "07d"), "models"
        )
        for key, model in self.models.items():
            print(f"loading {key}")
            if model is not None:
                load_checkpoints_path = os.path.join(ckpt_path, f"{key}.pt")
                # check if checkpoint exists
                if os.path.exists(load_checkpoints_path):
                    print(f"{key}: {ckpt_path} loaded")
                    model.load_state_dict(torch.load(load_checkpoints_path))
                    if isinstance(model, OffsetsSDF):
                        # load epsilons
                        for i, mlp in enumerate(model.mlps_eps):
                            load_checkpoints_path = os.path.join(
                                ckpt_path, f"{key}_eps_{i}.pt"
                            )
                            if os.path.exists(load_checkpoints_path):
                                print(f"{key}_eps_{i}: {load_checkpoints_path} loaded")
                                mlp.load_state_dict(torch.load(load_checkpoints_path))
                            else:
                                print(
                                    f"{key}_eps_{i}: {load_checkpoints_path} does not exist, skipping"
                                )
                else:
                    print(f"{key}: {load_checkpoints_path} does not exist, skipping")

        # if method has occupancy grid
        if self.hyper_params.use_occupancy_grid and self.occupancy_grid is not None:
            # check if it was saved
            occupancy_grid_values_path = os.path.join(ckpt_path, "grid_values.pt")
            occupancy_grid_occupancy_path = os.path.join(ckpt_path, "grid_occupancy.pt")
            if os.path.exists(occupancy_grid_values_path) and os.path.exists(
                occupancy_grid_occupancy_path
            ):
                self.occupancy_grid.set_grid_values(
                    torch.load(occupancy_grid_values_path)
                )
                self.occupancy_grid.set_grid_occupancy(
                    torch.load(occupancy_grid_occupancy_path)
                )
                print("occupancy_grid: loaded")
            else:
                # if not, reconstruct it from scratch
                self.update_occupancy_grid(
                    iter_nr=iter_nr,
                    decay=0.0,
                    random_voxels=False,
                    jitter_samples=False,
                )
                print("occupancy_grid: recomputed from scratch")

        if self.optimizer is not None:
            # load optimizer state
            save_path = os.path.join(
                self.load_checkpoints_path, format(iter_nr, "07d"), "models"
            )
            opt_ckpt_path = os.path.join(
                save_path, f"{self.optimizer.__class__.__name__.lower()}.pt"
            )
            if os.path.exists(opt_ckpt_path):
                print("loading optimizer state from ", opt_ckpt_path)
                optimizer_state_dict = torch.load(opt_ckpt_path)
                self.optimizer.load_state_dict(optimizer_state_dict)
            else:
                print(
                    f"\n[bold yellow]WARNING[/bold yellow]: optimizer state dict not found at {opt_ckpt_path}"
                )

        if self.grad_scaler is not None:
            # load grad_scaler state
            save_path = os.path.join(
                self.load_checkpoints_path, format(iter_nr, "07d"), "models"
            )
            gs_ckpt_path = os.path.join(
                save_path, f"{self.grad_scaler.__class__.__name__.lower()}.pt"
            )
            if os.path.exists(gs_ckpt_path):
                print("loading grad scaler state from ", gs_ckpt_path)
                grad_scaler_state_dict = torch.load(gs_ckpt_path)
                self.grad_scaler.load_state_dict(grad_scaler_state_dict)
            else:
                print(
                    f"\n[bold yellow]WARNING[/bold yellow]: grad scaler state dict not found at {gs_ckpt_path}"
                )

    def save(self, iter_nr):
        print("\nsaving checkpoints")

        if self.save_checkpoints_path is None:
            print("no save checkpoints path specified, skipping")
            return

        curr_ckpt_iter_str = format(iter_nr, "07d")
        save_path = os.path.join(
            self.save_checkpoints_path, curr_ckpt_iter_str, "models"
        )
        os.makedirs(save_path, exist_ok=True)

        # iterate over all models and save
        for key, model in self.models.items():
            if model is not None:
                model.save(save_path, override_name=key)
                print(f"{key}: saved")
            else:
                print(f"{key}: is None, skipping")

        # if method has occupancy grid, save it
        if self.occupancy_grid is not None:
            torch.save(
                self.occupancy_grid.get_grid_values(),
                os.path.join(save_path, "grid_values.pt"),
            )
            torch.save(
                self.occupancy_grid.get_grid_occupancy(),
                os.path.join(save_path, "grid_occupancy.pt"),
            )
            print("occupancy_grid: saved")

        if self.optimizer is not None:
            # save optimizer state
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(
                    save_path, f"{self.optimizer.__class__.__name__.lower()}.pt"
                ),
            )

        if self.grad_scaler is not None:
            # save grad scaler state
            torch.save(
                self.grad_scaler.state_dict(),
                os.path.join(
                    save_path, f"{self.grad_scaler.__class__.__name__.lower()}.pt"
                ),
            )

        print("[bold blue]INFO[/bold blue]: checkpoint saved: ", curr_ckpt_iter_str)

    def update_occupancy_grid(self, iter_nr, **kwargs):
        # has to be implemented by the method if it has an occupancy grid
        pass

    def render_rays_batchify(
        self,
        rays_o_all,
        rays_d_all,
        chunk_size=512,
        iter_nr=None,
        verbose=False,
        override={},
    ) -> dict:
        """Render rays

        Args:
            rays_o (torch.tensor): ray origins
            rays_d (torch.tensor): ray directions
            iter_nr (int): training iteration number
        """
        nr_chunks = math.ceil(rays_o_all.shape[0] / chunk_size)
        rays_o_list = torch.chunk(rays_o_all, nr_chunks)
        rays_d_list = torch.chunk(rays_d_all, nr_chunks)

        renders_batches_lists = {}

        # pbar = tqdm(
        #     rays_o_list,
        #     desc="rendering rays batches",
        #     ncols=100,
        #     leave=False
        # )
        # for i, _ in enumerate(pbar):
        for i, _ in enumerate(rays_o_list):
            # render batch

            batch_res = self.render_rays(
                rays_o=rays_o_list[i],
                rays_d=rays_d_list[i],
                iter_nr=iter_nr,
                override=override,
                debug_ray_idx=None,
                verbose=verbose,
            )

            # append batch rendered rays to a list of batches results

            # iterate over render modes
            renders_batch = batch_res["renders"]
            for render_mode, renders_dict in renders_batch.items():
                if renders_dict is not None:
                    # check if mode is already in dict
                    if render_mode not in renders_batches_lists.keys():
                        # init
                        renders_batches_lists[render_mode] = {}
                    # iterate over render keys in batch
                    for render_key, render in renders_dict.items():
                        if render is not None:
                            if (
                                render_key
                                not in renders_batches_lists[render_mode].keys()
                            ):
                                # init
                                renders_batches_lists[render_mode][render_key] = []
                            renders_batches_lists[render_mode][render_key].append(
                                render
                            )
                            # print(render_mode, render_key, render.shape)

        # concat lists to np.ndarrays
        renders = {}
        for render_mode, renders_dict in renders_batches_lists.items():
            renders[render_mode] = {}
            for render_key, render_batches_list in renders_dict.items():
                renders[render_mode][render_key] = torch.cat(render_batches_list, dim=0)
                # print(render_mode, render_key, renders[render_mode][render_key].shape)

        return renders

    @abstractmethod
    def collect_opt_params(self) -> list:
        pass

    @abstractmethod
    def render_rays(
        self,
        rays_o,
        rays_d,
        iter_nr=None,
        **kwargs,
    ) -> dict:
        """Render rays

        Args:
            rays_o (torch.tensor): ray origins
            rays_d (torch.tensor): ray directions
            iter_nr (int): training iteration number
        """
        pass

    @torch.no_grad()
    def render(
        self, camera, iter_nr=None, verbose=False, debug_pixel=None, **kwargs  # (y, x)
    ) -> dict:
        """base method renderer

        Args:
            camera (Camera): Camera object.
            iter_nr (int, optional): Used for c2f models. Defaults to None.

        Returns:
            pred (dict): Dictionary of renders for each rendering mode (np.ndarray).
        """

        jitter_pixels = self.hyper_params.jitter_test_rays
        chunk_size = self.hyper_params.test_rays_batch_size
        nr_rays_per_pixel = self.hyper_params.nr_test_rays_per_pixel

        # gen rays

        if self.profiler is not None:
            self.profiler.start("ray_gen")

        rays_o_all, rays_d_all, points_2d = get_camera_rays(
            camera,
            nr_rays_per_pixel=nr_rays_per_pixel,
            jitter_pixels=jitter_pixels,
            device="cuda",
        )
        if verbose:
            print("rays_o_all", rays_o_all.shape)
            print("rays_d_all", rays_d_all.shape)
        rays_o_all = rays_o_all.contiguous()
        rays_d_all = rays_d_all.contiguous()

        if self.profiler is not None:
            self.profiler.end("ray_gen")

        debug_pixel_idx = None
        debug_pixel_batch_idx = 0

        if chunk_size is not None:
            # split rays in chunks
            nr_chunks = math.floor(rays_o_all.shape[0] / chunk_size)
            split_sizes = [chunk_size] * nr_chunks + [
                rays_o_all.shape[0] - chunk_size * nr_chunks
            ]
            rays_o_list = list(
                torch.split(rays_o_all, split_size_or_sections=split_sizes, dim=0)
            )
            rays_d_list = list(
                torch.split(rays_d_all, split_size_or_sections=split_sizes, dim=0)
            )
            # rays_o_list = torch.chunk(rays_o_all, nr_chunks)
            # rays_d_list = torch.chunk(rays_d_all, nr_chunks)
            if debug_pixel is not None:
                y = debug_pixel[0]
                x = debug_pixel[1]
                pixel_idx = y * camera.width + x
                debug_pixel_batch_idx = math.floor(pixel_idx / chunk_size)
                debug_pixel_idx = pixel_idx - debug_pixel_batch_idx * chunk_size
        else:
            rays_o_list = [rays_o_all]
            rays_d_list = [rays_d_all]
            if debug_pixel is not None:
                y = debug_pixel[0]
                x = debug_pixel[1]
                pixel_idx = y * camera.width + x
                debug_pixel_idx = pixel_idx

        if debug_pixel is not None:
            print("debug_pixel", debug_pixel)
            print("width", camera.width)
            print("height", camera.height)
            print("nr_pixels", camera.width * camera.height)
            print("pixel_idx", pixel_idx)
            print("chunk_size", chunk_size)
            print("debug_pixel_idx", debug_pixel_idx)
            print("debug_pixel_batch_idx", debug_pixel_batch_idx)

        renders_batches_lists = {}

        if self.profiler is not None:
            self.profiler.start("render_frame")

        nr_samples = 0
        nr_rays = 0
        pbar = tqdm(
            rays_o_list,
            desc="rendering rays batches",
            # ncols=100,
            leave=False,
        )
        for i, _ in enumerate(pbar):
            # render batch

            if verbose:
                print(f"rendering batch {i+1}/{len(rays_o_list)}")
                print("rays_o", rays_o_list[i].shape)
                print("rays_d", rays_d_list[i].shape)

            batch_res = self.render_rays(
                rays_o=rays_o_list[i],
                rays_d=rays_d_list[i],
                iter_nr=iter_nr,
                override={},
                debug_ray_idx=debug_pixel_idx if i == debug_pixel_batch_idx else None,
                verbose=verbose,
            )

            # find out number of samples
            samples_3d = batch_res["samples_3d"]
            if samples_3d is not None:
                if isinstance(samples_3d, list) and len(samples_3d) > 0:
                    samples_3d = torch.cat(samples_3d, dim=0)
                if isinstance(samples_3d, torch.Tensor):
                    nr_samples = samples_3d.shape[0]

            nr_rays = rays_o_list[i].shape[0]

            # update progress bar
            pbar.set_postfix(
                {
                    "nr_samples": nr_samples,
                    "nr_rays": nr_rays,
                }
            )

            # append batch rendered rays to a list of batches results

            # iterate over render modes
            renders_batch = batch_res["renders"]
            for render_mode, renders_dict in renders_batch.items():
                if renders_dict is not None:
                    # check if mode is already in dict
                    if render_mode not in renders_batches_lists.keys():
                        # init
                        renders_batches_lists[render_mode] = {}
                    # iterate over render keys in batch
                    for render_key, render in renders_dict.items():
                        if render is not None:
                            if (
                                render_key
                                not in renders_batches_lists[render_mode].keys()
                            ):
                                # init
                                renders_batches_lists[render_mode][render_key] = []
                            renders_batches_lists[render_mode][render_key].append(
                                render.detach().cpu().numpy()
                            )
                            # print(
                            #     f"render_mode: {render_mode}, render_key: {render_key}, shape: {render.shape}"
                            # )

        # concat lists to np.ndarrays and average supersampling
        renders = {}
        for render_mode, renders_dict in renders_batches_lists.items():
            renders[render_mode] = {}
            for render_key, render_batches_list in renders_dict.items():
                # concatenate lists
                render_vals = np.concatenate(render_batches_list, axis=0)
                # average supersampling
                if nr_rays_per_pixel > 1:
                    render_vals = render_vals.reshape(
                        render_vals.shape[0] // nr_rays_per_pixel, nr_rays_per_pixel, -1
                    ).mean(axis=1)
                renders[render_mode][render_key] = render_vals
                # print(
                #     f"render_mode: {render_mode}, render_key: {render_key}, shape: {renders[render_mode][render_key].shape}")

        # --------------------------------------------------------

        if self.profiler is not None:
            self.profiler.end("render_frame")

        return renders

    @abstractmethod
    def forward(
        self,
        rays_o,
        rays_d,
        gt_rgb,
        gt_mask,
        iter_nr,
    ):
        pass
