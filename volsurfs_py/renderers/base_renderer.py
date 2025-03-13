from abc import ABC, abstractmethod
import torch
import numpy as np
import math
from tqdm import tqdm

from mvdatasets.utils.raycasting import get_camera_rays


class BaseRenderer(ABC):
    def __init__(self, profiler=None):
        """Base renderer

        args:
            profiler (Profiler, optional): Profiler object. Defaults to None.
        """

        self.profiler = profiler
        self.renders_options = {}

        self.active_shader = "None"
        self.active_render_mode = "None"

    # @abstractmethod
    # def shade(self, outputs) -> dict:
    #     """convert outputs to frame buffer values"""
    #     # TODO: continue from here
    #     pass

    @abstractmethod
    def render_rays(self, rays_o, rays_d, verbose=False) -> dict:
        """Render rays

        Args:
            rays_o (torch.tensor): ray origins
            rays_d (torch.tensor): ray directions
        """
        # TODO: continue from here
        pass

    @torch.no_grad()
    def render(self, camera, nr_rays_per_pixel=1, verbose=False) -> dict:
        """base renderer

        Args:
            camera (Camera): Camera object.
            chunk_size (int): Number of rays to be rendered together. Defaults to None.
            iter_nr (int, optional): Used for c2f models. Defaults to None.

        Returns:
            pred (dict): Dictionary of renders for each rendering mode (np.ndarray).
        """

        # gen rays

        if self.profiler is not None:
            self.profiler.start("ray_gen")

        rays_o_all, rays_d_all, points_2d = get_camera_rays(
            camera,
            nr_rays_per_pixel=nr_rays_per_pixel,
            jitter_pixels=nr_rays_per_pixel > 1,
            device="cuda",
        )
        if verbose:
            print("rays_o_all", rays_o_all.shape)
            print("rays_d_all", rays_d_all.shape)
        rays_o_all = rays_o_all.contiguous()
        rays_d_all = rays_d_all.contiguous()

        if self.profiler is not None:
            self.profiler.end("ray_gen")

        if self.profiler is not None:
            self.profiler.start("render_frame")

        # render
        res = self.render_rays(rays_o=rays_o_all, rays_d=rays_d_all, verbose=verbose)

        # iterate over render modes
        renders_torch = res["renders"]

        # concat lists to np.ndarrays
        renders_np = {}
        for render_mode, renders_dict in renders_torch.items():
            renders_np[render_mode] = {}
            for render_key, render_torch in renders_dict.items():

                # average supersampling
                render = render_torch
                if nr_rays_per_pixel > 1:
                    render = render.reshape(
                        render.shape[0] // nr_rays_per_pixel, nr_rays_per_pixel, -1
                    ).mean(dim=1)

                renders_np[render_mode][render_key] = render.cpu().numpy()

        if self.profiler is not None:
            self.profiler.end("render_frame")

        return renders_np
