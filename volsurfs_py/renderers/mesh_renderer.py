from rich import print

import os
import json
import torch
import numpy as np

from volsurfs_py.renderers.base_renderer import BaseRenderer
from volsurfs_py.utils.mesh_loaders import Mesh
from mvdatasets.utils.tensor_mesh import TensorMesh
from mvdatasets.utils.tensor_texture import TensorTexture
from raytracelib import RayTracer
from volsurfs_py.encodings.sphericalharmonics import SHEncoder


class MeshRenderer(BaseRenderer):
    def __init__(
        self,
        scene_path,
        t_near=1e-3,
        t_far=100,
        profiler=None,
    ):
        super().__init__(profiler=profiler)

        # open scene config json
        with open(f"{scene_path}/scene.json") as f:
            scene_config = json.load(f)
            for mesh in scene_config["meshes"]:
                mesh["mesh_path"] = os.path.join(scene_path, mesh["mesh_path"])
                for texture in mesh["textures"]:
                    texture["texture_path"] = os.path.join(
                        scene_path, texture["texture_path"]
                    )

        mesh_meta = scene_config["meshes"][0]

        mesh = Mesh(mesh_meta=mesh_meta)
        self.tensor_mesh = TensorMesh(mesh, device="cuda")
        self.tensor_texture = TensorTexture(
            texture_np=mesh.texture.image,
            lerp=True,
        )

        # build BVHs from meshes
        self.raytracer = RayTracer([self.tensor_mesh])

        self.t_near = t_near
        self.t_far = t_far

        #
        self.active_render_mode = "ray_traced"
        self.active_shader = "rgb"

        # bg color
        self.default_bg_color = (255, 255, 255)  # rgb
        self.bg_color = (
            torch.tensor(self.default_bg_color, dtype=torch.float32, device="cuda")
            / 255.0
        )

    @torch.no_grad()
    def shade(self, buffers_dict) -> dict:

        res = {}

        # get is_hit buffer
        is_hit = buffers_dict["is_hit"]

        for buffer_name, buffer in buffers_dict.items():

            if "normals" in buffer_name:
                # already normalized to [-1, 1]
                normals = buffer
                normals = (normals + 1) * 0.5
                normals[~is_hit] = self.bg_color
                res[buffer_name] = normals

            if "is_hit" in buffer_name:
                # convert hit flag to float
                res[buffer_name] = is_hit.float().unsqueeze(-1)

            if "uv" in buffer_name:
                # concatenate b channel (zeros)
                uvs = buffer
                uvs = torch.cat(
                    (uvs, torch.zeros_like(uvs[:, :1], device="cuda")), dim=-1
                )
                uvs[~is_hit] = self.bg_color
                res[buffer_name] = uvs

            if "rgb" in buffer_name:
                # clamp to [0, 1]
                rgb = buffer.clamp(0, 1)
                rgb[~is_hit] = self.bg_color
                res[buffer_name] = rgb

            if "alpha" in buffer_name:
                # clamp to [0, 1]
                alpha = buffer.clamp(0, 1)
                alpha[~is_hit] = 0.0
                res[buffer_name] = alpha

            if "view_dirs" in buffer_name:
                # already normalized to [-1, 1]
                view_dirs = buffer
                view_dirs = (view_dirs + 1) * 0.5
                view_dirs[~is_hit] = self.bg_color
                res[buffer_name] = view_dirs

        return res

    @torch.no_grad()
    def render_rays(self, rays_o, rays_d, verbose=False) -> dict:

        # get nr rays
        nr_rays = rays_o.shape[0]
        print("nr_rays", nr_rays)

        # results tensors
        # per mesh
        surfs_points = torch.zeros(nr_rays, 3)
        surfs_hits = torch.zeros(nr_rays).bool()
        surfs_normals = torch.zeros(nr_rays, 3)
        surfs_depths = torch.zeros(nr_rays, 1)
        surfs_uvs = torch.zeros(nr_rays, 2)
        rgb_fg = torch.zeros(nr_rays, 3)
        alpha_fg = torch.zeros(nr_rays, 1)

        # raytrace mesh
        mesh_hit = self.raytracer.trace(rays_o, rays_d)
        any_hit = mesh_hit["any_hit"]  # bool
        if any_hit:
            # primitive_ids = mesh_hit["ids"]  # [N,]
            faces_id = mesh_hit["triangles_id"]  # [N,]
            ray_t = mesh_hit["depth"]  # [N,]
            hits = mesh_hit["is_hit"]  # [N,]
            points = mesh_hit["positions"]  # [N, 3]
            normals = mesh_hit["normals"]  # [N, 3]
            barycentric = mesh_hit["barycentric"]  # [N, 3]

            # update hits
            surfs_hits = hits
            surfs_points[hits] = points[hits]
            surfs_depths[hits] = ray_t[hits].view(-1, 1)
            surfs_normals[hits] = normals[hits]

            # get uv coords
            vertices_uvs = self.tensor_mesh.get_faces_uvs()[faces_id]
            uv_points = torch.sum(
                barycentric.unsqueeze(-1) * vertices_uvs, dim=1
            )  # [N, 2]
            # update uvs
            surfs_uvs[hits] = uv_points[hits]

            # access texture
            # print("uv_points", uv_points[hits].shape)
            sh_coeffs = self.tensor_texture(uv_points[hits])

            # check nr_coeffs in last dim
            nr_coeffs = sh_coeffs.shape[-1] // 4
            if nr_coeffs == 1:
                sh_deg = 0
            elif nr_coeffs == 4:
                sh_deg = 1
            elif nr_coeffs == 9:
                sh_deg = 2
            elif nr_coeffs == 16:
                sh_deg = 3

            sh_coeffs = sh_coeffs.view(-1, 4, nr_coeffs)

            # to float16
            sh_coeffs = sh_coeffs.half()

            # evaluate view dependent channels
            raw_output = SHEncoder.eval(sh_coeffs, rays_d[hits], degree=sh_deg)
            rgba = torch.sigmoid(raw_output)

            # to float32
            rgba = rgba.float()

            rgb_fg[hits] = rgba[:, :3]
            alpha_fg[hits] = rgba[:, 3].view(-1, 1)

        buffers_dict = {
            "is_hit": surfs_hits,
            "normals": surfs_normals,
            "uvs": surfs_uvs,
            "rgb": rgb_fg,
            "alpha": alpha_fg,
            "view_dirs": rays_d,
        }

        shaded_buffers_dict = self.shade(buffers_dict)

        return {
            "renders": {
                "ray_traced": shaded_buffers_dict,
            }
        }
