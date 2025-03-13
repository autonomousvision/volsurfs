from rich import print
import torch
import torch.nn.functional as F
import os
import shutil
import open3d as o3d
import numpy as np

from volsurfs_py.schedulers.warmup import GradualWarmupScheduler
from volsurfs_py.models.color_sh import ColorSH
from volsurfs_py.models.rgb import RGB
from volsurfs_py.models.sh_neural_textures import SHNeuralTextures

# from volsurfs_py.models.nerfhash import NerfHash
from volsurfs_py.utils.losses import loss_l1
from volsurfs_py.methods.base_method import BaseMethod
from volsurfs_py.utils.background import render_contracted_bg
from volsurfs_py.models.nerfhash import NerfHash

# from volsurfs_py.utils.sampling import create_uncontracted_bg_samples
# from volsurfs_py.utils.background import render_contracted_bg
from volsurfs_py.utils.raycasting import intersect_bounding_primitive, reflect_rays
from volsurfs_py.utils.background import render_contracted_bg
from volsurfs_py.utils.mesh_loaders import load_meshes_indexed_from_path
from mvdatasets.utils.tensor_mesh import TensorMesh
from mvdatasets.utils.mesh import Mesh
from raytracelib import RayTracer


class VolSurfs(BaseMethod):

    method_name = "volsurfs"

    def __init__(
        self,
        train: bool,
        hyper_params,
        load_checkpoints_path,
        save_checkpoints_path,
        bounding_primitive,
        meshes_path=None,
        models_path=None,
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

        if meshes_path is None and start_iter_nr == 0:
            print(
                "\n[bold red]ERROR[/bold red]: --meshes_path must be specified if training from scratch"
            )
            exit(1)

        if models_path is None and start_iter_nr == 0:
            print(
                "\n[bold yellow]WARNING[/bold yellow]: --models_path not specified, training from scratch"
            )

        if start_iter_nr > 0:
            print(
                "\n[bold yellow]WARNING[/bold yellow]: --meshes_path is ignored if starting from checkpoint"
            )

        local_meshes_path = os.path.join(load_checkpoints_path, "meshes")

        # if is first iteration, copy used meshes in checkpoints folder
        if start_iter_nr == 0:

            # load meshes from --meshes_path
            meshes, meshes_paths = load_meshes_indexed_from_path(
                meshes_indices=self.hyper_params.meshes_indices,
                meshes_path=meshes_path,
                require_uvs=self.hyper_params.using_neural_textures,
                return_paths=True,
            )

            # copy meshes in checkpoints folder
            print("\ncopying meshes in checkpoints folder")
            # copy meshes in run checkpoints folder
            os.makedirs(local_meshes_path, exist_ok=True)
            # iterate over meshes paths
            for nr_mesh, mesh_path in enumerate(meshes_paths):
                mesh_filename = os.path.basename(mesh_path)
                mesh_format = mesh_filename.split(".")[-1]
                dest_path = os.path.join(
                    local_meshes_path, str(nr_mesh) + "." + mesh_format
                )
                shutil.copy(mesh_path, dest_path)
                print(f"copied {mesh_filename} to {dest_path}")

        else:

            # load meshes from local_meshes_path
            meshes = load_meshes_indexed_from_path(
                meshes_indices=None,
                meshes_path=local_meshes_path,
                require_uvs=self.hyper_params.using_neural_textures,
                return_paths=False,
            )

        # move meshes to gpu
        self.tensor_meshes = []
        for mesh in meshes:
            tensor_mesh = TensorMesh(mesh, device="cuda")
            self.tensor_meshes.append(tensor_mesh)

        # if self.bg_color is None:
        #     sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20, create_uv_map=True)
        #     mesh = Mesh(o3d_mesh=sphere_mesh)
        #     tensor_mesh = TensorMesh(mesh, device="cuda")
        #     self.tensor_meshes.append(tensor_mesh)

        self.nr_meshes = len(self.tensor_meshes)

        # build BVHs from meshes
        self.raytracer = RayTracer(self.tensor_meshes)

        #
        self.with_alpha_decay = self.hyper_params.with_alpha_decay

        # create appearance models
        # if appearance models are indep, instantiate one per mesh

        # instantiate color models

        if self.hyper_params.using_neural_textures:

            sh_range = self.hyper_params.sh_range

            # color
            for i in range(self.nr_meshes):

                model_rgb = SHNeuralTextures(
                    sh_deg=self.hyper_params.sh_degree,
                    nr_channels=3,
                    sh_range=sh_range,
                    anchor=self.hyper_params.using_neural_textures_anchor,
                    lerp=self.hyper_params.using_neural_textures_lerp,
                    deg_res=self.hyper_params.textures_res,
                    quantize_output=self.hyper_params.using_sh_quantization,
                    squeeze_output=self.hyper_params.using_sh_squeezing,
                    align_to_webgl=True,
                ).to("cuda")

                if hyper_params.are_volsurfs_colors_indep:
                    # one for each surf
                    self.models[f"rgb_{i}"] = model_rgb
                else:
                    # one for all surfs
                    self.models["rgb"] = model_rgb
                    break

            # transparency
            for i in range(self.nr_meshes):
                # if inner mesh is solid, do not predict alpha for it
                if hyper_params.is_inner_mesh_solid and i == 0:
                    model_alpha = None
                else:

                    if self.hyper_params.transp_view_dep:

                        model_alpha = SHNeuralTextures(
                            sh_deg=self.hyper_params.sh_degree,
                            nr_channels=1,
                            sh_range=sh_range,
                            anchor=self.hyper_params.using_neural_textures_anchor,
                            lerp=self.hyper_params.using_neural_textures_lerp,
                            deg_res=self.hyper_params.textures_res,
                            quantize_output=self.hyper_params.using_sh_quantization,
                            squeeze_output=self.hyper_params.using_sh_squeezing,
                            align_to_webgl=True,
                        ).to("cuda")

                    else:

                        model_alpha = SHNeuralTextures(
                            sh_deg=0,
                            nr_channels=1,
                            sh_range=sh_range,
                            anchor=self.hyper_params.using_neural_textures_anchor,
                            lerp=self.hyper_params.using_neural_textures_lerp,
                            deg_res=self.hyper_params.textures_res,
                            quantize_output=self.hyper_params.using_sh_quantization,
                            squeeze_output=self.hyper_params.using_sh_squeezing,
                            align_to_webgl=True,
                        ).to("cuda")

                if hyper_params.are_volsurfs_alphas_indep:
                    # one for each surf
                    self.models[f"alpha_{i}"] = model_alpha
                else:
                    # one for all surfs
                    self.models["alpha"] = model_alpha
                    break

        else:

            # legacy method

            for i in range(self.nr_meshes):

                if hyper_params.appearance_predict_sh_coeffs:
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
                        geom_feat_dep=False,
                        in_geom_feat_size=0,
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
                        geom_feat_dep=False,
                        in_geom_feat_size=0,
                        nr_iters_for_c2f=hyper_params.rgb_nr_iters_for_c2f,
                        use_lipshitz_mlp=False,
                        bb_sides=bounding_primitive.get_radius() * 2.0,
                    ).to("cuda")

                if hyper_params.are_volsurfs_colors_indep:
                    # one for each surf
                    self.models[f"rgb_{i}"] = model_rgb
                else:
                    # one for all surfs
                    self.models["rgb"] = model_rgb
                    break

            for i in range(self.nr_meshes):
                # if inner mesh is solid, do not predict alpha for it
                if hyper_params.is_inner_mesh_solid and i == 0:
                    model_alpha = None
                else:
                    if hyper_params.appearance_predict_sh_coeffs:
                        # SH coefficients prediction
                        assert (
                            hyper_params.transp_view_dep
                        ), "SH coeffs only implemented for view dependent alpha"
                        model_alpha = ColorSH(
                            in_channels=3,
                            out_channels=1,
                            mlp_layers_dims=hyper_params.rgb_mlp_layers_dims,
                            pos_encoder_type=hyper_params.rgb_pos_encoder_type,
                            sh_deg=hyper_params.sh_degree,
                            normal_dep=self.hyper_params.transp_normal_dep,
                            geom_feat_dep=False,
                            in_geom_feat_size=0,
                            nr_iters_for_c2f=hyper_params.rgb_nr_iters_for_c2f,
                            bb_sides=bounding_primitive.get_radius() * 2.0,
                        ).to("cuda")
                    else:
                        model_alpha = RGB(
                            in_channels=3,
                            out_channels=1,
                            mlp_layers_dims=hyper_params.rgb_mlp_layers_dims,
                            pos_encoder_type=hyper_params.rgb_pos_encoder_type,
                            dir_encoder_type=hyper_params.rgb_dir_encoder_type,
                            sh_deg=hyper_params.sh_degree,
                            view_dep=hyper_params.transp_view_dep,
                            normal_dep=self.hyper_params.transp_normal_dep,
                            geom_feat_dep=False,
                            in_geom_feat_size=0,
                            nr_iters_for_c2f=hyper_params.rgb_nr_iters_for_c2f,
                            use_lipshitz_mlp=False,
                            bb_sides=bounding_primitive.get_radius() * 2.0,
                        ).to("cuda")

                if hyper_params.are_volsurfs_alphas_indep:
                    # one for each surf
                    self.models[f"alpha_{i}"] = model_alpha
                else:
                    # one for all surfs
                    self.models["alpha"] = model_alpha
                    break

        # # instantiate bg model (optional)
        # if self.bg_color is None:

        #     if self.hyper_params.using_neural_textures:

        #         sh_range = self.hyper_params.sh_range

        #         model_bg = SHNeuralTextures(
        #                 sh_deg=self.hyper_params.sh_degree,
        #                 nr_channels=3,
        #                 sh_range=sh_range,
        #                 anchor=self.hyper_params.using_neural_textures_anchor,
        #                 lerp=self.hyper_params.using_neural_textures_lerp,
        #                 deg_res=self.hyper_params.textures_res,
        #                 quantize_output=self.hyper_params.using_sh_quantization,
        #                 squeeze_output=self.hyper_params.using_sh_squeezing,
        #                 align_to_webgl=True
        #             ).to("cuda")

        #     else:

        #         print("\n[bold red]ERROR[/bold red]: background 3D field appearance model not implemented")
        #         exit(1)

        #         # TODO: implement background field appearance model
        #         # model_bg = RGB(
        #         #     in_channels=3,
        #         #     out_channels=3,
        #         #     mlp_layers_dims=hyper_params.rgb_mlp_layers_dims,
        #         #     pos_encoder_type=hyper_params.rgb_pos_encoder_type,
        #         #     dir_encoder_type=hyper_params.rgb_dir_encoder_type,
        #         #     sh_deg=hyper_params.sh_degree,
        #         #     view_dep=hyper_params.rgb_view_dep,
        #         #     normal_dep=self.hyper_params.rgb_normal_dep,
        #         #     geom_feat_dep=False,
        #         #     in_geom_feat_size=0,
        #         #     nr_iters_for_c2f=hyper_params.rgb_nr_iters_for_c2f,
        #         #     use_lipshitz_mlp=False,
        #         #     bb_sides=bounding_primitive.get_radius()*2.0
        #         # ).to("cuda")

        # else:

        #     model_bg = None

        # self.models["bg"] = model_bg

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

        # if is first iteration and models path is given
        if models_path is not None and start_iter_nr == 0:
            # if not self.hyper_params.using_neural_textures:
            #     # load appearance checkpoints
            #     for key, model in self.models.items():
            #         if model is not None:
            #             if "rgb" in key or "alpha" in key:
            #                 ckpt_path = os.path.join(models_path, f"{key}.pt")
            #                 if os.path.exists(ckpt_path):
            #                     model.load_state_dict(torch.load(ckpt_path))
            #                     print(f"\n[bold green]SUCCESS[/bold green]: {key} loaded from {ckpt_path}")
            #                 else:
            #                     print(f"\n[bold red]ERROR[/bold red]: checkpoint {ckpt_path} does not exist")
            #                     exit(1)
            #         else:
            #             print(f"\n[bold yellow]WARNING[/bold yellow]: model {key} is None, skipping loading")
            # else:
            #     print("\n[bold red]ERROR[/bold red]: cannot load models_path when using neural textures")
            #     exit(1)

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

    def collect_opt_params(self) -> list:
        # group parameters for optimization
        opt_params = []
        for key, model in self.models.items():
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

    # do a forward pass through the model
    def render_rays(
        self,
        rays_o,
        rays_d,
        iter_nr=None,
        debug_ray_idx=None,
        **kwargs,
    ):
        """render a batch of rays with raytracing"""

        # intersect bounding primitive
        raycast = intersect_bounding_primitive(self.bounding_primitive, rays_o, rays_d)

        nr_rays = rays_o.shape[0]

        # intersect bounding primitive
        # raycast = intersect_bounding_primitive(
        #    self.bounding_primitive, rays_o, rays_d
        # )

        if debug_ray_idx is not None:
            debug_pixel_vis = torch.zeros(nr_rays, 1)
            debug_pixel_vis[debug_ray_idx] = 1.0
        else:
            debug_pixel_vis = None

        # results tensors
        # per mesh
        surfs_points = torch.zeros(nr_rays, self.nr_meshes, 3)
        surfs_hits = torch.zeros(nr_rays, self.nr_meshes).bool()
        surfs_normals = torch.zeros(nr_rays, self.nr_meshes, 3)
        # surfs_depths = torch.zeros(nr_rays, self.nr_meshes, 1)
        surfs_rgb = torch.zeros(nr_rays, self.nr_meshes, 3)
        surfs_alpha = torch.zeros(nr_rays, self.nr_meshes, 1)
        surfs_transmittance = torch.ones(nr_rays, self.nr_meshes, 1)
        surfs_blending_weights = torch.zeros(nr_rays, self.nr_meshes, 1)
        # optional
        if self.hyper_params.using_neural_textures:
            surfs_uvs = torch.zeros(nr_rays, self.nr_meshes, 2)
        else:
            surfs_uvs = None
        # aggregated
        rgb_fg = torch.zeros(nr_rays, 3)
        bg_transmittance = torch.ones(nr_rays, 1)
        # geometry
        points = None
        points_normals = None

        # RAYTRACE -----------------------------------------------------------

        if self.profiler is not None:
            self.profiler.start("meshes_raytracing")

        meshes_hits = []
        for i in range(self.nr_meshes):

            # raytrace mesh
            res = self.raytracer.trace(rays_o, rays_d, mesh_id=i)
            any_hit = res["any_hit"]  # bool
            if any_hit:
                meshes_hits.append(res)
            else:
                meshes_hits.append(None)

        if self.profiler is not None:
            self.profiler.end("meshes_raytracing")

        # SHADE --------------------------------------------------------------

        for i, mesh_hit in enumerate(meshes_hits):
            if mesh_hit is not None:

                # primitive_ids = mesh_hit["ids"]  # [N,]
                faces_id = mesh_hit["triangles_id"]  # [N,]
                ray_t = mesh_hit["depth"]  # [N,]
                hits = mesh_hit["is_hit"]  # [N,]
                points = mesh_hit["positions"]  # [N, 3]
                normals = mesh_hit["normals"]  # [N, 3]
                barycentric = mesh_hit["barycentric"]  # [N, 3]

                # update hits
                surfs_hits[:, i] = hits
                surfs_points[hits, i] = points[hits]
                # surfs_depths[hits, i] = ray_t[hits].view(-1, 1)
                surfs_normals[hits, i] = normals[hits]

                if surfs_uvs is not None:
                    # get uv coords
                    vertices_uvs = self.tensor_meshes[i].get_faces_uvs()[faces_id]
                    uv_points = torch.sum(
                        barycentric.unsqueeze(-1) * vertices_uvs, dim=1
                    )  # [N, 2]
                    # update uvs
                    surfs_uvs[hits, i] = uv_points[hits]
                    # TODO: remove
                    # surfs_uvs[hits, i] -= 0.5

                if self.profiler is not None:
                    self.profiler.start("ray_color_inference")

                # predict colors
                if self.hyper_params.are_volsurfs_colors_indep:
                    model_rgb = self.models[f"rgb_{i}"]
                else:
                    model_rgb = self.models["rgb"]

                # as in Ref-NeRF
                # reflected_dirs = reflect_rays(rays_dirs=rays_d[hits], normals_dirs=normals[hits])

                if self.hyper_params.using_neural_textures:
                    if surfs_uvs is None:
                        print(
                            "\n[bold red]ERROR[/bold red]: surfs_uvs is None, but using neural textures"
                        )
                        exit(1)
                    # query neural textures
                    surfs_points_rgb_pred = model_rgb(
                        uv_coords=surfs_uvs[hits, i], view_dirs=rays_d[hits]
                    )
                else:
                    # legacy
                    surfs_points_rgb_pred = model_rgb(
                        points=surfs_points[hits, i],
                        samples_dirs=rays_d[hits],  # reflected_dirs,
                        normals=surfs_normals[hits, i],
                        iter_nr=iter_nr,
                    )
                surfs_rgb[hits, i] = surfs_points_rgb_pred[:, :3]

                # predict alphas
                if self.hyper_params.are_volsurfs_alphas_indep:
                    model_alpha = self.models[f"alpha_{i}"]
                else:
                    model_alpha = self.models["alpha"]

                if model_alpha is None:
                    surfs_points_alpha_pred = torch.ones(
                        surfs_points[hits, i].shape[0], 1
                    )
                else:
                    if self.hyper_params.using_neural_textures:
                        if surfs_uvs is None:
                            print(
                                "\n[bold red]ERROR[/bold red]: surfs_uvs is None, but using neural textures"
                            )
                            exit(1)
                        # query neural textures
                        surfs_points_alpha_pred = model_alpha(
                            uv_coords=surfs_uvs[hits, i], view_dirs=rays_d[hits]
                        )
                    else:
                        # legacy
                        surfs_points_alpha_pred = model_alpha(
                            points=surfs_points[hits, i],
                            samples_dirs=rays_d[hits],  # reflected_dirs,
                            normals=surfs_normals[hits, i],
                            iter_nr=iter_nr,
                        )

                    # TODO: still testing
                    threshold = 10.0
                    alpha_decay = torch.ones_like(surfs_points_alpha_pred)
                    if self.with_alpha_decay:
                        with torch.no_grad():
                            # modulate transparency with intersection angle
                            dot = torch.sum(
                                -rays_d[hits] * surfs_normals[hits, i],
                                dim=1,
                                keepdim=True,
                            ).clamp(0.0, 1.0)
                            alpha_decay = torch.sigmoid(threshold * dot) * 2.0 - 1.0
                    surfs_points_alpha_pred = surfs_points_alpha_pred * alpha_decay

                surfs_alpha[hits, i] = surfs_points_alpha_pred

                if self.profiler is not None:
                    self.profiler.end("ray_color_inference")

        # outer to inner
        surfs_alpha = surfs_alpha.flip(1)
        surfs_rgb = surfs_rgb.flip(1)

        # cast to float16
        surfs_rgb = surfs_rgb.half()
        surfs_alpha = surfs_alpha.half()

        # SURFS + BG TRANSMITTANCES -------------------------------

        # surfs + bg transmittances (outer to inner)
        transmittances = torch.cumprod(1 - surfs_alpha, dim=1)

        if self.nr_meshes == 1:
            surfs_transmittance = torch.ones_like(transmittances)
            bg_transmittance = transmittances.squeeze(-1)
        else:
            surfs_transmittance = transmittances[:, :-1]
            surfs_transmittance = torch.cat(
                [torch.ones_like(surfs_transmittance[:, -1:]), surfs_transmittance],
                dim=1,
            )
            bg_transmittance = transmittances[:, -1:].squeeze(-1)

        # ALPHA BLENDING ---------------------------------------------------

        if self.profiler is not None:
            self.profiler.start("render_fg")

        # outer to inner
        surfs_blending_weights = surfs_transmittance * surfs_alpha

        # blending
        rgb_fg = (surfs_rgb * surfs_blending_weights).sum(dim=1)

        # inner to outer
        surfs_rgb = surfs_rgb.flip(1)
        surfs_blending_weights = surfs_blending_weights.flip(1)
        surfs_transmittance = surfs_transmittance.flip(1)
        surfs_alpha = surfs_alpha.flip(1)

        if self.profiler is not None:
            self.profiler.end("render_fg")

        if debug_ray_idx is not None:

            print("\n[bold yellow]DEBUG[/bold yellow]: debug_ray_idx")
            print(f"debug_ray_idx: {debug_ray_idx}")

            # print current textures values
            if self.hyper_params.using_neural_textures:
                for i in range(self.nr_meshes):

                    uv_coords = surfs_uvs[debug_ray_idx, i]
                    sample_rgb = surfs_rgb[debug_ray_idx, i] * 255.0
                    sample_alpha = surfs_alpha[debug_ray_idx, i]

                    print(f"mesh {i} uvs: {uv_coords}")
                    print(f"mesh {i} rgb: {sample_rgb}")
                    print(f"mesh {i} alpha: {sample_alpha}")

                    # predict colors
                    if self.hyper_params.are_volsurfs_colors_indep:
                        model_rgb = self.models[f"rgb_{i}"]
                    else:
                        model_rgb = self.models["rgb"]

                    # if using neural textures, query neural texture on uv coords
                    if self.hyper_params.using_neural_textures:
                        # query neural textures
                        tex_pred = model_rgb(uv_coords=uv_coords.unsqueeze(0)).squeeze(
                            -1
                        )
                        print(f"mesh {i} rgb_pred: {tex_pred * 255.0}")

                        textures = model_rgb.render(bake=True)
                        for texture in textures:
                            texture = texture.squeeze(-1)
                            texture = texture * 255.0
                            print(f"texture: {texture}")

            exit(0)

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

        # cast to float16
        rgb_bg = rgb_bg.half()

        # COMPOSITING --------------------------------------------------------
        pred_rgb = rgb_fg + bg_transmittance * rgb_bg

        # prepare output

        # get point samples on all hit surfaces
        points = surfs_points[surfs_hits].reshape(-1, 3)

        # get normal samples on all hit surfaces
        points_normals = surfs_normals[surfs_hits].reshape(-1, 3)

        # # print pred_rgb type (float16)
        # print(f"pred_rgb type: {pred_rgb.dtype}")
        # print(f"rgb_fg type: {rgb_fg.dtype}")
        # print(f"rgb_bg type: {rgb_bg.dtype}")
        # print(f"surfs_rgb type: {surfs_rgb.dtype}")
        # print(f"surfs_alpha type: {surfs_alpha.dtype}")
        # print(f"surfs_blending_weights type: {surfs_blending_weights.dtype}")
        # print(f"surfs_normals type: {surfs_normals.dtype}")
        # print(f"bg_transmittance type: {bg_transmittance.dtype}")

        # cast back to float32
        pred_rgb = pred_rgb.float()
        rgb_fg = rgb_fg.float()
        rgb_bg = rgb_bg.float()
        surfs_rgb = surfs_rgb.float()
        surfs_alpha = surfs_alpha.float()
        surfs_blending_weights = surfs_blending_weights.float()
        surfs_normals = surfs_normals.float()
        bg_transmittance = bg_transmittance.float()

        renders = {
            "rgb": pred_rgb,
            "rgb_fg": rgb_fg,
            "rgb_bg": rgb_bg,
            "surfs_alpha": surfs_alpha,
            "surfs_rgb": surfs_rgb,
            # "surfs_depths": surfs_depths,
            "surfs_normals": surfs_normals,
            "surfs_blending_weights": surfs_blending_weights,
            "bg_transmittance": bg_transmittance,
        }

        if surfs_uvs is not None:
            renders["surfs_uvs"] = surfs_uvs

        res = {
            "renders": {
                "ray_traced": renders,
            },
            "samples_3d": points,
            "samples_grad": points_normals,
        }

        return res

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

        if is_first_iter and self.scheduler_lr_decay is not None:
            if self.hyper_params.nr_warmup_iters > 0:
                self.lr_scheduler = GradualWarmupScheduler(
                    self.optimizer,
                    multiplier=1,
                    total_epoch=self.hyper_params.nr_warmup_iters,
                    after_scheduler=self.scheduler_lr_decay,
                )
            else:
                self.lr_scheduler = self.scheduler_lr_decay

        if self.profiler is not None:
            self.profiler.start("forward")

        # forward through the network and get the prediction
        res = self.render_rays(rays_o=rays_o, rays_d=rays_d, iter_nr=iter_nr)
        renders, pts = res["renders"], res["samples_3d"]
        pred_rgb = renders["ray_traced"]["rgb"]

        if self.profiler is not None:
            self.profiler.end("forward")

        if self.profiler is not None:
            self.profiler.start("losses")

        loss = 0.0
        loss_rgb = 0.0

        # rgb loss
        if self.hyper_params.is_training_masked:
            loss_rgb = loss_l1(gt_rgb, pred_rgb, mask=gt_mask)
        else:
            loss_rgb = loss_l1(gt_rgb, pred_rgb)
        loss += loss_rgb

        if self.profiler is not None:
            self.profiler.end("losses")

        losses = {"loss": loss, "rgb": loss_rgb}

        additional_info_to_log = {}

        return losses, additional_info_to_log, pts
