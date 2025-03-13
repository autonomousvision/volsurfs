#!/usr/bin/env python3

# python ./volsurfs_py/baker.py --method_name surf --dataset dtu --scene dtu_scan24 --exp_name default --run_id 2023-09-07-151930 --nr_meshes_to_extract 1

from rich import print
import torch
import os
import json
import numpy as np
import argparse
import open3d as o3d
from tqdm import tqdm
from PIL import Image
import mvdatasets as mvds
from volsurfs_py.params.paths_params import PathsParams
from volsurfs_py.params.cmd_params import CmdParams
from volsurfs_py.params.data_params import DataParams
from volsurfs_py.params.hyper_params import get_method_hyper_params
from volsurfs_py.utils.background import get_bg_color
from volsurfs_py.methods.volsurfs import VolSurfs
from volsurfs_py.methods.offsets_surfs import OffsetsSurfs
from volsurfs_py.utils.mesh_extraction import (
    extract_o3d_mesh_from_method,
    extract_o3d_meshes_from_offsets_surfs,
    simplify_o3d_mesh,
    save_o3d_mesh,
    load_o3d_mesh,
)
from volsurfs_py.utils.texture_extraction import (
    extract_textures,
    compute_o3d_mesh_atlas,
    dilate_texture,
)
from volsurfs_py.utils.mesh_from_depth import MeshExtractor
from volsurfs_py.utils.rendering import render_cameras
from volsurfs_py.utils.volsurfs_utils import (
    print_params,
    init_run,
    init_method,
    init_bounding_primitive,
)

# from volsurfs_py.models.colorcal import Colorcal
# mvdatasets imports
from mvdatasets.utils.images import save_numpy_as_png
from mvdatasets.utils.profiler import Profiler


def save_textures_as_png(texture_np, texture_idx, mesh_idx, textures_path):

    print(f"texture_np.shape: {texture_np.shape}")

    # check if texture is 4 channels
    if texture_np.shape[2] != 4:
        print(
            f"[bold red]ERROR[/bold red]: texture has {texture_np.shape[2]} channels, should have 4"
        )
        exit(1)

    # get number of features stored in the texture
    nr_features = texture_np.shape[3]

    textures_paths = []

    # iterate over features, save an image for each
    for i in range(nr_features):
        img_np = texture_np[:, :, :, i]
        img_filename = f"mesh_{mesh_idx}_texture_{texture_idx}_feature_{i}.png"
        # save as png
        save_numpy_as_png(img_np, textures_path, img_filename, append_format=False)
        textures_paths.append(os.path.join("textures", img_filename))

    return textures_paths


def main():
    print("\n[bold blue]BAKER[/bold blue]")

    # argparse
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument(
        "--method_name", help="Name of the method to use", required=True, type=str
    )
    parser.add_argument(
        "--dataset",
        help="Dataset like dtu, nerf_synthetic, ...",
        required=True,
        type=str,
    )
    parser.add_argument("--scene", help="Scene name", required=True, type=str)
    parser.add_argument(
        "--exp_name", help="Experiment config name", required=True, type=str
    )
    parser.add_argument(
        "--run_id",
        help="Run ID of a previous run to load (will load last checkpoint)",
        type=str,
    )
    parser.add_argument(
        "--extract_meshes",
        action="store_true",
        help="Set to extract meshes with marching cubes",
    )
    parser.add_argument(
        "--extract_bg_mesh",
        action="store_true",
        help="Set to extract the background mesh",
    )
    parser.add_argument(
        "--nr_meshes_to_extract",
        default=1,
        help="Number of meshes to extract (if > 1, will use the offset to \
            shift the level sets)",
        type=int,
    )
    parser.add_argument(
        "--delta_surfs",
        default=0.0025,
        help="Delta to shift the level sets",
        type=float,
    )
    parser.add_argument(
        "--extract_mesh_res",
        default=1000,
        help="Resolution of the extracted mesh (usually >= 700)",
        type=int,
    )
    parser.add_argument(
        "--extract_mesh_density_threshold",
        default=40,
        help="Density threshold to extract with marching cubes",
        type=float,
    )
    parser.add_argument(
        "--extract_level_set",
        default=0.0,
        help="SDF level set to extract with marching cubes",
        type=float,
    )
    # parser.add_argument(
    #     "--remove_invisible_faces",
    #     action="store_true",
    #     help="Set to remove faces not visibile from any training view",
    # )
    parser.add_argument(
        "--simplify_meshes",
        action="store_true",
        help="Set to export the simplify the extracted meshes",
    )
    parser.add_argument(
        "--simplification_faces_ratio",
        default=0.025,
        help="Simplification target number of faces in percentage",
        type=float,
    )
    parser.add_argument(
        "--compute_meshes_xatlas",
        action="store_true",
        help="Compute the atlas of the extracted (and simplified) mesh",
    )
    parser.add_argument(
        "--extract_textures",
        action="store_true",
        help="Set to export the appearance model as texture",
    )
    parser.add_argument(
        "--extract_texture_res",
        default=None,
        help="Resolution of extracted textures",
        type=int,
    )
    parser.add_argument(
        "--extract_texture_samples_per_texel",
        default=None,
        help="Number of random samples per texel when extracting textures",
        type=int,
    )
    parser.add_argument(
        "--dilate_texture",
        action="store_true",
        help="Set to run dilation on the extracted textures",
    )
    parser.add_argument(
        "--nr_dilation_iters",
        default=5,
        help="Size of the kernel used for dilation",
        type=int,
    )
    parser.add_argument(
        "--subsample_factor",
        type=int,
        default=-1,
        help="subsample factor for the dataset resolution",
    )
    args = CmdParams(parser.parse_args().__dict__)

    # paths
    paths = PathsParams(args)

    # init run
    run_id, start_iter_nr = init_run(args, paths, is_run_id_required=True)

    # read the hyper_params config file
    hyper_params = get_method_hyper_params(args["method_name"], paths["hyper_params"])

    # initialize the parameters used for data loading
    data_params = DataParams(
        paths["datasets"], args["dataset"], args["scene"], paths["data_config"]
    )
    if args["subsample_factor"] > 0:
        data_params.subsample_factor = args["subsample_factor"]

    # create profiler
    profiler = Profiler(verbose=False)

    # ADDITIONAL PATHS -------------------------------------------------

    # #
    # paths["last_checkpoint"] = os.path.join(
    #     paths["runs"], run_id, format(start_iter_nr, "07d")
    # )
    # # meshes path
    # paths["meshes"] = os.path.join(
    #     paths["last_checkpoint"],
    #     "meshes",
    # )
    # # tmp renders path
    # paths["tmp_renders"] = os.path.join(
    #     paths["last_checkpoint"],
    #     "tmp_renders",
    # )
    # # cleaned meshes path
    # paths["meshes_cleaned"] = os.path.join(
    #     paths["last_checkpoint"],
    #     "meshes_cleaned",
    # )
    # # simplified meshes path
    # paths["meshes_simplified"] = os.path.join(
    #     paths["last_checkpoint"],
    #     "meshes_simplified",
    # )
    # # uv simplified meshes path
    # paths["meshes_simplified_uvs"] = os.path.join(
    #     paths["last_checkpoint"],
    #     "meshes_simplified_uvs",
    # )
    # # textures path
    # paths["textures"] = os.path.join(
    #     paths["last_checkpoint"],
    #     "textures",
    # )

    # Define common paths using a dictionary
    subdirs = [
        "meshes",
        "tmp_renders",
        "meshes_cleaned",
        "meshes_simplified",
        "meshes_simplified_uvs",
        "textures",
    ]

    # Create paths
    paths["last_checkpoint"] = os.path.join(
        paths["runs"], run_id, format(start_iter_nr, "07d")
    )

    # Generate paths for subdirectories
    for subdir in subdirs:
        paths[subdir] = os.path.join(paths["last_checkpoint"], subdir)

    # PRINTING --------------------------------------------------------

    print_params(
        args=args, data_params=data_params, hyper_params=hyper_params, paths=paths
    )

    # DATA LOADING --------------------------------------------------------

    print("\n[bold blue]loading data[/bold blue]")
    mv_data = mvds.MVDataset(
        args["dataset"],
        args["scene"],
        data_params.datasets_path,
        splits=["train", "test"],
        config=data_params.dict(),
        verbose=True,
    )

    # if mv_data.scene_radius > 1.0:
    #    print("[bold red]ERROR[/bold red]: scene radius is too large, should be <= 1.0")
    #    exit(1)

    if hyper_params.is_training_masked or hyper_params.is_testing_masked:
        if not mv_data.has_masks():
            print("[bold red]ERROR[/bold red]: dataset does not have masks")
            exit(1)

    # bounding primitive

    bounding_primitive = init_bounding_primitive(mv_data, device)

    # MODEL --------------------------------------------------------

    # constant background color
    bg_color = get_bg_color(data_params.bg_color)

    method = init_method(
        method_name=args["method_name"],
        train=False,
        hyper_params=hyper_params,
        load_checkpoints_path=paths["load_checkpoints"],
        save_checkpoints_path=None,  # paths["save_checkpoints"],
        bounding_primitive=bounding_primitive,
        model_colorcal=None,  # model_colorcal,
        bg_color=bg_color,
        start_iter_nr=start_iter_nr,
        profiler=profiler,
    )

    # MESH EXTRACTION, SIMPLIFICATION, ATLAS COMPUTATION --------------------

    # foreground meshes
    if args["extract_meshes"] and args["nr_meshes_to_extract"] > 0:

        if (
            args["method_name"] == "surf"
            or args["method_name"] == "nerf"
            or args["method_name"] == "offsets_surfs"
        ):

            if args["nr_meshes_to_extract"] > 1:

                delta_surfs = args["delta_surfs"]

                surfs_const_offsets = delta_surfs * (args["nr_meshes_to_extract"] // 2)
                levelsets = torch.linspace(
                    -surfs_const_offsets,
                    surfs_const_offsets,
                    args["nr_meshes_to_extract"],
                )
                print(f"levelsets selected {levelsets}")

            else:

                levelsets = [args["extract_level_set"]]

        # elif args['method_name'] == "offsets_surfs":
        #     pass

        else:
            print(
                f"[bold red]ERROR[/bold red]: method {args['method_name']} does not support mesh extraction"
            )
            exit(1)

        print("\nextracting meshes")

        os.makedirs(paths["meshes"], exist_ok=True)

        # extract levelsets of the same field
        if args["method_name"] == "surf" or args["method_name"] == "nerf":

            for i, levelset in enumerate(levelsets):
                if isinstance(levelset, torch.Tensor):
                    levelset_ = round(levelset.item(), 4)
                else:
                    levelset_ = round(levelset, 4)
                print(f"\nmesh {i+1}/{len(levelsets)}, d={levelset_}")

                # returns open3d mesh
                o3d_mesh = extract_o3d_mesh_from_method(
                    method=method,
                    nr_points_per_dim=int(args["extract_mesh_res"]),
                    bounding_primitive=bounding_primitive,
                    levelset=levelset_,
                    iter_nr=start_iter_nr,
                    extract_main_surf=True,
                )

                # extracted mesh stats
                print("stats")
                print("    nr_vertices", len(o3d_mesh.vertices))
                print("    nr_faces", len(o3d_mesh.triangles))

                # save high res mesh
                mesh_filename = f"{levelset_}.ply"
                mesh_path = os.path.join(paths["meshes"], mesh_filename)
                save_o3d_mesh(o3d_mesh, mesh_path)

        # extract 0-level sets of different SDFs
        elif args["method_name"] == "offsets_surfs":

            if args["nr_meshes_to_extract"] > 1:

                # extract k levels sets of the main SDF

                for i, levelset in enumerate(levelsets):
                    if isinstance(levelset, torch.Tensor):
                        levelset_ = round(levelset.item(), 4)
                    else:
                        levelset_ = round(levelset, 4)
                    print(f"\nmesh {i+1}/{len(levelsets)}, d={levelset_}")

                    # returns open3d mesh
                    o3d_mesh = extract_o3d_mesh_from_method(
                        method=method,
                        nr_points_per_dim=int(args["extract_mesh_res"]),
                        bounding_primitive=bounding_primitive,
                        levelset=levelset_,
                        iter_nr=start_iter_nr,
                        extract_main_surf=True,
                    )

                    # extracted mesh stats
                    print("stats")
                    print("    nr_vertices", len(o3d_mesh.vertices))
                    print("    nr_faces", len(o3d_mesh.triangles))

                    # save high res mesh
                    mesh_filename = f"{i}.ply"
                    mesh_path = os.path.join(paths["meshes"], mesh_filename)
                    save_o3d_mesh(o3d_mesh, mesh_path)

            else:

                # extract the k-SDFs

                o3d_meshes = extract_o3d_meshes_from_offsets_surfs(
                    method=method,
                    nr_points_per_dim=int(args["extract_mesh_res"]),
                    bounding_primitive=bounding_primitive,
                    iter_nr=start_iter_nr,
                )

                for i, o3d_mesh in enumerate(o3d_meshes):

                    # extracted mesh stats
                    print("stats")
                    print("    nr_vertices", len(o3d_mesh.vertices))
                    print("    nr_faces", len(o3d_mesh.triangles))

                    # save high res mesh
                    mesh_filename = f"{i}.ply"
                    mesh_path = os.path.join(paths["meshes"], mesh_filename)
                    save_o3d_mesh(o3d_mesh, mesh_path)

        else:
            print(
                f"[bold red]ERROR[/bold red]: method {args['method_name']} mesh extraction procedure not defined"
            )
            exit(1)

    if args["extract_bg_mesh"]:

        # check if method has background
        if "bg" not in method.models or method.models["bg"] is None:
            print(
                f"[bold yellow]WARNING[/bold yellow]: method {args['method_name']} does not support background mesh extraction"
            )
            exit(1)

        else:
            # extracting from depth

            os.makedirs(paths["tmp_renders"], exist_ok=True)

            # only render if depths_fg.npz is not available
            if not os.path.exists(os.path.join(paths["tmp_renders"], "depths_fg.npz")):
                print("\nrendering training views")

                # render all training views
                output = render_cameras(
                    cameras=mv_data["train"],
                    method=method,
                    iter_nr=None,
                    render_modes=[
                        ("volumetric", ["rgb", "depth_fg", "depth_bg", "weights_sum"]),
                    ],
                )

                depths_fg = {
                    camera_idx: camera_vals["renders"]["volumetric"]["depth_fg"]
                    for camera_idx, camera_vals in output.items()
                }
                depths_bg = {
                    camera_idx: camera_vals["renders"]["volumetric"]["depth_bg"]
                    for camera_idx, camera_vals in output.items()
                }
                fg_mask = {
                    camera_idx: camera_vals["renders"]["volumetric"]["weights_sum"]
                    for camera_idx, camera_vals in output.items()
                }
                rgbs = {
                    camera_idx: camera_vals["renders"]["volumetric"]["rgb"]
                    for camera_idx, camera_vals in output.items()
                }

                # save as dictionaries
                np.savez(
                    os.path.join(paths["tmp_renders"], "depths_fg.npz"), **depths_fg
                )
                np.savez(
                    os.path.join(paths["tmp_renders"], "depths_bg.npz"), **depths_bg
                )
                np.savez(os.path.join(paths["tmp_renders"], "fg_mask.npz"), **fg_mask)
                np.savez(os.path.join(paths["tmp_renders"], "rgbs.npz"), **rgbs)
                print("all saved")

            else:

                # load them

                def load_npz_as_dict(file_path):
                    data = np.load(file_path, allow_pickle=True)
                    return {key: data[key] for key in data}

                # load npz
                depths_fg = load_npz_as_dict(
                    os.path.join(paths["tmp_renders"], "depths_fg.npz")
                )
                depths_bg = load_npz_as_dict(
                    os.path.join(paths["tmp_renders"], "depths_bg.npz")
                )
                fg_masks = load_npz_as_dict(
                    os.path.join(paths["tmp_renders"], "fg_mask.npz")
                )
                rgbs = load_npz_as_dict(os.path.join(paths["tmp_renders"], "rgbs.npz"))

            poses = {
                camera_idx: camera_vals["pose"]
                for camera_idx, camera_vals in output.items()
            }
            intrinsics = {
                camera_idx: camera_vals["intrinsics"]
                for camera_idx, camera_vals in output.items()
            }

            output = {}
            for key in depths_fg.keys():

                output[key] = {}

                rgb = rgbs[key]
                depth_fg = depths_fg[key]
                depth_bg = depths_bg[key]
                fg_mask = fg_masks[key]
                # depth = depth_fg * fg_mask + depth_bg * (1 - fg_mask)
                depth = depth_fg

                renders = {}
                renders["volumetric"] = {
                    "depth": depth,  # H, W, 1
                    "rgb": rgb,  # H, W, 3
                }
                output[key]["renders"] = renders

                output[key]["pose"] = poses[key]
                output[key]["intrinsics"] = intrinsics[key]

            # convert to lists of torch.tensor
            rgbs = []
            depths = []
            c2ws = []
            ixts = []

            for camera_idx, camera in output.items():
                rgbs.append(
                    torch.tensor(
                        camera["renders"]["volumetric"]["rgb"].transpose((2, 0, 1))
                    ).float()
                )  # 3, H, W
                depths.append(
                    torch.tensor(
                        camera["renders"]["volumetric"]["depth"].transpose((2, 0, 1))
                    ).float()
                )  # 1, H, W
                c2ws.append(camera["pose"])
                ixts.append(camera["intrinsics"])

            # TODO: continue

            # # convert to lists of torch.tensor
            # rgbs = []
            # depths = []
            # c2ws = []
            # ixts = []

            # for camera_idx, camera in output.items():
            #     rgbs.append(torch.tensor(camera["renders"]["volumetric"]["rgb"]).permute(2, 0, 1))
            #     depths.append(torch.tensor(camera["renders"]["volumetric"]["depth"]).permute(2, 0, 1))
            #     c2ws.append(camera["pose"])
            #     ixts.append(camera["intrinsics"])

            # for i in range(len(rgbs)):
            #     print(f"rgbs[{i}].shape: {rgbs[i].shape}")
            #     print(f"depths[{i}].shape: {depths[i].shape}")
            #     print(f"c2ws[{i}].shape: {c2ws[i].shape}")
            #     print(f"ixts[{i}].shape: {ixts[i].shape}")

            # extractor = MeshExtractor(depths, rgbs, c2ws, ixts)
            # mesh = extractor.extract_mesh_unbounded()

            # o3d.io.write_triangle_mesh(
            #     'test.obj',
            #     mesh,
            #     write_ascii=False,
            #     compressed=True,
            #     write_vertex_normals=False,
            #     write_vertex_colors=False,
            #     write_triangle_uvs=False,
            #     print_progress=True
            # )

        #     # returns open3d mesh
        #     o3d_mesh = extract_o3d_mesh_from_method(
        #         method=method,
        #         nr_points_per_dim=int(args["extract_mesh_res"]),
        #         is_bg=True,
        #         bounding_primitive=bounding_primitive,
        #         levelset=0.4,
        #         iter_nr=start_iter_nr,
        #     )

        #     # extracted mesh stats
        #     print("stats")
        #     print("    nr_vertices", len(o3d_mesh.vertices))
        #     print("    nr_faces", len(o3d_mesh.triangles))

        #     # save high res mesh
        #     mesh_filename = f"bg.ply"
        #     mesh_path = os.path.join(paths["meshes"], mesh_filename)
        # save_o3d_mesh(o3d_mesh, mesh_path)

    # # check if should remove invisible trianlges from meshes
    # if args["remove_invisible_faces"]:

    #     print("\nremoving invisible faces")

    #     os.makedirs(paths["meshes_cleaned"], exist_ok=True)

    #     # iterate over meshes
    #     meshes_filenames = os.listdir(paths["meshes"])
    #     print("meshes_path", paths["meshes"])
    #     print("meshes_filenames", meshes_filenames)

    #     if len(meshes_filenames) == 0:
    #         print("[bold red]ERROR[/bold red]: no meshes to clean")
    #         exit(1)

    #     for i, mesh_filename in enumerate(meshes_filenames):
    #         print(f"\nmesh {i+1}/{len(meshes_filenames)}")

    #         # load o3d mesh
    #         mesh_path = os.path.join(paths["meshes"], mesh_filename)
    #         o3d_mesh = load_o3d_mesh(mesh_path)
    #         if o3d_mesh is None:
    #             continue

    #         if mv_data is None:
    #             print("[bold red]ERROR[/bold red]: no cameras data to remove invisible faces")
    #             exit(1)

    #         # clean meshes
    #         cleaned_o3d_mesh = None

    #         # TODO: implement
    #         # cleaned_o3d_mesh = simplify_o3d_mesh(
    #         #     o3d_mesh,
    #         #     mv_data
    #         # )

    #         # extracted mesh stats
    #         print("stats")
    #         print("    nr_vertices", len(cleaned_o3d_mesh.vertices))
    #         print("    nr_faces", len(cleaned_o3d_mesh.triangles))

    #         # save simplified mesh
    #         save_o3d_mesh(cleaned_o3d_mesh, os.path.join(paths["meshes_cleaned"], mesh_filename))

    # check if should simplify mesh
    if args["simplify_meshes"]:

        print("\nsimplifying meshes")

        os.makedirs(paths["meshes_simplified"], exist_ok=True)

        # iterate over meshes
        meshes_filenames = os.listdir(paths["meshes"])
        print("meshes_path", paths["meshes"])
        print("meshes_filenames", meshes_filenames)

        if len(meshes_filenames) == 0:
            print("[bold red]ERROR[/bold red]: no meshes to simplify")
            exit(1)

        for i, mesh_filename in enumerate(meshes_filenames):
            print(f"\nmesh {i+1}/{len(meshes_filenames)}")

            # load o3d mesh
            mesh_path = os.path.join(paths["meshes"], mesh_filename)
            o3d_mesh = load_o3d_mesh(mesh_path)
            if o3d_mesh is None:
                continue

            # simplify mesh
            simplified_o3d_mesh = simplify_o3d_mesh(
                o3d_mesh,
                target_nr_faces_ratio=args["simplification_faces_ratio"],
            )

            # extracted mesh stats
            print("stats")
            print("    nr_vertices", len(simplified_o3d_mesh.vertices))
            print("    nr_faces", len(simplified_o3d_mesh.triangles))

            # save simplified mesh
            save_o3d_mesh(
                simplified_o3d_mesh,
                os.path.join(paths["meshes_simplified"], mesh_filename),
            )

            # deleting original mesh
            # TODO: make optional
            # os.remove(mesh_path)

    if args["compute_meshes_xatlas"]:

        print("\ncomputing xatlas")

        os.makedirs(paths["meshes_simplified_uvs"], exist_ok=True)

        # iterate over meshes
        meshes_filenames = os.listdir(paths["meshes_simplified"])
        print("meshes_path", paths["meshes_simplified"])
        print("meshes_filenames", meshes_filenames)

        if len(meshes_filenames) == 0:
            print("[bold red]ERROR[/bold red]: no meshes to compute xatlas")
            exit(1)

        for i, mesh_filename in enumerate(meshes_filenames):
            print(f"\nmesh {i+1}/{len(meshes_filenames)}")

            # load o3d mesh
            mesh_path = os.path.join(paths["meshes_simplified"], mesh_filename)
            simplified_o3d_mesh = load_o3d_mesh(mesh_path)
            if simplified_o3d_mesh is None:
                continue

            # compute atlas
            simplified_with_atlas_o3d_mesh, atlas_img_np = compute_o3d_mesh_atlas(
                simplified_o3d_mesh, bilinear=False, padding=3, verbose=True
            )

            # save simplified mesh
            save_o3d_mesh(
                simplified_with_atlas_o3d_mesh,
                os.path.join(
                    paths["meshes_simplified_uvs"], mesh_filename[:-4] + ".obj"
                ),
                write_uvs=True,
            )

            # save atlas img
            paths["meshes_simplified_uvs_img"] = os.path.join(
                paths["meshes_simplified_uvs"], "atlas"
            )
            os.makedirs(paths["meshes_simplified_uvs_img"], exist_ok=True)
            atlas_img_path = os.path.join(
                paths["meshes_simplified_uvs_img"], mesh_filename[:-4] + ".png"
            )
            atlas_img = Image.fromarray(atlas_img_np)
            atlas_img.save(atlas_img_path)

    # TEXTURE EXTRACTION ----------------------------------------------------

    if args["extract_textures"]:

        # check if method supports texture extraction
        if not (isinstance(method, VolSurfs)):
            print(
                f"[bold red]ERROR[/bold red]: method {args['method_name']} does not support texture extraction"
            )
            exit(1)

        # check if appearance model is a 3D or 2D field
        if (
            args["extract_texture_res"] is None
            and not method.hyper_params.using_neural_textures
        ):
            print(
                f"[bold red]ERROR[/bold red]: method {args['method_name']} requires a texture resolution"
            )
            exit(1)

        if (
            args["extract_texture_res"] is not None
            and method.hyper_params.using_neural_textures
        ):
            print(
                f"[bold red]ERROR[/bold red]: method {args['method_name']} with neural textures does not require a texture resolution"
            )
            exit(1)

        if args["dilate_texture"] and method.hyper_params.using_neural_textures:
            print(
                f"[bold error]WARNING[/bold error]: method {args['method_name']} with neural textures does not support dilation"
            )
            exit(1)

        # create destination folder
        os.makedirs(paths["textures"], exist_ok=True)

        meshes_info = []
        if method.hyper_params.using_neural_textures:

            for mesh_idx in tqdm(range(method.nr_meshes)):

                print(f"\nextracting texture from mesh {mesh_idx}")

                model_rgb = method.models[f"rgb_{mesh_idx}"]
                model_alpha = method.models[f"alpha_{mesh_idx}"]

                if model_alpha is None:
                    ignore_alpha = True
                else:
                    ignore_alpha = False

                # extract textures

                textures_rgb_np = model_rgb.render(bake=True)

                if model_alpha is not None:
                    textures_alpha_np = model_alpha.render(bake=True)
                else:
                    # full 1.0 alphas
                    textures_alpha_np = []
                    for texture_rgb_np in textures_rgb_np:
                        print(f"texture_rgb_np.shape: {texture_rgb_np.shape}")
                        texture_alpha_np = np.ones(
                            texture_rgb_np.shape[:-2] + (1, texture_rgb_np.shape[-1]),
                            dtype=np.float32,
                        )
                        print(f"texture_alpha_np.shape: {texture_alpha_np.shape}")
                        textures_alpha_np.append(texture_alpha_np)

                textures_info = []
                sh_range = method.hyper_params.sh_range
                for texture_idx, (texture_rgb_np, texture_alpha_np) in enumerate(
                    zip(textures_rgb_np, textures_alpha_np)
                ):

                    texture_np = np.concatenate(
                        [texture_rgb_np, texture_alpha_np], axis=2
                    )

                    # assert all values are between 0 and 1
                    if np.any(texture_np < 0.0) or np.any(texture_np > 1.0):
                        print(
                            f"[bold red]ERROR[/bold red]: texture has values outside [0, 1]"
                        )
                        exit(1)

                    # clip to [0, 1]
                    # texture_np = np.clip(texture_np, 0.0, 1.0)

                    # texture_np = np.rot90(texture_np, k=1)

                    # flip y axis
                    texture_np = np.flipud(texture_np).copy()

                    # save to disk
                    textures_paths = save_textures_as_png(
                        texture_np, texture_idx, mesh_idx, paths["textures"]
                    )

                    for texture_path in textures_paths:
                        textures_info.append(
                            {
                                "texture_path": texture_path,
                                "texture_scale": (
                                    -sh_range[texture_idx],
                                    sh_range[texture_idx],
                                ),
                                "texture_resolution": tuple(texture_np.shape[:2]),
                            }
                        )

                mesh_info = {
                    "mesh_path": os.path.join("meshes", str(mesh_idx) + ".obj"),
                    "textures": textures_info,
                    "ignore_alpha": ignore_alpha,
                }

                meshes_info.append(mesh_info)

        else:

            print(
                "[bold red]ERROR[/bold red]: legacy texture extraction method deprecated"
            )
            exit(1)

            # TODO: re-enable

            # # legacy method
            # nr_samples_per_texel = args["extract_texture_samples_per_texel"]

            # if nr_samples_per_texel is None:
            #     print(f"[bold red]ERROR[/bold red]: method {args['method_name']} requires a number of samples per texel")
            #     exit(1)

            # texture_res = args["extract_texture_res"]

            # for mesh_idx in tqdm(range(method.nr_meshes)):

            #     print(f"\nextracting texture from mesh {mesh_idx})")

            #     texture_rgb, texture_alpha = extract_textures(
            #         mesh_idx=mesh_idx,
            #         method=method,
            #         texture_res=texture_res,
            #         nr_samples_per_texel=nr_samples_per_texel,
            #         iter_nr=start_iter_nr,
            #     )
            #     texture_rgb_np = texture_rgb.cpu().numpy()

            #     # rotate it as WebGL likes
            #     texture_rgb_np = np.rot90(texture_rgb_np, k=1)
            #     if texture_alpha is not None:
            #         texture_alpha_np = texture_alpha.cpu().numpy()
            #         texture_alpha_np = np.rot90(texture_alpha_np, k=1)

            #     # run dilation (optional)
            #     if args["dilate_texture"]:
            #         # dilate rgb texture
            #         texture_rgb_np = dilate_texture(texture_rgb_np, args["nr_dilation_iters"])
            #         # dilate alpha texture
            #         if texture_alpha is not None:
            #             texture_alpha_np = dilate_texture(texture_alpha_np, args["nr_dilation_iters"])

            #     textures_rgb_info = save_textures_as_png(texture_rgb_np, mesh_idx, texture_res, nr_samples_per_texel, paths["textures"])
            #     if texture_alpha is not None:
            #         textures_alpha_info = save_textures_as_png(texture_alpha_np, mesh_idx, texture_res, nr_samples_per_texel, paths["textures"])

            #     textures_info = {
            #         "mesh_path": os.path.join("meshes", str(mesh_idx) + ".obj"),
            #         "rgb_textures": textures_rgb_info,
            #         "alpha_textures": textures_alpha_info
            #     }

            #     mesh_info = {
            #         "mesh_path": os.path.join("meshes", str(mesh_idx) + ".obj"),
            #         "textures": textures_info
            #     }
            #     meshes_info.append(mesh_info)

        # collect scene info
        scene_info = {}

        include_resolutions = True
        if include_resolutions:
            scene_info["resolution"] = ([mv_data.get_width(), mv_data.get_height()],)

        include_bg = True
        if include_bg:
            if data_params.bg_color is not None:
                scene_info["bg_color"] = data_params.bg_color
            else:
                scene_info["bg_color"] = "black"

        # add meshes info
        scene_info["meshes"] = meshes_info

        include_cameras = True
        if include_cameras:

            # convert cameras to opengl format and save them

            scene_info["cameras"] = {"test": {}, "train": {}}

            # train
            for camera in mv_data["train"]:
                camera_idx = camera.camera_idx
                projectionMatrix = camera.get_opengl_projection_matrix(
                    near=0.1, far=100.0
                )
                matrixWorld = camera.get_opengl_matrix_world()
                scene_info["cameras"]["train"][camera_idx] = {
                    "projectionMatrix": projectionMatrix.tolist(),
                    "matrixWorld": matrixWorld.tolist(),
                }

            # test
            for camera in mv_data["test"]:
                camera_idx = camera.camera_idx
                projectionMatrix = camera.get_opengl_projection_matrix(
                    near=0.1, far=100.0
                )
                matrixWorld = camera.get_opengl_matrix_world()
                scene_info["cameras"]["test"][camera_idx] = {
                    "projectionMatrix": projectionMatrix.tolist(),
                    "matrixWorld": matrixWorld.tolist(),
                }

        # save scene settings to scene file
        scene_info_path = os.path.join(paths["last_checkpoint"], "scene.json")

        # save as json
        with open(scene_info_path, "w") as f:
            json.dump(scene_info, f, indent=4)

        print(f"\nscene info saved to {scene_info_path}")

    # finished
    return


if __name__ == "__main__":
    # Set a random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)

    torch.set_default_dtype(torch.float32)

    # # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(seed)  # Set a random seed for GPU
    else:
        device = "cpu"
    torch.set_default_device(device)

    main()
