#!/usr/bin/env python3

# python ./volsurfs_py/visualizer.py --method_name surf --dataset dtu --scene dtu_scan24 --exp_name default --run_id 2023-09-07-151930

from rich import print
import torch
import os
import numpy as np
import argparse

# volsurfs imports
import volsurfs

# from volsurfs import Sphere
from volsurfs_py.utils.background import get_bg_color
from volsurfs_py.params.paths_params import PathsParams
from volsurfs_py.params.cmd_params import CmdParams
from volsurfs_py.params.data_params import DataParams
from volsurfs_py.params.hyper_params import get_method_hyper_params
from volsurfs_py.viewer.viewer import Viewer
from volsurfs_py.renderers.mesh_renderer import MeshRenderer

# raytracelib imports
from volsurfs_py.utils.training import get_last_checkpoint_in_path

# mvdatasets imports
from mvdatasets import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.geometry.primitives.bounding_sphere import BoundingSphere
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from volsurfs_py.utils.volsurfs_utils import print_params, init_run, init_method


def main():
    print("\n[bold blue]VISUALIZER[/bold blue]")

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
    # parser.add_argument("--texture_path", default="textures/smurf.png", type=str)
    # parser.add_argument("--width", type=int, default=1920, help="GUI width")
    # parser.add_argument("--height", type=int, default=1080, help="GUI height")
    # parser.add_argument(
    #     "--bg_color",
    #     help="Constant background color (white, black, random)",
    #     type=str,
    # )
    # parser.add_argument("--normalize_mesh", action="store_true")
    parser.add_argument("--continuous_update", action="store_true")
    # parser.add_argument(
    #     "--radius",
    #     type=float,
    #     default=1.4,
    #     help="default GUI camera radius from center",
    # )
    # parser.add_argument(
    #     "--fovy", type=float, default=50, help="default GUI camera fovy"
    # )
    parser.add_argument(
        "--subsample_factor",
        type=int,
        default=1,
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
    data_params.subsample_factor = args["subsample_factor"]

    # create profiler
    profiler = Profiler(verbose=False)

    # PRINTING ------------------------------------------------------------

    print_params(
        args=args, data_params=data_params, hyper_params=hyper_params, paths=paths
    )

    # DATA LOADING --------------------------------------------------------

    print("\n[bold blue]loading data[/bold blue]")
    mv_data = MVDataset(
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
            print(f"[bold red]ERROR[/bold red]: dataset does not have masks")
            exit(1)

    # bounding primitive
    bounding_primitive = BoundingBox(
        pose=np.eye(4),
        local_scale=np.array(
            [
                mv_data.scene_radius * 2,
                mv_data.scene_radius * 2,
                mv_data.scene_radius * 2,
            ]
        ),
        device=device,
        verbose=True,
    )

    # MODEL --------------------------------------------------------

    # # color calibration model
    # if use_color_calibration:
    #     model_colorcal = Colorcal(train_data_tensor_reel.rgb_reel.shape[0], 0)
    # else:
    model_colorcal = None

    # constant background color
    bg_color = get_bg_color(data_params.bg_color)

    method = init_method(
        method_name=args["method_name"],
        train=False,
        hyper_params=hyper_params,
        load_checkpoints_path=paths["load_checkpoints"],
        save_checkpoints_path=None,  # paths["save_checkpoints"],
        bounding_primitive=bounding_primitive,
        model_colorcal=model_colorcal,
        bg_color=bg_color,
        start_iter_nr=start_iter_nr,
        profiler=profiler,
    )

    # EVALUATION ----------------------------------------------------------

    # set eval mode
    method.set_eval_mode()

    exit()

    # WIP

    gui = GUI(
        method,
        mv_cameras=mv_data,
    )
    gui.render()

    # finished
    exit(0)


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
