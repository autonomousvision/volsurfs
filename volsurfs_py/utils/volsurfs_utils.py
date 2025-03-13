from rich import print
import numpy as np
import os
import datetime
import shutil
from volsurfs_py.utils.training import get_last_checkpoint_in_path
from volsurfs_py.methods.nerf import NeRF
from volsurfs_py.methods.surf import Surf
from volsurfs_py.methods.offsets_surfs import OffsetsSurfs
from volsurfs_py.methods.volsurfs import VolSurfs
from mvdatasets.geometry.primitives.bounding_sphere import BoundingSphere
from mvdatasets.geometry.primitives.bounding_box import BoundingBox


def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True


def print_params(**kwargs):
    for key, value in kwargs.items():
        print(f"\n[bold blue]{key}[/bold blue]")
        if isinstance(value, dict):
            for dict_key, dict_value in value.items():
                print(dict_key, dict_value)
        else:
            print(value)


def init_run(args, paths, is_run_id_required=False):
    """
    Initialize run_id and checkpoints folder.
    Updates paths in-place.
    """

    # if run_id is not given, create a new one
    if args["run_id"] is None:

        if is_run_id_required:
            print("[bold red]ERROR[/bold red]: run_id must be provided")
            exit(1)

        if args["continue_training"]:
            print(
                "[bold red]ERROR[/bold red]: run_id must be provided when continuing training"
            )
            exit(1)

        # get current one
        print("run_id not provided, creating new run")
        run_id = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

        # create checkpoints folder
        paths["save_checkpoints"] = os.path.join(
            paths["runs"], run_id
        )  # create checkpoints folder
        os.makedirs(paths["save_checkpoints"], exist_ok=True)

        # start from scratch
        paths["load_checkpoints"] = paths["save_checkpoints"]

        # copy config files to checkpoints folder
        os.makedirs(os.path.join(paths["save_checkpoints"], "config"), exist_ok=True)
        shutil.copyfile(
            paths["hyper_params"],
            os.path.join(paths["save_checkpoints"], "config", "hyper_params.cfg"),
        )
        shutil.copyfile(
            paths["train_config"],
            os.path.join(paths["save_checkpoints"], "config", "train_config.cfg"),
        )
        shutil.copyfile(
            paths["data_config"],
            os.path.join(paths["save_checkpoints"], "config", "data_config.cfg"),
        )

        # start from scratch
        start_iter_nr = 0

    else:
        # if run_id is given, load checkpoints from that folder

        run_id = args["run_id"]

        print(f"run_id provided: {run_id}")

        # check if checkpoints folder exists
        paths["load_checkpoints"] = os.path.join(paths["runs"], run_id)
        if not os.path.exists(paths["load_checkpoints"]):
            print("load checkpoints path does not exist: ", paths["load_checkpoints"])
            exit(1)

        last_checkpoint = get_last_checkpoint_in_path(paths["load_checkpoints"])
        if last_checkpoint is None:
            print(
                "[bold yellow]WARNING[/bold yellow]: no checkpoints found in ",
                paths["load_checkpoints"],
            )
            start_iter_nr = 0
        else:
            # latest checkpoint
            start_iter_nr = int(last_checkpoint)

        # load config files from checkpoint folder
        paths["hyper_params"] = os.path.join(
            paths["load_checkpoints"], "config", "hyper_params.cfg"
        )
        paths["train_config"] = os.path.join(
            paths["load_checkpoints"], "config", "train_config.cfg"
        )
        paths["data_config"] = os.path.join(
            paths["load_checkpoints"], "config", "data_config.cfg"
        )

        if args["train"] and not args["continue_training"]:
            # create a new run_id, use provided run_id only for loading checkpoints
            run_id = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S") + "_" + run_id

            # create checkpoints folder
            paths["save_checkpoints"] = os.path.join(
                paths["runs"], run_id
            )  # create checkpoints folder
            os.makedirs(paths["save_checkpoints"], exist_ok=True)

            # copy config files to checkpoints folder
            os.makedirs(
                os.path.join(paths["save_checkpoints"], "config"), exist_ok=True
            )
            shutil.copyfile(
                paths["hyper_params"],
                os.path.join(paths["save_checkpoints"], "config", "hyper_params.cfg"),
            )
            shutil.copyfile(
                paths["train_config"],
                os.path.join(paths["save_checkpoints"], "config", "train_config.cfg"),
            )
            shutil.copyfile(
                paths["data_config"],
                os.path.join(paths["save_checkpoints"], "config", "data_config.cfg"),
            )

        else:
            paths["save_checkpoints"] = paths["load_checkpoints"]

    print("run_id", run_id)
    print("start_iter_nr", start_iter_nr)

    return run_id, start_iter_nr


def init_method(
    method_name,
    train,
    hyper_params,
    load_checkpoints_path,
    save_checkpoints_path,
    bounding_primitive,
    model_colorcal=None,
    bg_color=None,
    start_iter_nr=0,
    profiler=None,
    **kwargs,
):
    if method_name == "nerf":
        method = NeRF(
            train,
            hyper_params,
            load_checkpoints_path,
            save_checkpoints_path,
            bounding_primitive,
            model_colorcal=model_colorcal,
            bg_color=bg_color,
            start_iter_nr=start_iter_nr,
            profiler=profiler,
        )
    elif method_name == "surf":

        init_sphere_radius = kwargs.get("init_sphere_radius", None)

        method = Surf(
            train,
            hyper_params,
            load_checkpoints_path,
            save_checkpoints_path,
            bounding_primitive,
            model_colorcal=model_colorcal,
            bg_color=bg_color,
            start_iter_nr=start_iter_nr,
            init_sphere_radius=init_sphere_radius,
            # contract_samples=contract_samples,
            profiler=profiler,
        )
    elif method_name == "volsurfs":
        meshes_path = kwargs.get("meshes_path", None)
        models_path = kwargs.get("models_path", None)
        method = VolSurfs(
            train,
            hyper_params,
            load_checkpoints_path,
            save_checkpoints_path,
            bounding_primitive,
            meshes_path=meshes_path,
            models_path=models_path,
            model_colorcal=model_colorcal,
            bg_color=bg_color,
            start_iter_nr=start_iter_nr,
            profiler=profiler,
        )
    elif method_name == "offsets_surfs":
        models_path = kwargs.get("models_path", None)
        method = OffsetsSurfs(
            train,
            hyper_params,
            load_checkpoints_path,
            save_checkpoints_path,
            bounding_primitive,
            models_path=models_path,
            model_colorcal=model_colorcal,
            bg_color=bg_color,
            start_iter_nr=start_iter_nr,
            profiler=profiler,
        )
    else:
        print(f"[bold red]ERROR[/bold red]: unknown method name {method_name}")
        exit(1)

    return method


def init_bounding_primitive(mv_data, device):
    """
    Initialize bounding primitive based on scene type.
    """
    if mv_data.scene_type == "bounded":
        # bounding box around the scene, scale depends on scene radius
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
        print(
            "[bold blue]INFO[/bold blue]: bounded scene, using AABB as bounding primitive"
        )
    elif mv_data.scene_type == "unbounded":
        # bounding sphere around the foreground of the scene (radius 0.5)
        bounding_primitive = BoundingSphere(
            pose=np.eye(4),
            local_scale=np.array([0.5, 0.5, 0.5]),
            device=device,
            verbose=True,
        )
        print(
            "[bold blue]INFO[/bold blue]: unbounded scene, using Sphere as bounding primitive"
        )
    else:
        print(
            "[bold red]ERROR[/bold red]: unknown scene type, should be 'bounded' or 'unbounded'"
        )
        exit(1)

    return bounding_primitive
