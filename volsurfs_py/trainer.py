#!/usr/bin/env python3

# python ./volsurfs_py/trainer.py --method_name surf --dataset dtu --scene dtu_scan24 --exp_name default [--run_id 2023-09-07-151930] [--train] [--eval_test] [--eval_train] [--bg_color white]

from rich import print
import matplotlib.pyplot as plt
import torch
import shutil
import os
import numpy as np
import datetime
import argparse
from tqdm import tqdm

# mvdatasets imports
import mvdatasets as mvds

# volsurfs imports
from volsurfs_py.params.paths_params import PathsParams
from volsurfs_py.params.cmd_params import CmdParams
from volsurfs_py.params.train_params import TrainParams
from volsurfs_py.params.data_params import DataParams
from volsurfs_py.utils.background import get_bg_color
from volsurfs_py.utils.volsurfs_utils import (
    print_params,
    init_run,
    init_method,
    init_bounding_primitive,
)
from volsurfs_py.callbacks.callback_utils import create_callbacks
from volsurfs_py.callbacks.training_state import TrainingState
from volsurfs_py.params.hyper_params import get_method_hyper_params
from volsurfs_py.utils.training import (
    estimate_test_loss,
    save_checkpoints,
    get_opt_params_gradients_norms,
    get_last_checkpoint_in_path,
)
from volsurfs_py.utils.evaluation import render_and_eval
from volsurfs_py.utils.rendering import render_from_camera

# from volsurfs_py.models.colorcal import Colorcal
# mvdatasets imports
from mvdatasets.utils.profiler import Profiler
from mvdatasets.utils.tensor_reel import TensorReel
from volsurfs_py.utils.visualization import (
    visualize_fields_sections,
    visualize_neural_textures,
)
from mvdatasets.utils.images import save_numpy_as_png


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# trains the model
def train(
    run_id,
    train_params,
    data_params,
    exp_name,
    train_data_tensor_reel,
    mv_data,
    method,
    teacher_method=None,
    profiler=None,
    start_iter_nr=0,
    bg_color=None,
    iter_finish_nr=0,
    eval_test=False,
    eval_train=False,
    keep_last_checkpoint_only=True,
):
    callbacks = create_callbacks(
        method,
        mv_data.scene_name,
        exp_name + "_distilled" if teacher_method is not None else exp_name,
        train_params,
        data_params,
        run_id,
        profiler,
    )

    # create phases
    phase = TrainingState()
    phase.iter_nr = start_iter_nr

    # training loop
    nr_training_rays_to_create = method.hyper_params.training_rays_batch_size
    nr_train_rays_per_pixel = method.hyper_params.nr_training_rays_per_pixel
    jitter_training_rays = method.hyper_params.jitter_training_rays
    callbacks.training_started()

    # TODO: this is only true for blender scenes, handle mipnerf360 and dtu differently
    intrinsics = mv_data["train"][0].get_intrinsics()
    width = mv_data["train"][0].width
    height = mv_data["train"][0].height
    camera_radius = mv_data.scene_radius

    new_test_loss_minimum = False
    test_loss_minimum = float("inf")

    pbar = tqdm(
        range(start_iter_nr, iter_finish_nr),
        desc="training",
        unit="iter",
        # ncols=100,
        leave=False,
    )
    for _ in pbar:
        callbacks.iter_started(phase=phase)

        is_last_iter = phase.iter_nr == iter_finish_nr - 1

        # set training mode
        method.set_train_mode()

        method.optimizer.zero_grad()

        if profiler is not None:
            profiler.start("training_ray_batch_gen")

        with torch.no_grad():

            # update the const bg color (if random)
            if bg_color == "random":
                rand_color = torch.rand(1, 3).cuda()
                method.bg_color = rand_color
                if teacher_method is not None:
                    teacher_method.bg_color = rand_color

            if teacher_method is not None:
                # sample random cameras on hemisphere
                from mvdatasets.utils.virtual_cameras import (
                    sample_cameras_on_hemisphere,
                )

                sampled_cameras = sample_cameras_on_hemisphere(
                    intrinsics=intrinsics,
                    width=width,
                    height=height,
                    radius=camera_radius,
                    nr_cameras=100,
                )
                # and create tensor reel
                random_cameras_tensor_reel = TensorReel(
                    sampled_cameras, width=width, height=height, device=device
                )
                # get rays from randomly sampled cameras
                (
                    _,
                    rays_o_rc,
                    rays_d_rc,
                    _,
                    _,
                ) = random_cameras_tensor_reel.get_next_rays_batch(
                    batch_size=int(nr_training_rays_to_create / 2),
                    jitter_pixels=True,
                )
                # get gt values from pre-trained model
                # TODO: add batchify option
                gt_vals_rc = teacher_method.render_rays_batchify(
                    rays_o_all=rays_o_rc,
                    rays_d_all=rays_d_rc,
                    chunk_size=teacher_method.hyper_params.test_rays_batch_size,
                )
                gt_rgb_rc = gt_vals_rc["volumetric"]["rgb"]
                gt_mask_rc = torch.ones_like(gt_rgb_rc[:, 0]).view(-1, 1)
            else:
                rays_o_rc = None
                rays_d_rc = None
                gt_rgb_rc = None
                gt_mask_rc = None

            # get rays and gt values (training cameras)
            (
                _,
                rays_o_tc,  # (N * nr_rays_per_pixel, 3)
                rays_d_tc,  # (N * nr_rays_per_pixel, 3)
                gt_vals_tc,
                _,
            ) = train_data_tensor_reel.get_next_rays_batch(
                batch_size=(
                    int(nr_training_rays_to_create / 2)
                    if teacher_method
                    else nr_training_rays_to_create
                ),
                jitter_pixels=jitter_training_rays,
                nr_rays_per_pixel=nr_train_rays_per_pixel,  # for supersampling
            )

            # get gt rgb and mask
            if "rgb" in gt_vals_tc:
                gt_rgb_tc = gt_vals_tc["rgb"]  # (N, 3)
            else:
                print("[bold red]ERROR[/bold red]: cameras must contain rgb")
                exit(1)
            if "mask" in gt_vals_tc:
                gt_mask_tc = gt_vals_tc["mask"]  # (N, 1)
            else:
                gt_mask_tc = torch.ones_like(gt_rgb_tc[:, 0]).view(-1, 1)

            # mask background if using mask, and color it with bg_color if specified
            if method.hyper_params.is_training_masked:
                gt_rgb_tc = gt_rgb_tc * gt_mask_tc
                if method.bg_color is not None:
                    gt_rgb_tc = gt_rgb_tc + (1 - gt_mask_tc) * method.bg_color

            if teacher_method is not None:
                # concatenate teacher and real data
                rays_o = torch.cat([rays_o_rc, rays_o_tc], dim=0)
                rays_d = torch.cat([rays_d_rc, rays_d_tc], dim=0)
                gt_rgb = torch.cat([gt_rgb_rc, gt_rgb_tc], dim=0)
                gt_mask = torch.cat([gt_mask_rc, gt_mask_tc], dim=0)
            else:
                rays_o = rays_o_tc
                rays_d = rays_d_tc
                gt_rgb = gt_rgb_tc
                gt_mask = gt_mask_tc

        if profiler is not None:
            profiler.end("training_ray_batch_gen")

        callbacks.before_forward_pass()

        # runs the forward pass with autocasting
        # with torch.cuda.amp.autocast():
        # forward and compute losses
        train_losses, additional_info_to_log, samples_3d_fg = method.forward(
            rays_o,
            rays_d,
            gt_rgb,
            gt_mask,
            phase.iter_nr,
            is_first_iter=phase.is_first_iter,
        )

        # log
        callbacks.after_forward_pass(
            phase=phase,
            train_losses=train_losses,
            additional_info_to_log=additional_info_to_log,
            lr=method.optimizer.param_groups[0]["lr"],
        )

        # check if train_losses["loss"] is a scalar tensor that requires grad
        if isinstance(train_losses["loss"], float):
            print(
                "[bold yellow]WARNING[/bold yellow]: loss if a float, skipping backward"
            )
        elif not train_losses["loss"].requires_grad:
            print(
                "[bold yellow]WARNING[/bold yellow]: loss does not require grad, skipping backward"
            )
        else:

            # backward
            callbacks.before_backward_pass()

            if method.grad_scaler is not None:
                # scales loss, then backward
                method.grad_scaler.scale(train_losses["loss"]).backward()
            else:
                train_losses["loss"].backward()

            # calculate gradient norm
            models_grad_norms = get_opt_params_gradients_norms(method)

            # step optimizer
            if method.grad_scaler is not None:
                # scaler.step() first unscales the gradients of the optimizer's assigned params,
                # if gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                method.grad_scaler.step(method.optimizer)
                # updates the scale for next iteration
                method.grad_scaler.update()
            else:
                method.optimizer.step()

            callbacks.after_backward_pass(
                phase=phase, models_grad_norms=models_grad_norms
            )

        if isinstance(train_losses["loss"], torch.Tensor):
            # convert loss to float
            train_losses["loss"] = train_losses["loss"].item()

        # dynamic nr_training_rays_to_create
        with torch.no_grad():
            is_nr_training_rays_dynamic = (
                method.hyper_params.is_nr_training_rays_dynamic
            )
            if is_nr_training_rays_dynamic and samples_3d_fg is not None:
                # adjust nr_training_rays_to_create based on how many samples we have
                cur_nr_samples = samples_3d_fg.shape[0]
                target_nr_of_training_samples = (
                    method.hyper_params.target_nr_of_training_samples
                )
                multiplier_nr_samples = (
                    float(target_nr_of_training_samples) / cur_nr_samples
                )
                nr_training_rays_to_create = int(
                    nr_training_rays_to_create * multiplier_nr_samples
                )

        # lr scheduler update
        if method.lr_scheduler is not None:
            method.lr_scheduler.step()

        # update progress bar
        postfix_dict = {
            "loss": train_losses["loss"],
            "nr_rays": nr_training_rays_to_create,
            "lr": method.optimizer.param_groups[0]["lr"],
        }
        if samples_3d_fg is not None:
            postfix_dict["nr_samples"] = samples_3d_fg.shape[0]
        pbar.set_postfix(postfix_dict)

        # set eval mode
        method.set_eval_mode()

        # estimate test loss
        test_losses = {}
        if (
            train_params.compute_test_loss
            and (
                (phase.iter_nr + 1) % train_params.compute_test_loss_freq == 0
                or is_last_iter
            )
            and train_params.with_wandb
        ):
            # TODO: estimate PSNR
            test_loss, test_losses = estimate_test_loss(
                method, mv_data, nr_training_rays_to_create, iter_nr=phase.iter_nr
            )
            # TODO: fix, happens too often
            # if test_loss < test_loss_minimum:
            #     test_loss_minimum = test_loss
            #     new_test_loss_minimum = True

        # save checkpoint
        if (
            new_test_loss_minimum
            or (
                phase.iter_nr != 0
                and train_params.save_checkpoints
                and ((phase.iter_nr + 1) % train_params.checkpoint_freq == 0)
            )
            or is_last_iter
        ):
            save_checkpoints(
                method,
                phase.iter_nr,
                remove_previous=keep_last_checkpoint_only,  # if not is_last_iter else False  # always keep last checkpoint
            )

        # render a random view from test set (without viewer, in wandb)
        wandb_imgs_np = {}
        if (
            phase.iter_nr == 0
            or (phase.iter_nr + 1) % train_params.render_freq == 0
            or is_last_iter
        ):
            print("\nrendering wandb test view")

            # get random camera from test_data cameras
            phase.test_cam_idx = (phase.test_cam_idx + 1) % len(mv_data["test"])
            cam_idx = phase.test_cam_idx
            camera = mv_data["test"][cam_idx]

            wandb_imgs_np = render_from_camera(
                camera,
                method,
                image_resize_target=300,
                iter_nr=phase.iter_nr,
                use_matplotlib_plots=True,
            )

            # (optional) visualize fields sections
            sections = visualize_fields_sections(method, phase.iter_nr)
            wandb_imgs_np["sections"] = sections

            # (optional) visualize neural textures
            neural_textures = visualize_neural_textures(method, phase.iter_nr)
            wandb_imgs_np["neural_textures"] = neural_textures

            # save images to disk
            save_dir = os.path.join(method.save_checkpoints_path, "wandb_imgs")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for collection_key, collection in wandb_imgs_np.items():
                for img_key, img_np in collection.items():
                    img_filename = (
                        f"{collection_key}_{img_key}_{format(phase.iter_nr, '07d')}"
                    )
                    save_numpy_as_png(img_np, save_dir, img_filename)

        # QUANTITATIVE EVALUATION --------------------------------------------

        eval_splits = []

        # check if should run on train set
        if (
            phase.iter_nr != 0
            and train_params.eval_train
            and (phase.iter_nr + 1 % train_params.eval_test_freq == 0)
            or (is_last_iter and eval_train)
        ):
            eval_splits.append("train")

        # check if should run on test set
        if (
            phase.iter_nr != 0
            and train_params.eval_test
            and (phase.iter_nr + 1 % train_params.eval_test_freq == 0)
            or (is_last_iter and eval_test)
        ):
            eval_splits.append("test")

        eval_res = render_and_eval(method, mv_data, eval_splits, iter_nr=phase.iter_nr)

        occupancy_stats = {}
        callbacks.iter_ended(
            phase=phase,
            test_losses=test_losses,
            imgs=wandb_imgs_np,
            eval_metrics=eval_res,
            occupancy_grid_stats=occupancy_stats,
        )

        phase.iter_nr += 1
        phase.is_first_iter = False

    callbacks.training_ended()

    print(
        "\n[bold green]SUCCESS[/bold green]: training ended after iter ", phase.iter_nr
    )
    return phase.iter_nr  # num completed iterations


def main(device="cuda"):
    print("\n[bold blue]TRAINER[/bold blue]")

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
        "--continue_training",
        action="store_true",
        help="Continue training from last checkpoint (if not, will start a new run)",
    )
    parser.add_argument(
        "--keep_last_checkpoint_only",
        action="store_true",
        help="Keep only the last checkpoint (if not, will keep all checkpoints)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model",
    )
    parser.add_argument(
        "--train_iters",
        help="Number of training iterations (overriding hyper-params value)",
        type=int,
    )
    parser.add_argument(
        "--lr",
        help="Learning rate (overriding hyper-params value)",
        type=float,
    )
    parser.add_argument(
        "--eval_test",
        action="store_true",
        help="Renders and eval all test views",
    )
    parser.add_argument(
        "--eval_train",
        action="store_true",
        help="Render and eval all train views",
    )
    parser.add_argument(
        "--models_path",
        help="Path to folder containing surf model checkpoint",
    )
    parser.add_argument("--meshes_path", help="Path to folder containing mesh files")
    # parser.add_argument(
    #     "--bg_color",
    #     help="Constant background color (white, black, random)",
    #     type=str,
    # )
    parser.add_argument(
        "--subsample_factor",
        type=int,
        default=-1,
        help="subsample factor for the dataset resolution",
    )
    parser.add_argument(
        "--teacher_run_id",
        help="Run ID of teacher method run to load (will load last checkpoint)",
        type=str,
    )
    parser.add_argument(
        "--teacher_exp_config", help="Teacher experiment config name", type=str
    )
    args = CmdParams(parser.parse_args().__dict__)

    # paths

    paths = PathsParams(args)

    # init run
    run_id, start_iter_nr = init_run(args, paths, is_run_id_required=False)

    # read the hyper_params config file
    hyper_params = get_method_hyper_params(args["method_name"], paths["hyper_params"])

    # override hyper_params
    if args["train_iters"] is not None:
        hyper_params.training_end_iter = args["train_iters"]
    if args["lr"] is not None:
        hyper_params.lr = args["lr"]
        hyper_params.lr_milestones = []

    # initialize the parameters used for training
    train_params = TrainParams(args["method_name"], paths["train_config"])

    # initialize the parameters used for data loading
    data_params = DataParams(
        paths["datasets"], args["dataset"], args["scene"], paths["data_config"]
    )
    if args["subsample_factor"] > 0:
        data_params.subsample_factor = args["subsample_factor"]

    # create profiler
    profiler = Profiler(verbose=False)

    # PRINTING --------------------------------------------------------

    print_params(
        args=args,
        train_params=train_params,
        data_params=data_params,
        hyper_params=hyper_params,
        paths=paths,
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

    # create tensor reel
    train_data_tensor_reel = TensorReel(mv_data["train"], device=device)

    # bounding primitive

    bounding_primitive = init_bounding_primitive(mv_data, device)

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
        train=args["train"],
        hyper_params=hyper_params,
        load_checkpoints_path=paths["load_checkpoints"],
        save_checkpoints_path=paths["save_checkpoints"],
        bounding_primitive=bounding_primitive,
        model_colorcal=model_colorcal,
        bg_color=bg_color,
        start_iter_nr=start_iter_nr,
        profiler=profiler,
        meshes_path=args["meshes_path"],
        models_path=args["models_path"],
        init_sphere_radius=mv_data.init_sphere_radius,
    )

    # TEACHER MODEL (optional) --------------------------------------------

    if paths["teacher_checkpoints"] is not None:
        last_checkpoint = get_last_checkpoint_in_path(paths["teacher_checkpoints"])
        if last_checkpoint is None:
            print("[bold red]ERROR[/bold red]: no teacher checkpoints found")
            exit(1)
        teacher_start_iter_nr = int(last_checkpoint)

        # read the hyper_params config file
        teacher_hyper_params = get_method_hyper_params(
            "nerf", paths["teacher_hyper_params"]
        )

        # teacher method
        from volsurfs_py.methods.nerf import NeRF

        teacher_method = NeRF(
            train=False,
            hyper_params=teacher_hyper_params,
            load_checkpoints_path=paths["teacher_checkpoints"],
            save_checkpoints_path=None,
            bounding_primitive=bounding_primitive,
            model_colorcal=model_colorcal,
            bg_color=bg_color,
            start_iter_nr=teacher_start_iter_nr,
            profiler=profiler,
        )
        print(f"teacher model loaded from iteration {teacher_start_iter_nr}")

    else:
        teacher_method = None

    # TRAINING --------------------------------------------------------

    if args["train"]:

        if start_iter_nr > 0:
            print("\nresuming training from iteration", start_iter_nr)
        else:
            print("\nstarting training from scratch")

        last_checkpoint_nr = train(
            run_id,
            train_params,
            data_params,
            args["exp_name"],
            train_data_tensor_reel,
            mv_data,
            method,
            teacher_method=teacher_method,
            start_iter_nr=start_iter_nr,
            profiler=profiler,
            bg_color=args["bg_color"],
            iter_finish_nr=hyper_params.training_end_iter,
            eval_test=args["eval_test"],
            eval_train=args["eval_train"],
            keep_last_checkpoint_only=args["keep_last_checkpoint_only"],
        )
        print("last iteration", last_checkpoint_nr)
    else:
        last_checkpoint_nr = start_iter_nr

    # EVALUATION --------------------------------------------------------

    if not args["train"]:

        # set eval mode
        method.set_eval_mode()

        eval_splits = []
        if args["eval_train"]:
            eval_splits.append("train")
        if args["eval_test"]:
            eval_splits.append("test")

        render_and_eval(method, mv_data, eval_splits, iter_nr=last_checkpoint_nr)

    # PROFILER --------------------------------------------------------

    # print profiler stats
    profiler.print_avg_times()

    # SPIRAL VIEW DIRS RENDERS ---------------------------------------------

    # TODO: re-implement
    # render_view_dirs_trajectory(camera, method, renders_path)

    # finished
    return run_id


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

    run_id = main(device=device)

    # return run_id to terminal
    print(run_id)
