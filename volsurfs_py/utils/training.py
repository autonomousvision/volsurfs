import os
import torch
import numpy as np
from copy import deepcopy
from rich import print

from volsurfs_py.utils.losses import loss_l1
from mvdatasets.utils.raycasting import get_random_camera_rays_and_frames


@torch.no_grad()
def estimate_test_loss(method, mv_data, chunk_size, iter_nr=None):
    """
    estimates the test loss by running evaluation on the test set a batch of rays from test set
    """

    # compute test loss
    test_losses = {}
    test_loss = 0.0

    # get random camera from test_data cameras
    cam_idx = np.random.randint(0, len(mv_data["test"]))
    camera = deepcopy(mv_data["test"][cam_idx])

    # get test rays and gt values
    (
        rays_o,
        rays_d,
        gt_rgb,
        gt_mask,
        pixels_2d,
    ) = get_random_camera_rays_and_frames(
        camera, nr_rays=chunk_size, jitter_pixels=False
    )
    rays_o = rays_o.cuda()
    rays_d = rays_d.cuda()
    assert gt_rgb is not None
    gt_rgb = gt_rgb.cuda()
    if gt_mask is not None:
        gt_mask = gt_mask.cuda()

    # render rays and compute rgb loss
    res = method.render_rays(rays_o=rays_o, rays_d=rays_d, iter_nr=iter_nr)
    renders = res["renders"]
    for render_mode, renders_dict in renders.items():
        if "rgb" in renders_dict:
            pred_rgb = renders_dict["rgb"]
            if method.hyper_params.is_testing_masked:
                assert gt_mask is not None
                loss_rgb = loss_l1(gt_rgb, pred_rgb, mask=gt_mask)
            else:
                loss_rgb = loss_l1(gt_rgb, pred_rgb)
            test_losses[f"{render_mode}_rgb"] = loss_rgb.item()
            test_loss += loss_rgb.item()

    return test_loss, test_losses


def save_checkpoints(method, iter_nr, remove_previous=True):
    """
    saves checkpoints
    """

    # if remove_previous, remove previous checkpoints
    if remove_previous:
        # remove previous checkpoints from method save_checkpoints path
        last_checkpoint = get_last_checkpoint_in_path(method.save_checkpoints_path)
        if last_checkpoint is None:
            print("[bold yellow]WARNING[/bold yellow]: no previous checkpoints found")
        # remove folder
        os.system(f"rm -rf {method.save_checkpoints_path}/{last_checkpoint}")
        print(
            "[bold blue]INFO[/bold blue]: removed previous checkpoint: ",
            last_checkpoint,
        )

    # save models
    method.save(iter_nr)


def get_last_checkpoint_in_path(checkpoints_path):

    # look for latest checkpoint in the folder
    list_of_checkpoints = os.listdir(checkpoints_path)
    if "config" in list_of_checkpoints:
        list_of_checkpoints.remove("config")
    if "wandb_imgs" in list_of_checkpoints:
        list_of_checkpoints.remove("wandb_imgs")
    if "meshes" in list_of_checkpoints:
        list_of_checkpoints.remove("meshes")
    if "results" in list_of_checkpoints:
        list_of_checkpoints.remove("results")
    print("checkpoints found: ", list_of_checkpoints)
    if len(list_of_checkpoints) == 0:
        print(
            "[bold yellow]WARNING[/bold yellow] no checkpoints found in ",
            checkpoints_path,
        )
        return None

    # to sort the list, convert it to int
    last_checkpoint = sorted(list_of_checkpoints, reverse=True)[0]
    print("latest checkpoint: ", last_checkpoint)

    return last_checkpoint


def get_opt_params_gradients_norms(method):
    models_grad_norms = []
    for model_params in method.opt_params:
        parameters = []
        for p in model_params["params"]:
            if p.requires_grad and p.grad is not None:
                parameters.append(p)
        params_norms = []
        for p in parameters:
            params_norms.append(torch.norm(p.grad.detach()))
        if len(params_norms) > 0:
            grad_norm = torch.norm(torch.stack(params_norms)).item()
        else:
            grad_norm = 0.0
        model_grad_norm = {"name": model_params["name"], "grad_norm": grad_norm}
        models_grad_norms.append(model_grad_norm)

    return models_grad_norms
