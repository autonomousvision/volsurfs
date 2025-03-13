import os
import torch
from PIL import Image
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from rich import print
import matplotlib.pyplot as plt
from volsurfs_py.utils.common import lin2hwc
from mvdatasets.utils.images import save_numpy_as_png

from volsurfs_py.utils.postprocessing import postprocess_renders


def save_renders(camera, imgs_np, out_img_path, override_filename=None):
    """save images to disk

    Args:
        imgs (dict): dict of render modes containing dicts of images
    """

    # save images
    for render_mode, imgs_dict in imgs_np.items():
        for img_key, img_np in imgs_dict.items():
            if override_filename is not None:
                img_filename = override_filename
            else:
                img_filename = format(camera.camera_idx, "03d")
            save_dir_path = os.path.join(out_img_path, render_mode, img_key)
            save_numpy_as_png(img_np, save_dir_path, img_filename)
            img_path = os.path.join(save_dir_path, img_filename + ".png")
            print(f"{render_mode}/{img_key}: saved to {img_path}")


@torch.no_grad()
def render_from_camera(
    camera,
    method,
    image_resize_target=None,
    iter_nr=None,
    use_matplotlib_plots=True,
    debug_pixel=None,
    postprocess=True,
    **kwargs,
):
    """renders the scene from the given camera"""

    camera_ = deepcopy(camera)

    if image_resize_target is not None:
        max_res = max(camera_.width, camera_.height)
        subsample_factor = max_res // image_resize_target
        if subsample_factor < 1:
            subsample_factor = 1
        camera_.resize(subsample_factor=subsample_factor)

    # render method
    renders_np = method.render(
        camera=camera_,
        iter_nr=iter_nr,
        debug_pixel=debug_pixel,
        verbose=False,
        **kwargs,
    )

    # get max nr samples per ray
    max_nr_samples_per_ray = method.hyper_params.max_nr_samples_per_ray
    if method.hyper_params.do_importance_sampling:
        max_nr_samples_per_ray += method.hyper_params.max_nr_imp_samples_per_ray

    if postprocess:
        # postprocess and convert to numpy
        processed_renders_np = postprocess_renders(
            renders_np,
            camera_,
            max_nr_samples_per_ray=max_nr_samples_per_ray,
            bg_color=method.bg_color,
            use_matplotlib_plots=use_matplotlib_plots,
            verbose=True,
        )
        return processed_renders_np
    else:
        return renders_np


def render_cameras(cameras, method, iter_nr=None, render_modes=[], **kwargs):
    """render all cameras and save images to disk"""

    print(f"rendering {len(cameras)} views")
    print(f"iter_nr for c2f: {iter_nr}")

    renders_all = {}
    # iterate over the cameras
    pbar = tqdm(
        range(len(cameras)),
        desc="rendering views",
        unit="img",
        # ncols=100,
        leave=False,
    )
    for i in pbar:
        # get camera
        camera = cameras[i]
        # update progress bar
        pbar.set_postfix({"camera": camera.camera_idx})
        renders_np = render_from_camera(
            camera,
            method,
            iter_nr=iter_nr,
            use_matplotlib_plots=False,
            debug_pixel=None,
            postprocess=False,
            **kwargs,
        )

        # render modes selection
        selected_renders_np = {}
        if len(render_modes) == 0:
            # render modes non specified, store all
            selected_renders_np = renders_np
        else:
            # only store specified render modes
            for render_mode in render_modes:
                if render_mode[0] not in renders_np:  # e.g. "volumetric"
                    print(
                        f"[bold yellow]WARNING[/bold yellow]: no {render_mode[0]} renders found"
                    )
                else:
                    selected_renders_np[render_mode[0]] = {}
                    for render_buffer in render_mode[1]:
                        if render_buffer in renders_np[render_mode[0]]:
                            # reshape to HWC
                            img_np = renders_np[render_mode[0]][render_buffer].reshape(
                                camera.height, camera.width, -1
                            )
                            selected_renders_np[render_mode[0]][render_buffer] = img_np
                        else:
                            # print warning
                            print(
                                f"[bold yellow]WARNING[/bold yellow]: no {render_buffer} render found"
                            )

        # get camera pose
        camera_pose = camera.get_pose()
        # get camera intrinsics
        camera_intrinsics = camera.get_intrinsics()
        # get camera idx
        camera_idx = str(camera.camera_idx)
        # store renders
        renders_all[camera_idx] = {
            "renders": selected_renders_np,
            "pose": camera_pose,
            "intrinsics": camera_intrinsics,
        }

    return renders_all


def render_cameras_and_save_from_cameras(
    cameras, method, renders_path, iter_nr=None, use_matplotlib_plots=True, **kwargs
):
    """render all cameras and save images to disk"""

    print(f"rendering {len(cameras)} views")
    print(f"saving renders to {renders_path}")
    print(f"iter_nr for c2f: {iter_nr}")

    # iterate over the cameras
    pbar = tqdm(
        range(len(cameras)),
        desc="rendering views",
        unit="img",
        # ncols=100,
        leave=False,
    )
    for i in pbar:
        # get camera
        camera = cameras[i]
        # update progress bar
        pbar.set_postfix({"camera": camera.camera_idx})

        # if camera.camera_idx != 100:
        #     continue

        # debug pixel is the center of the image
        debug_pixel = None
        # debug_pixel = (camera.height // 2, camera.width // 2)
        # debug_pixel = (951, 944)

        imgs_np = render_from_camera(
            camera,
            method,
            iter_nr=iter_nr,
            use_matplotlib_plots=use_matplotlib_plots,
            debug_pixel=debug_pixel,
            postprocess=True,
            **kwargs,
        )

        if renders_path is not None:
            save_renders(camera, imgs_np, renders_path)

        # TODO: remove
        # exit(0)


# TODO: needs to be updated
# def render_view_dirs_trajectory(camera, method, renders_path):
#     """render a fixed view with view dirs following a defined trajectory"""
#     # create spiral trajectory
#     # TODO
#     view_dirs = ...
#     # iterate over the view dirs
#     pbar = tqdm(view_dirs, desc="Rendering spiral", unit="img")
#     for i, view_dir in enumerate(pbar):
#         # output path
#         out_img_path = os.path.join(renders_path, "view_dirs", str(camera_idx))
#         render_and_save(
#             camera,
#             method,
#             out_img_path,
#             override_filename=str(i),
#             with_mask=method.hyper_params.with_mask,
#             bg_color=method.bg_color,
#         )
#     return
