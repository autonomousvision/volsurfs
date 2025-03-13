from rich import print
import numpy as np
import matplotlib.pyplot as plt
from volsurfs_py.utils.common import lin2hwsc, lin2hwc
from mpl_toolkits.axes_grid1 import make_axes_locatable


#
def postprocess_with_matplotlib(img_np, cmap="jet", vmin=None, vmax=None, dpi=72):
    # Testing matplotlib plots

    fig = plt.figure(
        figsize=(img_np.shape[1] / dpi, img_np.shape[0] / dpi),
        facecolor="white",
        dpi=dpi,
    )
    ax = fig.add_subplot(111)
    # define colormap range
    if vmin is None and vmax is None:
        img = ax.imshow(img_np, cmap=cmap)
    if vmin is not None and vmax is not None:
        img = ax.imshow(img_np, cmap=cmap, vmin=vmin, vmax=vmax)
    elif vmin is not None:
        img = ax.imshow(img_np, cmap=cmap, vmin=vmin)
    elif vmax is not None:
        img = ax.imshow(img_np, cmap=cmap, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)
    fig.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    w, h = fig.canvas.get_width_height()
    plot_np = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    plot_np = plot_np / 255.0
    plt.close(fig)
    return plot_np


#
def postprocess_renders(
    renders,
    camera,
    max_nr_samples_per_ray=None,
    bg_color=None,
    use_matplotlib_plots=True,
    verbose=True,
):
    """reshape predictions as images and postprocess"""

    height = camera.height
    width = camera.width

    mask_gt = camera.get_mask()  # uint8
    if mask_gt is not None:
        mask_gt = mask_gt / 255.0  # float32

    processed_renders = {}
    # for each render mode
    for render_mode, renders_dict in renders.items():

        if verbose:
            print(f"\npostprocessing {render_mode}")

        processed_renders[render_mode] = {}

        # for each renderered key
        mode_res = {}
        for render_key, render_lin in renders_dict.items():

            # render_lin can have shape:
            # - nr_pixels, nr_surfs, nr_channels
            # - nr_pixels, nr_channels
            # check which format it has
            nr_surfs = render_lin.shape[1] if np.ndim(render_lin) == 3 else 1

            if verbose:
                print(f"    current buffer {render_key}", render_lin.shape)

            # and reshape to h, w,, nr_surfs, nr_channels
            imgs_np = lin2hwsc(render_lin, height, width, nr_surfs)
            # print(f"    imgs_np.shape: {imgs_np.shape}")

            # iterate over all rendered surfaces and stack images on a row
            img_row_np = np.zeros([height, width * nr_surfs, 3])
            # print(f"    img_row_np.shape: {img_row_np.shape}")
            for i in range(nr_surfs):
                # get i-th surface render

                if verbose:
                    print(f"    processing surface {i}")

                img_np = imgs_np[:, :, i]
                print(f"{render_key}, shape: {img_np.shape}")

                # standard postprocessing
                if "normals" in render_key:
                    # postprocess normals
                    img_np = (img_np + 1.0) * 0.5

                elif "depth" in render_key or "interval" in render_key:
                    if use_matplotlib_plots:
                        # TODO: set vmax to max depth
                        img_np = postprocess_with_matplotlib(img_np, cmap="jet")
                    else:
                        # postprocess depth
                        if np.max(img_np) > 0.0:
                            # normalize depth
                            img_np = 1 - (img_np / np.max(img_np))

                elif "sum" in render_key:
                    if use_matplotlib_plots:
                        img_np = postprocess_with_matplotlib(
                            img_np, cmap="viridis", vmin=0.0
                        )
                    else:
                        # [0, 1] image
                        if np.max(img_np) > 0.0:
                            img_np = img_np / np.max(img_np)

                        # get the color map
                        color_map = plt.colormaps.get_cmap("viridis")
                        # apply the colormap
                        img_np = color_map(img_np[:, :, 0])[..., :3]

                elif "uv" in render_key:
                    # stack zeros for third channel
                    img_np = np.concatenate(
                        [img_np, np.zeros([img_np.shape[0], img_np.shape[1], 1])],
                        axis=-1,
                    )

                elif "nr_samples" in render_key:
                    if use_matplotlib_plots:
                        # TODO: set vmax to max nr of samples per ray
                        img_np = postprocess_with_matplotlib(
                            img_np, cmap="Purples", vmin=0, vmax=max_nr_samples_per_ray
                        )
                    else:
                        # postprocess nr_samples
                        if np.max(img_np) > 0.0:
                            # normalize nr_samples
                            img_np = img_np / np.max(img_np)

                        # get the color map
                        color_map = plt.colormaps.get_cmap("Purples")
                        # apply the colormap
                        img_np = color_map(img_np[:, :, 0])[..., :3]

                # repeat color channel if needed
                if img_np.shape[-1] == 1:
                    img_np = np.repeat(img_np, 3, axis=-1)

                print(f"    img_np.shape: {img_np.shape}")
                lb = i * width
                ub = (i + 1) * width
                img_row_np[:, lb:ub, :] = img_np

            mode_res[render_key] = img_row_np
        #
        processed_renders[render_mode] = mode_res

        # (only if rgb is available)
        if "rgb" in processed_renders[render_mode]:

            if not camera.has_rgbs():
                print(
                    "[bold red]ERROR[/bold red]: camera has no rgb image, this should not happen"
                )
                exit(1)

            gt_img = camera.get_rgb()  # uint8
            gt_img = gt_img / 255.0  # float32

            pred_img = processed_renders[render_mode]["rgb"]
            processed_renders[render_mode]["gt"] = gt_img

            if verbose:
                print("    computing error map")

            # compute error
            error_np = np.sum(np.abs(pred_img - gt_img), axis=-1)

            if use_matplotlib_plots:
                # plt image
                error_np = postprocess_with_matplotlib(
                    error_np, cmap="viridis", vmin=0.0
                )
            else:
                # [0, 1] image
                if np.max(error_np) > 0.0:
                    error_np = error_np / np.max(error_np)
            processed_renders[render_mode]["error"] = error_np

        # TODO: continue here
        # # mask if available
        # if camera.has_masks():

        #     gt_mask = camera.get_mask()  # uint8
        #     gt_mask = gt_mask / 255.0  # float32

        #     processed_renders[render_mode]["mask"] = gt_mask

        # # compute masked error if mask_gt is available
        # if mask_gt is not None:

        #     if bg_color is not None:
        #         # if a const bg color is specified, fill the background with that color
        #         bg_rgb = (
        #             bg_color.expand(camera.height, camera.width, 3).cpu().numpy()
        #         )
        #     else:
        #         # else fill the background with the gt rgb
        #         bg_rgb = np.zeros([camera.height, camera.width, 3])

        #     # mask gt
        #     gt_masked_img = gt_img * mask_gt
        #     gt_masked_img = gt_masked_img + (1 - mask_gt) * bg_rgb
        #     processed_renders[render_mode]["masked_gt"] = gt_masked_img

        #     # mask pred
        #     pred_masked_img = pred_img * mask_gt
        #     pred_masked_img = pred_masked_img + (1 - mask_gt) * bg_rgb
        #     processed_renders[render_mode]["masked_rgb"] = pred_masked_img

        #     if verbose:
        #         print("    computing masked error map")

        #     # compute error
        #     masked_error_np = np.sum(np.abs(pred_masked_img - gt_masked_img), axis=-1)

        #     if use_matplotlib_plots:
        #         # plt image
        #         masked_error_np = postprocess_with_matplotlib(masked_error_np, cmap="viridis", vmin=0.0)
        #     else:
        #         # [0, 1] image
        #         if np.max(masked_error_np) > 0.0:
        #             masked_error_np = masked_error_np / np.max(masked_error_np)
        #     processed_renders[render_mode]["masked_error"] = masked_error_np

        if verbose:
            print("done")

    return processed_renders
