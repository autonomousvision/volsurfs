import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _plt_fig_to_plot_np(fig):
    fig.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    w, h = fig.canvas.get_width_height()
    plot_np = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    plot_np = plot_np / 255.0
    plt.close(fig)
    return plot_np


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(
            0,
            1
            / 2
            * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))),
        )
        normalized_max = min(
            1,
            1
            / 2
            * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))),
        )
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [
            normalized_min,
            normalized_mid,
            normalized_max,
        ]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_2d_sdfs(sdfs, width, height):
    nr_sdfs = len(sdfs)
    figsize = (5 * nr_sdfs, 5)
    X, Y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    fig, axs = plt.subplots(1, nr_sdfs, figsize=figsize)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    for i, sdf in enumerate(sdfs):
        Z = sdf.reshape(width, height)
        contours = axs[i].contour(X, Y, Z, 3, colors="black")
        axs[i].clabel(contours, inline=True, fontsize=8)
        vmin = sdf.min()
        vmax = sdf.max()
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
        img = axs[i].imshow(
            Z, extent=[0, 1, 0, 1], origin="lower", cmap="bwr", alpha=0.8, norm=norm
        )
        fig.colorbar(img, ax=axs[i])
    plot_np = _plt_fig_to_plot_np(fig)
    return plot_np


def plot_2d_sdf(sdf, width, height):
    X, Y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    Z = sdf.reshape(width, height)
    contours = ax.contour(X, Y, Z, 3, colors="black")
    ax.clabel(contours, inline=True, fontsize=8)
    vmin = sdf.min()
    vmax = sdf.max()
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    img = ax.imshow(
        Z, extent=[0, 1, 0, 1], origin="lower", cmap="bwr", alpha=0.8, norm=norm
    )
    fig.colorbar(img, ax=ax)
    plot_np = _plt_fig_to_plot_np(fig)
    return plot_np


def plot_2d_sdfs_together(
    sdfs, width, height, main_sdf_idx=None, colors=["b", "black", "r"], dpi=72
):
    X, Y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    figsize = (width / dpi, height / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    for i, sdf in enumerate(sdfs):
        if isinstance(sdf, torch.Tensor):
            sdf = sdf.detach().cpu().numpy()
        Z = sdf.reshape(width, height)
        # TODO: make nr_levels dependent on main_sdf_idx
        # if main_sdf_idx is not None and i == main_sdf_idx:
        #     levels = [-0.005, 0.0, 0.005]
        # else:
        #     levels = [0.0]
        if len(sdfs) == 1:
            levels = 3
        else:
            levels = [0.0]
        contours = ax.contour(X, Y, Z, levels, colors=colors[i % len(colors)])
        ax.clabel(contours, inline=True, fontsize=12)
        # contours.collections[0].set_label("test")
    # plt.legend()

    plot_np = _plt_fig_to_plot_np(fig)
    return plot_np


def plot_2d_density(density, width, height, dpi=72):
    X, Y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    Z = density.cpu().numpy().reshape(width, height)
    figsize = (width / dpi, height / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(Z, extent=[0, 1, 0, 1], origin="lower")
    fig.colorbar(im, cax=cax, orientation="vertical")
    plot_np = _plt_fig_to_plot_np(fig)
    return plot_np


def plot_2d_occupancy(occupancy, width, height, dpi=72):
    Z = occupancy.cpu().numpy().reshape(width, height)
    figsize = (width / dpi, height / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    ax.imshow(Z, extent=[0, 1, 0, 1], origin="lower", cmap="gray")
    plot_np = _plt_fig_to_plot_np(fig)
    return plot_np
