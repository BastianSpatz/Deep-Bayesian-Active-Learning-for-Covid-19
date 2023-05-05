import math
import random

# import k3d
import matplotlib.pyplot as plt
import numpy as np
import torch
from pylab import rcParams

# rcParams["figure.figsize"] = 10, 10


def plot_slices(n_rows, n_cols, n_samples, volume):
    """Plot a montage of slices"""
    heights = [512] * n_rows
    widths = [512] * n_cols

    fig_width = 8.0  # inches
    fig_height = fig_width * sum(heights) / sum(widths)
    fig, ax = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    len_sclices = volume.shape[0]
    start_slice = (len_sclices - n_samples) // 2
    for i in range(n_rows):
        for j in range(n_cols):
            ax[i, j].imshow(volume[start_slice + (i * n_rows) + j])
            ax[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


def plot_volume(volumes):
    if not isinstance(volumes, list):
        volumes = [volumes]
    for volume in volumes:
        if isinstance(volume, torch.Tensor):
            # print(volume.shape)
            volume = volume.transpose(0, -1)
            # volume = volume.permute(0, 2, 3, 1)

            volume_arr = volume.cpu().detach().numpy()
        else:
            volume_arr = volume
        plot = k3d.plot()
        plt_volume = k3d.volume(
            volume_arr[::-1, :, :].astype(np.float32), alpha_coef=15
        )
        plot += plt_volume
        plot.display()
