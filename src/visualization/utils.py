import math
import random

import k3d
import matplotlib.pyplot as plt
import numpy as np
import torch
from pylab import rcParams
from torchvision import transforms

rcParams["figure.figsize"] = 10, 10


def plot_tensor_image(img_tensor):
    # if img_tensor.shape[0] != 3:
    #     img_tensor = img_tensor.permute(1, 2, 0)
    plt.imshow(img_tensor)


def plot_from_volume_tensor(
    vol, label, indices=None, n_samples=1, n_cols=3, title=None
):

    if indices is None:
        n_rows = int(math.ceil(n_samples / n_cols))
    else:
        n_rows = int(math.ceil(len(indices) / n_cols))

    fig = plt.figure()
    fig.suptitle(title or "Images")

    indices = random.sample(range(vol.shape[0]), n_samples)
    ax = []
    for i, idx in enumerate(sorted(indices), 1):
        image = vol[idx]
        ax.append(fig.add_subplot(n_rows, n_cols, i))
        ax[-1].set_title("slice: {} with label: {}".format(idx, label))
        ax[-1].axis("off")
        plot_tensor_image(image)
    plt.tight_layout()
    plt.show()


def plot_volume(volume):
    if isinstance(volume, torch.Tensor):
        # print(volume.shape)
        volume = volume.transpose(0, -1)
        # volume = volume.permute(0, 2, 3, 1)

        volume_arr = volume.cpu().detach().numpy()
    else:
        volume_arr = volume
    plt_volume = k3d.volume(volume_arr[::-1, :, :].astype(np.float32), alpha_coef=20)
    plot = k3d.plot()
    plot += plt_volume
    plot.display()
