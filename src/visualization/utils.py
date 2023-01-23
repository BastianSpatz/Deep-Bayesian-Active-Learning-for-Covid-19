import random
import math
import matplotlib.pyplot as plt


def plot_from_volume(vol, indices=None, n_samples=None, n_cols=3):
    if indices is None:
        if n_samples is None:
            raise ValueError("Please specifiy indices or the number of random samples to plot.")
        else:
            indices = random.sample(range(len(vol)), n_samples)
    n_rows = int(math.ceil(len(indices)/n_cols))
    fig = plt.figure()
    fig.suptitle('Images')

    ax = []
    for i, idx in enumerate(indices):
        image = vol[idx]
        ax.append(fig.add_subplot(n_rows, n_cols, i))
        ax[-1].set_title("slice: {}".format(idx))
        plt.imshow(image, interpolation='nearest')
    plt.show()
