import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import patches

def is_torch(tensor):
    return isinstance(tensor, torch.Tensor)

def plot_img(img, block=True, mask_weight=0.7, title="", save=None, norm_heatmap=False,
             bounds_to_labels=None, figsize=None, cmap=None, return_fig=False):
    """
    Plot image
    Args:
        img: numpy image
        block: Default True to plot in pycharm, set to False for notebooks
        mask_weight:
        title:
        save: if not None should be path to save figure to
        norm_heatmap: If True, normalizes the heatmap to the range 0-1
        bounds_to_labels: If given - splits to discrete ranges based on the bounds and plots legend
            Should be a dictionary of bounds (segment start values) to labels (text strings for legend)
        cmap: specific matplotlib color map
    """
    if len(img.shape) < 2:
        raise Exception("image shape is invalid: should be 2d or 3d image")

    if is_torch(img):
        img = np.transpose(img.data.numpy(), [1, 2, 0])

    img = np.squeeze(img)

    if len(img.shape) > 2:
        if img.shape[2] < 3:
            raise Exception("image shape is invalid, color channels should be of the size of 3")

    if img.dtype == np.bool:
        img = img.astype(np.int)
    if img.dtype == np.float16:
        img = img.astype(np.float32)

    if bounds_to_labels is not None:
        num_colors = len(bounds_to_labels.keys())
        cmap = 'tab20' if cmap is None else cmap
        cm = colors.ListedColormap(plt.get_cmap(cmap, num_colors).colors)
        bounds = list(bounds_to_labels.values())
        bounds = [b-0.5 for b in bounds] + [bounds[-1]+0.5]
        norm = colors.BoundaryNorm(bounds, num_colors)
        img_norm = img
    else:
        if img.max() > img.min():
            img_norm = (img - img.min()) / (img.max() - img.min())
        else:
            img_norm = np.ones(img.shape) * img.max()
        cm, norm = cmap, None

    fig = plt.figure(figsize=figsize)
    plt.title(title)
    im = plt.imshow(img_norm, cmap=cm, norm=norm)
    if bounds_to_labels is not None:
        cbar = fig.colorbar(im)
        cbar.ax.get_yaxis().set_ticks([])
        for j, label in enumerate(bounds_to_labels.keys()):
            cbar.ax.text(.5, (2 * j + 1) / (2*num_colors), label, va='center')

    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    plt.show(block=block)

    if return_fig:
        return fig