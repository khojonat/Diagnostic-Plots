import os

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_2d_hist(
    x: np.ndarray,
    y: np.ndarray,
    nbins: int = 512,
    output_path: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str | None = None,
    cmap: str = "viridis",
) -> str:
    """
    Produce a 2D histogram (density map) for arbitrary x/y data.

    Parameters
    ----------
    x, y : array_like
        Arrays of the same length containing the coordinates to histogram.
    nbins : int, optional
        Number of bins along each axis (default: 512).
    output_path : str, optional
        Full path to save the PNG. If None, saves to 'Plots/density_2d.png'.
    xlabel, ylabel : str, optional
        Axis labels for the plot.
    title : str, optional
        Plot title.
    cmap : str, optional
        Matplotlib colormap name.

    Returns
    -------
    str
        Path to the saved image file.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape for a 2D histogram.")

    H, xedges, yedges = np.histogram2d(x, y, bins=nbins)

    fig, ax = plt.subplots(figsize=(6, 6))

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Counts")

    fig.tight_layout()

    if output_path is None:
        os.makedirs("Plots", exist_ok=True)
        output_path = os.path.join("Plots", "density_2d.png")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported and used via plot_2d_hist()."
    )

