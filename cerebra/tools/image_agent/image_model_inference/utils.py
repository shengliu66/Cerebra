import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.filters import threshold_otsu

black_yellow_cmap = LinearSegmentedColormap.from_list("black_yellow", ["black", "yellow"])

def visualize_saliency(
    saliency: torch.Tensor,
    save_path: str,
    image: torch.Tensor = None,
    num_samples: int = 9,
    overlay_alpha: float = 0.5,
    cmap=black_yellow_cmap,
    max_cols: int = 3,
    figscale: float = 2.2,
    title: str = "Saliency Map",
    smooth_sigma: float = 0.5,
):
    def _to_np3d(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().float().numpy()
        x = np.squeeze(np.asarray(x))
        assert x.ndim == 3, f"Expected 3D array, got {x.shape}"
        return x

    S = _to_np3d(saliency)

    if smooth_sigma > 0:
        S = ndimage.gaussian_filter(S, sigma=smooth_sigma)
    
    
    # Adaptive brain masking if image is provided
    if image is not None:
        I = _to_np3d(image)
        assert I.shape == S.shape, f"Image shape {I.shape} must match saliency {S.shape}"
        
        # Calculate adaptive threshold using Otsu's method
        adaptive_threshold = threshold_otsu(I)
        brain_mask = 1#(I > adaptive_threshold)
        S = S * brain_mask
        
        # Renormalize within brain
        brain_values = S[brain_mask]
        if len(brain_values) > 0 and brain_values.max() > brain_values.min():
            smin, smax = brain_values.min(), brain_values.max()
            S = (S - smin) / (smax - smin + 1e-6)
            S = S * brain_mask  # Keep only brain regions
        
        # Normalize image for display
        imin, imax = I.min(), I.max()
        I = (I - imin) / (imax - imin + 1e-6)
    else:
        # Original normalization
        smin, smax = S.min(), S.max()
        S = (S - smin) / (smax - smin + 1e-6)
        I = None
    
    D, H, W = S.shape
    # idxs = np.linspace(0, W - 1, num_samples, dtype=int)
    idxs = np.linspace(20, 60, num_samples, dtype=int)
    idxs = np.unique(idxs[(idxs >= 0) & (idxs < W)])
    n = len(idxs)

    cols = min(max_cols, n)
    
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * figscale, rows * figscale),
        constrained_layout=True
    )
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for k, j in enumerate(idxs):
        ax = axes[k // cols, k % cols]
        if I is not None:
            ax.imshow(I[:, :, j], cmap="gray")
            ax.imshow(S[:, :, j], cmap=cmap, alpha=overlay_alpha)
        else:
            ax.imshow(S[:, :, j], cmap=cmap)
        ax.set_title(f"Slice={j}", fontsize=10)
        ax.axis("off")

    # Hide unused axes
    for t in range(n, rows * cols):
        axes[t // cols, t % cols].axis("off")

    fig.suptitle(title, fontsize=12, y=1.02)

    plt.savefig(save_path)
    plt.close()


def make_notice_png(save_path="saliency_unavailable.png",
                    text="Saliency map for this image not available"):
    plt.figure(figsize=(8, 2))
    plt.text(0.5, 0.5, text,
             ha="center", va="center",
             fontsize=16, color="white")
    plt.axis("off")
    plt.gca().set_facecolor("black")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.5, dpi=150)
    plt.close()
    return save_path
