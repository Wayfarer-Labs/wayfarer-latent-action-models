"""
As mentioned in Sec. 2.3, AdaWorld can also easily create a flexible number of control options through latent action
clustering.

Specifically, we process our Procgen and Gym Retro training set using the latent action encoder to obtain the
corresponding latent actions.

To generate different control options, we apply K-means clustering to all latent actions, setting
the number of clustering centers to the desired number of control options.

To examine the controllability of varying actions derived with AdaWorld, we adopt the 
∆PSNR metric following Genie (Bruce et al., 2024).
Table 7 shows the ∆PSNR of the latent action decoder predictions. The larger the ∆PSNR,
the more the predictions are affected by the action conditions and therefore the world model is more controllable.

The results in Table 7 demonstrate that the control options derived with AdaWorld represent distinct meanings
and exhibit comparable controllability to the discrete counterpart, while the latter
does not support a customizable number of actions, as it is fixed once trained.
"""
import  umap
import  torch

import  numpy               as np
from    torch               import Tensor
from    pathlib             import Path
import  matplotlib.pyplot   as plt
from    kmeans_pytorch      import kmeans
from    latent_action_models.utils import _sample_colors


def umap_visualization(
    latent_actions_mu_n1d: Tensor,
    *,
    colors: list[tuple[int, int, int]] | None = None,
    legend: dict[str, tuple[int, int, int]] | None = None, # MODIFIED: Add legend argument
    vis_filename: str = 'umap_visualization.png',
    vis_dir: Path = Path('visualizations/'),
) -> tuple[np.ndarray, plt.Figure]:
    
    latent_np: np.ndarray = latent_actions_mu_n1d.squeeze(1).detach().cpu().numpy()
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.10, metric="euclidean", verbose=False)
    embedding_n2: np.ndarray = reducer.fit_transform(latent_np)

    fig, ax = plt.subplots(figsize=(12, 10)) # Made figure larger for legend

    if colors and legend:
        colors_arr = np.array(colors)
        for label, color_rgb in legend.items():
            # Find all points that have this color
            indices = np.where((colors_arr == color_rgb).all(axis=1))[0]
            if len(indices) > 0:
                ax.scatter(
                    embedding_n2[indices, 0],
                    embedding_n2[indices, 1],
                    s=5,
                    alpha=0.7,
                    linewidths=0,
                    color=np.array(color_rgb) / 255.0,
                    label=label
                )
        ax.legend(title="Action Categories", markerscale=4, fontsize=9)
    else:
        # Fallback to a simple plot if no legend is provided
        ax.scatter(embedding_n2[:, 0], embedding_n2[:, 1], s=5, alpha=0.6)

    ax.set(title="UMAP Projection of Latent Actions", xlabel="UMAP-1", ylabel="UMAP-2")
    fig.tight_layout()

    vis_dir.mkdir(parents=True, exist_ok=True)
    viz_path = vis_dir / vis_filename
    fig.savefig(viz_path, dpi=300)
    plt.close(fig)

    print(f"[AdaWorld] UMAP plot saved to: {viz_path.resolve()}")
    return embedding_n2, fig


def kmeans_cluster_latent_actions(latent_actions_mu_n1d:    Tensor,
                                  num_clusters:             int,
                                  save_dir:                 Path | None = None,
                                  filename:                 str         = 'latent_action_centers.pt') -> tuple[Tensor, Tensor]:
    
    if isinstance(save_dir, str):   save_dir = Path(save_dir)
    if save_dir:                    save_dir.mkdir(parents=True, exist_ok=True)

    cluster_ids, cluster_centers = kmeans(  X            = latent_actions_mu_n1d,
                                            num_clusters = num_clusters,
                                            distance     = 'euclidean' )
    if save_dir:
        torch.save  (cluster_centers, save_dir / filename)
        print       (f'Saved clusters with {num_clusters=} to {save_dir / filename}')
    
    return cluster_ids, cluster_centers


if __name__ == "__main__":
    import torch
    import numpy as np

    # 1) Generate (10_000, 1, 10) random gaussians
    latent_actions_mu_n1d = torch.randn(10_000, 1, 10)

    # 2) For each of the 10_000, add a random integer in the integer range [-1, 3]
    random_ints              = torch.randint(-1, 2, (10_000, 1, 1), dtype=latent_actions_mu_n1d.dtype)
    latent_actions_mu_n1d   += (random_ints * 3)

    # 3) For each of the 10_000, add a random float in the range [-1, 1]
    random_floats           = (torch.rand(10_000, 1, 1) * 2 - 1).to(latent_actions_mu_n1d.dtype)
    latent_actions_mu_n1d  += (random_floats * 1.2)

    cluster_ids, cluster_centers = kmeans_cluster_latent_actions(latent_actions_mu_n1d, 3, Path("clusters/"))
    umap_visualization(latent_actions_mu_n1d, cluster_ids_n=cluster_ids)
