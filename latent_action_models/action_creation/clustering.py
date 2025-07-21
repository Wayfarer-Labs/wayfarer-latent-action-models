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
import  os
import  umap
import  torch
import  numpy as np
from    torch import Tensor
from    pathlib import Path
import  matplotlib.pyplot as plt
from    kmeans_pytorch import kmeans

def umap_visualization(latent_actions_mu_n1d:   Tensor,
                       visualization_path:      os.PathLike) -> None:
    latent_np: np.ndarray = latent_actions_mu_n1d.detach().cpu().numpy()

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.10,
        metric="euclidean",
        verbose=False,
    )
    embedding_2d = reducer.fit_transform(latent_np)   # shape (N, 2)

    # -- Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        s=5, alpha=0.6, linewidths=0)

    ax.set(title="UMAP Projection of Latent Actions", xlabel="UMAP-1", ylabel="UMAP-2")
    fig.tight_layout()

    viz_path = Path(visualization_path)
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(viz_path, dpi=300)
    plt.close(fig)

    print(f"[AdaWorld] UMAP plot saved to: {viz_path.resolve()}")


def kmeans_cluster_latent_actions(latent_actions_mu_n1d:    Tensor,
                                  num_clusters:             int,
                                  save_dir:                 Path | None = None,
                                  filename:                 str         = 'latent_action_centers.pt') -> tuple[Tensor, Tensor]:
    
    cluster_ids, cluster_centers = kmeans(  X            = latent_actions_mu_n1d,
                                            num_clusters = num_clusters,
                                            distance     = 'euclidean' )
    if save_dir:
        torch.save  (cluster_centers, save_dir / filename)
        print       (f'Saved clusters with {num_clusters=} to {save_dir / filename}')
    
    return cluster_ids, cluster_centers
