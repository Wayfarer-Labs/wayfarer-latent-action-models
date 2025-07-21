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
import  random
import  umap
import  torch
from    itertools           import product as cartesian_product
from    toolz.itertoolz     import take, random_sample
from    typing              import Iterable
import  numpy               as np
from    torch               import Tensor
from    pathlib             import Path
import  matplotlib.pyplot   as plt
from    kmeans_pytorch      import kmeans


_PALETTE    = (10, 35, 65, 125, 165, 225)


def _sample_colors( n_colors: int = 8, *,
                    seed: int | None = None,
                    palette: tuple[int, ...] = _PALETTE) -> list[tuple[int, int, int]]:

    total       = len(palette)**3
    assert n_colors <= total, f'{n_colors} > {total} non-unique colors.'

    rng     = random.Random(seed)
    cube    = list(cartesian_product(palette, repeat=3))
    sample  = take(n_colors, random_sample(n_colors / total, cube, random_state=rng))
    picked  = list(sample)

    if len(picked) < n_colors:
        remainder = rng.sample(list(cube), n_colors - len(picked))
        picked.extend(remainder)

    rng.shuffle(picked) ; return picked


def umap_visualization(latent_actions_mu_n1d:   Tensor,
                        cluster_ids_n:           Tensor | None = None,
                        *, 
                        vis_filename:            str = 'umap_visualization.png',
                        vis_dir:                 Path = Path('visualizations/'),) -> tuple[np.ndarray, np.ndarray]:
    
    if cluster_ids_n is None: cluster_ids_n = torch.zeros(latent_actions_mu_n1d.shape[0], dtype=torch.long)
    latent_np: np.ndarray                   = latent_actions_mu_n1d.squeeze(1).detach().cpu().numpy()

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.10,
        metric="euclidean",
        verbose=False,
    )
    embedding_n2: np.ndarray = reducer.fit_transform(latent_np)   # shape (N, 2)

    clusters = np.unique(cluster_ids_n.detach().cpu().numpy())

    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster, color in zip(clusters, _sample_colors(n_colors=len(clusters))):
        cluster_elements_n2 = embedding_n2[cluster_ids_n == cluster]
        colors_n3           = np.array([color] * len(cluster_elements_n2)) / 255.
        ax.scatter( cluster_elements_n2[:, 0],
                    cluster_elements_n2[:, 1],
                    s=5, alpha=0.6, linewidths=0, c=colors_n3,
                    label=f"Cluster {cluster}")

    ax.set(title="UMAP Projection of Latent Actions", xlabel="UMAP-1", ylabel="UMAP-2")
    ax.legend(title="Clusters", markerscale=4, fontsize=8)

    fig.tight_layout()

    vis_dir.mkdir(parents=True, exist_ok=True)
    viz_path = vis_dir / vis_filename
    fig.savefig(viz_path, dpi=300)
    plt.close(fig)

    print(f"[AdaWorld] UMAP plot saved to: {viz_path.resolve()}")
    return embedding_n2, cluster_ids_n


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
