import  os
from    tqdm import tqdm
import  pathlib
import  torch
from    torch              import Tensor
from    torch.utils.data   import Dataset

from latent_action_models.models.latent_action_model import LatentActionModel, ActionEncodingInfo


def generate_latent_actions(model:      LatentActionModel,
                            dataset:    Dataset,
                            dst_dir:    pathlib.Path | None = None,
                            filename:   str                 = 'latent_actions.pt') -> Tensor:

    latent_actions_list_1n1d: list[Tensor] = []

    for video_nchw in tqdm( iter(dataset),
                            desc="Generating latent actions..." ):
        video_1nchw: Tensor         = video_nchw.unsqueeze(0)
        info: ActionEncodingInfo    = model     .encode(video_1nchw)
        latent_actions_list_1n1d                .append(info['mean_bn1d'])

    print       (f'Concatenating...')
    latent_actions_bn1d = torch.cat(latent_actions_list_1n1d, dim=0)

    if dst_dir:
        print   (f'Saving...')
        os      .makedirs(dst_dir, exist_ok=True)
        torch   .save(latent_actions_bn1d, dst_dir / filename)

    return latent_actions_bn1d
