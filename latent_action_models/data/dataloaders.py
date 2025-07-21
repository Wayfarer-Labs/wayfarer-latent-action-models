from    multimethod import multimethod
import  torch
from    typing                         import Literal
from    torch                          import Tensor
from    torch.utils.data               import Dataset, DataLoader

from    latent_action_models.configs   import DataConfig


class RandomDataset(Dataset):
    def __init__(self, config: DataConfig) -> None:
        super().__init__()
        self.config     = config
        self.height     = self.width = config.resolution
        self.format     = config.output_format
        self.n_frames   = config.num_frames

    def __len__(self):
        return 1_000

    def __getitem__(self, index) -> Tensor:
        video_bnchw = torch.randn(self.n_frames, 3, self.height, self.width)
        video_bnchw += abs(video_bnchw.min())
        video_bnchw /= video_bnchw.max()
        assert video_bnchw.max() <= 1. and video_bnchw.min() >= 0.
        return video_bnchw


@multimethod
def _dataset(env_source: str, config: DataConfig) -> Dataset:
    return RandomDataset(config)


def create_dataloader(config: DataConfig) -> DataLoader:
    return DataLoader(_dataset(config.env_source, config),
                      batch_size=config.batch_size,
                      shuffle=config.randomize)