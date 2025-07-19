import multimethod
from typing import Literal
from torch.utils.data import Dataset, DataLoader
from latent_action_models.configs import DataConfig


class AtariDataset(Dataset):
    def __init__(self, config: DataConfig):
        self.config = config

    def __len__(self):
        return self.config.samples_per_epoch


    def __getitem__(self, index):
        return self.config.samples_per_epoch


@multimethod
def _dataset(env_source: Literal["atari"], config: DataConfig) -> Dataset:
    raise NotImplementedError


@multimethod
def _dataset(env_source: Literal["call_of_duty"], config: DataConfig) -> Dataset:
    raise NotImplementedError


def create_dataloader(config: DataConfig) -> DataLoader:
    return DataLoader(_dataset(config.env_source, config),
                      batch_size=config.batch_size,
                      shuffle=config.randomize)