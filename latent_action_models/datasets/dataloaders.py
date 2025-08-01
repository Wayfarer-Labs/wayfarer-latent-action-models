import  json
from    multimethod import multimethod
import  torch
from    pathlib                         import Path
from    typing                          import Literal
from    torch                           import Tensor
from    torch.utils.data                import Dataset, DataLoader, IterableDataset

from    latent_action_models.configs                                import DataConfig
from    latent_action_models.datasets.clip_metadata_generator       import _dataset_clips, ClipEntry
from    latent_action_models.utils                                  import init_distributed
from    latent_action_models.datasets.decord_dataset                import DecordVideoDataset
from    latent_action_models.datasets.robotics_1x_dataset           import Robotics_1X_Dataset
from    latent_action_models.datasets.video_loader                  import VideoServerIterableDataset, video_collate_fn

CLIPS_BASE_DIR = Path.cwd() / 'latent_action_models' / 'datasets' / 'indices' 

class RandomDataset(Dataset):
    def __init__(self, config: DataConfig) -> None:
        super().__init__()
        self.config     = config
        self.height     = self.width = config.resolution
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
def _dataset(dataset: Literal["random"], config: DataConfig, rank: int = 0, world: int = 1) -> Dataset:
    return RandomDataset(config)


@multimethod
def _dataset(dataset: Literal["gta_4"], config: DataConfig, rank: int = 0, world: int = 1) -> Dataset:
    if CLIPS_BASE_DIR / 'gta4_clips.jsonl':
        clips   = [ClipEntry(**json.loads(line)) for line in open(CLIPS_BASE_DIR / 'gta4_clips.jsonl').readlines()]
    else: clips = _dataset_clips(dataset)

    return DecordVideoDataset(clips, resolution=config.resolution, num_frames=config.num_frames)


@multimethod
def _dataset(dataset: Literal["call_of_duty"], config: DataConfig, rank: int = 0, world: int = 1) -> Dataset:
    if CLIPS_BASE_DIR / 'cod_clips.jsonl':
        clips   = [ClipEntry(**json.loads(line)) for line in open(CLIPS_BASE_DIR / 'cod_clips.jsonl').readlines()]
    else: clips = _dataset_clips(dataset)

    return DecordVideoDataset(clips, resolution=config.resolution, num_frames=config.num_frames)


@multimethod
def _dataset(dataset: Literal["1x_robotics"], config: DataConfig, rank: int = 0, world: int = 1) -> Dataset:
    return Robotics_1X_Dataset(config)


@multimethod
def _dataset(dataset: Literal["owl_data"], config: DataConfig, rank: int = 0, world: int = 1) -> IterableDataset:
    return VideoServerIterableDataset(shuffle_buffer=64, num_workers=64)


def create_dataloader(config: DataConfig) -> DataLoader:
    rank, world, _  = init_distributed()
    dataset         = _dataset(config.dataset_name, config, rank, world)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=video_collate_fn if config.dataset_name == "owl_data" else None,
    )


if __name__ == "__main__":
    data            = DataConfig().from_dict({
        "dataset_name": "1x_robotics",
        "resolution": 256,
        "num_frames": 2,
        "batch_size": 64,
        "num_workers": 8,
    })
    dl              = create_dataloader(data)
    import time
    for i in range(10):
        start = time.time()
        video_bnchw,*_     = next(iter(dl))
        print(f'{i}: {video_bnchw.shape} {time.time() - start}')
        print(f'checksum: {video_bnchw.sum()}')
