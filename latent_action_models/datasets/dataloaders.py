import  json
from    multimethod import multimethod
import  torch
from    pathlib                         import Path
from    typing                          import Literal
from    torch                           import Tensor
from    torch.utils.data                import Dataset, DataLoader

from    latent_action_models.configs                            import DataConfig
from    latent_action_models.datasets.clip_metadata_generator   import _dataset_clips, ClipEntry
from    latent_action_models.utils                              import init_distributed
from    latent_action_models.datasets.decord_dataset            import DecordVideoDataset

CLIPS_BASE_DIR = Path.cwd() / 'latent_action_models' / 'data' / 'indices' 

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


def create_dataloader(config: DataConfig) -> DataLoader:
    rank, world, _  = init_distributed()
    dataset         = _dataset(config.dataset_name, config, rank, world)

    return DataLoader(dataset, batch_size=config.batch_size)


if __name__ == "__main__":
    data            = DataConfig().from_dict({
        "dataset_name": "gta_4",
        "resolution": 256,
        "num_frames": 2,
        "batch_size": 8,
        "num_threads": 4,
    })
    dl              = create_dataloader(data)
    
    for _ in range(10):
        video_bnchw     = next(iter(dl))
        print(video_bnchw.shape)
        print(f'checksum: {video_bnchw.sum()}')
