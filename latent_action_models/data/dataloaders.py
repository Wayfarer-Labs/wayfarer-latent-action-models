import  json
from    multimethod import multimethod
import  torch
from    pathlib                         import Path
from    typing                          import Literal
from    torch                           import Tensor
from    torch.utils.data                import Dataset, DataLoader

from    latent_action_models.configs                        import DataConfig
from    latent_action_models.data.dali_pipeline             import DALI_VideoDataset
from    latent_action_models.data.clip_metadata_generator   import _dataset_clips, ClipEntry
from    latent_action_models.utils                          import get_world_size

CLIPS_BASE_DIR = Path.cwd() / 'latent_action_models' / 'data' / 'indices' 

def _shard_for_rank(elements: list, rank: int = 0) -> list:
    world_size  = get_world_size()
    per_rank    = len(elements) // world_size
    start, end  = rank * per_rank, (rank + 1) * per_rank if rank < world_size - 1 else len(elements)
    return elements[start:end]


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
def _dataset(dataset: Literal["random"], config: DataConfig, rank: int = 0) -> Dataset:
    return RandomDataset(config)

@multimethod
def _dataset(dataset: Literal["gta_4"], config: DataConfig, rank: int = 0) -> Dataset:
    if CLIPS_BASE_DIR / 'gta4_clips.jsonl':
        clips   = [ClipEntry(**json.loads(line)) for line in open(CLIPS_BASE_DIR / 'gta4_clips.jsonl').readlines()]
    else: clips = _dataset_clips(dataset)

    clips = _shard_for_rank(clips, rank=rank)

    return DALI_VideoDataset(clips,
                             resolution=config.resolution,
                             num_frames=config.num_frames,
                             batch_size=config.batch_size,
                             num_threads=config.num_threads)


@multimethod
def _dataset(dataset: Literal["call_of_duty"], config: DataConfig, rank: int = 0) -> Dataset:
    if CLIPS_BASE_DIR / 'cod_clips.jsonl':
        clips   = [ClipEntry(**json.loads(line)) for line in open(CLIPS_BASE_DIR / 'cod_clips.jsonl').readlines()]
    else: clips = _dataset_clips(dataset)

    clips = _shard_for_rank(clips, rank=rank)


def create_dataloader(config: DataConfig, rank: int = 0) -> DataLoader:
    # TODO: make sure video is rescaled to 0/1 
    dataset = _dataset(config.env_source, config, rank)

    if isinstance(dataset, DALI_VideoDataset):
        return DataLoader(_dataset(config.env_source, config),
                          batch_size=1,
                          shuffle=config.randomize)
    else:
        return DataLoader(_dataset(config.env_source, config),
                          batch_size=config.batch_size,
                          shuffle=config.randomize)
