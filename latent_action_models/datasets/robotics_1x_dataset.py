import  json
import  torch
import  typing as tp
from    pathlib import Path
from    typing import Literal
from    torch.utils.data    import Dataset, IterableDataset
from    torchvision.io      import decode_image
from    torchvision         import transforms
from    toolz               import valfilter

from latent_action_models.configs import DataConfig

DATASET_ROOT = Path("/mnt") / "data" / "sami" / "1x_dataset" / "original"
IMAGE_PATH   = Path("/mnt") / "data" / "sami" / "1x_dataset" / "data"

class Robotics_1X_Dataset(Dataset):
    def __init__(self, config: DataConfig) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int) -> tp.Any:
        pass


class Robotics_1X_Dataset(IterableDataset):
    def __init__(self,
                config: DataConfig,
                root: Path = IMAGE_PATH,
                split: Literal['train', 'val'] = 'train'):
        super().__init__()

        self.config         = config
        self.image_root     = root / split / 'images'
        self.metadata_root  = root / split / 'metadata'

        self.min_frames_per_episode  = config.num_frames
        self.resolution              = config.resolution
        self.resize_fn               = transforms.Resize(self.resolution)

        self.all_metadata: list[dict]   = [
            json.load(open(str(path)))
            for path in self.metadata_root.glob('metadata_*.json')
        ]

        self.metadata: dict = {
            meta['shard_idx']: meta for meta in self.all_metadata
        } 
        # NOTE turns out each metadata has a global episode, instead of a local one.
        # e.g. metadata_0.json ends at episode 519, metadata_1.jsson starts at 520.
        # This means we can randomly sample an episode, and then a frame number, and load.
        self.episode_to_nframes = {
            int(e): meta['num_frames_per_episode'][e]
            for meta in self.all_metadata
            for e in meta['num_frames_per_episode']
        }
        self.episode_to_nframes = valfilter(
            lambda x: x >= self.min_frames_per_episode,
            self.episode_to_nframes,
        )
        
        self.episode_to_shard = {
            e: meta['shard_idx']
            for meta in self.all_metadata
            for e in meta['episodes']
            if e in self.episode_to_nframes
        }

        assert set(self.episode_to_shard) == set(self.episode_to_nframes)
        print(f'Loaded metadata for {len(self.episode_to_shard)} episodes totaling {sum(self.episode_to_nframes.values())} frames')

    def random_filenames(self) -> list[Path]:
        import random
        episode      = random.choice(list(self.episode_to_shard.keys()))
        start_frame  = random.randint(0, self.episode_to_nframes[episode]-self.min_frames_per_episode)
        end_frame    = start_frame + self.min_frames_per_episode
        assert         end_frame <= self.episode_to_nframes[episode]
        shard        = self.episode_to_shard[episode]

        return [
            self.image_root / f'rgb_sh{shard}_ep{episode}_fr{i}.jpeg'
            for i in range(start_frame, end_frame)
        ]

    def get_random_video(self) -> tuple[torch.Tensor, type(None), type(None)]:
        filenames   = self.random_filenames()
        images      = [decode_image(str(filename.absolute())) for filename in filenames]
        images      = [self.resize_fn(image) for image in images]
        video_nchw  = torch.stack(images)
        return video_nchw, 'nopath', torch.tensor(0)

    def __iter__(self):
        while True: yield self.get_random_video()
