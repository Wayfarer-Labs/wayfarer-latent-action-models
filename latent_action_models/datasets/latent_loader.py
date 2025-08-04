import  torch
import  random
import  traceback
from    functools           import cache
from    torch               import Tensor
from    torch.utils.data    import IterableDataset, DataLoader
from    pathlib             import Path
from    typing              import Generator


LATENT_BASE_DIR = Path('/mnt/data/datasets/1x_latents/')


class ChunkSizeException(Exception):
    pass


class LatentIterableDataset(IterableDataset):
    def __init__(self, base_dir: Path = LATENT_BASE_DIR, num_frames: int = 2, stride: int = 1):
        self.base_dir       = base_dir
        self.num_episodes   = len(list(self.base_dir.glob('*')))
        self.num_frames     = num_frames
        self.stride         = stride

    @cache
    def _sample_chunk_paths(self, episode_idx: int) -> list[tuple[Path, int]]:
        episode_path    = self.base_dir / f'{episode_idx}' / 'splits'
        return [
            (
                path,
                torch.load(path, map_location='meta').shape[0]
            )
            for path in episode_path.glob('*_rgblatent.pt')
        ]

    def sample(self) -> tuple[Tensor, dict]:
        episode_idx             = random.randint(0, self.num_episodes-1)
        chunk_path, chunk_size  = random.choice (self._sample_chunk_paths(episode_idx))
        # -- index into the chunk
        window_size             = self.num_frames * self.stride
        
        if chunk_size < window_size:
            raise ChunkSizeException(f'Chunk of size {chunk_size} at {chunk_path} smaller than {window_size=}')
        
        max_start               = max(chunk_size - window_size - 1, 0)
        start_idx               = random.randint(0, max_start)
        end_idx                 = start_idx + window_size
        chunk                   = torch.load(chunk_path)[start_idx:end_idx:self.stride]

        return (
            chunk, 
            {
                'chunk_path':   chunk_path,
                'chunk_size':   chunk_size,
                'episode':      episode_idx,
                'start_idx':    start_idx,
                'end_idx':      end_idx
            }
        )

    def __iter__(self) -> Generator[tuple[Tensor, dict], None, None]:
        while True: 
            try:
                yield self.sample()
            except ChunkSizeException: continue
            except (FileNotFoundError, IndexError, ValueError): 
                traceback.print_exc() ; continue


def video_collate_fn(batch: list[tuple[Tensor, dict]]) -> tuple[Tensor, tuple[dict, ...]]:
    latents, metadata = zip(*batch)
    return torch.stack(latents, dim=0), metadata



if __name__ == "__main__":
    import time
    dataset = LatentIterableDataset()
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=video_collate_fn, num_workers=64)

    print("Testing VideoServerIterableDataset DataLoader...")
    t0 = time.time()
    for i, (videos, metadatas) in enumerate(dataloader):
        print(f"Batch {i}: videos.shape={videos.shape}, min={videos.min().item():.3f}, max={videos.max().item():.3f}, mean={videos.mean().item():.3f}")
        #print(f"Metadata[0]: {metadatas[0]}")
        if i >= 10:
            break
    print(f"Done. Time elapsed: {time.time() - t0:.2f}s")