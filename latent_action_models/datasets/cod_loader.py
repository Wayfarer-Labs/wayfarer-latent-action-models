import  os
import  csv
import  uuid
import  torch
import  random
import  traceback
import  hashlib
from    functools           import cache
from    tqdm                import tqdm
from    toolz               import first, valfilter
from    itertools           import product as cartesian_product, chain
from    torch               import Tensor
from    collections         import defaultdict
from    torch.utils.data    import IterableDataset, Dataset, DataLoader
from    pathlib             import Path
from    typing              import Generator, Literal
from    torchvision.transforms import Resize


LATENT_TRAIN_DIR    = Path('/mnt/data/datasets/cod_yt_latents')
MANIFEST_PATH       = Path('/mnt/data/sami/cache/lam_manifests')

class ChunkSizeException(Exception):
    pass

# Top-level worker usable by both threads and processes
def _collect_episode_chunks(epi_dir_str: str, keep_episodes: list[int], window_span: int) -> list[tuple[str, int, int]]:
    epi_dir = Path(epi_dir_str)
    try:
        epi_id = int(epi_dir.name)
    except Exception:
        return []
    if epi_id not in keep_episodes:
        return []
    splits = epi_dir / "splits"
    if not splits.exists():
        return []

    episode_chunks: list[tuple[str, int, int]] = []
    for chunk in sorted(splits.glob("*_rgb.pt"), key=lambda p: p.name):
        L = _probe_chunk_length(chunk)
        count = L - window_span
        if count <= 0:
            continue
        episode_chunks.append((str(chunk), epi_id, int(L)))
    return episode_chunks


def _probe_chunk_length(pt_path: Path) -> int:
    try:
        x = torch.load(pt_path, map_location='meta')
        return int(x.shape[0])
    except Exception:
        x = torch.load(pt_path, map_location='cpu')
        return int(x.shape[0])


class CoD_Dataset(Dataset):
    def __init__(
        self,
        base_dir: Path = LATENT_TRAIN_DIR,
        split: Literal['train', 'val'] = 'train',
        val_split: float    = 0.1,
        num_frames: int     = 2,
        stride: int         = 1,
        is_latent: bool     = False,
        seed: int           = 42,
        parallel_backend: Literal['thread', 'process'] = 'thread',
        max_workers: int | None = None,
        resolution: int | None = None,
    ):
        self.split              = split
        self.base_dir           = base_dir
        self.num_frames         = num_frames
        self.stride             = stride
        self.is_latent          = is_latent
        self.seed               = seed
        self.parallel_backend   = parallel_backend
        self.max_workers        = max_workers
        self.file_suffix        = '_rgblatent.pt' if is_latent else 'rgb.pt'
        self.resolution         = resolution
        if self.resolution is not None: assert not self.is_latent, "Resolution is only supported for RGB data"
        self.resize_fn          = Resize(resolution) if resolution is not None else None

        self.overall_episodes = [
            int(epi_dir.name)
            for epi_dir in self.base_dir.iterdir()
            if epi_dir.is_dir()
        ]
        random.Random(self.seed).shuffle(self.overall_episodes)
        self.train_episodes = self.overall_episodes[:int(len(self.overall_episodes) * (1 - val_split))]
        self.val_episodes   = self.overall_episodes[int(len(self.overall_episodes) * (1 - val_split)):]

        self._train_manifest = self._load_or_build_manifest('train')
        self._val_manifest   = self._load_or_build_manifest('val')

        self._train_chunks  = self._train_manifest['chunks']
        self._train_windows = self._train_manifest['windows']
        self._train_meta    = self._train_manifest['meta']

        self._val_chunks  = self._val_manifest['chunks']
        self._val_windows = self._val_manifest['windows']
        self._val_meta    = self._val_manifest['meta']


    def _load_or_build_manifest(self, split: Literal['train', 'val']) -> dict[str, ...]:
        episodes = self.train_episodes if split == 'train' else self.val_episodes
        manifest_path = MANIFEST_PATH / f"{split}_manifest_stride{self.stride}_numframes{self.num_frames}.pt"
        if manifest_path.exists():
            return torch.load(manifest_path, map_location="cpu")
        else:
            manifest = self.build_manifest(self.num_frames, self.stride, self.base_dir, keep_episodes=episodes, backend=self.parallel_backend, max_workers=self.max_workers)
            tmp_path = manifest_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            torch.save(manifest, tmp_path)
            os.replace(tmp_path, manifest_path)
            return manifest

    @property
    def windows(self) -> Tensor:
        return self._train_windows if self.split == 'train' else self._val_windows
    
    @property
    def chunks(self) -> Tensor:
        return self._train_chunks if self.split == 'train' else self._val_chunks
    
    @property
    def meta(self) -> Tensor:
        return self._train_meta if self.split == 'train' else self._val_meta

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, idx: int):
        windows_map, chunks_map = (self._train_windows, self._train_chunks) if self.split == 'train' else (self._val_windows, self._val_chunks)
        cid, start = windows_map[idx].tolist()
        cid   = int(cid)
        start = int(start)

        end_exclusive = start + (self.num_frames - 1) * self.stride + 1

        path        = chunks_map["paths"][cid]

        chunk = torch.load(path, map_location="cpu")  # expect shape [T, ...]
        T     = int(chunk.shape[0])

        if end_exclusive > T:
            raise IndexError(
                f"Window out of range for chunk_id={cid} ({path}): need up to {end_exclusive}, length={T}. "
                "Rebuild manifest."
            )
        
        window = chunk[start:end_exclusive:self.stride]

        if window.shape[0] != self.num_frames:
            raise RuntimeError(
                f"Expected {self.num_frames} frames, got {window.shape[0]} "
                f"(cid={cid}, start={start}, stride={self.stride})."
            )

        info = {
            "chunk_id":    cid,
            "episode_idx": int(chunks_map["episode_idx"][cid].item()),
            "path":        path,
            "start_idx":   start,
            "stride":      self.stride,
            "num_frames":  self.num_frames,
        }
        return window, info
 
 
    def build_manifest_threads(self, num_frames: int, stride: int, base_dir: Path, keep_episodes: list[int], max_workers: int | None = None) -> dict[str, ...]:
        return self.build_manifest(num_frames, stride, base_dir, keep_episodes, backend='thread', max_workers=max_workers)


    def build_manifest_processes(self, num_frames: int, stride: int, base_dir: Path, keep_episodes: list[int], max_workers: int | None = None) -> dict[str, ...]:
        return self.build_manifest(num_frames, stride, base_dir, keep_episodes, backend='process', max_workers=max_workers)


    def build_manifest(self, num_frames: int, stride: int, base_dir: Path, keep_episodes: list[int], backend: Literal['thread', 'process'] = 'thread', max_workers: int | None = None) -> dict[str, ...]:
        # -- build manifest if not previously computed
        window_span = (num_frames-1) * stride # distance from start to last included frame
        chunk_paths:    list[str]  = []
        episode_ids:    list[int]   = []
        lengths:        list[int]   = []

        # Choose executor type
        if backend == 'thread':
            from concurrent.futures import ThreadPoolExecutor as Executor
        elif backend == 'process':
            from concurrent.futures import ProcessPoolExecutor as Executor
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Get all episode directories
        episode_dirs = sorted(
            [p for p in base_dir.iterdir() if p.is_dir()],
            key=lambda p: int(p.name)
        )
        episode_dir_strs = [str(p) for p in episode_dirs]

        # Process episodes in parallel
        worker_count = min(64, len(episode_dir_strs)) if max_workers is None else max_workers
        if worker_count <= 0: worker_count = 1
        from concurrent.futures import as_completed

        with Executor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_collect_episode_chunks, epi_dir_str, keep_episodes, window_span): epi_dir_str
                for epi_dir_str in episode_dir_strs
            }
            for future in tqdm(
                as_completed(futures),
                total=len(episode_dir_strs),
                desc=f"Processing episodes: {stride=} {num_frames=} ({backend})..."
            ):
                try:
                    episode_chunks = future.result()
                except Exception:
                    traceback.print_exc()
                    continue
                
                for chunk_path, episode_id, length in episode_chunks:
                    chunk_paths .append(chunk_path)
                    episode_ids .append(episode_id)
                    lengths     .append(length)

        assert chunk_paths

        C           = len(chunk_paths)
        episode_t   = torch.tensor(episode_ids, dtype=torch.int32)
        length_t    = torch.tensor(lengths,     dtype=torch.int32)

        windows_list: list[torch.Tensor] = []
        for cid, L in enumerate(lengths):
            count           = L - window_span
            # starts in [0, count-1], inclusive
            starts          = torch.arange(count, dtype=torch.int32)
            cids            = torch.full_like(starts, fill_value=int(cid))
            windows_list   += [torch.stack((cids, starts), dim=1)]

        windows = torch.cat(windows_list, dim=0) if windows_list else torch.empty((0, 2), dtype=torch.int32)

        meta = dict(
            num_frames  = int(num_frames),
            stride      = int(stride),
            window_span = int(window_span),
            num_chunks  = int(C),
            num_windows = int(windows.shape[0]),
            base_dir    = str(base_dir),
        )
        
        chunks = dict(
            episode_idx = episode_t,    # [C] int32
            length      = length_t,     # [C] int32
            paths       = [str(p) for p in chunk_paths],  # list[str], indexed by chunk_id
        )

        manifest = dict(
            meta    = meta,
            chunks  = chunks,
            windows = windows,              # [N,2] int32: [chunk_id, start_idx]
        )

        return manifest


class LatentIterableDataset(IterableDataset):
    def __init__(self, base_dir: Path = LATENT_TRAIN_DIR, num_frames: int = 2, stride: int = 1):
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
            for path in episode_path.glob('*' + self.file_suffix)
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


def latent_collate_fn(batch: list[tuple[Tensor, dict]]) -> tuple[Tensor, tuple[dict, ...]]:
    latents, metadata = zip(*batch)
    return torch.stack(latents, dim=0), metadata



if __name__ == "__main__":
    import time
    dataset = CoD_Dataset(stride = 4, parallel_backend='process', max_workers=16)
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=latent_collate_fn, num_workers=0)

    print("Testing VideoServerIterableDataset DataLoader...")
    t0 = time.time()
    for i, (videos, metadatas) in enumerate(dataloader):
        print(f"Batch {i}: videos.shape={videos.shape}, min={videos.min().item():.3f}, max={videos.max().item():.3f}, mean={videos.mean().item():.3f}")
        #print(f"Metadata[0]: {metadatas[0]}")
        if i >= 6400:
            break
    print(f"Done. Time elapsed: {time.time() - t0:.2f}s")