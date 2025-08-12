import  os
import  csv
import  uuid
import  torch
import  random
import  traceback
import  hashlib
from    functools           import cache
from    tqdm                import tqdm
from    toolz               import first
from    itertools           import product as cartesian_product, chain
from    torch               import Tensor
from    collections         import defaultdict
from    torch.utils.data    import IterableDataset, Dataset, DataLoader
from    pathlib             import Path
from    typing              import Generator


LATENT_BASE_DIR = Path('/mnt/data/datasets/1x_latents/')
MANIFEST_PATH   = Path('latent_action_models/datasets/manifest')

class ChunkSizeException(Exception):
    pass

def _probe_chunk_length(pt_path: Path) -> int:
    try:
        x = torch.load(pt_path, map_location='meta')
        return int(x.shape[0])
    except Exception:
        x = torch.load(pt_path, map_location='cpu')
        return int(x.shape[0])


class LatentDataset(Dataset):
    def __init__(self,
                base_dir: Path      = LATENT_BASE_DIR,
                num_frames: int     = 2,
                stride: int         = 1,
        ):

        self.base_dir       = base_dir
        self.num_episodes   = len(list(self.base_dir.glob('*')))
        self.num_frames     = num_frames
        self.stride         = stride

        manifest            = LatentDataset.build_manifest(self.num_frames, self.stride, self.base_dir)

        self._chunks        = manifest['chunks']
        self._windows       = manifest['windows']
        self._meta          = manifest['chunks']

    def __len__(self) -> int:
        return int(self._windows.shape[0])

    def __getitem__(self, idx: int):
        cid, start = self._windows[idx].tolist()
        cid   = int(cid)
        start = int(start)

        end_exclusive = start + (self.num_frames - 1) * self.stride + 1

        path  = self._chunks["paths"][cid]
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
            "episode_idx": int(self._chunks["episode_idx"][cid].item()),
            "path":        path,
            "start_idx":   start,
            "stride":      self.stride,
            "num_frames":  self.num_frames,
        }
        return window, info


    @classmethod
    def build_manifest(cls, num_frames: int, stride: int, base_dir: Path) -> dict[str, ...]:
        MANIFEST_PATH.mkdir(parents=True, exist_ok=True)
        
        manifest_name   = f"manifest_stride{stride}_numframes{num_frames}.pt"
        manifest_path   = MANIFEST_PATH / manifest_name

        if manifest_path.exists():
            return torch.load(manifest_path, map_location="cpu")

        # -- build manifest if not previously computed
        window_span = (num_frames-1) * stride # distance from start to last included frame
        chunk_paths:    list[Path]  = []
        episode_ids:    list[int]   = []
        lengths:        list[int]   = []

        for epi_dir in sorted(
            [p for p in base_dir.iterdir() if p.is_dir()],
            key=lambda p: int(p.name)
        ):
            splits = epi_dir / "splits"
            
            if not splits.exists():  continue

            for chunk in sorted(splits.glob("*_rgblatent.pt"), key=lambda p: p.name):
                try:                L = _probe_chunk_length(chunk)
                except Exception:   continue
                
                # number of valid start positions for exactly num_frames with given stride
                count = L - window_span
                print(L, count)
                if count <= 0: continue

                chunk_paths += [chunk]
                episode_ids += [int(epi_dir.name)]
                lengths     += [int(L)]
            
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

        # -- atomic save
        tmp_path = manifest_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        torch   .save(manifest, tmp_path)        # binary, compact, fast
        os      .replace(tmp_path, manifest_path)   # atomic on POSIX
        return manifest


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


def latent_collate_fn(batch: list[tuple[Tensor, dict]]) -> tuple[Tensor, tuple[dict, ...]]:
    latents, metadata = zip(*batch)
    return torch.stack(latents, dim=0), metadata



if __name__ == "__main__":
    import time
    dataset = LatentDataset(stride = 4)
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=latent_collate_fn, num_workers=64)

    print("Testing VideoServerIterableDataset DataLoader...")
    t0 = time.time()
    for i, (videos, metadatas) in enumerate(dataloader):
        print(f"Batch {i}: videos.shape={videos.shape}, min={videos.min().item():.3f}, max={videos.max().item():.3f}, mean={videos.mean().item():.3f}")
        #print(f"Metadata[0]: {metadatas[0]}")
        if i >= 6400:
            break
    print(f"Done. Time elapsed: {time.time() - t0:.2f}s")