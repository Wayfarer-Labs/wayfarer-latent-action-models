from __future__ import annotations
import torch
from typing     import Sequence
from toolz      import groupby, valmap

import nvidia.dali.fn           as fn
import nvidia.dali.types        as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.pipeline       import pipeline_def

from latent_action_models.data.clip_metadata_generator import ClipEntry

__all__ = ["DALI_VideoDataset", "create_dali_video_dataset"]


def _infer_ddp_shard() -> tuple[int, int]:
    """Return (rank, world_size) even when torch.distributed is not initialised."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    else:
        return 0, 1


@pipeline_def
def _dali_pipe(filenames: Sequence[str], *, num_frames: int, stride: int, resize: int):
    """Internal DALI pipeline definition (one per GPU)."""
    videos = fn.readers.video(
        device="gpu",  # GPU decode only
        filenames=filenames,
        sequence_length=num_frames,
        stride=stride,
        num_shards=len(filenames),  # filenames already sharded, but DALI needs >0
        random_shuffle=True,
        skip_vfr_check=True,
        dtype=types.DALIDataType.FLOAT,
    )
    videos = fn.resize(videos, resize_x=resize, resize_y=resize)
    videos = videos / 255.0
    videos = fn.transpose(videos, perm=[0, 3, 1, 2])  # F,C,H,W
    return videos


class DALI_VideoDataset(torch.utils.data.IterableDataset):
    """Thin PyTorch wrapper around a *single-GPU* DALI pipeline.

    Parameters
    ----------
    clips : list[ClipEntry]
        Pre-parsed clip metadata (all **same codec**).
    num_frames : int, default 16
        Logical frames per sample (after stride).
    batch_size : int, default 8
    num_threads : int, default 4
    resolution : int, default 256
        Output height & width after resize.
    shuffle : bool, default True
        Whether to shuffle the *order* of clips each epoch.
    """

    def __init__(self,
                 clips:         list[ClipEntry],
                 *,
                 num_frames:    int = 16,
                 batch_size:    int = 8,
                 num_threads:   int = 4,
                 resolution:    int = 256,
                 shuffle:       bool = True,
                 rank:          int = 0,
                 world:         int = 1) -> None:

        self.clips      = clips
        self.batch_size = batch_size
        self.shuffle    = shuffle
        device_id       = torch.device('cuda', rank)
        # shard filenames by rank to avoid overlap
        filenames       = [c.video for c in clips][rank::world]

        assert filenames, f"Rank {rank} received 0 clips after sharding!"

        # Derive stride from first clip (they are homogeneous in codec & FPS here)
        stride          = max(1, round(clips[0].fps / 30))

        # Build DALI pipeline
        self._pipe = _dali_pipe(
            filenames,
            num_frames=num_frames,
            stride=stride,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            resize=resolution,
        )
        self._pipe.build()

        self._it = iter(DALIGenericIterator([self._pipe], ["data"], auto_reset=True))

    # ------------------------------------------------------------------
    def __iter__(self): return self

    def __next__(self) -> torch.Tensor:
        videos_bnchw = next(self._it)[0]["data"]
        return videos_bnchw


def create_dali_video_dataset(clips: list[ClipEntry], *, keep_codec: str = "h264", **ds_kwargs) -> DALI_VideoDataset:
    """Filter `clips` by `keep_codec`, log ignored counts, and return dataset."""
    # Count by codec
    by_codec    = groupby(lambda c: c.codec, clips)
    counts      = valmap(len, by_codec)
    
    kept            = by_codec.get(keep_codec, [])
    skipped_total   = sum(cnt for k, cnt in counts.items() if k != keep_codec)
    print(f"[DALI_VideoDataset] Keeping {len(kept)} clips with codec '{keep_codec}'. Skipped {skipped_total} clips: "
          + ", ".join(f"{k}={v}" for k, v in counts.items() if k != keep_codec))

    if not kept: raise ValueError(f"No clips left after filtering for codec '{keep_codec}' - counts: {counts}")

    return DALI_VideoDataset(kept, **ds_kwargs)
