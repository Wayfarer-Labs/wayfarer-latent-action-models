from __future__ import annotations

import  json, random, numpy as np, torch
from    typing              import Iterator


import  nvidia.dali.fn              as fn
import  nvidia.dali.types           as types
from    nvidia.dali.data_node       import DataNode
from    nvidia.dali.plugin.pytorch  import DALIGenericIterator
from    nvidia.dali.pipeline        import Pipeline, pipeline_def

from    latent_action_models.data.clip_metadata_generator import ClipEntry


class ClipIterator:
    """
    Yields (encoded_bytes, start, end, stride) for DALI external_source.
    """
    def __init__(self, 
                 clips:     list[ClipEntry],
                 shuffle:   bool = True) -> None:
        self._clips = clips
        self._shuffle = shuffle

    def __iter__(self) -> Iterator[tuple[np.ndarray, int, int, int]]:
        order = list(range(len(self._clips)))

        if self._shuffle: random.shuffle(order)

        for idx in order:
            c = self._clips[idx]
            with open(c.video, "rb") as f: buf = np.frombuffer(f.read(), dtype=np.uint8)
            yield buf, c.start, c.end + 1, c.stride


def make_pipeline(source:       ClipIterator,
                  batch_size:   int,
                  num_frames:   int,
                  output_type = types.RGB,
                  num_threads:  int = 4,
                  resolution:   int = 256,
                  device_id:    int = 0) -> Pipeline:

    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    def _factory(source: ClipIterator, output_type=types.RGB) -> DataNode:
        vid_buf, start, end, stride = fn.external_source(
            source=source,
            num_outputs=num_frames,
            batch=False, # one sample at a time
            dtype=[types.UINT8, types.INT32, types.INT32, types.INT32],
            device="cpu"
        )
        frames = fn.experimental.decoders.video(
            vid_buf,
            start_frame=start,
            end_frame=end,
            stride=stride,
            device="mixed",
            output_type=output_type
        )
        frames = fn.resize(frames, resize_x=resolution, resize_y=resolution)
        frames = fn.transpose(frames, perm=[0,3,1,2])  # (N,C,H,W)
        return frames

    return _factory(source, output_type)


class DALI_VideoDataset(torch.utils.data.IterableDataset):
    """
    Thin PyTorch wrapper around a single-GPU DALI pipeline.
    """
    def __init__(self,
                 clips:         list[ClipEntry],
                 num_frames:    int = 16,
                 batch_size:    int = 8,
                 num_threads:   int = 4,
                 resolution:    int = 256,
                 shuffle:       bool = True) -> None:
        self._source            = ClipIterator(clips, shuffle)
        self._pipe: Pipeline    = make_pipeline(source      = self._source,
                                                num_frames  = num_frames,
                                                batch_size  = batch_size,
                                                num_threads = num_threads,
                                                resize      = resolution,
                                                device_id   = torch.cuda.current_device()) ; self._pipe.build()
        self._it                = iter(DALIGenericIterator([self._pipe], ["video"], auto_reset=True))

    def __iter__(self): return self

    def __next__(self) -> torch.Tensor:
        sample_nchw = next(self._it)[0]["video"]   # shape (N,C,H,W), already on GPU
        return sample_nchw
