import zmq
import torch

class VideoServerLoader:
    """
    Loads videos sequentially from video loading server.
    Returned objects are dicts with keys:
    - frames: [nhwc] uint8 [0,255] rgb ndarray
    - metadata: dict with keys:
        - start_frame: int
        - end_frame: int
        - start_ts: float
        - end_ts: float
        - vid_path: str
        - vid_name: str

    Note that num_workers is really num_ports to know how many queues we can connect to.
    """
    def __init__(self, num_workers=64):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        for i in range(num_workers):
            self.socket.connect(f"tcp://127.0.0.1:{5555 + i}")

    def get_next(self):
        # Receives a Python object sent by the server
        payload = self.socket.recv_pyobj()
        return payload

        # INSERT_YOUR_CODE

import threading
import queue
import random
from torch.utils.data import IterableDataset, DataLoader

class VideoServerIterableDataset(IterableDataset):
    """
    Iterable dataset that can fetch videos from queue on video loading server.
    Note that num_workers is really num_ports to know how many queues we can connect to.
    Set this to whatever the setting was when launching the data server.

    Shuffle buffer is used to fetch many videos then shuffle them.
    Note that each worker on server is looking at a single video, so randomness is determined by
    how many separate workers there are.

    Returned objects are video tensors [b,n,c,h,w] and a list of dictionaries.
    Dicts have info on where the videos came from. See above object documentation for details. 
    """
    def __init__(self, shuffle_buffer=64, num_workers=64):
        super().__init__()
        self.shuffle_buffer = shuffle_buffer
        self.loader = VideoServerLoader(num_workers=num_workers)
        self._buffer_queue = queue.Queue(maxsize=2)  # Double buffer
        self._stop_event = threading.Event()
        self._prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._prefetch_thread.start()

    def _prefetch_loop(self):
        while not self._stop_event.is_set():
            buffer = []
            for _ in range(self.shuffle_buffer):
                try:
                    item = self.loader.get_next()
                    buffer.append(item)
                except Exception as e:
                    print(f"Error fetching from VideoServerLoader: {e}")
            random.shuffle(buffer)
            # Block if queue is full
            self._buffer_queue.put(buffer)

    def __iter__(self):
        while True:
            buffer = self._buffer_queue.get()
            for item in buffer:
                yield item

    def __del__(self):
        self._stop_event.set()
        if hasattr(self, "_prefetch_thread"):
            self._prefetch_thread.join(timeout=1)

def video_collate_fn(batch):
    """
    batch: list of dicts with keys 'frames' [n,h,w,c] and 'metadata'
    Returns:
        videos: [b, n, c, h, w] float32, normalized to [-1,1]
        metadatas: list of dicts
    """
    import numpy as np
    import torch

    videos = []
    metadatas = []
    for item in batch:
        frames = item['frames']  # [n, h, w, c], uint8
        # Convert to float32, [0,255] -> [0,1] -> [-1,1]
        frames = torch.from_numpy(frames).float() / 255.0 * 2.0 - 1.0  # [n, h, w, c]
        frames = frames.permute(0, 3, 1, 2)  # [n, c, h, w]
        videos.append(frames)
        metadatas.append(item['metadata'])
    videos = torch.stack(videos, dim=0)  # [b, n, c, h, w]
    return videos, metadatas

if __name__ == "__main__":
    import time

    dataset = VideoServerIterableDataset(shuffle_buffer=32)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=video_collate_fn, num_workers=0)

    print("Testing VideoServerIterableDataset DataLoader...")
    t0 = time.time()
    for i, (videos, metadatas) in enumerate(dataloader):
        print(f"Batch {i}: videos.shape={videos.shape}, min={videos.min().item():.3f}, max={videos.max().item():.3f}, mean={videos.mean().item():.3f}")
        #print(f"Metadata[0]: {metadatas[0]}")
        if i >= 10:
            break
    print(f"Done. Time elapsed: {time.time() - t0:.2f}s")