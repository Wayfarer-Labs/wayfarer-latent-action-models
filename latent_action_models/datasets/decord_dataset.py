from __future__ import annotations
import torch
import decord
import random
import numpy as np
from torch.utils.data import Dataset

# Assuming ClipEntry is defined as you provided
from latent_action_models.datasets.clip_metadata_generator import ClipEntry


class DecordVideoDataset(Dataset):
    """
    A PyTorch Dataset for reading random video clips, with support for
    Variable Frame Rate (VFR) video to ensure time synchronization.
    """
    def __init__(self,
                 clips:         list[ClipEntry],
                 *,
                 num_frames:    int = 16,
                 target_fps:    int = 30, # Define a target FPS for clip duration
                 resolution:    int = 256):

        self.clips = clips
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.resolution = resolution
        decord.bridge.set_bridge('torch')
        self.ctx = decord.cpu(0)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int]:
        clip_info = self.clips[idx]
        
        # Open the video and get the precise timestamp for every frame
        vr = decord.VideoReader(clip_info.video, ctx=self.ctx, width=self.resolution, height=self.resolution)
        total_frames = len(vr)
        # get_frame_timestamp returns a (N, 2) array of [start_time, end_time] for each frame. We want the start time.
        timestamps = vr.get_frame_timestamp(np.arange(total_frames))[:, 0]

        # --- NEW: VFR-safe random sampling logic ---
        # Calculate the desired duration of the clip in seconds
        clip_duration = self.num_frames / self.target_fps
        video_duration = timestamps[-1]

        # Determine the latest possible start time for the clip to fit
        latest_possible_start_time = video_duration - clip_duration
        
        # Choose a new, random start time
        random_start_time = random.uniform(0, max(0, latest_possible_start_time))

        # Find the frame index that is closest to our random start time
        # np.searchsorted finds the insertion point, which is exactly what we need
        start_frame_idx = np.searchsorted(timestamps, random_start_time)
        
        # To avoid issues with stride in VFR, we find frame indices by time
        end_time = random_start_time + clip_duration
        end_frame_idx = np.searchsorted(timestamps, end_time)

        # We need exactly num_frames. We can get them by taking evenly spaced indices
        # between the calculated start and end frame indices.
        frame_indices = np.linspace(start_frame_idx, end_frame_idx, self.num_frames, dtype=int)
        # Ensure indices are within the valid range
        frame_indices = np.clip(frame_indices, 0, total_frames - 1)
        
        frames = vr.get_batch(frame_indices)

        # Preprocess the tensor
        video_tensor = frames.permute(0, 3, 1, 2)
        video_tensor = video_tensor / 255.0

        # Return the tensor and the metadata (true start frame) for the clip
        return video_tensor, clip_info.video, start_frame_idx


if __name__ == "__main__":
    from pathlib import Path
    from latent_action_models.datasets.clip_metadata_generator import _dataset_clips, _from_file, _to_file
    # gtaclips = _dataset_clips('gta_4')
    clips = _from_file(Path.cwd() / 'latent_action_models' / 'datasets' / 'indices' / 'gta4_clips.jsonl')
    ds = DecordVideoDataset(clips)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=10, num_workers=0
    )
    x = next(iter(dl))