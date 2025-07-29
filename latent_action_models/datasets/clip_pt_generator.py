import torch
import decord
import ray # NEW: Import Ray
from pathlib import Path
from tqdm import tqdm

# Set the target directory containing your .mkv files
_VIDEO_DIR = Path('/mnt/data/shahbuland/video-proc-2/datasets/gta_nas')

# --- NEW: Define the work for a single video as a Ray remote function ---
@ray.remote
def convert_single_video(video_path: Path) -> str:
    """
    Converts a single .mkv file to a .pt file.
    This function will be executed in parallel by Ray workers.
    """
    # This setup is done once per worker process
    decord.bridge.set_bridge('torch')
    ctx = decord.cpu(0)

    output_path = video_path.with_suffix('.pt')

    if output_path.exists():
        return f"SKIPPED: {video_path.name} already exists."

    try:
        video_reader = decord.VideoReader(str(video_path), ctx=ctx)
        all_frames = video_reader.get_batch(range(len(video_reader)))
        torch.save(all_frames, output_path)
        return f"SUCCESS: Converted {video_path.name}"
    except Exception as e:
        return f"ERROR: Failed to convert {video_path.name}. Reason: {e}"


def convert_videos_to_pt_parallel(directory: Path):
    """
    Finds all .mkv files and converts them to .pt in parallel using Ray.
    """
    mkv_files = sorted(list(directory.glob("*.mkv")))
    if not mkv_files:
        print(f"WARNING: No .mkv files found in {directory}")
        return

    print(f"Found {len(mkv_files)} .mkv files to process in {directory}")

    # --- NEW: Launch all conversion tasks in parallel ---
    # .remote() immediately returns a "future" (ObjectRef) and doesn't block
    futures = [convert_single_video.remote(path) for path in mkv_files]
    
    # --- NEW: Wait for tasks to complete and show progress ---
    # We iterate through the futures and call ray.get() on each.
    # This blocks until the result for that specific future is ready.
    for future in tqdm(futures, desc="Converting .mkv to .pt"):
        # You can optionally print the result of each task as it finishes
        # print(ray.get(future))
        ray.get(future) # This waits for the task to complete

    print("\nConversion process finished.")


if __name__ == "__main__":
    # --- NEW: Initialize Ray ---
    # By default, Ray will try to use all available cores.
    ray.init()

    try:
        convert_videos_to_pt_parallel(_VIDEO_DIR)
    finally:
        # --- NEW: Shutdown Ray ---
        ray.shutdown()