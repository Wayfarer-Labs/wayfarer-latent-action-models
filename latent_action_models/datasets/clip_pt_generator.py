import torch
from torchcodec.decoders import VideoDecoder
from pathlib import Path
from tqdm import tqdm
import traceback


def convert_single_video_gpu_chunked(
    video_path: Path,
    device: str = "cuda:0"
) -> str:
    """
    Converts a single .mkv file to a .pt file using the GPU for decoding,
    processing in chunks to conserve GPU memory.
    """
    output_path = video_path.with_suffix('.pt')

    if output_path.exists():
        return f"SKIPPED: {video_path.name} already exists."

    try:
        reader = VideoDecoder(str(video_path), 'cuda')
        # num_frames = len(reader)
        
        cpu_chunks = []
        
        # MODIFIED: Added tqdm wrapper around the frame reader
        # This will show a progress bar for the frames within this single video.
        for frame in tqdm(reader, desc=f"  -> Decoding {video_path.name}", leave=False):
            gpu_frame = frame['data']
            cpu_chunks.append(gpu_frame.cpu())

        video_tensor = torch.stack(cpu_chunks).to(torch.uint8)
        torch.save(video_tensor, output_path)
        
        return f"SUCCESS: Converted {video_path.name}"
    except Exception:
        # ... (error handling is the same) ...
        print(f"\nERROR: Failed to convert {video_path.name}. Full traceback below:")
        traceback.print_exc()
        return f"ERROR: Failed to convert {video_path.name}."


def convert_videos_serially(directory: Path):
    mkv_files = sorted(list(directory.glob("*.mkv")))
    if not mkv_files:
        print(f"WARNING: No .mkv files found in {directory}")
        return

    print(f"Found {len(mkv_files)} .mkv files to process serially.")

    for path in tqdm(mkv_files, desc="Converting .mkv to .pt"):
        # Call the renamed function
        status = convert_single_video_gpu_chunked(path)

    print("\nConversion process finished.")


if __name__ == "__main__":
    _VIDEO_DIR = Path('/mnt/data/shahbuland/video-proc-2/datasets/gta_nas')
    convert_videos_serially(_VIDEO_DIR)