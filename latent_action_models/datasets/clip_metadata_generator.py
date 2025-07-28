import  json
import  ffmpeg
from    tqdm                import tqdm
from    multimethod         import multimethod
from    dataclasses         import dataclass, asdict
from    typing              import Literal, Optional
from    pathlib             import Path


_TARGET_FPS = 30
_GTA4_PATH  = Path('/mnt/data/shahbuland/video-proc-2/datasets/gta_nas')
_COD_PATH   = Path('/mnt/data/shahbuland/video-proc-2/datasets/cod-yt')


def _probe(vid_path: Path, accurate=False):
    common = dict(
        cmd='ffprobe', v='error', select_streams='v:0',
        show_entries='stream=nb_frames,r_frame_rate,duration,codec_name'  # narrow output
    )
    if accurate: common['count_frames'] = None  # VERY slow

    info        = ffmpeg.probe(str(vid_path), **common)
    s           = info['streams'][0]
    num, den    = map(int, s['r_frame_rate'].split('/'))
    fps         = num / den if den else 0.0
    codec       = s['codec_name']

    frames = (v := info['format']).get('nb_frames')
    if frames in (None, 'N/A') and accurate: frames = v.get('nb_read_frames')
    if frames in (None, 'N/A'):              frames = int(float(v['duration']) * fps +0.5) if 'duration' in v else None
    
    if frames is None:  return frames,      fps, codec
    else:               return int(frames), fps, codec


@dataclass(frozen=True, slots=True)
class ClipEntry:
    video:  str
    csv:    str | None
    start:  int                 # inclusive, native FPS
    end:    int                 # inclusive
    fps:    float               # native FPS
    codec:  str

    @property
    def stride(self)    -> int: return max(1, round(self.fps / _TARGET_FPS))
    
    @property
    def jsonl(self)     -> str:
        return json.dumps(asdict(self))


@multimethod
def _dataset_clips( dataset:    Literal["call_of_duty"],
                    directory:  Path = _COD_PATH,
                    limit:      Optional[int] = None) -> list[ClipEntry]:
    # NOTE Should read .mp4s
    videos = sorted(directory.glob('*.mp4'))

    if limit is not None: videos = videos[:limit]

    if not videos:
        print(f'WARNING: No videos found with {limit=}')
        return []
    
    clips: list[ClipEntry] = []

    for video in tqdm(videos, desc="Reading clips for COD..."):
        frames, fps, codec = _probe(Path(video), accurate=False)
        print(f'{codec=}')
        if None in (frames, fps, codec):
            print(f'WARNING: Bad video found at {video} - ({frames, fps, codec}) no FPS/Frame count detected. Diagnose with ffprobe!')
            continue
            
        clip_entry = ClipEntry(video=str(video), csv=None, start=0, end=frames, fps=fps, codec=codec)
        clips     += [clip_entry]
    
    return clips


@multimethod
def _dataset_clips( dataset:    Literal["gta_4"],
                    directory:  Path = _GTA4_PATH,
                    limit:      Optional[int] = None) -> list[ClipEntry]:
    videos = sorted(directory.glob('*.mkv'))

    if limit is not None: videos = videos[:limit]

    if not videos:
        print(f'WARNING: No videos found with {limit=}')
        return []
    
    clips: list[ClipEntry] = []

    for video in tqdm(videos, desc="Reading clips for GTA4..."):
        frames, fps, codec = _probe(Path(video), accurate=False)
        csv_path    = str(video).replace('.mkv', '.csv')
        csv_path    = csv_path if Path(csv_path).exists() else None 

        if None in (frames, fps, codec):
            print(f'WARNING: Bad video found at {video} - ({frames, fps, codec}) no FPS/Frame count detected. Diagnose with ffprobe!')
            continue
        print(codec)
        clip_entry = ClipEntry(video=str(video), csv=csv_path, start=0, end=frames, fps=fps, codec=codec)
        clips     += [clip_entry]
    
    return clips

def _to_file(path: Path, clips: list[ClipEntry]) -> None: 
    with open(path, 'w') as f: 
        f.writelines((clip.jsonl+'\n' for clip in clips))

def _from_file(path: Path) -> list[ClipEntry]:
    clips = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            clips.append(ClipEntry(**data))
    return clips


if __name__ == "__main__":
    gta_clips = _dataset_clips('gta_4')
    _to_file(Path.cwd() / 'latent_action_models' / 'datasets' / 'indices' / 'gta4_clips.jsonl', gta_clips)
    cod_clips = _dataset_clips('call_of_duty')
    _to_file(Path.cwd() / 'latent_action_models' / 'datasets' / 'indices' / 'cod_clips.jsonl',  cod_clips)
