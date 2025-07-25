import  json
import  ffmpeg
import  pyarrow
import  traceback
from    itertools       import starmap
from    typing          import Optional, Generator, Iterable
import  pandas          as pd
from    pandas          import json_normalize
from    tqdm            import tqdm
from    multimethod     import multimethod
from    toolz           import first, memoize, juxt, identity, pipe, curry
from    toolz.curried   import juxt, map
from    pathlib         import Path
from    dataclasses     import dataclass, asdict

from latent_action_models.data_exploration.parse_csv    import process_dir as get_actions, ParsedEvent
from latent_action_models.data_exploration.utils        import DATA_ROOT, is_ffmpeg_installed
from latent_action_models.data_exploration              import cache

starmap             = curry(starmap)
ENABLE_CACHE        = True
CACHED_ACTIONS_PATH = Path(first(cache.__path__)) / 'events.jsonl'
PARQUET_PATH        = Path(first(cache.__path__)) / 'gta_events.parquet'
FFPROBE_ARGS        = dict( cmd='ffprobe', v='error', select_streams='v:0',
                            show_entries='stream=nb_frames,r_frame_rate,duration,codec_name')
ROW_PROGRESS_BAR    = tqdm(desc='Reading dataframe rows...')


if ENABLE_CACHE:
    get_actions = curry(lambda _, limit: (ParsedEvent(**json.loads(line)) for line in open(CACHED_ACTIONS_PATH, 'r').readlines()))


@dataclass
class VideoMetadata:
    video_path: str
    start:      Optional[int]   = None # inclusive, native FPS
    end:        Optional[int]   = None # inclusive
    fps:        Optional[float] = None # native FPS
    codec:      Optional[str]   = None
    exc:        Optional[str]   = None

    def __post_init__(self):
        ROW_PROGRESS_BAR.update(1)
        ROW_PROGRESS_BAR.set_description(f'Reading dataframe rows... {self.video_path}')


@multimethod
@memoize
def process_video(video_path: Path) -> VideoMetadata:
    is_ffmpeg_installed()
    metadata = VideoMetadata(video_path=str(video_path))
    exc      = None

    try: 
        info            = ffmpeg.probe(str(video_path), **FFPROBE_ARGS)
        streams, fmt    = first(info['streams']), info['format']
        num, den        = map(int, streams['r_frame_rate'].split('/'))
        fps             = num / den
        num_frames      = fmt.get('nb_frames')

        if num_frames in (None, 'N/A') and 'duration' in fmt:
            num_frames = int(float(fmt['duration']) * fps +0.5)
        
        metadata.fps    = fps
        metadata.codec  = streams['codec_name']
        metadata.start  = 0
        metadata.end    = num_frames

    except:     exc = traceback.format_exc()
    finally:    metadata.exc = exc ; return metadata


@multimethod
def process_video(action: ParsedEvent):
    return process_video(Path(action.origin_vid))


# -- denormalized
@dataclass
class DataframeRow:
    video_metadata: VideoMetadata
    action_data:    ParsedEvent


def get_dataframe_rows(root: Path = DATA_ROOT, *, file_limit=None) -> Generator[DataframeRow, None, None]:
    yield from pipe(
        root,
        get_actions(limit=file_limit),      # generates ParsedEvents
        map(juxt(process_video, identity)), # yields tuples of ParsedEvent, VideoMetadata
        starmap(DataframeRow)               # yields DataframeRow 
    )


def to_parquet(rows: Iterable[DataframeRow], out_path: Path,
               *,
               compression: str = "snappy") -> pd.DataFrame:
    print(f'Materializing...')
    _rows   = list(asdict(row) for row in rows)

    print(f'Creating dataframe...')
    df      = json_normalize(_rows, sep='.')
    
    df.to_parquet(
        out_path,
        engine      = 'pyarrow',
        compression = compression,
        index       = False
    )
    
    return df



if __name__ == "__main__":
    to_parquet(get_dataframe_rows(file_limit=None), out_path=PARQUET_PATH)