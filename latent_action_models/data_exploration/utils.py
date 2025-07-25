import  subprocess
from    pathlib         import Path
from    typing          import Iterable, Optional, Generator
from    toolz           import first, last, memoize
from    multimethod     import parametric


DATA_ROOT = Path("/mnt/data/shahbuland/video-proc-2/datasets/gta_nas")


def iter_csv(root: Path = DATA_ROOT) -> Generator[Path, None, None]:
    yield from root.glob('*.csv')

def iter_mkv(root: Path = DATA_ROOT) -> Generator[Path, None, None]:
    yield from root.glob('*.mkv')

def _first_member(s: Iterable[str], seek: str) -> Optional[str]:
    return first(c for c in s if seek in c)


def is_keypress     (raw_row: str)  -> bool:
    return any(('down'   in last(raw_row.split(',')),
                'up'     in last(raw_row.split(','))))

def is_mouse_delta  (raw_row: str)  -> bool:
    return 'mouse event' in last(raw_row.split(',')).lower()

@memoize
def is_ffmpeg_installed():
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        assert result.returncode == 0 or "ffprobe version" in result.stdout.lower(), \
            f"ffprobe not found or not working: {result.stderr or result.stdout}"

        assert "not found" not in result.stderr.lower() and "not found" not in result.stdout.lower(), \
            f"ffprobe not found: {result.stderr or result.stdout}"

    except FileNotFoundError: raise AssertionError("ffprobe command not found in PATH")
 


MouseDeltaRow   = parametric(str, is_mouse_delta)
KeypressRow     = parametric(str, is_keypress)
