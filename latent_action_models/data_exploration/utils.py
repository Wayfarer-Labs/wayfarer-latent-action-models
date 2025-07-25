from    pathlib         import Path
from    typing          import Iterable, Optional, Generator
from    toolz           import first, last


DATA_ROOT = Path("/mnt/data/shahbuland/video-proc-2/datasets/gta_nas")


def iter_csv(root: Path = DATA_ROOT) -> Generator[Path, None, None]:
    yield from root.glob('*.csv')


def _first_member(s: Iterable[str], seek: str) -> Optional[str]:
    return first(c for c in s if seek in c)


def is_keypress     (raw_row: str)  -> bool:
    return any(('down'   in last(raw_row.split(',')),
                'up'     in last(raw_row.split(','))))

def is_mouse_delta  (raw_row: str)  -> bool:
    return 'mouse event' in last(raw_row.split(',')).lower()
