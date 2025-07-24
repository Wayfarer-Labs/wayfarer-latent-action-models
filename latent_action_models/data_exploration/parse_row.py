from optparse import Option
import  pathlib, csv, re, sys
from    pathlib         import Path
from    multimethod     import overload
from    typing          import Iterable, Literal, Optional, Generator, TypedDict, Callable
from    dataclasses     import dataclass, field, asdict
from    toolz           import excepts as _excepts, pipe, map, last, first
from    toolz.curried   import map, concatv
from    functools       import wraps


DATA_ROOT = Path("/mnt/data/shahbuland/video-proc-2/datasets/gta_nas")


def excepts(exc: type(Exception), handler: Callable):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs): return _excepts(exc, fn, handler)
        return wrapper
    return decorator


def iter_csv(root: Path = DATA_ROOT) -> Generator[Path, None, None]:
    yield from root.glob('*.csv')


class RowContext(TypedDict):
    origin_csv:     Path
    origin_vid:     Path
    lineno:         int
    raw_row:        str


@dataclass
class ParsedEvent:
    origin_csv:     Path
    origin_vid:     Path
    lineno:         int
    raw:            str                 = field(repr=False)
    exc:            Optional[str]       = None
    t_sec:          Optional[float]
    event_type:     Optional[Literal["mouse", "key"]]
    keys_active:    Optional[list[str]]
    key_pressed:    Optional[str]
    transition:     Optional[str]
    dx:             Optional[int]
    dy:             Optional[int]
    wheel:          Optional[float]

def _first_member   (s: Iterable[str], seek: str) -> Optional[str]:
    return first(c for c in s if seek in c)

def _from_ctxt      (ctxt: RowContext) -> ParsedEvent:
    return ParsedEvent( origin_csv = ctxt['origin_csv'],
                        origin_vid = ctxt['origin_vid'],
                        lineno     = ctxt['lineno'],
                        raw        = ctxt['raw'])


def is_mouse_delta  (raw_row: str)  -> bool:
    return last(raw_row.split(',')) == 'mouse event'


@excepts            (Exception, _from_ctxt)
@overload
def _parse_row      (raw_row: is_mouse_delta, ctxt: RowContext) -> ParsedEvent:
    timestamp, data, _  = raw_row.split(',')
    dx, dy, wheel       = tuple(_first_member(data.split(';'), key)
                                for key in ('pos_x', 'pos_y', 'wheel'))
    
    if None in (dx, dy, wheel):         print(f'[!!] ({dx=} {dy=} {wheel=})')

    sign_x, sign_y      = 1, 1

    if '-' in dx: sign_x *= -1
    if '-' in dy: sign_y *= -1
    
    if not any('+' in dx, '-' in dy):   print(f'[!!] expected sign: {dx=} {dy=}')

    dx, dy, wheel       = tuple(last(e.split(':')).strip('+-')
                                for  e in (dx, dy, wheel))
    dx, dy, wheel       = tuple(float(e)
                                for  e in (dx, dy, wheel))

    event               = _from_ctxt(ctxt)
    event.event_type    = 'mouse'
    event.t_sec         = float(timestamp)
    (   event.dx,
        event.dy,
        event.wheel )   = dx, dy, wheel

    return event

def is_keypress     (raw_row: str)  -> bool:
    return any('down'   in last(raw_row.split(',')),
               'up'     in last(raw_row.split(',')))

@excepts            (Exception, _from_ctxt)
@overload
def _parse_row      (raw_row: is_keypress,    ctxt: RowContext) -> ParsedEvent:
    timestamp, keys_active, key_pressed = raw_row.split(',')
    

def _preprocess_raw (raw_row: str)  -> str: return raw_row.strip().lower()

def process_file(path: Path)        -> Generator[ParsedEvent, None, None]:
    with open(path, 'r') as f:
        yield from (
            _parse_row(
                _preprocess_raw(line),
                ctxt=RowContext(
                origin_csv  = path,
                origin_vid  = path.with_suffix('.mkv'),
                lineno      = i,
                raw_row     = line
        )) 
            for i,  line in enumerate(f.readlines())
            if      line != ''
        )


def process_dir (root: Path = DATA_ROOT)   -> Generator[ParsedEvent, None, None]:
    yield from pipe(
        root,
        iter_csv,
        map(process_file),
        concatv
    )

if __name__ == "__main__":
    events      = list(process_dir())
    bad_events  = [e for e in events if e.exc is not    None]
    good_events = [e for e in events if e.exc is        None]
    pass
