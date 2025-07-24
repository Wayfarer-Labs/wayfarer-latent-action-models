import  traceback
from    tqdm            import tqdm
from    pathlib         import Path
from    multimethod     import multimethod, parametric
from    typing          import Iterable, Literal, Optional, Generator, TypedDict, Callable
from    dataclasses     import dataclass, field, asdict
from    toolz           import excepts as _excepts, pipe, map, last, first
from    toolz.curried   import map, concatv, mapcat
from    functools       import wraps, singledispatch


DATA_ROOT       = Path("/mnt/data/shahbuland/video-proc-2/datasets/gta_nas")
FILE_PROGRESS_BAR    = tqdm(desc='Processing files...')
LINE_PROGRESS_BAR    = tqdm(desc='Processing lines...')


def excepts(exc: type(Exception), handler: Callable):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return _excepts(exc, fn,
                            lambda _: handler(*args, **kwargs))(*args, **kwargs)
        return wrapper
    return decorator


def iter_csv(root: Path = DATA_ROOT) -> Generator[Path, None, None]:
    yield from root.glob('*.csv')

@dataclass()
class RowContext:
    origin_csv:     Path
    origin_vid:     Path
    lineno:         int
    raw_row:        str


@dataclass()
class ParsedEvent:
    origin_csv:         Path
    origin_vid:         Path
    lineno:             int
    raw:                str                                 = field(repr=False)
    t_sec:              Optional[float]                     = None
    event_type:         Optional[Literal["mouse", "key"]]   = None
    keys_active:        Optional[list[str]]                 = None
    key_pressed:        Optional[str]                       = None
    key_pressed_dir:    Optional[Literal['up', 'down']]     = None
    transition:         Optional[str]                       = None
    dx:                 Optional[int]                       = None
    dy:                 Optional[int]                       = None
    wheel:              Optional[float]                     = None
    exc:                Optional[str]                       = None

def is_keypress     (raw_row: str)  -> bool:
    return any(('down'   in last(raw_row.split(',')),
                'up'     in last(raw_row.split(','))))

def is_mouse_delta  (raw_row: str)  -> bool:
    return 'mouse event' in last(raw_row.split(','))


MouseDeltaRow   = parametric(str, is_mouse_delta)
KeypressRow     = parametric(str, is_keypress)


@multimethod
def _parse_row      (raw_row: str,      ctxt: RowContext) -> ParsedEvent:
    raise Exception(f'Cannot detect event: \"{raw_row}\" - {ctxt}')


def _from_ctxt(raw_row: str, ctxt: RowContext) -> ParsedEvent:
    return ParsedEvent( origin_csv  = ctxt.origin_csv,
                        origin_vid  = ctxt.origin_vid,
                        lineno      = ctxt.lineno,
                        raw         = ctxt.raw_row,
                        exc         = exc   if (exc := traceback.format_exc()) != 'NoneType: None\n'
                                            else None)


def _first_member(s: Iterable[str], seek: str) -> Optional[str]:
    return first(c for c in s if seek in c)


@excepts(Exception, _from_ctxt)
@multimethod
def _parse_row(raw_row: MouseDeltaRow, ctxt: RowContext) -> ParsedEvent:
    timestamp, data, _  = raw_row.split(',')
    dx, dy, wheel       = tuple(_first_member(data.split(';'), key)
                                for key in ('pos_x', 'pos_y', 'wheel'))

    if None in (dx, dy, wheel):         print(f'[!!] ({dx=} {dy=} {wheel=})')

    sign_x, sign_y        = 1, 1
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


@excepts(Exception, _from_ctxt)
@multimethod
def _parse_row(raw_row: KeypressRow, ctxt: RowContext) -> ParsedEvent:
    (   timestamp,
        keys_raw,
        keypresses ) = raw_row.split(',')

    if 'none' in keys_raw.lower():  keys_active = []
    else:                           keys_active = keys_raw.split('+')

    key_pressed, key_pressed_dir    = keypresses.lower().split(' ')

    event                           = _from_ctxt(ctxt)
    event.event_type                = 'key'
    event.t_sec                     = float(timestamp)
    event.keys_active               = keys_active
    event.key_pressed               = key_pressed
    event.key_pressed_dir           = key_pressed_dir
    return event


def _preprocess_raw (raw_row: str)  -> str: return raw_row.strip().lower()

def process_file(path: Path)        -> Generator[ParsedEvent, None, None]:
    with open(path, 'r') as f:
        FILE_PROGRESS_BAR.set_description(f'Processing files... {path} - lines: {len(f.readlines())}') ; f.seek(0)
        yield from (
            _parse_row(
                _preprocess_raw(line),
                RowContext(
                    origin_csv  = path,
                    origin_vid  = path.with_suffix('.mkv'),
                    lineno      = i,
                    raw_row     = line
                )
            ) 
            for i,  line in enumerate(f.readlines())
            if     (line != '') ^ bool(LINE_PROGRESS_BAR.update(1))
        )

        FILE_PROGRESS_BAR.update(1)


def process_dir (root: Path = DATA_ROOT)   -> Generator[ParsedEvent, None, None]:
    yield from pipe(
        root,
        iter_csv,
        mapcat(process_file)
    )

if __name__ == "__main__":
    events      = list(process_dir())
    bad_events  = [e for e in events if e.exc is not    None]
    good_events = [e for e in events if e.exc is        None]
    pass
