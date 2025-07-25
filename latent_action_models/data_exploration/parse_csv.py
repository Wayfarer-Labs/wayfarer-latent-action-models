from __future__ import annotations

import  traceback
import  json
from    tqdm            import tqdm
from    pathlib         import Path
from    multimethod     import multimethod
from    typing          import Literal, Optional, Generator
from    dataclasses     import dataclass, field, asdict
from    toolz           import curry, pipe, last, identity
from    toolz.curried   import mapcat, take

from    latent_action_models.data_exploration.utils import (
    iter_csv, _first_member,
    MouseDeltaRow, KeypressRow
)

DATA_ROOT         = Path("/mnt/data/shahbuland/video-proc-2/datasets/gta_nas")
FILE_PROGRESS_BAR = tqdm(desc='Processing files...')
LINE_PROGRESS_BAR = tqdm(desc='Processing lines...')


@dataclass
class RowContext:
    origin_csv:         str
    origin_vid:         str
    lineno:             int
    raw_row:            str


@dataclass
class ParsedEvent:
    origin_csv:         str
    origin_vid:         str
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



def file_context(context: RowContext) -> ParsedEvent:
    return ParsedEvent( origin_csv  = context.origin_csv,
                        origin_vid  = context.origin_vid,
                        lineno      = context.lineno,
                        raw         = context.raw_row,
                        exc         = exc   if (exc := traceback.format_exc()) != 'NoneType: None\n'
                                            else None)

def _preprocess_raw (raw_row: str) -> str:
    return raw_row.strip().lower()

@multimethod
def _parse_row(raw_row: str, context: RowContext) -> ParsedEvent:
    raise Exception(f'Cannot detect event: \"{raw_row}\" - {context}')


@multimethod
def _parse_row(raw_row: MouseDeltaRow, context: RowContext) -> ParsedEvent:
    timestamp, data, _  = raw_row.split(',')
    dx, dy, wheel       = tuple(_first_member(data.split(';'), key)
                                for key in ('pos_x', 'pos_y', 'wheel'))

    sign_x, sign_y        = 1, 1
    if '-' in dx: sign_x *= -1
    if '-' in dy: sign_y *= -1
    
    dx, dy, wheel       = tuple(last(e.split(':')).strip('+-')
                                for  e in (dx, dy, wheel))
    dx, dy, wheel       = tuple(float(e)
                                for  e in (dx, dy, wheel))

    event               = file_context(context)
    event.event_type    = 'mouse'
    event.t_sec         = float(timestamp)
    (   event.dx,
        event.dy,
        event.wheel )   = dx, dy, wheel

    return event

@multimethod
def _parse_row(raw_row: KeypressRow, context: RowContext) -> ParsedEvent:
    (   timestamp,
        keys_raw,
        keypresses ) = raw_row.split(',')

    if 'none' in keys_raw.lower():  keys_active = []
    else:                           keys_active = keys_raw.split('+')

    key_pressed, key_pressed_dir    = keypresses.lower().split(' ')

    event                           = file_context(context)
    event.event_type                = 'key'
    event.t_sec                     = float(timestamp)
    event.keys_active               = keys_active
    event.key_pressed               = key_pressed
    event.key_pressed_dir           = key_pressed_dir
    return event


def process_file(path: Path)        -> Generator[ParsedEvent, None, None]:
    with open(path, 'r') as f:
        FILE_PROGRESS_BAR.set_description(
            f'Processing files... {path} - lines: {len(f.readlines())}')
        f.seek(0)
        
        for i, line in enumerate(f):

            if line in ('', '\n'): continue

            context = RowContext(
                origin_csv  = str(path),
                origin_vid  = str(vid_path := path.with_suffix('.mkv')),
                lineno      = i,
                raw_row     = line
            )

            if not vid_path.exists(): continue

            try:    yield _parse_row(_preprocess_raw(line), context)
            except: yield file_context(context)

            LINE_PROGRESS_BAR.update(1)

        FILE_PROGRESS_BAR.update(1)


@curry
def process_dir (root: Path, *, limit = None)   -> Generator[ParsedEvent, None, None]:
    yield from pipe(
        root,
        iter_csv,
        take(limit) if limit else identity,
        mapcat(process_file)
    )


if __name__ == "__main__":
    limit       = 1
    total_lines = sum(len(open(f).readlines()) for f in iter_csv(DATA_ROOT))
    print(f'Total lines: {total_lines}')
    events      = list(process_dir(DATA_ROOT, limit=None))
    bad_events  = [e for e in events if e.exc is not    None]
    print(f'Error rate with limit {limit}: {len(bad_events) / len(events) * 100}%')
    good_events = [e for e in events if e.exc is        None]

    with open('events.jsonl', 'w+') as f:
        for e in good_events:
            f.write(json.dumps(asdict(e)) + '\n')

    with open('bad_events.jsonl', 'w+') as f:
        for e in bad_events:
            f.write(json.dumps(asdict(e)) + '\n')
