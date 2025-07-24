from   __future__ import annotations
import csv, re, sys, pathlib
from   collections import namedtuple

DATA_ROOT = pathlib.Path("/mnt/data/shahbuland/video-proc-2/datasets/gta_nas")

# ── regexes ──────────────────────────────────────────────────────────────
KEY_RE   = re.compile(r"^(?P<key>[A-Z0-9]+) (?P<state>down|up)$")
MOUSE_RE = re.compile(
    r"^pos_x:\s*(?P<dx>[+-]?\d+);\s*pos_y:\s*(?P<dy>[+-]?\d+);\s*wheel:\s*(?P<wheel>[+-]?\d+\.\d+)$"
)

ParsedEvent = namedtuple("ParsedEvent",
    "t_sec event_type keys_active key transition dx dy wheel raw"
)

def parse_row(ts: str, col1: str, col2: str) -> ParsedEvent | None:
    """Return a ParsedEvent or None if the line cannot be interpreted."""
    ts, col1, col2 = ts.strip(), col1.strip(), col2.strip()

    # Guard: timestamp must be convertible to float --------------------------------
    if not ts:
        return None
    try:
        t_sec = float(ts)
    except ValueError:
        return None

    # Mouse event ------------------------------------------------------------------
    if col2.lower().startswith("mouse"):
        m = MOUSE_RE.match(col1)
        if not m:
            return None
        return ParsedEvent(
            t_sec       = t_sec,
            event_type  = "mouse",
            keys_active = None,
            key         = None,
            transition  = None,
            dx          = int(m["dx"]),
            dy          = int(m["dy"]),
            wheel       = float(m["wheel"]),
            raw         = f"{col1} | {col2}"
        )

    # Key event --------------------------------------------------------------------
    m = KEY_RE.match(col2)
    if not m:
        return None
    keys_active = None if col1 == "None" else col1.split('+')
    return ParsedEvent(
        t_sec       = t_sec,
        event_type  = "key",
        keys_active = keys_active,
        key         = m["key"],
        transition  = m["state"],
        dx          = 0,
        dy          = 0,
        wheel       = 0.0,
        raw         = f"{col1} | {col2}")


# ── file‑level routine ───────────────────────────────────────────────────

def parse_file(path: pathlib.Path, preview: int = 10):
    good, bad = [], []
    with path.open(newline='', encoding='utf-8', errors='replace') as fh:
        reader = csv.reader(fh)
        for row_no, row in enumerate(reader, 1):
            if len(row) != 3:
                bad.append((row_no, row))
                continue
            evt = parse_row(*row)
            (good if evt else bad).append(evt or (row_no, row))

    total = len(good) + len(bad)
    pct   = 100 * len(bad) / total if total else 0
    print(f"{'[!] ' if bad else '[✓] '}"  # visual cue
          f"{path.name} → {len(bad):,}/{total:,} failed ({pct:.2f}%)")
    if bad:
        print(f"     • showing first {min(preview,len(bad))} failures")
        for row_no, raw in bad[:preview]:
            print(f"       line {row_no}: {raw}")
    return len(good), len(bad)

# ── directory walker ─────────────────────────────────────────────────────

def main():
    total_good = total_bad = 0
    for csv_path in sorted(DATA_ROOT.rglob("*.csv")):
        good, bad = parse_file(csv_path)
        total_good += good
        total_bad  += bad

    grand_total = total_good + total_bad
    print("\n―――― aggregate ――――")
    if total_bad:
        pct = 100 * total_bad / grand_total
        print(f"Parsed {total_good:,}/{grand_total:,} lines successfully; "
              f"{total_bad:,} ({pct:.2f}%) still failing.")
    else:
        print(f"All {grand_total:,} lines parsed successfully — nice!")

if __name__ == "__main__":
    main()
