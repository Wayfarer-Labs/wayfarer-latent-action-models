from __future__ import annotations

import polars as pl
from pathlib import Path
from toolz   import first


from latent_action_models.data_exploration import cache

RAW_PARQUET    = Path(first(cache.__path__)) / "gta_events.parquet"
OUT_PARQUET    = Path(first(cache.__path__)) / "gta_frames_clean.parquet"

ALLOWED_KEYS   = {"w", "a", "s", "d", "mouse1", "mouse2"}
JUNK_PATTERNS  = ("pos_x:", "wheelmouse")     # filter these out early
KEY_PREFIX     = "keypress_"                  # final column prefix
DEFAULT_FPS    = 29.97


def resolve(df: pl.DataFrame, prefix: str, field: str) -> pl.Expr:
    flat1 = f"{prefix}_{field}"
    flat2 = f"{prefix}.{field}"
    if flat1 in df.columns:                                   return pl.col(flat1)
    if flat2 in df.columns:                                   return pl.col(flat2)
    if prefix in df.columns and isinstance(df.schema[prefix], pl.Struct):
        return pl.col(prefix).struct.field(field)
    raise KeyError(f"cannot find {prefix}.{field}")


def cumulative_sum(expr: pl.Expr) -> pl.Expr:
    return getattr(expr, "cumsum", lambda: pl.cum_sum(expr))()


def flatten_events(df: pl.DataFrame) -> pl.DataFrame:
    mapping = [  # (struct, field, new_name)
        ("video_metadata", "video_path",      "video_path"),
        ("video_metadata", "start",           "vid_start"),
        ("video_metadata", "fps",             "fps"),
        ("action_data",    "t_sec",           "t_sec"),
        ("action_data",    "dx",              "dx"),
        ("action_data",    "dy",              "dy"),
        ("action_data",    "wheel",           "wheel"),
        ("action_data",    "keys_active",     "keys_active"),
        ("action_data",    "key_pressed",     "key_pressed"),
        ("action_data",    "key_pressed_dir", "key_pressed_dir"),
    ]
    cols  = [resolve(df, s, f).alias(n) for s, f, n in mapping]
    drops = [c for c in df.columns if c.startswith("video_metadata")
                                   or c.startswith("action_data")]
    return df.with_columns(cols).select(pl.exclude(drops))

def discover_keys(df: pl.DataFrame) -> list[str]:
    observed = (df.select("keys_active")
                  .explode("keys_active")
                  .drop_nulls()
                  .to_series()
                  .str.to_lowercase()
                  .unique()
                  .to_list())

    missing = sorted(ALLOWED_KEYS - set(observed))
    if missing: print("⚠️ expected keys never seen in data:", ", ".join(missing))
    return sorted(ALLOWED_KEYS)


def add_running_state(sub: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    for k in keys:
        press  = ((pl.col("key_pressed") == k) & (pl.col("key_pressed_dir") == "down")).cast(pl.Int8)
        rels   = ((pl.col("key_pressed") == k) & (pl.col("key_pressed_dir") == "up"  )).cast(pl.Int8)
        delta  = press - rels
        tmp    = f"__delta_{k}"
        sub    = (
            sub.with_columns(delta.alias(tmp))
               .with_columns(
                   pl.col(tmp).cum_sum().clip(0, 1).fill_null(0).alias(KEY_PREFIX + k)
               )
               .drop(tmp)
        )
    return sub

def aggregate_one_video(sub: pl.DataFrame, video: str, spf: float, keys: list[str]) -> pl.DataFrame:
    sub = add_running_state(sub, keys)
    return (
        sub.group_by("frame_idx")
           .agg([
               pl.col("dx")   .sum(),
               pl.col("dy")   .sum(),
               pl.col("wheel").sum(),
               *[pl.col(KEY_PREFIX + k).last() for k in keys],
           ])
           .with_columns([
               pl.lit(video).alias("video_path"),
               (pl.col("frame_idx") * spf + sub["vid_start"][0]).alias("t_sec"),
           ])
    )

def process_video(group_key, sub: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    video = group_key[0] if isinstance(group_key, (list, tuple)) else group_key
    fps   = sub["fps"][0] or DEFAULT_FPS
    spf   = 1.0 / fps
    sub   = (
        sub.with_columns(
            ((pl.col("t_sec") - pl.col("vid_start")) / spf)
              .round()
              .cast(pl.Int64)
              .alias("frame_idx"))
           .sort("frame_idx")
    )
    return aggregate_one_video(sub, video, spf, keys)


def build_frames(df: pl.DataFrame) -> pl.DataFrame:
    keys   = discover_keys(df)
    frames = [process_video(vk, sub, keys) for vk, sub in df.group_by("video_path")]
    return pl.concat(frames)


def main() -> None:
    print("⏳ reading", RAW_PARQUET)
    raw        = pl.read_parquet(RAW_PARQUET)
    flat       = flatten_events(raw)
    frame_df   = build_frames(flat)

    print("💾 writing", OUT_PARQUET.name)
    frame_df.write_parquet(OUT_PARQUET, compression="zstd")
    print("✔  rows:", len(frame_df), "  cols:", len(frame_df.columns))

if __name__ == "__main__":
    main()
