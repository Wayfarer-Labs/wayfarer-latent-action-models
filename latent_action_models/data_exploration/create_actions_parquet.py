from __future__ import annotations

import  polars as pl
from    pathlib     import Path
from    toolz       import first


from latent_action_models.data_exploration import cache

RAW_PARQUET  = Path(first(cache.__path__)) / "gta_events.parquet"
OUT_PARQUET  = Path(first(cache.__path__)) / "gta_frames.parquet"


def resolve(df: pl.DataFrame, prefix: str, field: str, sep: str = '.') -> pl.Expr:
    if (flat_name := f"{prefix}{sep}{field}") in df.columns:                return pl.col(flat_name)
    if prefix in df.columns and isinstance(df.schema[prefix], pl.Struct):   return pl.col(prefix).struct.field(field)                # real struct
    raise KeyError(f"cannot find {prefix}.{field}")


def flatten(df: pl.DataFrame) -> pl.DataFrame:
    """Project out the handful of columns we care about and drop the rest."""
    mapping = [
        ("video_metadata", "video_path",         "video_path"),
        ("video_metadata", "start",              "vid_start"),
        ("video_metadata", "fps",                "fps"),
        ("action_data",    "t_sec",              "t_sec"),
        ("action_data",    "dx",                 "dx"),
        ("action_data",    "dy",                 "dy"),
        ("action_data",    "wheel",              "wheel"),
        ("action_data",    "keys_active",        "keys_active"),
        ("action_data",    "key_pressed",        "key_pressed"),
        ("action_data",    "key_pressed_dir",    "key_pressed_dir"),
    ]
    exprs = [resolve(df, src, fld).alias(dst) for src, fld, dst in mapping]
    drop  = [c for c in df.columns if c.startswith("video_metadata") or
                                   c.startswith("action_data")]
    return df.with_columns(exprs).select(pl.exclude(drop))


def all_keys(df: pl.DataFrame) -> list[str]:
    """Return the universe of unique key/button names."""
    return (df.select("keys_active")
              .explode("keys_active")
              .unique()
              .drop_nulls()
              .to_series()
              .to_list())


def with_running_state(sub: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    for k in keys:
        down         = ((pl.col("key_pressed") == k) &
                        (pl.col("key_pressed_dir") == "down")).cast(pl.Int8)
        up           = ((pl.col("key_pressed") == k) &
                        (pl.col("key_pressed_dir") == "up"  )).cast(pl.Int8)
        delta_alias  = f"__delta_{k}"
        sub          = (
            sub
            .with_columns((down - up).alias(delta_alias))
            .with_columns(
                pl.cum_sum(delta_alias)                       # running count
                  .clip(lower_bound=0, upper_bound=1)        # force 0/1
                  .alias(k)
            )
            .drop(delta_alias)
        )
    return sub


def aggregate_frames(sub: pl.DataFrame, video: str, spf: float, keys: list[str]) -> pl.DataFrame:
    sub = with_running_state(sub, keys)
    return (
        sub.group_by("frame_idx")
           .agg([
               pl.col("dx")   .sum(),
               pl.col("dy")   .sum(),
               pl.col("wheel").sum(),
               *[pl.col(k).last() for k in keys],
           ])
           .with_columns([
               pl.lit(video)                                       .alias("video_path"),
               (pl.col("frame_idx") * spf + sub["vid_start"][0]) .alias("t_sec"),
           ])
    )


def process_video(video_key, sub: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    video = first(video_key) if isinstance(video_key, (list, tuple)) else video_key
    fps   = sub["fps"][0] or 29.97
    spf   = 1.0 / fps

    sub = (sub
           .with_columns(
               ((pl.col("t_sec") - pl.col("vid_start")) / spf)
                 .round()
                 .cast(pl.Int64)
                 .alias("frame_idx"))
           .sort("frame_idx"))

    return aggregate_frames(sub, video, spf, keys)


def build_frame_table(df: pl.DataFrame) -> pl.DataFrame:
    keys   = all_keys(df)
    frames = [process_video(vk, sub, keys) for vk, sub in df.group_by("video_path")]
    return pl.concat(frames)


def main() -> None:
    df        = pl.read_parquet(RAW_PARQUET)
    flat      = flatten(df)
    frame_df  = build_frame_table(flat)

    frame_df.write_parquet(OUT_PARQUET, compression="zstd")
    print(
        f"Wrote {OUT_PARQUET} with {len(frame_df):,} rows "
        f"and {len(frame_df.columns) - 5} key columns."
    )


if __name__ == "__main__":
    from latent_action_models.data_exploration.create_raw_parquet import to_parquet, get_dataframe_rows

    # to_parquet(get_dataframe_rows(file_limit=None), out_path=RAW_PARQUET)

    main()
