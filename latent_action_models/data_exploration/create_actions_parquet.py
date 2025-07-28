from __future__ import annotations

import polars as pl
from pathlib import Path
from toolz   import first

from latent_action_models.data_exploration import cache


RAW_PARQUET         = Path(first(cache.__path__)) / "gta_events.parquet"
OUT_PARQUET         = Path(first(cache.__path__)) / "gta_frames_clean.parquet"
OUT_PARQUET_SCHEMA  = {
    'frame_idx': pl.Int64, 'dx': pl.Float64, 'dy': pl.Float64, 'wheel': pl.Float64,
    'keypress_a': pl.Int64, 'keypress_d': pl.Int64, 'keypress_mouse1': pl.Int64,
    'keypress_mouse2': pl.Int64, 'keypress_s': pl.Int64, 'keypress_w': pl.Int64, 'keypress_shift': pl.Int64,
    'video_path': pl.Utf8, 't_sec': pl.Float64
}
FINAL_COLUMN_ORDER = list(OUT_PARQUET_SCHEMA.keys())

ALLOWED_KEYS   = {"w", "a", "s", "d", "mouse1", "mouse2", "shift"}
# JUNK_PATTERNS  = ("pos_x:", "wheelmouse")
KEY_PREFIX     = "keypress_"
DEFAULT_FPS    = 29.97


def resolve(df: pl.DataFrame, prefix: str, field: str) -> pl.Expr:
    # This helper is fine as-is
    flat1 = f"{prefix}_{field}"
    flat2 = f"{prefix}.{field}"
    if flat1 in df.columns:                                   return pl.col(flat1)
    if flat2 in df.columns:                                   return pl.col(flat2)
    if prefix in df.columns and isinstance(df.schema[prefix], pl.Struct):
        return pl.col(prefix).struct.field(field)
    raise KeyError(f"cannot find {prefix}.{field}")


def flatten_events(df: pl.DataFrame) -> pl.DataFrame:
    # This function is mostly the same as before.
    mapping = [
        ("video_metadata", "video_path",      "video_path"),
        ("video_metadata", "start",           "vid_start"),
        ("video_metadata", "fps",             "fps"),
        ("action_data",    "t_sec",           "t_sec"),
        ("action_data",    "dx",              "dx"),
        ("action_data",    "dy",              "dy"),
        ("action_data",    "wheel",           "wheel"),
        ("action_data",    "keys_active",     "keys_active"),
        # We don't really need these two anymore, but it's safe to leave them
        ("action_data",    "key_pressed",     "key_pressed"),
        ("action_data",    "key_pressed_dir", "key_pressed_dir"),
    ]
    
    # The problematic filter line has been REMOVED from here.
    
    cols  = [resolve(df, s, f).alias(n) for s, f, n in mapping]
    drops = [c for c in df.columns if c.startswith("video_metadata") or c.startswith("action_data")]
    
    flat_df = df.with_columns(cols).select(pl.exclude(drops))

    # Normalize the keys_active list to be all lowercase.
    return flat_df.with_columns(
        pl.col("keys_active").list.eval(pl.element().str.to_lowercase())
    )

def discover_keys(df: pl.DataFrame) -> list[str]:
    # This function is fine as-is
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


def aggregate_one_video(sub: pl.DataFrame, video: str, spf: float, keys: list[str]) -> pl.DataFrame:
    # This function is completely rewritten to be robust and produce dense output.
    
    # 1. Aggregate all events by frame. For each frame, we sum the mouse deltas
    #    and find the `keys_active` list from the very last event in that frame.
    sparse_agg = (
        sub.group_by("frame_idx")
           .agg([
               pl.col("dx").sum(),
               pl.col("dy").sum(),
               pl.col("wheel").sum().cast(pl.Float64),
               pl.col("keys_active").sort_by(pl.col("t_sec")).last().alias("final_keys"),
           ])
           .sort("frame_idx")
    )
    
    # 2. Create the final keypress columns based on the `final_keys` list.
    #    For each key, we check if it's present in the list for that frame.
    key_state_cols = [
        pl.col("final_keys").list.contains(k).cast(pl.Int64).alias(KEY_PREFIX + k)
        for k in keys
    ]
    sparse_agg_with_keys = sparse_agg.with_columns(key_state_cols).drop("final_keys")

    # 3. Create a dense timeline to ensure we have a row for every single frame,
    #    even those with no events.
    min_frame, max_frame = sub["frame_idx"].min(), sub["frame_idx"].max()
    if min_frame is None or max_frame is None:
        return pl.DataFrame(schema=OUT_PARQUET_SCHEMA).select(FINAL_COLUMN_ORDER)
    
    dense_frames = pl.DataFrame({"frame_idx": range(min_frame, max_frame + 1)})

    # 4. Use `join_asof` to map the aggregated states onto the dense timeline.
    #    This correctly propagates the last known state to frames with no events.
    dense_agg = dense_frames.join_asof(sparse_agg_with_keys, on="frame_idx")

    # 5. Add metadata and enforce final schema and column order.
    final_df = dense_agg.with_columns([
        pl.lit(video).alias("video_path"),
        (pl.col("frame_idx") * spf + sub["vid_start"][0]).alias("t_sec"),
    ])

    final_df = final_df.with_columns([
        pl.col(c).fill_null(0.0) if final_df[c].dtype.is_float() else pl.col(c).fill_null(0)
        for c in final_df.columns
        if c not in ["frame_idx", "video_path"] # Don't fill these columns
    ])

    return final_df.select(FINAL_COLUMN_ORDER)



def process_video(group_key, sub: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    # This function is simplified, as the complex logic is now in aggregate_one_video
    video = group_key[0] if isinstance(group_key, (list, tuple)) else group_key
    fps   = sub["fps"][0] or DEFAULT_FPS
    spf   = 1.0 / fps
    
    # Add frame_idx to the event stream
    sub = sub.with_columns(
        ((pl.col("t_sec") - pl.col("vid_start")) / spf)
            .round()
            .cast(pl.Int64)
            .alias("frame_idx")
    )
    
    if sub.is_empty():
        return pl.DataFrame(schema=OUT_PARQUET_SCHEMA)
        
    return aggregate_one_video(sub, video, spf, keys)


def build_frames(df: pl.DataFrame) -> pl.DataFrame:
    # This function is mostly the same
    keys   = discover_keys(df)
    frames = [process_video(vk, sub, keys) for vk, sub in df.group_by("video_path")]
    return pl.concat(frames)


def main() -> None:
    print("⏳ reading", RAW_PARQUET)
    raw = pl.read_parquet(RAW_PARQUET)

    flat = flatten_events(raw)
    frame_df = build_frames(flat)

    print("\n✅ Sample of processed data:")
    print(frame_df)

    print("\n💾 Writing", OUT_PARQUET.name)
    # Ensure final columns are in the desired order before writing
    final_cols = ['frame_idx', 'dx', 'dy', 'wheel'] + sorted([c for c in frame_df.columns if c.startswith(KEY_PREFIX)]) + ['video_path', 't_sec']
    frame_df.select(final_cols).write_parquet(OUT_PARQUET, compression="zstd")
    print("✔  rows:", len(frame_df), "  cols:", len(frame_df.columns))

if __name__ == "__main__":
    main()