from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN

try:
    import polars as pl
except Exception:
    pl = None


RAW_SAMPLE_RATE = 30
HEADER_ROWS_TO_SKIP = 11
RAW_COLUMN_NAMES = ["x", "y", "z"]


def samples_per_minute(fs: float) -> int:
    return max(1, int(round(fs * 60)))


def samples_per_hour(fs: float) -> int:
    return samples_per_minute(fs) * 60


def samples_for_minutes(fs: float, minutes: float) -> int:
    return max(1, int(round(fs * 60 * minutes)))


def infer_sample_rate_from_datetime(datetime_series: pd.Series) -> float | None:
    timestamps = pd.to_datetime(datetime_series, errors="coerce").dropna().sort_values()
    if len(timestamps) < 2:
        return None

    delta_seconds = timestamps.diff().dt.total_seconds().dropna()
    if delta_seconds.empty:
        return None

    step_seconds = float(delta_seconds.median())
    if step_seconds <= 0:
        return None
    return 1.0 / step_seconds


def read_raw_parquet(parquet_path: Path) -> tuple[pd.DataFrame, float, str]:
    parquet_columns = ["datetime", "accelerometer_x", "accelerometer_y", "accelerometer_z"]
    required_columns = {"accelerometer_x", "accelerometer_y", "accelerometer_z"}

    if pl is not None:
        df = pl.read_parquet(parquet_path, columns=parquet_columns)
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"{parquet_path.name}: missing required parquet columns: {sorted(missing_columns)}"
            )

        selected = df.select(
            [
                "datetime",
                pl.col("accelerometer_x").alias("x"),
                pl.col("accelerometer_y").alias("y"),
                pl.col("accelerometer_z").alias("z"),
            ]
        )
        pandas_df = pd.DataFrame(selected.to_dict(as_series=False))
    else:
        try:
            pandas_df = pd.read_parquet(parquet_path, columns=parquet_columns)
        except Exception as exc:
            raise ImportError(
                "Reading parquet files requires either polars or a pandas parquet engine "
                "(pyarrow or fastparquet)."
            ) from exc

        missing_columns = required_columns - set(pandas_df.columns)
        if missing_columns:
            raise ValueError(
                f"{parquet_path.name}: missing required parquet columns: {sorted(missing_columns)}"
            )

        pandas_df = pandas_df.rename(
            columns={
                "accelerometer_x": "x",
                "accelerometer_y": "y",
                "accelerometer_z": "z",
            }
        )

    fs = infer_sample_rate_from_datetime(pandas_df["datetime"]) if "datetime" in pandas_df.columns else None
    inferred_fs = fs or RAW_SAMPLE_RATE
    return pandas_df[["x", "y", "z"]].copy(), inferred_fs, "raw"


def read_actigraphy_csv(csv_path: Path) -> tuple[pd.DataFrame, float, str]:
    preview = pd.read_csv(csv_path, nrows=5)
    preview_columns = {str(col).strip() for col in preview.columns}

    aggregated_avg_cols = {
        "accelerometer_x_avg",
        "accelerometer_y_avg",
        "accelerometer_z_avg",
    }

    if aggregated_avg_cols.issubset(preview_columns):
        df = pd.read_csv(csv_path)
        df.columns = [str(col).strip() for col in df.columns]
        renamed = df.rename(
            columns={
                "accelerometer_x_avg": "x",
                "accelerometer_y_avg": "y",
                "accelerometer_z_avg": "z",
            }
        )
        fs = infer_sample_rate_from_datetime(renamed["datetime"]) if "datetime" in renamed.columns else None
        inferred_fs = fs or (1.0 / 60.0)
        label = "30-second aggregated" if inferred_fs >= (1.0 / 30.0) else "aggregated"
        return renamed[["x", "y", "z"]].copy(), inferred_fs, label

    df = pd.read_csv(csv_path, skiprows=HEADER_ROWS_TO_SKIP, header=None)
    df = df.iloc[:, :3].copy()
    df.columns = RAW_COLUMN_NAMES
    return df, RAW_SAMPLE_RATE, "raw"


def read_actigraphy_file(path: Path) -> tuple[pd.DataFrame, float, str]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return read_raw_parquet(path)
    if suffix == ".csv":
        return read_actigraphy_csv(path)
    raise ValueError(f"Unsupported file type '{path.suffix}' for {path.name}. Use .csv or .parquet.")


def select_window(df: pd.DataFrame, start: int, window: int) -> pd.DataFrame:
    stop = min(start + window, len(df))
    if start < 0 or start >= len(df) or stop <= start:
        raise ValueError(
            f"Requested window [{start}, {start + window}) is outside the data range "
            f"for {len(df)} rows."
        )
    return df.iloc[start:stop].reset_index(drop=True)


def build_3d_line_figure(df: pd.DataFrame, title: str, color: str = "#1f77b4") -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df["x"],
                y=df["y"],
                z=df["z"],
                mode="lines",
                line={"width": 2, "color": color},
                name=title,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene={"xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def build_cluster_figure(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    cluster_ids = sorted(df["cluster"].unique())

    for cluster_id in cluster_ids:
        cluster_df = df[df["cluster"] == cluster_id]
        name = "noise" if cluster_id == -1 else f"cluster {cluster_id}"
        color = "#7f7f7f" if cluster_id == -1 else None
        fig.add_trace(
            go.Scatter3d(
                x=cluster_df["x"],
                y=cluster_df["y"],
                z=cluster_df["z"],
                mode="lines",
                line={"width": 4, **({"color": color} if color else {})},
                name=name,
            )
        )

    fig.update_layout(
        title=title,
        scene={"xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def choose_temporal_radius_seconds(fs_after_downsampling: float) -> float:
    if fs_after_downsampling >= 1:
        return 1.0
    return 60.0


def choose_downsample_factor(input_kind: str, requested_factor: int | None) -> int:
    if requested_factor is not None:
        return max(1, requested_factor)
    if input_kind == "raw":
        return 6
    return 1


def choose_min_samples(fs_after_downsampling: float, requested_min_samples: int | None) -> int:
    if requested_min_samples is not None:
        return max(1, requested_min_samples)

    if fs_after_downsampling < 1:
        return max(2, samples_per_minute(fs_after_downsampling))
    return max(1, int(round(fs_after_downsampling)))


def choose_cluster_window_minutes(input_kind: str, requested_minutes: float | None) -> float:
    if requested_minutes is not None:
        return max(0.25, requested_minutes)
    return 60.0


def start_index_from_time(fs: float, start_hour: float, start_minute: float) -> int:
    total_minutes = (start_hour * 60.0) + start_minute
    if total_minutes < 0:
        raise ValueError("Start time must be zero or positive.")
    return max(0, int(round(total_minutes * 60.0 * fs)))


def parse_window_hours(window_hours_text: str | None, default_start_hour: float) -> list[float]:
    if not window_hours_text or not window_hours_text.strip():
        return [default_start_hour]

    hours = []
    for part in window_hours_text.split(","):
        value = part.strip()
        if not value:
            continue
        try:
            hour = float(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid value '{value}' in --window-hours. Use a comma-separated list like 2,6,12."
            ) from exc
        if hour < 0:
            raise ValueError("--window-hours values must be zero or positive.")
        hours.append(hour)

    if not hours:
        return [default_start_hour]
    return hours


def format_hour_token(start_hour: float, start_minute: float) -> str:
    hour_text = f"{start_hour:g}".replace(".", "p")
    minute_text = f"{start_minute:g}".replace(".", "p")
    return f"{hour_text}h_{minute_text}m"


def save_window_outputs(
    output_dir: Path,
    window_label: str,
    first_fig: go.Figure,
    second_fig: go.Figure,
    cluster_fig: go.Figure,
) -> dict[str, Path]:
    output_files = {
        "raw_window": output_dir / f"raw_window_3d_{window_label}.html",
        "downsampled_window": output_dir / f"downsampled_window_3d_{window_label}.html",
        "clustered_window": output_dir / f"clustered_window_3d_{window_label}.html",
    }
    first_fig.write_html(output_files["raw_window"])
    second_fig.write_html(output_files["downsampled_window"])
    cluster_fig.write_html(output_files["clustered_window"])
    return output_files


def build_time_aware_features(
    df: pd.DataFrame,
    fs_after_downsampling: float,
    eps: float,
    temporal_radius_seconds: float,
) -> np.ndarray:
    xyz = df[["x", "y", "z"]].to_numpy(dtype=float, copy=True)
    elapsed_seconds = np.arange(len(df), dtype=float) / fs_after_downsampling
    time_feature = (elapsed_seconds / temporal_radius_seconds) * eps
    return np.column_stack([xyz, time_feature])


def find_temporal_split_index(labels: np.ndarray) -> int | None:
    best_split = None
    best_gap = 0

    for cluster_id in sorted(label for label in np.unique(labels) if label != -1):
        positions = np.flatnonzero(labels == cluster_id)
        if len(positions) < 2:
            continue

        gaps = np.diff(positions)
        gap_indices = np.flatnonzero(gaps > 1)
        for gap_index in gap_indices:
            left_end = int(positions[gap_index])
            right_start = int(positions[gap_index + 1])
            gap_size = right_start - left_end - 1
            if gap_size > best_gap:
                split_index = left_end + 1 + (gap_size // 2)
                if 0 < split_index < len(labels):
                    best_gap = gap_size
                    best_split = split_index

    return best_split


def remap_cluster_labels(labels: np.ndarray, start_cluster_id: int) -> tuple[np.ndarray, int]:
    remapped = np.full(len(labels), -1, dtype=int)
    next_cluster_id = start_cluster_id

    for original_cluster_id in sorted(label for label in np.unique(labels) if label != -1):
        remapped[labels == original_cluster_id] = next_cluster_id
        next_cluster_id += 1

    return remapped, next_cluster_id


def cluster_with_time_splitting(
    df: pd.DataFrame,
    eps: float,
    min_samples: int,
    fs_after_downsampling: float,
    temporal_radius_seconds: float,
    start_cluster_id: int = 0,
) -> tuple[np.ndarray, int]:
    if len(df) == 0:
        return np.array([], dtype=int), start_cluster_id

    features = build_time_aware_features(df, fs_after_downsampling, eps, temporal_radius_seconds)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)

    split_index = find_temporal_split_index(labels)
    if split_index is None or split_index <= 0 or split_index >= len(df):
        return remap_cluster_labels(labels, start_cluster_id)

    left_labels, next_cluster_id = cluster_with_time_splitting(
        df.iloc[:split_index].reset_index(drop=True),
        eps=eps,
        min_samples=min_samples,
        fs_after_downsampling=fs_after_downsampling,
        temporal_radius_seconds=temporal_radius_seconds,
        start_cluster_id=start_cluster_id,
    )
    right_labels, next_cluster_id = cluster_with_time_splitting(
        df.iloc[split_index:].reset_index(drop=True),
        eps=eps,
        min_samples=min_samples,
        fs_after_downsampling=fs_after_downsampling,
        temporal_radius_seconds=temporal_radius_seconds,
        start_cluster_id=next_cluster_id,
    )
    return np.concatenate([left_labels, right_labels]), next_cluster_id


def open_html_outputs(output_files: dict[str, Path]) -> None:
    for output_path in output_files.values():
        webbrowser.open_new_tab(output_path.resolve().as_uri())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Python equivalent of cluster_demo.R for actigraphy data."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the actigraphy input file (.csv or raw .parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cluster_demo_outputs"),
        help="Directory for the generated HTML plots.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Keep every Nth sample before clustering. Defaults to 6 for raw data and 1 for aggregated epoch files.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="DBSCAN eps parameter.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Override DBSCAN min_samples. By default, raw data uses about 1 second of samples and epoch data uses about 1 minute.",
    )
    parser.add_argument(
        "--start-hour",
        type=float,
        default=6.0,
        help="Hour offset from the start of the recording for both plots. Default: 6.",
    )
    parser.add_argument(
        "--start-minute",
        type=float,
        default=0.0,
        help="Additional minute offset from the chosen start hour. Default: 0.",
    )
    parser.add_argument(
        "--raw-window-minutes",
        type=float,
        default=1.0,
        help="Length of the raw preview window in minutes. Default: 1.",
    )
    parser.add_argument(
        "--cluster-window-minutes",
        type=float,
        default=None,
        help="Length of the downsampled clustering window in minutes. Defaults to 60 minutes to match cluster_demo.R.",
    )
    parser.add_argument(
        "--window-hours",
        type=str,
        default=None,
        help="Comma-separated hour offsets to generate multiple windows in one run, for example: 2,6,12.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the generated Plotly figures in a browser as well as saving them.",
    )
    args = parser.parse_args()

    df, fs_original, input_kind = read_actigraphy_file(args.input_path)
    downsample_factor = choose_downsample_factor(input_kind, args.downsample)
    df_downsampled = df.iloc[::downsample_factor].reset_index(drop=True)
    fs_cluster = fs_original / downsample_factor
    cluster_window_minutes = choose_cluster_window_minutes(input_kind, args.cluster_window_minutes)
    min_samples = choose_min_samples(fs_cluster, args.min_samples)
    temporal_radius_seconds = choose_temporal_radius_seconds(fs_cluster)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_hours = parse_window_hours(args.window_hours, args.start_hour)
    all_output_files = []

    for start_hour in requested_hours:
        start_idx = start_index_from_time(fs_original, start_hour, args.start_minute)
        raw_window_samples = samples_for_minutes(fs_original, args.raw_window_minutes)
        first_window = select_window(df, start=start_idx, window=raw_window_samples)
        first_fig = build_3d_line_figure(
            first_window,
            (
                f"{input_kind.title()} {args.raw_window_minutes:g}-minute window "
                f"at {start_hour:g}h {args.start_minute:g}m"
            ),
        )

        cluster_start_idx = start_index_from_time(fs_cluster, start_hour, args.start_minute)
        cluster_window_samples = samples_for_minutes(fs_cluster, cluster_window_minutes)
        cluster_window = select_window(df_downsampled, start=cluster_start_idx, window=cluster_window_samples)
        second_fig = build_3d_line_figure(
            cluster_window,
            (
                f"Downsampled {cluster_window_minutes:g}-minute window "
                f"at {start_hour:g}h {args.start_minute:g}m"
            ),
        )

        cluster_window = cluster_window.copy()
        cluster_labels, _ = cluster_with_time_splitting(
            cluster_window,
            eps=args.eps,
            min_samples=min_samples,
            fs_after_downsampling=fs_cluster,
            temporal_radius_seconds=temporal_radius_seconds,
        )
        cluster_window["cluster"] = cluster_labels
        cluster_fig = build_cluster_figure(
            cluster_window,
            (
                f"Time-aware DBSCAN clusters on downsampled {cluster_window_minutes:g}-minute window "
                f"at {start_hour:g}h {args.start_minute:g}m"
            ),
        )

        window_label = format_hour_token(start_hour, args.start_minute)
        output_files = save_window_outputs(args.output_dir, window_label, first_fig, second_fig, cluster_fig)
        all_output_files.append(output_files)

        print(f"Saved raw window plot to: {output_files['raw_window']}")
        print(f"Saved downsampled window plot to: {output_files['downsampled_window']}")
        print(f"Saved clustered window plot to: {output_files['clustered_window']}")
        print(
            "Cluster counts:\n"
            f"{cluster_window['cluster'].value_counts().sort_index().to_string()}"
        )

    print(f"Detected input kind: {input_kind}")
    print(f"Detected sample rate: {fs_original:.6f} samples/sec")
    print(
        "Window hours used: "
        + ", ".join(f"{start_hour:g}h {args.start_minute:g}m" for start_hour in requested_hours)
    )
    print(f"Raw preview window: {args.raw_window_minutes:g} minutes")
    print(f"Cluster window: {cluster_window_minutes:g} minutes")
    print(f"Downsample factor used: {downsample_factor}")
    print(f"DBSCAN min_samples used: {min_samples}")
    print(f"Temporal radius used: {temporal_radius_seconds:g} seconds")

    if args.show:
        for output_files in all_output_files:
            open_html_outputs(output_files)


if __name__ == "__main__":
    main()
