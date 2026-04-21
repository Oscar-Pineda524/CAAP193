from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN as SklearnDBSCAN

try:
    import cupy as cp
    from cuml.cluster import DBSCAN as CumlDBSCAN
except Exception:
    cp = None
    CumlDBSCAN = None

try:
    import polars as pl
except Exception:
    pl = None


INPUT_DIR = Path("RAW_Parquet-selected")
OUTPUT_DIR = Path("cluster_csvs")
RAW_SAMPLE_RATE = 30
DEFAULT_DBSCAN_EPS = 0.014
DEFAULT_MIN_CLUSTER_SECONDS = 0.8
DEFAULT_MIN_CLUSTER_MINUTES = 0.75
DEFAULT_ALLOWED_SPLIT_GAP_SECONDS = 0.4
DEFAULT_CLUSTER_CSV_MIN_SAMPLES = 4


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
    datetime_column = ["datetime"] if "datetime" in pandas_df.columns else []
    return pandas_df[datetime_column + ["x", "y", "z"]].copy(), inferred_fs, "raw"


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
    return DEFAULT_CLUSTER_CSV_MIN_SAMPLES


def choose_cluster_window_minutes(input_kind: str, requested_minutes: float | None) -> float:
    if requested_minutes is not None:
        return max(0.25, requested_minutes)
    return 60.0


def prompt_for_initial() -> str:
    while True:
        initial = input("Enter subject initial: ").strip().upper()
        if len(initial) == 1 and initial.isalpha():
            return initial
        print("Please enter a single letter, for example A.")


def get_matching_parquet_files(input_dir: Path, initial: str) -> list[Path]:
    return sorted(input_dir.glob(f"{initial}_*_RAW_enriched.parquet"))


def day_or_night_label(timestamp: pd.Timestamp) -> str:
    time_value = timestamp.time()
    if pd.Timestamp("06:00:00").time() <= time_value < pd.Timestamp("18:00:00").time():
        return "day"
    return "night"


def split_segment_by_day(
    segment_df: pd.DataFrame,
    cluster_label: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for cluster_date, day_df in segment_df.groupby(segment_df["datetime"].dt.normalize(), sort=True):
        start_dt = day_df["datetime"].iloc[0]
        end_dt = day_df["datetime"].iloc[-1]
        rows.append(
            {
                "cluster_date": cluster_date.date(),
                "cluster_start_date_time": start_dt,
                "cluster_end_date_time": end_dt,
                "count": int(len(day_df)),
                "day_or_night": day_or_night_label(start_dt),
                "source_cluster_label": cluster_label,
            }
        )

    return rows


def summarize_clusters(clustered_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_start = None

    cluster_values = clustered_df["cluster"].astype(int).tolist()

    for idx, cluster_label in enumerate(cluster_values):
        if cluster_label == -1:
            if current_start is not None:
                segment = clustered_df.iloc[current_start:idx].reset_index(drop=True)
                rows.extend(split_segment_by_day(segment, int(cluster_values[current_start])))
                current_start = None
            continue

        if current_start is None:
            current_start = idx
            continue

        previous_label = cluster_values[idx - 1]
        if cluster_label != previous_label:
            segment = clustered_df.iloc[current_start:idx].reset_index(drop=True)
            rows.extend(split_segment_by_day(segment, int(previous_label)))
            current_start = idx

    if current_start is not None:
        segment = clustered_df.iloc[current_start:].reset_index(drop=True)
        rows.extend(split_segment_by_day(segment, int(cluster_values[current_start])))

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "cluster_date",
                "cluster_number",
                "cluster_start_date_time",
                "cluster_end_date_time",
                "count",
                "day_or_night",
            ]
        )

    # Keep DBSCAN labels intact, but suppress very short reported segments in the CSV.
    summary_df = summary_df[summary_df["count"] >= DEFAULT_CLUSTER_CSV_MIN_SAMPLES].reset_index(drop=True)
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "cluster_date",
                "cluster_number",
                "cluster_start_date_time",
                "cluster_end_date_time",
                "count",
                "day_or_night",
            ]
        )

    summary_df = summary_df.sort_values(
        ["cluster_start_date_time", "cluster_end_date_time"]
    ).reset_index(drop=True)
    summary_df["cluster_number"] = (
        summary_df.groupby("cluster_date").cumcount() + 1
    )

    return summary_df[
        [
            "cluster_date",
            "cluster_number",
            "cluster_start_date_time",
            "cluster_end_date_time",
            "count",
            "day_or_night",
        ]
    ]


def summarize_clustered_window(clustered_window_df: pd.DataFrame) -> pd.DataFrame:
    if clustered_window_df.empty:
        return pd.DataFrame(
            columns=[
                "cluster_date",
                "cluster_number",
                "cluster_start_date_time",
                "cluster_end_date_time",
                "count",
                "day_or_night",
            ]
        )

    return summarize_clusters(clustered_window_df)


def resolve_compute_device(requested_device: str) -> str:
    normalized = requested_device.strip().lower()
    if normalized not in {"auto", "cpu", "gpu"}:
        raise ValueError("--device must be one of: auto, cpu, gpu.")

    gpu_available = cp is not None and CumlDBSCAN is not None
    if normalized == "auto":
        return "gpu" if gpu_available else "cpu"
    if normalized == "gpu" and not gpu_available:
        raise ImportError(
            "GPU mode requested, but RAPIDS libraries are not available. "
            "Install cupy and cuml in a CUDA-compatible environment, or use --device cpu."
        )
    return normalized


def build_time_aware_features_for_device(
    df: pd.DataFrame,
    fs_after_downsampling: float,
    eps: float,
    temporal_radius_seconds: float,
    device: str,
):
    xyz = df[["x", "y", "z"]].to_numpy(dtype=float, copy=True)

    if device == "gpu":
        xyz_gpu = cp.asarray(xyz)
        elapsed_seconds = cp.arange(len(df), dtype=cp.float64) / fs_after_downsampling
        time_feature = (elapsed_seconds / temporal_radius_seconds) * eps
        return cp.column_stack([xyz_gpu, time_feature])

    elapsed_seconds = np.arange(len(df), dtype=float) / fs_after_downsampling
    time_feature = (elapsed_seconds / temporal_radius_seconds) * eps
    return np.column_stack([xyz, time_feature])


def labels_to_numpy(labels) -> np.ndarray:
    if cp is not None and isinstance(labels, cp.ndarray):
        return cp.asnumpy(labels).astype(int, copy=False)
    if hasattr(labels, "to_numpy"):
        return labels.to_numpy().astype(int, copy=False)
    return np.asarray(labels, dtype=int)


def merge_tiny_label_runs(labels: np.ndarray, min_run_length: int) -> np.ndarray:
    if len(labels) == 0 or min_run_length <= 1:
        return labels

    merged = labels.copy()

    while True:
        runs: list[tuple[int, int, int]] = []
        start_idx = 0
        current_label = int(merged[0])

        for idx in range(1, len(merged)):
            label = int(merged[idx])
            if label != current_label:
                runs.append((start_idx, idx, current_label))
                start_idx = idx
                current_label = label
        runs.append((start_idx, len(merged), current_label))

        changed = False
        for run_index, (start_idx, end_idx, label) in enumerate(runs):
            run_length = end_idx - start_idx
            if label == -1 or run_length >= min_run_length:
                continue
            if run_index == 0 or run_index == len(runs) - 1:
                continue

            left_start, left_end, left_label = runs[run_index - 1]
            right_start, right_end, right_label = runs[run_index + 1]
            left_length = left_end - left_start
            right_length = right_end - right_start

            if left_label == -1 or right_label == -1:
                continue
            if left_length < min_run_length or right_length < min_run_length:
                continue

            replacement_label = left_label if left_length >= right_length else right_label
            merged[start_idx:end_idx] = replacement_label
            changed = True
            break

        if not changed:
            return merged


def bridge_single_noise_gaps(labels: np.ndarray, min_run_length: int) -> np.ndarray:
    if len(labels) < 3 or min_run_length <= 0:
        return labels

    bridged = labels.copy()

    while True:
        runs: list[tuple[int, int, int]] = []
        start_idx = 0
        current_label = int(bridged[0])

        for idx in range(1, len(bridged)):
            label = int(bridged[idx])
            if label != current_label:
                runs.append((start_idx, idx, current_label))
                start_idx = idx
                current_label = label
        runs.append((start_idx, len(bridged), current_label))

        changed = False
        for run_index, (start_idx, end_idx, label) in enumerate(runs):
            run_length = end_idx - start_idx
            if label != -1 or run_length != 1:
                continue
            if run_index == 0 or run_index == len(runs) - 1:
                continue

            left_start, left_end, left_label = runs[run_index - 1]
            right_start, right_end, right_label = runs[run_index + 1]
            left_length = left_end - left_start
            right_length = right_end - right_start

            if left_label == -1 or right_label == -1:
                continue
            if left_label != right_label:
                continue
            if left_length < min_run_length or right_length < min_run_length:
                continue

            bridged[start_idx:end_idx] = left_label
            changed = True
            break

        if not changed:
            return bridged


def find_temporal_split_index(
    labels: np.ndarray,
    fs_after_downsampling: float,
    allowed_gap_seconds: float = DEFAULT_ALLOWED_SPLIT_GAP_SECONDS,
) -> int | None:
    best_split = None
    best_gap = 0
    allowed_gap_samples = max(0, int(round(fs_after_downsampling * allowed_gap_seconds)))

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
            if gap_size <= allowed_gap_samples:
                continue
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


def cluster_with_time_splitting_for_device(
    df: pd.DataFrame,
    eps: float,
    min_samples: int,
    fs_after_downsampling: float,
    temporal_radius_seconds: float,
    device: str,
    start_cluster_id: int = 0,
) -> tuple[np.ndarray, int]:
    if len(df) == 0:
        return np.array([], dtype=int), start_cluster_id

    features = build_time_aware_features_for_device(
        df=df,
        fs_after_downsampling=fs_after_downsampling,
        eps=eps,
        temporal_radius_seconds=temporal_radius_seconds,
        device=device,
    )

    if device == "gpu":
        labels = labels_to_numpy(CumlDBSCAN(eps=eps, min_samples=min_samples).fit_predict(features))
    else:
        labels = labels_to_numpy(SklearnDBSCAN(eps=eps, min_samples=min_samples).fit_predict(features))

    split_index = find_temporal_split_index(
        labels,
        fs_after_downsampling=fs_after_downsampling,
    )
    if split_index is None or split_index <= 0 or split_index >= len(df):
        return remap_cluster_labels(labels, start_cluster_id)

    left_labels, next_cluster_id = cluster_with_time_splitting_for_device(
        df.iloc[:split_index].reset_index(drop=True),
        eps=eps,
        min_samples=min_samples,
        fs_after_downsampling=fs_after_downsampling,
        temporal_radius_seconds=temporal_radius_seconds,
        device=device,
        start_cluster_id=start_cluster_id,
    )
    right_labels, next_cluster_id = cluster_with_time_splitting_for_device(
        df.iloc[split_index:].reset_index(drop=True),
        eps=eps,
        min_samples=min_samples,
        fs_after_downsampling=fs_after_downsampling,
        temporal_radius_seconds=temporal_radius_seconds,
        device=device,
        start_cluster_id=next_cluster_id,
    )
    return np.concatenate([left_labels, right_labels]), next_cluster_id


def cluster_full_file(
    parquet_path: Path,
    eps: float,
    downsample: int | None,
    min_samples: int | None,
    cluster_window_minutes: float | None,
    device: str,
) -> tuple[pd.DataFrame, float, int, int, float, float]:
    df, fs_original, input_kind = read_raw_parquet(parquet_path)
    if "datetime" not in df.columns:
        raise ValueError(f"{parquet_path.name} is missing the required datetime column.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{parquet_path.name} does not contain valid datetime rows.")

    downsample_factor = choose_downsample_factor(input_kind, downsample)
    df_downsampled = df.iloc[::downsample_factor].reset_index(drop=True)
    fs_cluster = fs_original / downsample_factor
    chosen_cluster_window_minutes = choose_cluster_window_minutes(
        input_kind,
        cluster_window_minutes,
    )
    cluster_window_samples = samples_for_minutes(fs_cluster, chosen_cluster_window_minutes)
    chosen_min_samples = choose_min_samples(fs_cluster, min_samples)
    temporal_radius_seconds = choose_temporal_radius_seconds(fs_cluster)

    window_summaries: list[pd.DataFrame] = []
    for start_idx in range(0, len(df_downsampled), cluster_window_samples):
        window_df = df_downsampled.iloc[start_idx:start_idx + cluster_window_samples].reset_index(drop=True)
        if window_df.empty:
            continue

        labels, _ = cluster_with_time_splitting_for_device(
            window_df[["x", "y", "z"]],
            eps=eps,
            min_samples=chosen_min_samples,
            fs_after_downsampling=fs_cluster,
            temporal_radius_seconds=temporal_radius_seconds,
            device=device,
        )
        labels = merge_tiny_label_runs(labels, chosen_min_samples)
        labels = bridge_single_noise_gaps(labels, chosen_min_samples)

        clustered_window_df = window_df.copy()
        clustered_window_df["cluster"] = labels
        window_summary = summarize_clustered_window(clustered_window_df)
        if not window_summary.empty:
            window_summaries.append(window_summary)

    if window_summaries:
        summary_df = pd.concat(window_summaries, ignore_index=True)
        summary_df = summary_df.sort_values(
            ["cluster_start_date_time", "cluster_end_date_time"]
        ).reset_index(drop=True)
        summary_df["cluster_number"] = (
            summary_df.groupby("cluster_date").cumcount() + 1
        )
        summary_df = summary_df[
            [
                "cluster_date",
                "cluster_number",
                "cluster_start_date_time",
                "cluster_end_date_time",
                "count",
                "day_or_night",
            ]
        ]
    else:
        summary_df = pd.DataFrame(
            columns=[
                "cluster_date",
                "cluster_number",
                "cluster_start_date_time",
                "cluster_end_date_time",
                "count",
                "day_or_night",
            ]
        )

    return (
        summary_df,
        fs_original,
        downsample_factor,
        chosen_min_samples,
        temporal_radius_seconds,
        chosen_cluster_window_minutes,
    )


def build_output_path(output_dir: Path, parquet_path: Path) -> Path:
    stem = parquet_path.name.replace("_RAW_enriched.parquet", "")
    return output_dir / f"{stem}_clusters.csv"


def choose_worker_count(requested_workers: int | None) -> int:
    if requested_workers is not None:
        return max(1, requested_workers)

    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count - 1))


def process_parquet_file(
    parquet_path: Path,
    output_dir: Path,
    eps: float,
    downsample: int | None,
    min_samples: int | None,
    cluster_window_minutes: float | None,
    device: str,
) -> dict[str, object]:
    summary_df, fs_original, downsample_factor, chosen_min_samples, temporal_radius_seconds, chosen_cluster_window_minutes = cluster_full_file(
        parquet_path=parquet_path,
        eps=eps,
        downsample=downsample,
        min_samples=min_samples,
        cluster_window_minutes=cluster_window_minutes,
        device=device,
    )

    output_path = build_output_path(output_dir, parquet_path)
    summary_df.to_csv(output_path, index=False)

    return {
        "parquet_name": parquet_path.name,
        "output_path": output_path,
        "row_count": len(summary_df),
        "sample_rate": fs_original,
        "downsample_factor": downsample_factor,
        "min_samples": chosen_min_samples,
        "temporal_radius_seconds": temporal_radius_seconds,
        "cluster_window_minutes": chosen_cluster_window_minutes,
        "device": device,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-file cluster summary CSVs from RAW_Parquet-selected parquet files "
            "using the same clustering logic as PCA_Cluster_Analysis_csv_allowgap.py."
        )
    )
    parser.add_argument(
        "--initial",
        type=str,
        default=None,
        help="Single-letter subject initial. If omitted, the script prompts for it.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing RAW parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where cluster CSV files will be written.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=DEFAULT_DBSCAN_EPS,
        help="DBSCAN eps parameter. Default matches PCA_Cluster_Analysis_csv_allowgap.py.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Optional override for the downsample factor. Default matches the existing script.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Optional override for DBSCAN min_samples. Default is 4.",
    )
    parser.add_argument(
        "--cluster-window-minutes",
        type=float,
        default=None,
        help="Window size used for sequential clustering across the file. Default matches the existing script.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device for DBSCAN: auto, cpu, or gpu. GPU mode requires RAPIDS/cuML with CUDA.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parquet files to process in parallel. Default chooses a small CPU-friendly value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial = (args.initial or prompt_for_initial()).strip().upper()
    if len(initial) != 1 or not initial.isalpha():
        raise ValueError("Initial must be a single letter.")
    device = resolve_compute_device(args.device)

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = get_matching_parquet_files(input_dir, initial)
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {input_dir} for subject initial '{initial}'."
        )

    print(f"Found {len(parquet_files)} file(s) for subject {initial}:")
    for parquet_path in parquet_files:
        print(f"  - {parquet_path.name}")

    worker_count = choose_worker_count(args.workers)
    print(f"\nUsing {worker_count} worker(s) on {device}.")

    if worker_count == 1 or len(parquet_files) == 1:
        for parquet_path in parquet_files:
            print(f"\nProcessing {parquet_path.name}...")
            result = process_parquet_file(
                parquet_path=parquet_path,
                output_dir=output_dir,
                eps=args.eps,
                downsample=args.downsample,
                min_samples=args.min_samples,
                cluster_window_minutes=args.cluster_window_minutes,
                device=device,
            )

            print(f"Saved {result['row_count']} cluster row(s) to {result['output_path']}")
            print(f"  sample rate: {result['sample_rate']:.6f} Hz")
            print(f"  downsample factor: {result['downsample_factor']}")
            print(f"  DBSCAN eps: {args.eps:g}")
            print(f"  DBSCAN min_samples: {result['min_samples']}")
            print(f"  temporal radius: {result['temporal_radius_seconds']:g} seconds")
            print(f"  cluster window: {result['cluster_window_minutes']:g} minutes")
            print(f"  compute device: {result['device']}")
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_path = {
            executor.submit(
                process_parquet_file,
                parquet_path,
                output_dir,
                args.eps,
                args.downsample,
                args.min_samples,
                args.cluster_window_minutes,
                device,
            ): parquet_path
            for parquet_path in parquet_files
        }

        for future in as_completed(future_to_path):
            parquet_path = future_to_path[future]
            print(f"\nCompleted {parquet_path.name}.")
            result = future.result()
            print(f"Saved {result['row_count']} cluster row(s) to {result['output_path']}")
            print(f"  sample rate: {result['sample_rate']:.6f} Hz")
            print(f"  downsample factor: {result['downsample_factor']}")
            print(f"  DBSCAN eps: {args.eps:g}")
            print(f"  DBSCAN min_samples: {result['min_samples']}")
            print(f"  temporal radius: {result['temporal_radius_seconds']:g} seconds")
            print(f"  cluster window: {result['cluster_window_minutes']:g} minutes")
            print(f"  compute device: {result['device']}")


if __name__ == "__main__":
    main()
