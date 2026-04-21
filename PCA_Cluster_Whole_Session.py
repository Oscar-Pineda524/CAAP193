from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import polars as pl
except Exception:
    pl = None

from PCA_Cluster_Analysis_parquet import (
    choose_min_samples,
    choose_temporal_radius_seconds,
    cluster_with_time_splitting,
    infer_sample_rate_from_datetime,
)


SURGERY_FILE = SCRIPT_DIR / "surgery_dates.csv"

# The existing PCA parquet script uses a downsample factor of 6 for 30 Hz raw data,
# which yields an effective sample rate of 5 Hz.
DEFAULT_DOWNSAMPLE_FACTOR = 6
DEFAULT_EPS = 0.01


def load_surgery_metadata(path: Path) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing surgery metadata file: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    required_cols = {"subject", "surgery_date"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"surgery_dates.csv is missing required columns: {sorted(missing_cols)}. "
            f"Found columns: {list(df.columns)}"
        )

    subject_map: dict[str, dict[str, object]] = {}
    initial_map: dict[str, dict[str, object]] = {}

    for _, row in df.iterrows():
        subject = str(row["subject"]).strip()
        if not subject:
            continue

        surgery_date = pd.to_datetime(row["surgery_date"], format="%m/%d/%Y", errors="coerce")
        if pd.isna(surgery_date):
            raise ValueError(f"Invalid surgery_date for subject '{subject}' in surgery_dates.csv")

        group = str(row.get("group", "Unknown")).strip().upper()
        if group not in {"T", "C"}:
            group = "Unknown"

        record = {
            "subject": subject,
            "subject_initial": subject[0].upper(),
            "surgery_date": surgery_date.normalize(),
            "group": group,
        }
        subject_map[subject] = record
        initial_map[record["subject_initial"]] = record

    return subject_map, initial_map


def resolve_subject_metadata(
    subject_hint: str,
    subject_map: dict[str, dict[str, object]],
    initial_map: dict[str, dict[str, object]],
) -> dict[str, object]:
    normalized = str(subject_hint).strip()
    if not normalized:
        raise ValueError("Please provide a subject initial or subject name.")

    candidates = [normalized, normalized.upper(), normalized.capitalize(), normalized[:1].upper()]
    for candidate in candidates:
        if candidate in subject_map:
            return subject_map[candidate].copy()
        if len(candidate) == 1 and candidate in initial_map:
            return initial_map[candidate].copy()

    raise ValueError(
        f"No surgery metadata found for '{subject_hint}'. "
        "Use a full subject name or its initial from surgery_dates.csv."
    )


def resolve_data_dir(requested: Path | None) -> Path:
    if requested is not None:
        data_dir = requested.expanduser().resolve()
        if not data_dir.exists() or not data_dir.is_dir():
            raise ValueError(f"Provided data directory is not valid: {data_dir}")
        return data_dir

    for candidate_name in ("30hz_RAW", "30hz_raw"):
        candidate = SCRIPT_DIR / candidate_name
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find a 30 Hz raw-data folder. Expected one of: "
        f"{SCRIPT_DIR / '30hz_RAW'} or {SCRIPT_DIR / '30hz_raw'}"
    )


def normalize_subject_initial(subject_text: str) -> str:
    match = re.search(r"[A-Za-z]", str(subject_text))
    if not match:
        raise ValueError(f"Could not extract a subject initial from '{subject_text}'.")
    return match.group(0).upper()


def prompt_for_subject_initial() -> str:
    response = input("Enter the subject initial to analyze (for example A): ").strip()
    return normalize_subject_initial(response)


def parse_date_from_filename(path: Path) -> pd.Timestamp | pd.NaT:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group(1), errors="coerce")


def find_subject_files(data_dir: Path, subject_initial: str) -> list[Path]:
    matches = []
    for path in sorted(data_dir.glob("*.parquet")):
        if normalize_subject_initial(path.name) == subject_initial:
            matches.append(path)
    return matches


def collect_downsampled_session(path: Path, downsample_factor: int) -> pl.DataFrame:
    if pl is None:
        raise ImportError(
            "This script requires polars to read parquet data efficiently. "
            "Install polars or run in an environment where it is available."
        )

    return (
        pl.scan_parquet(path)
        .with_row_index("row_idx")
        .filter((pl.col("row_idx") % downsample_factor) == 0)
        .select(
            [
                pl.col("datetime"),
                pl.col("accelerometer_x").alias("x"),
                pl.col("accelerometer_y").alias("y"),
                pl.col("accelerometer_z").alias("z"),
            ]
        )
        .collect()
    )


def unique_recording_days(df: pl.DataFrame) -> list[pd.Timestamp]:
    days = (
        df.select(pl.col("datetime").dt.date().alias("recording_day"))
        .unique()
        .sort("recording_day")
        .get_column("recording_day")
        .to_list()
    )
    return [pd.Timestamp(day) for day in days if day is not None]


def filter_day(df: pl.DataFrame, recording_day: pd.Timestamp) -> pl.DataFrame:
    next_day = recording_day + pd.Timedelta(days=1)
    return df.filter(
        (pl.col("datetime") >= recording_day.to_pydatetime())
        & (pl.col("datetime") < next_day.to_pydatetime())
    )


def polars_day_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": df["datetime"].to_list(),
            "x": df["x"].to_list(),
            "y": df["y"].to_list(),
            "z": df["z"].to_list(),
        }
    )


def summarize_clusters_for_day(
    day_df: pd.DataFrame,
    eps: float,
    min_samples_override: int | None,
) -> dict[str, object]:
    if day_df.empty:
        return {
            "cluster_count": 0,
            "noise_points": 0,
            "n_samples": 0,
            "fs_cluster": np.nan,
            "min_samples": np.nan,
            "duration_hours": 0.0,
        }

    fs_cluster = infer_sample_rate_from_datetime(day_df["datetime"]) or np.nan
    if pd.isna(fs_cluster) or fs_cluster <= 0:
        raise ValueError("Unable to infer a valid sample rate from the downsampled datetime column.")

    temporal_radius_seconds = choose_temporal_radius_seconds(float(fs_cluster))
    min_samples = choose_min_samples(float(fs_cluster), min_samples_override)
    cluster_labels, _ = cluster_with_time_splitting(
        day_df,
        eps=eps,
        min_samples=min_samples,
        fs_after_downsampling=float(fs_cluster),
        temporal_radius_seconds=temporal_radius_seconds,
    )

    unique_clusters = sorted(label for label in np.unique(cluster_labels) if label != -1)
    noise_points = int(np.sum(cluster_labels == -1))

    if len(day_df) > 1:
        duration_hours = float(
            (
                pd.to_datetime(day_df["datetime"]).iloc[-1]
                - pd.to_datetime(day_df["datetime"]).iloc[0]
            ).total_seconds()
            / 3600.0
        )
    else:
        duration_hours = 0.0

    return {
        "cluster_count": int(len(unique_clusters)),
        "noise_points": noise_points,
        "n_samples": int(len(day_df)),
        "fs_cluster": float(fs_cluster),
        "min_samples": int(min_samples),
        "duration_hours": duration_hours,
    }


def analyze_subject_sessions(
    subject_metadata: dict[str, object],
    files: list[Path],
    downsample_factor: int,
    eps: float,
    min_samples_override: int | None,
    max_recording_days: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for file_path in files:
        print(f"Loading {file_path.name} with downsample factor {downsample_factor}...")
        session_df = collect_downsampled_session(file_path, downsample_factor)
        recording_days = unique_recording_days(session_df)
        if max_recording_days is not None:
            recording_days = recording_days[: max(0, max_recording_days)]

        filename_date = parse_date_from_filename(file_path)

        for recording_day in recording_days:
            print(f"  Clustering {file_path.name} for {recording_day.date()}...")
            day_pd = polars_day_to_pandas(filter_day(session_df, recording_day))
            cluster_summary = summarize_clusters_for_day(
                day_pd,
                eps=eps,
                min_samples_override=min_samples_override,
            )

            rows.append(
                {
                    "subject": subject_metadata["subject"],
                    "subject_initial": subject_metadata["subject_initial"],
                    "group": subject_metadata["group"],
                    "surgery_date": pd.Timestamp(subject_metadata["surgery_date"]).date().isoformat(),
                    "source_file": file_path.name,
                    "source_file_date": filename_date.date().isoformat() if not pd.isna(filename_date) else None,
                    "recording_day": recording_day.date().isoformat(),
                    "days_from_surgery": int(
                        (recording_day.normalize() - pd.Timestamp(subject_metadata["surgery_date"])).days
                    ),
                    **cluster_summary,
                }
            )

    if not rows:
        raise ValueError("No recording days were available to analyze for the selected subject.")

    return pd.DataFrame(rows).sort_values(["recording_day", "source_file"]).reset_index(drop=True)


def build_progress_figure(summary_df: pd.DataFrame, subject_metadata: dict[str, object]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(12, 6))

    grouped = list(summary_df.groupby("source_file", sort=False))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(grouped))))

    for color, (source_file, source_df) in zip(colors, grouped):
        source_df = source_df.sort_values("days_from_surgery")
        ax.plot(
            source_df["days_from_surgery"],
            source_df["cluster_count"],
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=color,
            label=source_file,
        )

    ax.axvline(0, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Days Before / After Surgery")
    ax.set_ylabel("Clusters Per Recorded Day")
    ax.set_title(
        f"{subject_metadata['subject']} whole-session cluster progression\n"
        f"Surgery date: {pd.Timestamp(subject_metadata['surgery_date']).date().isoformat()}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the PCA cluster-identification workflow on a single subject's 30 Hz raw parquet files "
            "and plot how the number of clusters per recorded day changes relative to surgery."
        )
    )
    parser.add_argument(
        "--subject-initial",
        type=str,
        default=None,
        help="Subject initial to analyze, for example A.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing the raw parquet files. Defaults to 30hz_RAW beside this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "pca_whole_session_outputs",
        help="Directory where the subject summary CSV and plot will be saved.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=DEFAULT_DOWNSAMPLE_FACTOR,
        help=(
            "Keep every Nth sample before clustering. Default is 6, which converts 30 Hz raw data to 5 Hz "
            "and matches the existing PCA parquet workflow."
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=DEFAULT_EPS,
        help="DBSCAN eps parameter. Default: 0.01.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Optional DBSCAN min_samples override. By default it follows the existing PCA parquet logic.",
    )
    parser.add_argument(
        "--max-recording-days",
        type=int,
        default=None,
        help="Optional limit for how many recording days to analyze per parquet file. Helpful for quick validation.",
    )
    args = parser.parse_args()

    subject_map, initial_map = load_surgery_metadata(SURGERY_FILE)
    data_dir = resolve_data_dir(args.data_dir)
    subject_initial = (
        normalize_subject_initial(args.subject_initial)
        if args.subject_initial is not None
        else prompt_for_subject_initial()
    )
    subject_metadata = resolve_subject_metadata(subject_initial, subject_map, initial_map)

    files = find_subject_files(data_dir, subject_initial)
    if not files:
        raise FileNotFoundError(
            f"No parquet files starting with '{subject_initial}' were found in {data_dir}."
        )

    subject_output_dir = args.output_dir.expanduser().resolve() / str(subject_metadata["subject"])
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = analyze_subject_sessions(
        subject_metadata=subject_metadata,
        files=files,
        downsample_factor=max(1, args.downsample),
        eps=args.eps,
        min_samples_override=args.min_samples,
        max_recording_days=args.max_recording_days,
    )

    csv_path = subject_output_dir / f"{subject_initial}_whole_session_cluster_counts.csv"
    png_path = subject_output_dir / f"{subject_initial}_whole_session_cluster_progression.png"
    summary_df.to_csv(csv_path, index=False)

    fig, _ = build_progress_figure(summary_df, subject_metadata)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Subject: {subject_metadata['subject']} ({subject_metadata['subject_initial']})")
    print(f"Group: {subject_metadata['group']}")
    print(f"Surgery date: {pd.Timestamp(subject_metadata['surgery_date']).date().isoformat()}")
    print(f"Input directory: {data_dir}")
    print(f"Parquet files analyzed: {len(files)}")
    print(f"Summary CSV saved to: {csv_path}")
    print(f"Progression plot saved to: {png_path}")


if __name__ == "__main__":
    main()
