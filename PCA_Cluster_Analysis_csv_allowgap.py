from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime
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
DEFAULT_DBSCAN_EPS = 0.014
DEFAULT_MIN_CLUSTER_SECONDS = 0.8
DEFAULT_MIN_CLUSTER_MINUTES = 0.75
DEFAULT_ALLOWED_SPLIT_GAP_SECONDS = 0.4


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
    datetime_column = ["datetime"] if "datetime" in pandas_df.columns else []
    return pandas_df[datetime_column + ["x", "y", "z"]].copy(), inferred_fs, "raw"


def read_actigraphy_csv(csv_path: Path) -> tuple[pd.DataFrame, float, str]:
    preview = pd.read_csv(csv_path, nrows=5)
    preview_columns = {str(col).strip() for col in preview.columns}

    raw_sensor_cols = {
        "accelerometer_x",
        "accelerometer_y",
        "accelerometer_z",
    }
    aggregated_avg_cols = {
        "accelerometer_x_avg",
        "accelerometer_y_avg",
        "accelerometer_z_avg",
    }

    if raw_sensor_cols.issubset(preview_columns):
        df = pd.read_csv(csv_path)
        df.columns = [str(col).strip() for col in df.columns]
        renamed = df.rename(
            columns={
                "accelerometer_x": "x",
                "accelerometer_y": "y",
                "accelerometer_z": "z",
            }
        )
        datetime_column = ["datetime"] if "datetime" in renamed.columns else []
        selected_columns = datetime_column + ["x", "y", "z"]
        return renamed[selected_columns].copy(), RAW_SAMPLE_RATE, "raw"

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
        datetime_column = ["datetime"] if "datetime" in renamed.columns else []
        return renamed[datetime_column + ["x", "y", "z"]].copy(), inferred_fs, label

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


def parse_clock_time(time_text: str) -> pd.Timedelta:
    text = time_text.strip()
    for fmt in ("%H:%M", "%H:%M:%S", "%H:%M:%S.%f"):
        try:
            parsed = datetime.strptime(text, fmt)
            return pd.Timedelta(
                hours=parsed.hour,
                minutes=parsed.minute,
                seconds=parsed.second,
                microseconds=parsed.microsecond,
            )
        except ValueError:
            continue
    raise ValueError(
        f"Invalid time '{time_text}'. Use HH:MM, HH:MM:SS, or HH:MM:SS.sss."
    )


def parse_specific_datetime_window(
    date_text: str,
    start_time_text: str,
    end_time_text: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        base_date = pd.Timestamp(date_text).normalize()
    except Exception as exc:
        raise ValueError(f"Invalid date '{date_text}'. Use YYYY-MM-DD.") from exc

    if pd.isna(base_date):
        raise ValueError(f"Invalid date '{date_text}'. Use YYYY-MM-DD.")

    start_dt = base_date + parse_clock_time(start_time_text)
    end_dt = base_date + parse_clock_time(end_time_text)

    if end_dt <= start_dt:
        end_dt += pd.Timedelta(days=1)

    return start_dt, end_dt


def require_datetime_series(df: pd.DataFrame, context: str) -> pd.Series:
    if "datetime" not in df.columns:
        raise ValueError(
            f"{context} requires a 'datetime' column in the input file. "
            "Use a parquet or aggregated CSV with timestamps, or use the hour-offset options instead."
        )

    timestamps = pd.to_datetime(df["datetime"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{context} requires valid timestamps in every row of the 'datetime' column.")
    return timestamps


def select_window_by_datetime(
    df: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    context: str,
) -> pd.DataFrame:
    timestamps = require_datetime_series(df, context)
    mask = (timestamps >= start_dt) & (timestamps < end_dt)
    if not mask.any():
        min_dt = timestamps.min()
        max_dt = timestamps.max()
        raise ValueError(
            f"Requested datetime window [{start_dt}, {end_dt}) is outside the data range "
            f"for {context}. Available datetime range: "
            f"[{min_dt}, {max_dt}]."
        )
    return df.loc[mask].reset_index(drop=True)


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


def cluster_color_map(cluster_ids: list[int]) -> dict[int, str]:
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#8c564b",
        "#e377c2",
        "#bcbd22",
        "#7f7f7f",
    ]
    colors = {-1: "#7f7f7f"}
    next_color = 0
    for cluster_id in cluster_ids:
        if cluster_id == -1:
            continue
        colors[cluster_id] = palette[next_color % len(palette)]
        next_color += 1
    return colors


def build_cluster_figure(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    cluster_ids = sorted(df["cluster"].unique())
    cluster_colors = cluster_color_map(cluster_ids)

    for cluster_id in cluster_ids:
        cluster_df = df[df["cluster"] == cluster_id]
        name = "noise" if cluster_id == -1 else f"cluster {cluster_id}"
        fig.add_trace(
            go.Scatter3d(
                x=cluster_df["x"],
                y=cluster_df["y"],
                z=cluster_df["z"],
                mode="lines",
                line={"width": 4, "color": cluster_colors[cluster_id]},
                name=name,
            )
        )

    fig.update_layout(
        title=title,
        scene={"xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def axis_range_with_padding(values: pd.Series, padding_fraction: float = 0.05) -> list[float]:
    minimum = float(values.min())
    maximum = float(values.max())
    spread = maximum - minimum
    padding = spread * padding_fraction if spread > 0 else max(abs(minimum) * padding_fraction, 0.05)
    return [minimum - padding, maximum + padding]


def build_cluster_animation_figure(
    df: pd.DataFrame,
    title: str,
    marker_size: int = 6,
) -> tuple[go.Figure, dict[int, str], list[int]]:
    animation_x_range = [-1, 2]
    animation_y_range = [-1, 1]
    animation_z_range = [-2, 1]
    fig = go.Figure()
    cluster_ids = sorted(df["cluster"].unique())
    cluster_colors = cluster_color_map(cluster_ids)

    for cluster_id in cluster_ids:
        name = "noise" if cluster_id == -1 else f"cluster {cluster_id}"
        fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line={"width": 6, "color": cluster_colors[cluster_id]},
                name=name,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    initial_cluster = int(df["cluster"].iloc[0])
    initial_color = cluster_colors[initial_cluster]
    initial_point = df.iloc[0]
    fig.add_trace(
        go.Scatter3d(
            x=[initial_point["x"]],
            y=[initial_point["y"]],
            z=[initial_point["z"]],
            mode="markers",
            marker={"size": marker_size, "color": initial_color},
            name="current position",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        scene={
            "xaxis": {"title": "x", "range": animation_x_range, "autorange": False},
            "yaxis": {"title": "y", "range": animation_y_range, "autorange": False},
            # Keep the animation focused on the z-band of interest so brief jerk
            # outliers can move off-frame without flattening the movement scale.
            "zaxis": {"title": "z", "range": animation_z_range, "autorange": False},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        uirevision="cluster-animation",
    )
    return fig, cluster_colors, cluster_ids


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
        return max(2, samples_for_minutes(fs_after_downsampling, DEFAULT_MIN_CLUSTER_MINUTES))
    return max(1, int(round(fs_after_downsampling * DEFAULT_MIN_CLUSTER_SECONDS)))


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


def format_datetime_token(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%d_%H-%M-%S")


def format_datetime_window_label(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> str:
    return f"{format_datetime_token(start_dt)}_to_{format_datetime_token(end_dt)}"


def format_datetime_window_text(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> str:
    return f"{start_dt.strftime('%Y-%m-%d %H:%M:%S')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S')}"


def save_window_outputs(
    output_dir: Path,
    window_label: str,
    first_fig: go.Figure,
    second_fig: go.Figure,
    cluster_fig: go.Figure,
    cluster_animation_path: Path | None = None,
) -> dict[str, Path]:
    output_files = {
        "raw_window": output_dir / f"raw_window_3d_{window_label}.html",
        "downsampled_window": output_dir / f"downsampled_window_3d_{window_label}.html",
        "clustered_window": output_dir / f"clustered_window_3d_{window_label}.html",
    }
    if cluster_animation_path is not None:
        output_files["clustered_window_animation"] = cluster_animation_path
    first_fig.write_html(output_files["raw_window"])
    second_fig.write_html(output_files["downsampled_window"])
    cluster_fig.write_html(output_files["clustered_window"])
    return output_files


def build_animation_time_arrays(
    df: pd.DataFrame,
    fs_after_downsampling: float,
) -> tuple[np.ndarray, list[str]]:
    if "datetime" in df.columns:
        timestamps = pd.to_datetime(df["datetime"], errors="coerce")
        if timestamps.notna().all():
            elapsed_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
            time_labels = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3].tolist()
            return elapsed_seconds, time_labels

    elapsed_seconds = np.arange(len(df), dtype=float) / fs_after_downsampling
    time_labels = [f"T+{seconds:0.3f}s" for seconds in elapsed_seconds]
    return elapsed_seconds, time_labels


def write_cluster_animation_html(
    output_path: Path,
    df: pd.DataFrame,
    title: str,
    fs_after_downsampling: float,
    tail_seconds: float,
) -> None:
    if df.empty:
        raise ValueError("Cannot build an animation for an empty clustered window.")

    fig, cluster_colors, cluster_ids = build_cluster_animation_figure(df, title)
    div_id = f"cluster-animation-{output_path.stem}"
    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id=div_id)

    elapsed_seconds, time_labels = build_animation_time_arrays(df, fs_after_downsampling)
    x_values = df["x"].astype(float).tolist()
    y_values = df["y"].astype(float).tolist()
    z_values = df["z"].astype(float).tolist()
    cluster_labels = df["cluster"].astype(int).tolist()
    cluster_ids_js = [int(cluster_id) for cluster_id in cluster_ids]
    cluster_colors_js = {str(cluster_id): color for cluster_id, color in cluster_colors.items()}
    total_duration_seconds = float(elapsed_seconds[-1]) if len(elapsed_seconds) > 1 else 0.0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f7f7f7;
      color: #1f1f1f;
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 20px;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
      margin: 12px 0 18px;
    }}
    button, select {{
      font: inherit;
      padding: 8px 12px;
    }}
    .status {{
      display: grid;
      gap: 6px;
      margin-bottom: 12px;
      font-size: 14px;
    }}
    .hint {{
      font-size: 13px;
      color: #555;
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>{title}</h1>
    <p class="hint">This animation plays at real time by default and keeps a rolling, cluster-colored trace of only the latest visible window so the PCA motion stays readable against external subject footage.</p>
    <div class="controls">
      <button id="play-button" type="button">Play</button>
      <button id="pause-button" type="button">Pause</button>
      <button id="restart-button" type="button">Restart</button>
      <label for="speed-select">Speed</label>
      <select id="speed-select">
        <option value="0.25">0.25x</option>
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x real time</option>
        <option value="2">2x</option>
        <option value="4">4x</option>
      </select>
    </div>
    <div class="status">
      <div id="clock-label"></div>
      <div id="cluster-label"></div>
      <div id="elapsed-label"></div>
    </div>
    {plot_html}
  </div>
  <script>
    const plot = document.getElementById({json.dumps(div_id)});
    const xValues = {json.dumps(x_values)};
    const yValues = {json.dumps(y_values)};
    const zValues = {json.dumps(z_values)};
    const clusterLabels = {json.dumps(cluster_labels)};
    const clusterIds = {json.dumps(cluster_ids_js)};
    const clusterColors = {json.dumps(cluster_colors_js)};
    const elapsedSeconds = {json.dumps(elapsed_seconds.tolist())};
    const timeLabels = {json.dumps(time_labels)};
    const tailSeconds = {json.dumps(float(tail_seconds))};
    const totalDurationSeconds = {json.dumps(total_duration_seconds)};
    const markerTraceIndex = clusterIds.length;

    const playButton = document.getElementById("play-button");
    const pauseButton = document.getElementById("pause-button");
    const restartButton = document.getElementById("restart-button");
    const speedSelect = document.getElementById("speed-select");
    const clockLabel = document.getElementById("clock-label");
    const clusterLabel = document.getElementById("cluster-label");
    const elapsedLabel = document.getElementById("elapsed-label");

    let playing = false;
    let playbackRate = 1.0;
    let elapsedAtPauseMs = 0;
    let playStartedAtMs = 0;
    let rafId = null;
    let lastRenderedIndex = -1;

    function findSampleIndex(targetSeconds) {{
      let left = 0;
      let right = elapsedSeconds.length - 1;
      while (left < right) {{
        const mid = Math.ceil((left + right) / 2);
        if (elapsedSeconds[mid] <= targetSeconds) {{
          left = mid;
        }} else {{
          right = mid - 1;
        }}
      }}
      return left;
    }}

    function formatElapsed(totalSeconds) {{
      const wholeSeconds = Math.max(0, Math.floor(totalSeconds));
      const hours = Math.floor(wholeSeconds / 3600);
      const minutes = Math.floor((wholeSeconds % 3600) / 60);
      const seconds = wholeSeconds % 60;
      return [hours, minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":");
    }}

    function updateStatus(index) {{
      clockLabel.textContent = `Timestamp: ${{timeLabels[index]}}`;
      clusterLabel.textContent = `Cluster: ${{clusterLabels[index]}}`;
      elapsedLabel.textContent = `Elapsed: ${{formatElapsed(elapsedSeconds[index])}} / ${{formatElapsed(totalDurationSeconds)}} | Visible window: ${{formatElapsed(tailSeconds)}}`;
    }}

    function buildClusterWindowTrace(startIndex, endIndex, targetCluster) {{
      const xs = [];
      const ys = [];
      const zs = [];
      let drawingSegment = false;

      for (let i = startIndex; i <= endIndex; i += 1) {{
        if (clusterLabels[i] === targetCluster) {{
          xs.push(xValues[i]);
          ys.push(yValues[i]);
          zs.push(zValues[i]);
          drawingSegment = true;
          continue;
        }}

        if (drawingSegment) {{
          xs.push(null);
          ys.push(null);
          zs.push(null);
          drawingSegment = false;
        }}
      }}

      return {{x: xs, y: ys, z: zs}};
    }}

    function renderAt(targetSeconds) {{
      const index = findSampleIndex(targetSeconds);
      if (index === lastRenderedIndex) {{
        updateStatus(index);
        return;
      }}

      const clusterColor = clusterColors[String(clusterLabels[index])] || "#1f77b4";
      const tailStartSeconds = Math.max(0, elapsedSeconds[index] - tailSeconds);
      const tailStartIndex = findSampleIndex(tailStartSeconds);
      const traceIndexes = [];
      const xUpdates = [];
      const yUpdates = [];
      const zUpdates = [];

      for (let traceIndex = 0; traceIndex < clusterIds.length; traceIndex += 1) {{
        const clusterId = clusterIds[traceIndex];
        const tracePoints = buildClusterWindowTrace(tailStartIndex, index, clusterId);
        traceIndexes.push(traceIndex);
        xUpdates.push(tracePoints.x);
        yUpdates.push(tracePoints.y);
        zUpdates.push(tracePoints.z);
      }}

      Plotly.restyle(plot, {{
        x: xUpdates,
        y: yUpdates,
        z: zUpdates,
      }}, traceIndexes);

      Plotly.restyle(plot, {{
        x: [[xValues[index]]],
        y: [[yValues[index]]],
        z: [[zValues[index]]],
        marker: [{{size: 6, color: clusterColor}}],
      }}, [markerTraceIndex]);

      lastRenderedIndex = index;
      updateStatus(index);
    }}

    function animationTick(nowMs) {{
      if (!playing) {{
        return;
      }}

      const elapsedNowMs = elapsedAtPauseMs + ((nowMs - playStartedAtMs) * playbackRate);
      const targetSeconds = Math.min(totalDurationSeconds, elapsedNowMs / 1000);
      renderAt(targetSeconds);

      if (targetSeconds >= totalDurationSeconds) {{
        playing = false;
        elapsedAtPauseMs = totalDurationSeconds * 1000;
        rafId = null;
        return;
      }}

      rafId = window.requestAnimationFrame(animationTick);
    }}

    function playAnimation() {{
      if (playing) {{
        return;
      }}
      playing = true;
      playStartedAtMs = performance.now();
      rafId = window.requestAnimationFrame(animationTick);
    }}

    function pauseAnimation() {{
      if (!playing) {{
        return;
      }}
      playing = false;
      elapsedAtPauseMs += (performance.now() - playStartedAtMs) * playbackRate;
      if (rafId !== null) {{
        window.cancelAnimationFrame(rafId);
        rafId = null;
      }}
    }}

    function restartAnimation() {{
      pauseAnimation();
      elapsedAtPauseMs = 0;
      lastRenderedIndex = -1;
      renderAt(0);
    }}

    playButton.addEventListener("click", playAnimation);
    pauseButton.addEventListener("click", pauseAnimation);
    restartButton.addEventListener("click", restartAnimation);
    speedSelect.addEventListener("change", (event) => {{
      const wasPlaying = playing;
      pauseAnimation();
      playbackRate = Number(event.target.value) || 1.0;
      if (wasPlaying) {{
        playAnimation();
      }}
    }});

    renderAt(0);
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


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
    ##eps is the neighborhood radius and min_samples is the minimum local density needed for a cluster:
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)

    split_index = find_temporal_split_index(
        labels,
        fs_after_downsampling=fs_after_downsampling,
    )
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
        default=DEFAULT_DBSCAN_EPS,
        help="DBSCAN eps parameter. Default: 0.014 for slightly more inclusive clustering.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help=(
            "Override DBSCAN min_samples. By default, raw data uses about 0.8 seconds "
            "of samples and aggregated data uses about 0.75 minutes."
        ),
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
        "--date",
        type=str,
        default=None,
        help="Calendar date for an exact window, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Start clock time for an exact window, for example 06:00 or 06:00:30.",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default=None,
        help="End clock time for an exact window, for example 07:00 or 07:15:30.",
    )
    parser.add_argument(
        "--animation-tail-seconds",
        type=float,
        default=20.0,
        help="Rolling visible window for the real-time cluster animation. Default: 60 seconds.",
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
    exact_window_args = [args.date, args.start_time, args.end_time]
    using_exact_window = any(value is not None for value in exact_window_args)

    if using_exact_window and not all(value is not None for value in exact_window_args):
        raise ValueError("Exact window mode requires --date, --start-time, and --end-time together.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_output_files = []
    window_summaries = []

    if using_exact_window:
        start_dt, end_dt = parse_specific_datetime_window(args.date, args.start_time, args.end_time)
        window_text = format_datetime_window_text(start_dt, end_dt)
        first_window = select_window_by_datetime(
            df,
            start_dt,
            end_dt,
            context=f"{input_kind} input",
        )
        cluster_window = select_window_by_datetime(
            df_downsampled,
            start_dt,
            end_dt,
            context=f"downsampled {input_kind} input",
        )
        window_requests = [
            {
                "window_label": format_datetime_window_label(start_dt, end_dt),
                "summary_label": window_text,
                "first_window": first_window,
                "cluster_window": cluster_window,
                "first_title": f"{input_kind.title()} window from {window_text}",
                "second_title": f"Downsampled window from {window_text}",
                "cluster_title": f"Time-aware DBSCAN clusters on downsampled window from {window_text}",
                "animation_title": f"Real-time PCA cluster animation on downsampled window from {window_text}",
            }
        ]
    else:
        requested_hours = parse_window_hours(args.window_hours, args.start_hour)
        window_requests = []
        for start_hour in requested_hours:
            start_idx = start_index_from_time(fs_original, start_hour, args.start_minute)
            raw_window_samples = samples_for_minutes(fs_original, args.raw_window_minutes)
            first_window = select_window(df, start=start_idx, window=raw_window_samples)

            cluster_start_idx = start_index_from_time(fs_cluster, start_hour, args.start_minute)
            cluster_window_samples = samples_for_minutes(fs_cluster, cluster_window_minutes)
            cluster_window = select_window(df_downsampled, start=cluster_start_idx, window=cluster_window_samples)
            window_requests.append(
                {
                    "window_label": format_hour_token(start_hour, args.start_minute),
                    "summary_label": f"{start_hour:g}h {args.start_minute:g}m",
                    "first_window": first_window,
                    "cluster_window": cluster_window,
                    "first_title": (
                        f"{input_kind.title()} {args.raw_window_minutes:g}-minute window "
                        f"at {start_hour:g}h {args.start_minute:g}m"
                    ),
                    "second_title": (
                        f"Downsampled {cluster_window_minutes:g}-minute window "
                        f"at {start_hour:g}h {args.start_minute:g}m"
                    ),
                    "cluster_title": (
                        f"Time-aware DBSCAN clusters on downsampled {cluster_window_minutes:g}-minute window "
                        f"at {start_hour:g}h {args.start_minute:g}m"
                    ),
                    "animation_title": (
                        f"Real-time PCA cluster animation on downsampled {cluster_window_minutes:g}-minute window "
                        f"at {start_hour:g}h {args.start_minute:g}m"
                    ),
                }
            )

    for window_request in window_requests:
        first_window = window_request["first_window"]
        cluster_window = window_request["cluster_window"]
        first_fig = build_3d_line_figure(first_window, window_request["first_title"])
        second_fig = build_3d_line_figure(cluster_window, window_request["second_title"])

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
            window_request["cluster_title"],
        )

        window_label = window_request["window_label"]
        animation_output_path = args.output_dir / f"clustered_window_animation_3d_{window_label}.html"
        write_cluster_animation_html(
            animation_output_path,
            cluster_window,
            window_request["animation_title"],
            fs_after_downsampling=fs_cluster,
            tail_seconds=max(0.5, args.animation_tail_seconds),
        )
        output_files = save_window_outputs(
            args.output_dir,
            window_label,
            first_fig,
            second_fig,
            cluster_fig,
            cluster_animation_path=animation_output_path,
        )
        all_output_files.append(output_files)
        window_summaries.append(window_request["summary_label"])

        print(f"Saved raw window plot to: {output_files['raw_window']}")
        print(f"Saved downsampled window plot to: {output_files['downsampled_window']}")
        print(f"Saved clustered window plot to: {output_files['clustered_window']}")
        print(f"Saved clustered window animation to: {output_files['clustered_window_animation']}")
        print(
            "Cluster counts:\n"
            f"{cluster_window['cluster'].value_counts().sort_index().to_string()}"
        )

    print(f"Detected input kind: {input_kind}")
    print(f"Detected sample rate: {fs_original:.6f} samples/sec")
    if using_exact_window:
        print("Exact datetime windows used: " + ", ".join(window_summaries))
    else:
        print("Window hours used: " + ", ".join(window_summaries))
        print(f"Raw preview window: {args.raw_window_minutes:g} minutes")
        print(f"Cluster window: {cluster_window_minutes:g} minutes")
    print(f"Downsample factor used: {downsample_factor}")
    print(f"DBSCAN eps used: {args.eps:g}")
    print(f"DBSCAN min_samples used: {min_samples}")
    print(f"Temporal radius used: {temporal_radius_seconds:g} seconds")

    if args.show:
        for output_files in all_output_files:
            open_html_outputs(output_files)


if __name__ == "__main__":
    main()
