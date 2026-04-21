from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


DEFAULT_CLUSTER_DIR = Path("cluster_csvs")
DEFAULT_SURGERY_DATES = Path("surgery_dates.csv")
DEFAULT_OUTPUT_DIR = Path("cluster_trends_outputs")
DEFAULT_RAW_DIR = Path("RAW_Parquet-selected")

GROUP_LABELS = {"T": "Treatment", "C": "Control"}
GROUP_BASE_COLORS = {"Treatment": "#e67e22", "Control": "#2e86c1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build cluster-count trend summaries, mixed models, and plots from "
            "per-file cluster CSV exports."
        )
    )
    parser.add_argument(
        "--cluster-dir",
        type=Path,
        default=DEFAULT_CLUSTER_DIR,
        help="Directory containing *_clusters.csv files.",
    )
    parser.add_argument(
        "--surgery-dates",
        type=Path,
        default=DEFAULT_SURGERY_DATES,
        help="CSV containing subject, surgery_date, and group columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where trend outputs will be written.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=(
            "Directory containing matching *_RAW_enriched.parquet files. "
            "Used to keep only full recording days that span 00:00:00 to 23:59:59."
        ),
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.8,
        help="Transparency for scatter points in the group trend plot.",
    )
    return parser.parse_args()


def build_subject_lookup(surgery_df: pd.DataFrame) -> pd.DataFrame:
    surgery_df = surgery_df.copy()
    surgery_df["subject"] = surgery_df["subject"].astype(str).str.strip()
    surgery_df["subject_initial"] = surgery_df["subject"].str[0].str.upper()
    surgery_df["surgery_date"] = pd.to_datetime(surgery_df["surgery_date"], errors="coerce")
    surgery_df["group"] = surgery_df["group"].astype(str).str.strip().str.upper()

    required = {"subject", "subject_initial", "surgery_date", "group"}
    missing = required - set(surgery_df.columns)
    if missing:
        raise ValueError(f"surgery_dates.csv is missing expected columns: {sorted(missing)}")

    if surgery_df["surgery_date"].isna().any():
        bad_subjects = surgery_df.loc[surgery_df["surgery_date"].isna(), "subject"].tolist()
        raise ValueError(f"Invalid surgery_date values for subjects: {bad_subjects}")

    if surgery_df["subject_initial"].duplicated().any():
        dupes = surgery_df.loc[
            surgery_df["subject_initial"].duplicated(keep=False),
            ["subject_initial", "subject"],
        ]
        raise ValueError(
            "Subject initials must be unique to match cluster CSV filenames. "
            f"Conflicts: {dupes.to_dict(orient='records')}"
        )

    unknown_groups = sorted(set(surgery_df["group"]) - set(GROUP_LABELS))
    if unknown_groups:
        raise ValueError(f"Unexpected group code(s) in surgery_dates.csv: {unknown_groups}")

    surgery_df["group_label"] = surgery_df["group"].map(GROUP_LABELS)
    return surgery_df[["subject", "subject_initial", "surgery_date", "group", "group_label"]]


def metric_output_stem(metric_name: str) -> str:
    return metric_name.replace("_per_day", "")


def period_output_stem(period_label: str) -> str:
    return "all" if period_label == "all" else period_label


def get_full_days_for_session(raw_path: Path) -> set[pd.Timestamp]:
    raw_df = pd.read_parquet(raw_path, columns=["datetime"])
    raw_df["datetime"] = pd.to_datetime(raw_df["datetime"], errors="coerce")
    raw_df = raw_df.dropna(subset=["datetime"])
    if raw_df.empty:
        return set()

    start_dt = raw_df["datetime"].iloc[0]
    end_dt = raw_df["datetime"].iloc[-1]

    first_full_day = start_dt.normalize()
    if start_dt.time() != pd.Timestamp("00:00:00").time():
        first_full_day += pd.Timedelta(days=1)

    last_full_day = end_dt.normalize()
    if end_dt.time() < pd.Timestamp("23:59:59").time():
        last_full_day -= pd.Timedelta(days=1)

    if first_full_day > last_full_day:
        return set()

    return set(pd.date_range(first_full_day, last_full_day, freq="D"))


def load_cluster_rows(cluster_dir: Path, raw_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(cluster_dir.glob("*_clusters.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *_clusters.csv files were found in {cluster_dir}")

    cluster_frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        session_stem = csv_path.name.replace("_clusters.csv", "")
        raw_path = raw_dir / f"{session_stem}_RAW_enriched.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Matching raw parquet file not found for {csv_path.name}: expected {raw_path}"
            )

        full_days = get_full_days_for_session(raw_path)
        if not full_days:
            continue

        subject_initial = csv_path.stem.split("_", 1)[0].upper()
        cluster_df = pd.read_csv(csv_path, usecols=["cluster_date", "count", "day_or_night"])
        cluster_df["cluster_date"] = pd.to_datetime(cluster_df["cluster_date"], errors="coerce")
        cluster_df = cluster_df.dropna(subset=["cluster_date"])
        cluster_df["day_or_night"] = cluster_df["day_or_night"].astype(str).str.strip().str.lower()
        cluster_df = cluster_df[cluster_df["cluster_date"].isin(full_days)].reset_index(drop=True)
        if cluster_df.empty:
            continue

        cluster_df["subject_initial"] = subject_initial
        cluster_df["source_file"] = csv_path.name
        cluster_frames.append(cluster_df)

    if not cluster_frames:
        raise ValueError("Cluster CSV files were found, but none contained valid full-day cluster rows.")

    combined = pd.concat(cluster_frames, ignore_index=True)
    return combined


def summarize_daily_metric(cluster_rows_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    if metric_name == "clusters_per_day":
        daily_df = (
            cluster_rows_df.groupby(["subject_initial", "cluster_date"], as_index=False)
            .size()
            .rename(columns={"size": metric_name})
        )
    elif metric_name == "mean_cluster_count_per_day":
        daily_df = (
            cluster_rows_df.groupby(["subject_initial", "cluster_date"], as_index=False)["count"]
            .mean()
            .rename(columns={"count": metric_name})
        )
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    return daily_df


def filter_cluster_rows_for_period(cluster_rows_df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    if period_label == "all":
        return cluster_rows_df.copy()
    if period_label not in {"day", "night"}:
        raise ValueError(f"Unsupported period label: {period_label}")
    return cluster_rows_df.loc[cluster_rows_df["day_or_night"] == period_label].copy()


def prepare_analysis_dataframe(
    metric_daily_df: pd.DataFrame,
    surgery_df: pd.DataFrame,
    metric_name: str,
) -> pd.DataFrame:
    merged = metric_daily_df.merge(surgery_df, on="subject_initial", how="left")
    missing_subjects = sorted(
        merged.loc[merged["subject"].isna(), "subject_initial"].dropna().unique().tolist()
    )
    if missing_subjects:
        raise ValueError(
            "These cluster CSV subject initials are missing from surgery_dates.csv: "
            f"{missing_subjects}"
        )

    merged["days_since_surgery"] = (
        merged["cluster_date"] - merged["surgery_date"]
    ).dt.days.astype(int)
    merged["group_label"] = pd.Categorical(
        merged["group_label"],
        categories=["Control", "Treatment"],
        ordered=True,
    )
    merged["subject"] = merged["subject"].astype(str)
    return merged.sort_values(["group_label", "subject", "cluster_date"]).reset_index(drop=True)


def fit_mixed_model(data: pd.DataFrame, formula: str):
    model = smf.mixedlm(formula, data=data, groups=data["subject"])
    return model.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)


def build_subject_color_map(data: pd.DataFrame) -> dict[str, str]:
    color_map: dict[str, str] = {}
    for group_label in ["Treatment", "Control"]:
        group_subjects = sorted(data.loc[data["group_label"] == group_label, "subject"].unique())
        if not group_subjects:
            continue
        base = GROUP_BASE_COLORS[group_label]
        palette = sns.light_palette(base, n_colors=max(3, len(group_subjects) + 2), reverse=True)
        shades = palette[1 : len(group_subjects) + 1]
        for subject, color in zip(group_subjects, shades, strict=False):
            color_map[subject] = color
    return color_map


def save_daily_counts(data: pd.DataFrame, output_dir: Path, metric_name: str) -> Path:
    output_path = output_dir / f"daily_{metric_output_stem(metric_name)}.csv"
    data.to_csv(output_path, index=False)
    return output_path


def save_daily_counts_for_period(
    data: pd.DataFrame,
    output_dir: Path,
    metric_name: str,
    period_label: str,
) -> Path:
    output_path = output_dir / (
        f"daily_{metric_output_stem(metric_name)}_{period_output_stem(period_label)}.csv"
    )
    data.to_csv(output_path, index=False)
    return output_path


def save_model_summaries(
    combined_result,
    data: pd.DataFrame,
    output_dir: Path,
    metric_name: str,
    period_label: str = "all",
) -> Path:
    output_path = output_dir / (
        f"{metric_output_stem(metric_name)}_{period_output_stem(period_label)}_mixed_model_summaries.txt"
    )
    outcome_label = metric_name
    sections = [
        (
            f"Combined mixed model ({period_label}): "
            f"{outcome_label} ~ days_since_surgery * group_label"
        ),
        str(combined_result.summary()),
    ]

    for group_label in ["Treatment", "Control"]:
        group_df = data.loc[data["group_label"] == group_label].copy()
        if group_df["subject"].nunique() < 2:
            sections.extend(
                [
                    "",
                    f"{group_label}-only mixed model was skipped because fewer than 2 subjects were available.",
                ]
            )
            continue

        group_result = fit_mixed_model(group_df, f"{outcome_label} ~ days_since_surgery")
        sections.extend(
            [
                "",
                (
                    f"{group_label}-only mixed model ({period_label}): "
                    f"{outcome_label} ~ days_since_surgery"
                ),
                str(group_result.summary()),
            ]
        )

    output_path.write_text("\n".join(sections), encoding="utf-8")
    return output_path


def plot_group_trends(
    data: pd.DataFrame,
    combined_result,
    subject_colors: dict[str, str],
    output_dir: Path,
    point_alpha: float,
    metric_name: str,
    period_label: str = "all",
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    params = combined_result.params

    for group_label in ["Treatment", "Control"]:
        group_df = data.loc[data["group_label"] == group_label].copy()
        if group_df.empty:
            continue

        for subject in sorted(group_df["subject"].unique()):
            subject_df = group_df.loc[group_df["subject"] == subject]
            ax.scatter(
                subject_df["days_since_surgery"],
                subject_df[metric_name],
                s=42,
                alpha=point_alpha,
                color=subject_colors[subject],
                edgecolors="none",
                label=subject if group_label == "Treatment" else None,
            )

        # Use the mixed-model fixed effects to derive the group mean slope, then
        # anchor that slope to the observed group mean for a cleaner visual summary.
        slope = float(params.get("days_since_surgery", 0.0))
        if group_label == "Treatment":
            slope += float(params.get("days_since_surgery:group_label[T.Treatment]", 0.0))

        x_grid = np.linspace(group_df["days_since_surgery"].min(), group_df["days_since_surgery"].max(), 200)
        x_center = float(group_df["days_since_surgery"].mean())
        y_center = float(group_df[metric_name].mean())
        y_line = y_center + slope * (x_grid - x_center)
        ax.plot(
            x_grid,
            y_line,
            color=GROUP_BASE_COLORS[group_label],
            linewidth=3,
            label=f"{group_label} mean slope",
        )

    treatment_proxy = plt.Line2D([0], [0], marker="o", linestyle="", color=GROUP_BASE_COLORS["Treatment"])
    control_proxy = plt.Line2D([0], [0], marker="o", linestyle="", color=GROUP_BASE_COLORS["Control"])
    treatment_line = plt.Line2D([0], [0], color=GROUP_BASE_COLORS["Treatment"], linewidth=3)
    control_line = plt.Line2D([0], [0], color=GROUP_BASE_COLORS["Control"], linewidth=3)
    ax.legend(
        [treatment_proxy, control_proxy, treatment_line, control_line],
        [
            "Treatment subjects (orange shades)",
            "Control subjects (blue shades)",
            "Treatment mean slope",
            "Control mean slope",
        ],
        loc="best",
        frameon=True,
    )

    if metric_name == "clusters_per_day":
        title = "Clusters Per Day Relative to Surgery"
        ylabel = "Number of Clusters Per Day"
    else:
        title = "Mean Cluster Count Per Day Relative to Surgery"
        ylabel = "Mean Cluster Count Per Day"

    title_suffix = "All Clusters" if period_label == "all" else f"{period_label.title()} Clusters"
    ax.set_title(f"{title}\n{title_suffix}")
    ax.set_xlabel("Days Since Surgery")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.8)
    fig.tight_layout()

    output_path = output_dir / (
        f"group_{metric_output_stem(metric_name)}_{period_output_stem(period_label)}_trends.png"
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_subject_trends(
    data: pd.DataFrame,
    subject_colors: dict[str, str],
    output_dir: Path,
    metric_name: str,
    period_label: str = "all",
) -> Path:
    subject_dir = output_dir / (
        f"subject_trends_{metric_output_stem(metric_name)}_{period_output_stem(period_label)}"
    )
    subject_dir.mkdir(parents=True, exist_ok=True)

    for subject in sorted(data["subject"].unique()):
        subject_df = data.loc[data["subject"] == subject].copy()
        subject_df = subject_df.sort_values("days_since_surgery").reset_index(drop=True)
        if subject_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5.5))
        subject_color = subject_colors[subject]

        ax.scatter(
            subject_df["days_since_surgery"],
            subject_df[metric_name],
            s=50,
            color=subject_color,
            alpha=0.9,
            edgecolors="none",
        )

        if len(subject_df) >= 2:
            coeffs = np.polyfit(subject_df["days_since_surgery"], subject_df[metric_name], deg=1)
            x_grid = np.linspace(
                subject_df["days_since_surgery"].min(),
                subject_df["days_since_surgery"].max(),
                200,
            )
            y_grid = np.polyval(coeffs, x_grid)
            ax.plot(x_grid, y_grid, color=GROUP_BASE_COLORS[str(subject_df["group_label"].iloc[0])], linewidth=2.5)

        if metric_name == "clusters_per_day":
            title_metric = "Clusters Per Day"
            ylabel = "Number of Clusters Per Day"
        else:
            title_metric = "Mean Cluster Count Per Day"
            ylabel = "Mean Cluster Count Per Day"

        ax.set_title(
            f"{subject}: {title_metric} Over Time\n"
            f"{period_label.title()} clusters\n"
            f"{subject_df['group_label'].iloc[0]} group"
        )
        ax.set_xlabel("Days Since Surgery")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linewidth=0.8)
        fig.tight_layout()

        safe_name = subject.replace(" ", "_")
        fig.savefig(
            subject_dir / f"{safe_name}_{metric_output_stem(metric_name)}_{period_output_stem(period_label)}.png",
            dpi=300,
        )
        plt.close(fig)

    return subject_dir


def write_analysis_notes(output_dir: Path) -> Path:
    notes = textwrap.dedent(
        """
        Notes
        -----
        - clusters_per_day is the number of cluster rows recorded for each subject on each cluster_date.
        - Only full recording days are included. A day is kept only when the matching raw parquet file
          spans the entire calendar day from 00:00:00 through at least 23:59:59.
        - days_since_surgery is computed as cluster_date - surgery_date in whole days.
        - The script generates two day-level outcomes:
          clusters_per_day and mean_cluster_count_per_day.
        - Each outcome is generated three ways:
          all clusters, day-only clusters, and night-only clusters.
        - Each combined mixed model uses a random intercept for subject:
          outcome ~ days_since_surgery * group_label
        - Subject-level trend plots use a simple linear fit per subject for visualization.
        """
    ).strip()
    output_path = output_dir / "README.txt"
    output_path.write_text(notes + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    surgery_df = pd.read_csv(args.surgery_dates)
    surgery_lookup = build_subject_lookup(surgery_df)
    cluster_rows_df = load_cluster_rows(args.cluster_dir, args.raw_dir)
    notes_path = write_analysis_notes(args.output_dir)

    metric_names = ["clusters_per_day", "mean_cluster_count_per_day"]
    period_labels = ["all", "day", "night"]
    output_records: list[str] = []

    for metric_name in metric_names:
        for period_label in period_labels:
            period_cluster_rows_df = filter_cluster_rows_for_period(cluster_rows_df, period_label)
            if period_cluster_rows_df.empty:
                output_records.append(
                    f"Skipped {metric_name} for {period_label}: no matching cluster rows were available."
                )
                continue

            metric_daily_df = summarize_daily_metric(period_cluster_rows_df, metric_name)
            if metric_daily_df.empty:
                output_records.append(
                    f"Skipped {metric_name} for {period_label}: no daily values were available."
                )
                continue

            analysis_df = prepare_analysis_dataframe(metric_daily_df, surgery_lookup, metric_name)
            if analysis_df.empty or analysis_df["subject"].nunique() < 2:
                output_records.append(
                    f"Skipped {metric_name} for {period_label}: not enough subject-day data for modeling."
                )
                continue

            subject_colors = build_subject_color_map(analysis_df)
            combined_result = fit_mixed_model(
                analysis_df,
                f"{metric_name} ~ days_since_surgery * group_label",
            )

            if period_label == "all":
                daily_counts_path = save_daily_counts(analysis_df, args.output_dir, metric_name)
            else:
                daily_counts_path = save_daily_counts_for_period(
                    analysis_df,
                    args.output_dir,
                    metric_name,
                    period_label,
                )
            summary_path = save_model_summaries(
                combined_result,
                analysis_df,
                args.output_dir,
                metric_name,
                period_label,
            )
            group_plot_path = plot_group_trends(
                analysis_df,
                combined_result,
                subject_colors,
                args.output_dir,
                args.point_alpha,
                metric_name,
                period_label,
            )
            subject_plot_dir = plot_subject_trends(
                analysis_df,
                subject_colors,
                args.output_dir,
                metric_name,
                period_label,
            )
            output_records.extend(
                [
                    f"Saved {metric_name} ({period_label}) daily values to: {daily_counts_path}",
                    f"Saved {metric_name} ({period_label}) mixed model summaries to: {summary_path}",
                    f"Saved {metric_name} ({period_label}) group trend plot to: {group_plot_path}",
                    f"Saved {metric_name} ({period_label}) per-subject trend plots to: {subject_plot_dir}",
                ]
            )
    for line in output_records:
        print(line)
    print(f"Saved notes to: {notes_path}")


if __name__ == "__main__":
    main()
