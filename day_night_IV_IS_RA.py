import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    _HAS_STATSMODELS = True
except Exception:
    smf = None
    _HAS_STATSMODELS = False


SCRIPT_DIR = Path(__file__).resolve().parent
SURGERY_FILE = SCRIPT_DIR / "surgery_dates.csv"

DAY_START = 6
DAY_END = 18
PERIODS = {
    "day": {"label": "Day", "color": "#d55e00"},
    "night": {"label": "Night", "color": "#0072b2"},
}
METRICS = {
    "IS": "Interdaily Stability (IS)",
    "IV": "Intradaily Variability (IV)",
    "RA": "Relative Amplitude (RA, classic M10/L5)",
}
Y_LIMITS = {
    "IS": (0.0, 0.7),
    "IV": (0.0, 1.0),
    "RA": (0.0, 0.25),
}


def load_surgery_metadata(path: Path):
    surgery_map = {}
    surgery_initial_map = {}

    if not path.exists():
        raise FileNotFoundError(f"Missing surgery metadata file: {path}")

    df_dates = pd.read_csv(path)
    df_dates.columns = df_dates.columns.str.strip().str.lower()

    required_cols = {"subject", "surgery_date"}
    missing_cols = required_cols - set(df_dates.columns)
    if missing_cols:
        raise ValueError(
            f"surgery_dates.csv is missing required columns: {sorted(missing_cols)}. "
            f"Found columns: {list(df_dates.columns)}"
        )

    for _, row in df_dates.iterrows():
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
            "surgery_date": surgery_date,
            "group": group,
        }
        surgery_map[subject] = record
        surgery_initial_map[record["subject_initial"]] = record

    return surgery_map, surgery_initial_map


surgery_map, surgery_initial_map = load_surgery_metadata(SURGERY_FILE)


root_path = input("Enter path to the DATA folder that contains subject CSV files or subject subfolders: ").strip()
DATA_ROOT = Path(root_path).expanduser().resolve()
if not DATA_ROOT.exists() or not DATA_ROOT.is_dir():
    raise ValueError(f"Provided path is not a valid directory: {DATA_ROOT}")

PLOT_ROOT = SCRIPT_DIR / f"day_night_plots_{DATA_ROOT.name}"
PLOT_ROOT.mkdir(parents=True, exist_ok=True)


def parse_subject_and_date_from_filename(path: Path):
    fname = path.name
    m_date = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
    session_date = pd.to_datetime(m_date.group(1), errors="coerce") if m_date else pd.NaT

    m_old = re.search(r"^(.*?)\s+(\d+)\s+\(\d{4}-\d{2}-\d{2}\)", fname)
    if m_old:
        return m_old.group(1).strip(), m_old.group(2).strip(), session_date, fname

    label = None
    subj_id = None
    if m_date:
        prefix = fname[:m_date.start()]
        prefix = re.sub(r"[\s_\-\(\)]+$", "", prefix)
        prefix = re.sub(r"^[\s_\-\(\)]+", "", prefix)
        label = prefix.strip() if prefix.strip() else None

        m_id = re.search(r"(\d+)", prefix)
        subj_id = m_id.group(1) if m_id else None

        if label:
            label_clean = re.sub(r"[_\-]*\d+[_\-]*", "", label).strip()
            label = label_clean if label_clean else label

    return label, subj_id, session_date, fname


def parse_subject_and_group_from_folder(folder_name: str):
    m = re.search(r"^(.*?)\((T|C)\)", folder_name)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return folder_name.strip(), "Unknown"


def resolve_subject_metadata(subject_hint: str, group_hint: str = "Unknown"):
    subject_hint = str(subject_hint).strip()
    group_hint = str(group_hint).strip().upper()
    if group_hint not in {"T", "C"}:
        group_hint = "Unknown"

    candidates = []
    if subject_hint:
        candidates.extend([subject_hint, subject_hint.upper(), subject_hint.capitalize(), subject_hint[:1].upper()])

    for candidate in candidates:
        if candidate in surgery_map:
            record = surgery_map[candidate].copy()
            if record["group"] == "Unknown" and group_hint in {"T", "C"}:
                record["group"] = group_hint
            return record
        if len(candidate) == 1 and candidate in surgery_initial_map:
            record = surgery_initial_map[candidate].copy()
            if record["group"] == "Unknown" and group_hint in {"T", "C"}:
                record["group"] = group_hint
            return record

    raise ValueError(
        f"No surgery date found for subject '{subject_hint}' in surgery_dates.csv. "
        "The subject can be listed as the full name or its initial."
    )


def parse_subject_key_from_filename(path: Path):
    label, _, _, _ = parse_subject_and_date_from_filename(path)
    if label:
        cleaned = re.sub(r"[^A-Za-z]", "", str(label)).strip()
        if cleaned:
            return cleaned.upper()

    stem = path.stem
    m = re.search(r"([A-Za-z])", stem)
    if m:
        return m.group(1).upper()
    return None


def load_actilife_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    elif "datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
        )
    else:
        raise ValueError(
            f"{path.name}: expected 'Datetime' column or ('Date' and 'Time'). "
            f"Found columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")

    if all(c in df.columns for c in ["accelerometer_x_sum", "accelerometer_y_sum", "accelerometer_z_sum"]):
        axis_cols = ["accelerometer_x_sum", "accelerometer_y_sum", "accelerometer_z_sum"]
    elif all(c in df.columns for c in ["Axis1", "Axis2", "Axis3"]):
        axis_cols = ["Axis1", "Axis2", "Axis3"]
    else:
        raise ValueError(
            f"{path.name}: expected accelerometer_x/y/z_sum or Axis1/2/3 columns. "
            f"Found columns: {list(df.columns)}"
        )

    df["MeanAxis"] = df[axis_cols].mean(axis=1)
    return df


def safe_div(num, den):
    if den is None or (isinstance(den, float) and np.isnan(den)) or den == 0:
        return np.nan
    return num / den


def subset_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    hours = df["Datetime"].dt.hour
    if period == "day":
        mask = (hours >= DAY_START) & (hours < DAY_END)
    elif period == "night":
        mask = (hours >= DAY_END) | (hours < DAY_START)
    else:
        raise ValueError(f"Unsupported period: {period}")
    return df.loc[mask].copy()


def compute_is_iv_ra(df: pd.DataFrame):
    hourly = df.set_index("Datetime")["MeanAxis"].resample("1h").mean().dropna()

    if len(hourly) < 12 or np.var(hourly.values, ddof=0) == 0:
        is_value = np.nan
    else:
        hourly_profile = hourly.groupby(hourly.index.hour).mean()
        is_value = safe_div(np.var(hourly_profile.values, ddof=0), np.var(hourly.values, ddof=0))

    x = df["MeanAxis"].to_numpy()
    vx = np.var(x, ddof=0)
    if len(x) < 2 or vx == 0:
        iv_value = np.nan
    else:
        iv_value = safe_div(np.mean(np.diff(x) ** 2), vx)

    if len(hourly) < 10:
        ra_value = np.nan
    else:
        m10 = hourly.rolling(window=10, min_periods=10).mean().max(skipna=True)
        l5 = hourly.rolling(window=5, min_periods=5).mean().min(skipna=True)
        if pd.isna(m10) or pd.isna(l5):
            ra_value = np.nan
        else:
            ra_value = safe_div(m10 - l5, m10 + l5)

    return is_value, iv_value, ra_value


def compute_slope(x_vals: pd.Series, y_vals: pd.Series) -> float:
    mask = (~pd.isna(x_vals)) & (~pd.isna(y_vals))
    if mask.sum() < 2:
        return np.nan
    x = x_vals[mask].to_numpy(dtype=float)
    y = y_vals[mask].to_numpy(dtype=float)
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def add_trend_line(ax, x_vals, y_vals, color):
    mask = (~pd.isna(x_vals)) & (~pd.isna(y_vals))
    if mask.sum() < 2:
        return
    x = pd.Series(x_vals)[mask].to_numpy(dtype=float)
    y = pd.Series(y_vals)[mask].to_numpy(dtype=float)
    coeffs = np.polyfit(x, y, 1)
    y_fit = np.polyval(coeffs, x)
    order = np.argsort(x)
    ax.plot(x[order], y_fit[order], linestyle="--", linewidth=2, color=color)


def discover_subject_batches(data_root: Path):
    subject_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    root_csv_files = sorted(data_root.glob("*.csv"))
    subject_batches = []

    if subject_dirs:
        print(f"Found {len(subject_dirs)} subject folders in: {data_root}\n")
        for subj_dir in subject_dirs:
            folder_subject, folder_group = parse_subject_and_group_from_folder(subj_dir.name)
            try:
                metadata = resolve_subject_metadata(folder_subject, folder_group)
            except ValueError as exc:
                print(f"  [WARN] {exc} Skipping folder '{subj_dir.name}'.")
                continue
            subject_batches.append({
                "source_name": subj_dir.name,
                "subject": metadata["subject"],
                "group": metadata["group"],
                "surgery_date": metadata["surgery_date"],
                "paths": sorted(subj_dir.glob("*.csv")),
                "plot_dir_name": subj_dir.name,
            })
    elif root_csv_files:
        print(f"Found {len(root_csv_files)} CSV files directly inside: {data_root}\n")
        grouped_paths = {}
        for path in root_csv_files:
            subject_key = parse_subject_key_from_filename(path)
            if not subject_key:
                print(f"  [WARN] Could not determine subject initial from filename: {path.name}")
                continue
            grouped_paths.setdefault(subject_key, []).append(path)

        for subject_key in sorted(grouped_paths):
            try:
                metadata = resolve_subject_metadata(subject_key)
            except ValueError as exc:
                print(f"  [WARN] {exc} Skipping files for initial '{subject_key}'.")
                continue
            subject_batches.append({
                "source_name": subject_key,
                "subject": metadata["subject"],
                "group": metadata["group"],
                "surgery_date": metadata["surgery_date"],
                "paths": sorted(grouped_paths[subject_key]),
                "plot_dir_name": metadata["subject"],
            })
    else:
        raise ValueError(f"No CSV files or subject subfolders found inside: {data_root}")

    return subject_batches


def fit_mixedlm_and_plot(master_sessions_df: pd.DataFrame, metric: str, period: str, ylabel: str, title: str, out_name: str, plot_root: Path):
    df = master_sessions_df.copy()
    metric_col = f"{period}_{metric}"

    df = df.dropna(subset=["subject", "group", "days_from_surgery", metric_col])
    df = df[df["group"].isin(["T", "C"])].copy()
    if df.empty:
        print(f"[MIXEDLM] No data available for {period} {metric}; skipping mixed model plot.")
        return

    df["days_from_surgery"] = df["days_from_surgery"].astype(float)
    df["group"] = pd.Categorical(df["group"], categories=["C", "T"])

    subjects = sorted(df["subject"].unique().tolist())
    cmap = plt.get_cmap("tab20")
    subj_to_color = {subject: cmap(i % 20) for i, subject in enumerate(subjects)}

    fig, ax = plt.subplots(figsize=(9, 5))
    for subject in subjects:
        sdf = df[df["subject"] == subject]
        ax.scatter(sdf["days_from_surgery"], sdf[metric_col], label=subject, color=subj_to_color[subject], alpha=0.9)
        if len(sdf) >= 2:
            slope = np.polyfit(sdf["days_from_surgery"].to_numpy(), sdf[metric_col].to_numpy(), 1)[0]
            xs = np.array([sdf["days_from_surgery"].min(), sdf["days_from_surgery"].max()])
            ys = np.polyval([slope, np.mean(sdf[metric_col]) - slope * np.mean(sdf["days_from_surgery"])], xs)
            ax.plot(xs, ys, color=subj_to_color[subject], linewidth=1, alpha=0.5)

    if _HAS_STATSMODELS:
        try:
            model = smf.mixedlm(
                f"{metric_col} ~ days_from_surgery * group",
                df,
                groups=df["subject"],
                re_formula="~days_from_surgery",
            )
            res = model.fit(reml=False, method="lbfgs", disp=False)
            print(f"[MIXEDLM] {period} {metric} fit complete.")
            print(res.summary())

            summary_path = plot_root / f"mixedlm_{period}_{metric}_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as handle:
                handle.write(str(res.summary()))

            x_grid = np.linspace(df["days_from_surgery"].min(), df["days_from_surgery"].max(), 200)
            for group in ["C", "T"]:
                pred_df = pd.DataFrame({"days_from_surgery": x_grid, "group": group, "subject": subjects[0]})
                y_hat = res.predict(pred_df)
                ax.plot(x_grid, y_hat, linewidth=3, alpha=0.9, label=f"MixedLM fit ({group})")
        except Exception as exc:
            print(f"[MIXEDLM] Could not fit mixed model for {period} {metric}: {exc}")
    else:
        print(f"[MIXEDLM] statsmodels not available; skipping model fit for {period} {metric}.")

    ax.set_xlabel("Days from surgery")
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\n(All subjects; points colored by subject)")
    ax.axvline(0, linestyle=":", linewidth=1)
    if metric in Y_LIMITS:
        ax.set_ylim(Y_LIMITS[metric])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=8)
    plt.tight_layout()

    out_path = plot_root / out_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Mixed model plot saved: {out_path}")


subject_batches = discover_subject_batches(DATA_ROOT)
master_summary_rows = []
master_sessions_rows = []

for batch in subject_batches:
    print("=" * 60)
    print(f"Subject source: {batch['source_name']}")

    subject_name = batch["subject"]
    group_label = batch["group"]
    surgery_date = batch["surgery_date"]
    paths = batch["paths"]

    if not paths:
        print("  [SKIP] No CSV files found for this subject.")
        continue

    print(f"  Found {len(paths)} CSV files")
    for path in paths[:10]:
        print("   -", path.name)
    if len(paths) > 10:
        print("   ...")

    plot_dir = PLOT_ROOT / batch["plot_dir_name"]
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in paths:
        _label, subj_id, session_date, fname = parse_subject_and_date_from_filename(path)
        try:
            df = load_actilife_csv(path)
        except Exception as exc:
            print(f"  [WARN] Could not load {path.name}: {exc}")
            continue

        row = {
            "subject": subject_name,
            "id": subj_id,
            "group": group_label,
            "session_date": session_date.date() if pd.notna(session_date) else None,
            "file": fname,
            "n_minutes_total": len(df),
        }

        for period in PERIODS:
            period_df = subset_period(df, period)
            is_value, iv_value, ra_value = compute_is_iv_ra(period_df)
            row[f"{period}_IS"] = is_value
            row[f"{period}_IV"] = iv_value
            row[f"{period}_RA"] = ra_value
            row[f"{period}_n_minutes"] = len(period_df)

        rows.append(row)

    if not rows:
        print("  [SKIP] No usable CSV files after loading/processing.")
        continue

    per_session = pd.DataFrame(rows)
    per_session["session_date"] = pd.to_datetime(per_session["session_date"], errors="coerce")
    per_session = per_session.dropna(subset=["session_date"]).copy()
    per_session["days_from_surgery"] = (per_session["session_date"] - surgery_date).dt.days
    per_session = per_session.sort_values("session_date").reset_index(drop=True)

    if per_session.empty:
        print("  [SKIP] All sessions were dropped because no YYYY-MM-DD date was found in filenames.")
        continue

    summary_row = {
        "folder": batch["source_name"],
        "subject": subject_name,
        "group": group_label,
        "surgery_date": str(surgery_date.date()),
        "n_sessions": int(per_session["days_from_surgery"].notna().sum()),
    }

    for period in PERIODS:
        for metric in METRICS:
            metric_col = f"{period}_{metric}"
            slope_col = f"{period}_{metric}_slope"
            summary_row[f"{metric_col}_mean"] = per_session[metric_col].mean(skipna=True)
            summary_row[f"{metric_col}_sd"] = per_session[metric_col].std(skipna=True)
            summary_row[slope_col] = compute_slope(per_session["days_from_surgery"], per_session[metric_col])

    master_summary_rows.append(summary_row)

    print("\nPer-session day/night circadian metrics (this subject):")
    print(per_session)
    print("\nGrouped day/night summary (this subject):")
    print(pd.DataFrame([summary_row]))

    per_session_out = plot_dir / "per_session_day_night_metrics.csv"
    per_session.to_csv(per_session_out, index=False)

    for _, row in per_session.iterrows():
        long_row = {
            "subject": subject_name,
            "group": group_label,
            "surgery_date": str(surgery_date.date()),
            "session_date": row["session_date"],
            "days_from_surgery": row["days_from_surgery"],
        }
        for period in PERIODS:
            for metric in METRICS:
                long_row[f"{period}_{metric}"] = row[f"{period}_{metric}"]
        master_sessions_rows.append(long_row)

    for metric, ylabel in METRICS.items():
        fig, ax = plt.subplots(figsize=(9, 4))
        for period, period_meta in PERIODS.items():
            metric_col = f"{period}_{metric}"
            ax.scatter(
                per_session["days_from_surgery"],
                per_session[metric_col],
                color=period_meta["color"],
                label=period_meta["label"],
                alpha=0.9,
            )
            add_trend_line(ax, per_session["days_from_surgery"], per_session[metric_col], period_meta["color"])

        ax.set_xlabel("Days from surgery")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{ylabel}: Day vs Night\n"
            f"Subject: {subject_name}  |  Group: {group_label}  |  Surgery: {surgery_date.date()}"
        )
        if metric in Y_LIMITS:
            ax.set_ylim(Y_LIMITS[metric])
        ax.axvline(0, linestyle=":", linewidth=1)
        ax.legend()
        plt.tight_layout()

        out_path = plot_dir / f"{metric}_day_night_scatter_trend.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path}")


print("\n" + "=" * 60)
print("DONE. Outputs saved under:")
print(f"  {PLOT_ROOT}")

if master_summary_rows:
    master_df = pd.DataFrame(master_summary_rows).sort_values(["group", "subject"])
    master_out = PLOT_ROOT / "master_day_night_summary.csv"
    master_df.to_csv(master_out, index=False)
    print(f"Master summary saved: {master_out}")
else:
    print("No subjects were processed successfully; no master summary written.")


if master_summary_rows:
    master_df = pd.DataFrame(master_summary_rows)

    for metric, ylabel in METRICS.items():
        slope_cols = [f"day_{metric}_slope", f"night_{metric}_slope"]
        grouped_slope_df = master_df[["subject", "group", *slope_cols]].copy()
        grouped_slope_df = grouped_slope_df[grouped_slope_df["group"].isin(["T", "C"])]
        grouped_slope_df = grouped_slope_df.melt(
            id_vars=["subject", "group"],
            value_vars=slope_cols,
            var_name="period",
            value_name="slope",
        ).dropna(subset=["slope"])

        if grouped_slope_df.empty:
            print(f"No grouped slope data available for {metric}; skipping treatment/control comparison.")
            continue

        grouped_slope_df["period"] = grouped_slope_df["period"].map({
            f"day_{metric}_slope": "Day",
            f"night_{metric}_slope": "Night",
        })
        grouped_slope_df["group"] = grouped_slope_df["group"].map({"T": "Treatment", "C": "Control"})

        order = ["Treatment", "Control"]
        period_offsets = {"Day": -0.18, "Night": 0.18}
        period_colors = {"Day": PERIODS["day"]["color"], "Night": PERIODS["night"]["color"]}
        x_positions = np.arange(len(order))

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for period in ["Day", "Night"]:
            period_sub = grouped_slope_df[grouped_slope_df["period"] == period]
            means = period_sub.groupby("group")["slope"].mean().reindex(order)
            sems = period_sub.groupby("group")["slope"].sem().reindex(order)
            ax.bar(
                x_positions + period_offsets[period],
                means.to_numpy(),
                width=0.34,
                yerr=sems.fillna(0).to_numpy(),
                capsize=6,
                color=period_colors[period],
                label=period,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(order)
        ax.set_title(f"Mean slope of {metric} vs Days from surgery by Group and Period")
        ax.set_ylabel(f"{metric} slope (per day)")
        ax.set_xlabel("Group")
        ax.axhline(0, linewidth=1)
        ax.legend()
        plt.tight_layout()

        out_path = PLOT_ROOT / f"compare_{metric}_slope_day_vs_night_by_group.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Day/night slope comparison by group saved: {out_path}")


if master_sessions_rows:
    master_sessions_df = pd.DataFrame(master_sessions_rows)
    mixed_out = PLOT_ROOT / "mixed_model_plots"
    mixed_out.mkdir(parents=True, exist_ok=True)

    for period, period_meta in PERIODS.items():
        for metric, ylabel in METRICS.items():
            fit_mixedlm_and_plot(
                master_sessions_df,
                metric,
                period,
                ylabel,
                f"{period_meta['label']} {metric} vs Days from surgery",
                f"mixed_{period}_{metric}_all_subjects.png",
                mixed_out,
            )

    master_sessions_out = mixed_out / "master_sessions_day_night_long.csv"
    master_sessions_df.to_csv(master_sessions_out, index=False)
    print(f"Master per-session long table saved: {master_sessions_out}")
else:
    print("No per-session rows collected; skipping mixed model plots.")
