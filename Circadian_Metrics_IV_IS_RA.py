import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit, logit

# Optional: mixed-effects modeling
try:
    import statsmodels.formula.api as smf
    _HAS_STATSMODELS = True
except Exception:
    smf = None
    _HAS_STATSMODELS = False

SCRIPT_DIR = Path(__file__).resolve().parent
SURGERY_FILE = SCRIPT_DIR / "surgery_dates.csv"

surgery_map = {}
surgery_initial_map = {}
if SURGERY_FILE.exists():
    df_dates = pd.read_csv(SURGERY_FILE)
    df_dates.columns = df_dates.columns.str.strip().str.lower()

    required_cols = {"subject", "surgery_date"}
    missing_cols = required_cols - set(df_dates.columns)
    if missing_cols:
        raise ValueError(
            f"surgery_dates.csv is missing required columns: {sorted(missing_cols)}. "
            f"Found columns: {list(df_dates.columns)}"
        )

    for _, r in df_dates.iterrows():
        subject = str(r["subject"]).strip()
        if not subject:
            continue

        surgery_date = pd.to_datetime(r["surgery_date"], format="%m/%d/%Y", errors="coerce")
        if pd.isna(surgery_date):
            raise ValueError(f"Invalid surgery_date for subject '{subject}' in surgery_dates.csv")

        group = str(r.get("group", "Unknown")).strip().upper()
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


# =============================
# DATA LOCATION (ask user)
# =============================
root_path = input("Enter path to the DATA folder that contains subject CSV files or subject subfolders: ").strip()
DATA_ROOT = Path(root_path).expanduser().resolve()

if not DATA_ROOT.exists() or not DATA_ROOT.is_dir():
    raise ValueError(f"Provided path is not a valid directory: {DATA_ROOT}")

# Output base directory (auto-named)
SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_ROOT = SCRIPT_DIR / f"scatter_plots_{DATA_ROOT.name}"
PLOT_ROOT.mkdir(parents=True, exist_ok=True)


# -------------------------
# Helpers
# -------------------------
def parse_subject_and_date_from_filename(path: Path):
    """
    Flexible parser that extracts session date from ANY filename containing YYYY-MM-DD.
    Examples supported:
      - 'Vincent 39035 (2024-03-01)_60s.csv'
      - 'A_2023-11-28_60s.csv'
      - 'A-2023-11-28.csv'
    Returns: (label, subj_id, session_date, fname)
      - label/subj_id are optional and may be None (we primarily use folder-based subject/group).
    """
    fname = path.name

    # Date anywhere in filename
    m_date = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
    session_date = pd.to_datetime(m_date.group(1), errors="coerce") if m_date else pd.NaT

    # Original pattern: "<label> <id> (<date>)"
    m_old = re.search(r"^(.*?)\s+(\d+)\s+\(\d{4}-\d{2}-\d{2}\)", fname)
    if m_old:
        label = m_old.group(1).strip()
        subj_id = m_old.group(2).strip()
        return label, subj_id, session_date, fname

    # Fallback: label is prefix before date (trim separators)
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
    """
    Parses subject name and group from a folder name like:
      'Avocado(T)-60s-Clean' -> ('Avocado', 'T')
      'Bombash(C)-60s-Clean' -> ('Bombash', 'C')
    Returns (subject_name, group) where group is 'T'/'C'/'Unknown'.
    """
    m = re.search(r"^(.*?)\((T|C)\)", folder_name)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # If no explicit (T)/(C), fall back to whole name
    return folder_name.strip(), "Unknown"

def extract_group_from_path(path: Path) -> str:
    """Detects '(T)' or '(C)' anywhere in the path (including folder names)."""
    s = str(path)
    if "(T)" in s:
        return "T"
    if "(C)" in s:
        return "C"
    return "Unknown"


def resolve_subject_metadata(subject_hint: str, group_hint: str = "Unknown"):
    """
    Resolve a subject from either a full subject name or a single-letter initial.
    Prefers the metadata in surgery_dates.csv, including group assignments there.
    """
    subject_hint = str(subject_hint).strip()
    group_hint = str(group_hint).strip().upper()
    if group_hint not in {"T", "C"}:
        group_hint = "Unknown"

    candidates = []
    if subject_hint:
        candidates.append(subject_hint)
        candidates.append(subject_hint.upper())
        candidates.append(subject_hint.capitalize())
        candidates.append(subject_hint[:1].upper())

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
    """
    Extract a subject identifier from a filename.
    For new files like 'A_2024-03-18_60s.csv', this returns 'A'.
    """
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
    """
    Robust CSV loader.
    Supports:
    - No metadata rows
    - Datetime column named 'Datetime' or 'datetime' (YYYY-MM-DD HH:MM:SS)
    - accelerometer_x_sum / y_sum / z_sum OR Axis1/2/3
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Build Datetime
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    elif "datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce"
        )
    else:
        raise ValueError(
            f"{path.name}: expected 'Datetime' column or ('Date' and 'Time'). "
            f"Found columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")

    # Determine accelerometer columns
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



def compute_IS_IV_RA(df: pd.DataFrame):
    # ---- IS ----
    hourly = df.set_index("Datetime")["MeanAxis"].resample("1h").mean().dropna()
    if len(hourly) < 24 or np.var(hourly.values, ddof=0) == 0:
        IS = np.nan
    else:
        hourly_profile = hourly.groupby(hourly.index.hour).mean()
        IS = safe_div(
            np.var(hourly_profile.values, ddof=0),
            np.var(hourly.values, ddof=0)
        )

    # ---- IV ----
    x = df["MeanAxis"].to_numpy()
    vx = np.var(x, ddof=0)
    if len(x) < 2 or vx == 0:
        IV = np.nan
    else:
        IV = safe_div(np.mean(np.diff(x) ** 2), vx)

    # ---- RA (classic M10/L5 version) ----
    # M10 = most active 10 consecutive hours
    # L5  = least active 5 consecutive hours
    if len(hourly) < 10:
        RA = np.nan
    else:
        m10_series = hourly.rolling(window=10, min_periods=10).mean()
        l5_series = hourly.rolling(window=5, min_periods=5).mean()

        M10 = m10_series.max(skipna=True)
        L5 = l5_series.min(skipna=True)

        if pd.isna(M10) or pd.isna(L5):
            RA = np.nan
        else:
            RA = safe_div((M10 - L5), (M10 + L5))

    return IS, IV, RA

def compute_slope(x_vals: pd.Series, y_vals: pd.Series) -> float:
    """Returns slope from simple linear regression y = a + b*x. x should be numeric."""
    mask = (~pd.isna(x_vals)) & (~pd.isna(y_vals))
    if mask.sum() < 2:
        return np.nan
    x = x_vals[mask].to_numpy(dtype=float)
    y = y_vals[mask].to_numpy(dtype=float)
    b, a = np.polyfit(x, y, 1)  # returns [slope, intercept]
    return float(b)



def normalize_at_day(df: pd.DataFrame, metric: str, day: float = 60):
    """
    Normalize a metric relative to its estimated value at a target day from surgery.
    Uses a simple linear fit per subject to estimate the metric value at `day`,
    then returns metric - estimated_value_at_day.
    """
    x_vals = df["days_from_surgery"]
    y_vals = df[metric]

    mask = (~pd.isna(x_vals)) & (~pd.isna(y_vals))
    if mask.sum() < 2:
        return pd.Series([np.nan] * len(df), index=df.index)

    x = x_vals[mask].to_numpy(dtype=float)
    y = y_vals[mask].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    value_at_day = slope * day + intercept

    return df[metric] - value_at_day

def add_trend_line(ax, x_vals, y_vals):
    """Adds a linear trend line where x_vals is numeric (e.g., days from surgery)."""
    mask = (~pd.isna(x_vals)) & (~pd.isna(y_vals))
    if mask.sum() < 2:
        return

    x = pd.Series(x_vals)[mask].to_numpy(dtype=float)
    y = pd.Series(y_vals)[mask].to_numpy(dtype=float)

    coeffs = np.polyfit(x, y, 1)
    y_fit = np.polyval(coeffs, x)

    order = np.argsort(x)
    ax.plot(x[order], y_fit[order], linestyle="--", linewidth=2)


def _format_unique(series: pd.Series, fallback: str) -> str:
    vals = [str(v) for v in series.dropna().unique().tolist() if str(v).strip() != ""]
    if len(vals) == 0:
        return fallback
    if len(vals) == 1:
        return vals[0]
    return ", ".join(vals)


# =============================
# FIXED Y-AXIS LIMITS (EDIT IF NEEDED)
# =============================
Y_LIMITS = {
    "IS": (0.0, 0.7),
    "IV": (0.0, 0.6),
    "RA": (0.0, 1.0),
}


# =============================
# DISCOVER INPUTS
# =============================
subject_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
root_csv_files = sorted(DATA_ROOT.glob("*.csv"))

subject_batches = []

if len(subject_dirs) > 0:
    print(f"Found {len(subject_dirs)} subject folders in: {DATA_ROOT}\n")
    for subj_dir in subject_dirs:
        folder_subject, folder_group = parse_subject_and_group_from_folder(subj_dir.name)
        try:
            metadata = resolve_subject_metadata(folder_subject, folder_group)
        except ValueError as e:
            print(f"  [WARN] {e} Skipping folder '{subj_dir.name}'.")
            continue
        paths = sorted(subj_dir.glob("*.csv"))
        subject_batches.append({
            "source_name": subj_dir.name,
            "subject": metadata["subject"],
            "group": metadata["group"],
            "surgery_date": metadata["surgery_date"],
            "paths": paths,
            "plot_dir_name": subj_dir.name,
        })
elif len(root_csv_files) > 0:
    print(f"Found {len(root_csv_files)} CSV files directly inside: {DATA_ROOT}\n")
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
        except ValueError as e:
            print(f"  [WARN] {e} Skipping files for initial '{subject_key}'.")
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
    raise ValueError(f"No CSV files or subject subfolders found inside: {DATA_ROOT}")

master_summary_rows = []
master_sessions_rows = []  # per-session rows across all subjects

for batch in subject_batches:
    print("=" * 60)
    print(f"Subject source: {batch['source_name']}")

    subject_name = batch["subject"]
    group_label = batch["group"]
    SURGERY_DATE = batch["surgery_date"]

    paths = batch["paths"]
    if len(paths) == 0:
        print("  [SKIP] No CSV files found for this subject.")
        continue

    print(f"  Found {len(paths)} CSV files")
    for p in paths[:10]:
        print("   -", p.name)
    if len(paths) > 10:
        print("   ...")

    # Per-subject output directory
    PLOT_DIR = PLOT_ROOT / batch["plot_dir_name"]
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # PROCESS FILES (per subject)
    # -------------------------
    rows = []
    for p in paths:
        label, subj_id, session_date, fname = parse_subject_and_date_from_filename(p)
        try:
            df = load_actilife_csv(p)
        except Exception as e:
            print(f"  [WARN] Could not load {p.name}: {e}")
            continue

        IS, IV, RA = compute_IS_IV_RA(df)

        rows.append({
            "subject": subject_name,
            "id": subj_id,
            "group": group_label,
            "session_date": session_date.date() if pd.notna(session_date) else None,
            "file": fname,
            "IS": IS,
            "IV": IV,
            "RA": RA,
            "n_minutes": len(df),
        })

    if len(rows) == 0:
        print("  [SKIP] No usable CSV files after loading/processing.")
        continue

    per_session = pd.DataFrame(rows)
    per_session["session_date"] = pd.to_datetime(per_session["session_date"], errors="coerce")
    # Drop sessions where we could not extract a date from the filename
    per_session = per_session.dropna(subset=["session_date"]).copy()
    per_session["days_from_surgery"] = (per_session["session_date"] - SURGERY_DATE).dt.days
    per_session = per_session.sort_values("session_date").reset_index(drop=True)

    # Normalized versions anchored at 60 days from surgery
    per_session["IS_norm60"] = normalize_at_day(per_session, "IS", 60)
    per_session["IV_norm60"] = normalize_at_day(per_session, "IV", 60)
    per_session["RA_norm60"] = normalize_at_day(per_session, "RA", 60)

    if per_session.empty:
        print("  [SKIP] All sessions were dropped because no YYYY-MM-DD date was found in filenames.")
        continue

    # Slopes (per subject) of each metric vs days from surgery
    IS_slope = compute_slope(per_session["days_from_surgery"], per_session["IS"])
    IV_slope = compute_slope(per_session["days_from_surgery"], per_session["IV"])
    RA_slope = compute_slope(per_session["days_from_surgery"], per_session["RA"])

    # -------------------------
    # SUMMARY (per subject)
    # -------------------------
    summary_row = {
        "folder": batch["source_name"],
        "subject": subject_name,
        "group": group_label,
        "surgery_date": str(SURGERY_DATE.date()),
        "n_sessions": int(per_session["days_from_surgery"].notna().sum()),
        "IS_mean": per_session["IS"].mean(skipna=True),
        "IS_sd": per_session["IS"].std(skipna=True),
        "IV_mean": per_session["IV"].mean(skipna=True),
        "IV_sd": per_session["IV"].std(skipna=True),
        "RA_mean": per_session["RA"].mean(skipna=True),
        "RA_sd": per_session["RA"].std(skipna=True),
        "IS_slope": IS_slope,
        "IV_slope": IV_slope,
        "RA_slope": RA_slope,
    }
    master_summary_rows.append(summary_row)

    print("\nPer-session circadian metrics (this subject):")
    print(per_session)
    print("\nGrouped longitudinal summary (this subject):")
    print(pd.DataFrame([summary_row]))

    # Save per-session metrics table for the subject
    per_session_out = PLOT_DIR / "per_session_metrics.csv"
    per_session.to_csv(per_session_out, index=False)

    # Append per-session rows to master list (for mixed model + across-subject plots)
    for _i, _r in per_session.iterrows():
        master_sessions_rows.append({
            "subject": subject_name,
            "group": group_label,
            "surgery_date": str(SURGERY_DATE.date()),
            "session_date": _r["session_date"],
            "days_from_surgery": _r["days_from_surgery"],
            "IS": _r["IS"],
            "IV": _r["IV"],
            "RA": _r["RA"],
            "IS_norm60": _r["IS_norm60"],
            "IV_norm60": _r["IV_norm60"],
            "RA_norm60": _r["RA_norm60"],
        })


    # -------------------------
    # SCATTER PLOTS WITH TREND LINE (saved)
    # -------------------------
    for metric, ylabel, title in [
        ("IS", "Interdaily Stability (IS)", "Longitudinal IS across sessions"),
        ("IV", "Intradaily Variability (IV)", "Longitudinal IV across sessions"),
        ("RA", "Relative Amplitude (RA, classic M10/L5)", "Longitudinal RA across sessions"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))

        # Color by group
        if group_label == "T":
            color = "red"
        elif group_label == "C":
            color = "blue"
        else:
            color = "gray"

        ax.scatter(
            per_session["days_from_surgery"],
            per_session[metric],
            color=color
        )

        add_trend_line(ax, per_session["days_from_surgery"], per_session[metric])
        for line in ax.get_lines():
            line.set_color(color)

        ax.set_xlabel("Days from surgery")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nSubject: {subject_name}  |  Group: {group_label}  |  Surgery: {SURGERY_DATE.date()}")

        # Fix Y-axis scale across runs
        if metric in Y_LIMITS:
            ax.set_ylim(Y_LIMITS[metric])

        # Mark surgery day
        ax.axvline(0, linestyle=":", linewidth=1)

        plt.tight_layout()

        out_path = PLOT_DIR / f"{metric}_scatter_trend.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path}")


    # -------------------------
    # NORMALIZED SCATTER PLOTS WITH TREND LINE (saved)
    # -------------------------
    for metric, ylabel, title in [
        ("IS_norm60", "Normalized IS (relative to day 60)", "Normalized IS across sessions"),
        ("IV_norm60", "Normalized IV (relative to day 60)", "Normalized IV across sessions"),
        ("RA_norm60", "Normalized RA (classic M10/L5, relative to day 60)", "Normalized RA across sessions"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))

        if group_label == "T":
            color = "red"
        elif group_label == "C":
            color = "blue"
        else:
            color = "gray"

        ax.scatter(
            per_session["days_from_surgery"],
            per_session[metric],
            color=color
        )

        add_trend_line(ax, per_session["days_from_surgery"], per_session[metric])
        for line in ax.get_lines():
            line.set_color(color)

        ax.set_xlabel("Days from surgery")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nSubject: {subject_name}  |  Group: {group_label}  |  Surgery: {SURGERY_DATE.date()}")
        ax.axvline(0, linestyle=":", linewidth=1)
        ax.axhline(0, linestyle="--", linewidth=1)

        plt.tight_layout()

        out_path = PLOT_DIR / f"{metric}_scatter_trend.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path}")


def _fit_mixedlm_and_plot(master_sessions_df: pd.DataFrame, metric: str, ylabel: str, title: str, out_name: str, plot_root: Path):
    """
    Fits a linear mixed model (random intercept + random slope by subject) and saves a plot:
      - Scatter of all per-session points, colored by subject
      - Fixed-effect prediction lines for T and C (if available)
    """
    df = master_sessions_df.copy()

    # Keep only rows with needed columns
    df = df.dropna(subset=["subject", "group", "days_from_surgery", metric])
    df = df[df["group"].isin(["T", "C"])].copy()

    if df.empty:
        print(f"[MIXEDLM] No data available for {metric}; skipping mixed model plot.")
        return

    # Ensure correct dtypes
    df["days_from_surgery"] = df["days_from_surgery"].astype(float)
    df["group"] = pd.Categorical(df["group"], categories=["C", "T"])

    # Assign each subject a unique color (deterministic order)
    subjects = sorted(df["subject"].unique().tolist())
    cmap = plt.get_cmap("tab20")
    subj_to_color = {s: cmap(i % 20) for i, s in enumerate(subjects)}

    fig, ax = plt.subplots(figsize=(9, 5))

    for s in subjects:
        sdf = df[df["subject"] == s]
        ax.scatter(sdf["days_from_surgery"], sdf[metric], label=s, color=subj_to_color[s], alpha=0.9)

        # Optional thin per-subject OLS line (helps visualize individual trends)
        if len(sdf) >= 2:
            b = np.polyfit(sdf["days_from_surgery"].to_numpy(), sdf[metric].to_numpy(), 1)[0]
            xs = np.array([sdf["days_from_surgery"].min(), sdf["days_from_surgery"].max()])
            ys = np.polyval([b, np.mean(sdf[metric]) - b*np.mean(sdf["days_from_surgery"])], xs)
            ax.plot(xs, ys, color=subj_to_color[s], linewidth=1, alpha=0.5)

    # Fit mixed model if statsmodels is available
    if _HAS_STATSMODELS:
        try:
            # Random intercept + random slope by subject
            model = smf.mixedlm(f"{metric} ~ days_from_surgery * group", df, groups=df["subject"], re_formula="~days_from_surgery")
            res = model.fit(reml=False, method="lbfgs", disp=False)

            print(f"[MIXEDLM] {metric} fit complete.")
            print(res.summary())

            # Fixed-effect predictions for each group across a grid
            x_grid = np.linspace(df["days_from_surgery"].min(), df["days_from_surgery"].max(), 200)
            for g in ["C", "T"]:
                pred_df = pd.DataFrame({"days_from_surgery": x_grid, "group": g, "subject": subjects[0]})
                y_hat = res.predict(pred_df)  # fixed effects + random for subject[0]; close enough for visualization
                ax.plot(x_grid, y_hat, linewidth=3, alpha=0.9, label=f"MixedLM fit ({g})")
        except Exception as e:
            print(f"[MIXEDLM] Could not fit mixed model for {metric}: {e}")
    else:
        print(f"[MIXEDLM] statsmodels not available; skipping model fit for {metric}.")

    ax.set_xlabel("Days from surgery")
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\n(All subjects; points colored by subject)")
    ax.axvline(0, linestyle=":", linewidth=1)

    # Legend can get large; keep it outside
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize=8)
    plt.tight_layout()

    out_path = plot_root / out_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Mixed model plot saved: {out_path}")



def _transform_metric_for_glmm(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str]:
    """
    Prepare a metric for a Gaussian mixed model on a transformed scale.
    This approximates a GLMM for bounded/positive continuous outcomes:
      - IS, RA in (0,1): logit transform
      - IV > 0: log transform

    Returns a copy of df with a new response column and the response column name.
    """
    out = df.copy()
    eps = 1e-6

    if metric in ["IS", "RA"]:
        resp_col = f"{metric}_glmm_resp"
        clipped = out[metric].clip(eps, 1 - eps)
        out[resp_col] = logit(clipped)
    elif metric == "IV":
        resp_col = f"{metric}_glmm_resp"
        clipped = out[metric].clip(eps)
        out[resp_col] = np.log(clipped)
    else:
        raise ValueError(f"Unsupported GLMM metric: {metric}")

    return out, resp_col


def _inverse_transform_glmm_response(y_vals: np.ndarray, metric: str) -> np.ndarray:
    """Map transformed fitted values back to the original metric scale."""
    if metric in ["IS", "RA"]:
        return expit(y_vals)
    elif metric == "IV":
        return np.exp(y_vals)
    else:
        raise ValueError(f"Unsupported GLMM metric: {metric}")


def _fit_glmmish_and_plot(master_sessions_df: pd.DataFrame, metric: str, ylabel: str, title: str, out_name: str, plot_root: Path):
    """
    Fits an approximate generalized linear mixed model by transforming the response:
      - IS, RA: logit transform
      - IV: log transform

    Then fits a linear mixed model on the transformed response with:
      response ~ days_from_surgery * group
      random intercept + random slope by subject

    Saves a plot with:
      - all subject points colored by subject
      - transformed-scale mixed-model fits back-transformed to original units
    """
    df = master_sessions_df.copy()
    df = df.dropna(subset=["subject", "group", "days_from_surgery", metric])
    df = df[df["group"].isin(["T", "C"])].copy()

    if df.empty:
        print(f"[GLMM] No data available for {metric}; skipping approximate GLMM plot.")
        return

    df["days_from_surgery"] = df["days_from_surgery"].astype(float)
    df["group"] = pd.Categorical(df["group"], categories=["C", "T"])

    subjects = sorted(df["subject"].unique().tolist())
    cmap = plt.get_cmap("tab20")
    subj_to_color = {s: cmap(i % 20) for i, s in enumerate(subjects)}

    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot raw points on original scale
    for s in subjects:
        sdf = df[df["subject"] == s]
        ax.scatter(
            sdf["days_from_surgery"],
            sdf[metric],
            label=s,
            color=subj_to_color[s],
            alpha=0.9
        )

    if _HAS_STATSMODELS:
        try:
            model_df, resp_col = _transform_metric_for_glmm(df, metric)

            model = smf.mixedlm(
                f"{resp_col} ~ days_from_surgery * group",
                model_df,
                groups=model_df["subject"],
                re_formula="~days_from_surgery"
            )
            res = model.fit(reml=False, method="lbfgs", disp=False)

            print(f"[GLMM-approx] {metric} fit complete.")
            print(res.summary())

            # Save summary text
            summary_path = plot_root / f"glmm_{metric}_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(str(res.summary()))
            print(f"Approximate GLMM summary saved: {summary_path}")

            x_grid = np.linspace(df["days_from_surgery"].min(), df["days_from_surgery"].max(), 200)
            for g in ["C", "T"]:
                pred_df = pd.DataFrame({
                    "days_from_surgery": x_grid,
                    "group": g,
                    "subject": subjects[0],  # placeholder for predict
                })
                y_hat_trans = res.predict(pred_df)
                y_hat_orig = _inverse_transform_glmm_response(np.asarray(y_hat_trans), metric)
                ax.plot(
                    x_grid,
                    y_hat_orig,
                    linewidth=3,
                    alpha=0.95,
                    label=f"Approx. GLMM fit ({g})"
                )

        except Exception as e:
            print(f"[GLMM] Could not fit approximate GLMM for {metric}: {e}")
    else:
        print(f"[GLMM] statsmodels not available; skipping approximate GLMM fit for {metric}.")

    ax.set_xlabel("Days from surgery")
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\\n(All subjects; points colored by subject)")
    ax.axvline(0, linestyle=":", linewidth=1)

    # Keep sensible y-axis limits on original scale
    if metric in Y_LIMITS:
        ax.set_ylim(Y_LIMITS[metric])

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize=8)
    plt.tight_layout()

    out_path = plot_root / out_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Approximate GLMM plot saved: {out_path}")

print("\n" + "=" * 60)
print("DONE. Outputs saved under:")
print(f"  {PLOT_ROOT}")

# Save master summary across all subjects
if len(master_summary_rows) > 0:
    master_df = pd.DataFrame(master_summary_rows).sort_values(["group", "subject"])
    master_out = PLOT_ROOT / "master_summary.csv"
    master_df.to_csv(master_out, index=False)
    print(f"Master summary saved: {master_out}")
else:
    print("No subjects were processed successfully; no master summary written.")



# -------------------------
# GROUP COMPARISON PLOTS (mean slope T vs C)
# -------------------------
if len(master_summary_rows) > 0:
    master_df = pd.DataFrame(master_summary_rows)

    # Keep only rows with valid group labels
    master_df = master_df[master_df["group"].isin(["T", "C"])].copy()

    def _plot_group_slope(metric_key: str, title: str, ylabel: str, out_name: str):
        sub = master_df.dropna(subset=[metric_key])
        if sub.empty:
            print(f"No data available for {metric_key}; skipping plot.")
            return

        grp = sub.groupby("group")[metric_key]
        means = grp.mean()
        sems = grp.sem()  # standard error of the mean

        # Ensure order T then C (or whatever exists)
        order = [g for g in ["T", "C"] if g in means.index.tolist()]
        means = means.loc[order]
        sems = sems.loc[order]

        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.bar(order, means.values, yerr=sems.values, capsize=6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Group")
        ax.axhline(0, linewidth=1)

        plt.tight_layout()
        out_path = PLOT_ROOT / out_name
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Group comparison plot saved: {out_path}")

    _plot_group_slope("IS_slope", "Mean slope of IS vs Days from surgery (T vs C)", "IS slope (per day)", "compare_IS_slope_T_vs_C.png")
    _plot_group_slope("IV_slope", "Mean slope of IV vs Days from surgery (T vs C)", "IV slope (per day)", "compare_IV_slope_T_vs_C.png")
    _plot_group_slope("RA_slope", "Mean slope of RA vs Days from surgery (T vs C)", "RA slope (per day)", "compare_RA_slope_T_vs_C.png")




# -------------------------
# ACROSS-SUBJECT MIXED MODEL PLOTS
# -------------------------
if len(master_sessions_rows) > 0:
    master_sessions_df = pd.DataFrame(master_sessions_rows)
    mixed_out = PLOT_ROOT / "mixed_model_plots"
    mixed_out.mkdir(parents=True, exist_ok=True)

    _fit_mixedlm_and_plot(master_sessions_df, "IS", "Interdaily Stability (IS)", "IS vs Days from surgery", "mixed_IS_all_subjects.png", mixed_out)
    _fit_mixedlm_and_plot(master_sessions_df, "IV", "Intradaily Variability (IV)", "IV vs Days from surgery", "mixed_IV_all_subjects.png", mixed_out)
    _fit_mixedlm_and_plot(master_sessions_df, "RA", "Relative Amplitude (RA, classic M10/L5)", "RA vs Days from surgery", "mixed_RA_all_subjects.png", mixed_out)
    _fit_mixedlm_and_plot(master_sessions_df, "IS_norm60", "Normalized IS (relative to day 60)", "Normalized IS vs Days from surgery", "mixed_IS_norm60_all_subjects.png", mixed_out)
    _fit_mixedlm_and_plot(master_sessions_df, "IV_norm60", "Normalized IV (relative to day 60)", "Normalized IV vs Days from surgery", "mixed_IV_norm60_all_subjects.png", mixed_out)
    _fit_mixedlm_and_plot(master_sessions_df, "RA_norm60", "Normalized RA (classic M10/L5, relative to day 60)", "Normalized RA vs Days from surgery", "mixed_RA_norm60_all_subjects.png", mixed_out)

    # Save the long-format per-session dataset too
    master_sessions_out = mixed_out / "master_sessions_long.csv"
    master_sessions_df.to_csv(master_sessions_out, index=False)
    print(f"Master per-session long table saved: {master_sessions_out}")
else:
    print("No per-session rows collected; skipping mixed model plots.")



# -------------------------
# APPROXIMATE GLMM PLOTS (transformed-response mixed models)
# -------------------------
if len(master_sessions_rows) > 0:
    master_sessions_df = pd.DataFrame(master_sessions_rows)
    glmm_out = PLOT_ROOT / "glmm_plots"
    glmm_out.mkdir(parents=True, exist_ok=True)

    _fit_glmmish_and_plot(master_sessions_df, "IS", "Interdaily Stability (IS)", "Approximate GLMM for IS vs Days from surgery", "glmm_IS_all_subjects.png", glmm_out)
    _fit_glmmish_and_plot(master_sessions_df, "IV", "Intradaily Variability (IV)", "Approximate GLMM for IV vs Days from surgery", "glmm_IV_all_subjects.png", glmm_out)
    _fit_glmmish_and_plot(master_sessions_df, "RA", "Relative Amplitude (RA, classic M10/L5)", "Approximate GLMM for RA vs Days from surgery", "glmm_RA_all_subjects.png", glmm_out)
else:
    print("No per-session rows collected; skipping approximate GLMM plots.")
