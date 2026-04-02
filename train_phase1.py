import os
import glob
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# =========================
# CONFIG
# =========================
DATA_DIR = "data/cleaned_csvs"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

TIME_COL = "time"

# Choose one main level signal
# Recommended: "level"
MAIN_SIGNAL = "level"

# Resample interval
RESAMPLE_RULE = "5min"

# Future prediction horizon
PREDICTION_HORIZON_STEPS = 6   # 6 * 5min = 30 min

# Hazard threshold strategy:
# use percentile on training set to define "hazardous level"
HAZARD_PERCENTILE = 0.90

RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# HELPERS
# =========================
def to_bool_int(series):
    """Convert TRUE/FALSE or boolean-like values to 0/1."""
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype(int)
    )


def read_one_csv(path):
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Rename known variants to one standard name
    rename_map = {
        "value_no_errors": "value_no_err"
    }
    df = df.rename(columns=rename_map)

    required = [
        "time", "raw_value", "value_no_err", "man_remove", "ffill",
        "stamp", "outbound", "frozen", "outlier", "depth_s", "level"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")

    # Keep only needed columns, plus optional ones if present
    keep_cols = required.copy()
    optional_cols = ["frozen_high"]
    for c in optional_cols:
        if c in df.columns:
            keep_cols.append(c)

    df = df[keep_cols].copy()

    # Parse time
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).sort_values("time")

    # Numeric columns
    numeric_cols = ["raw_value", "value_no_err", "depth_s", "level"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Boolean flags
    flag_cols = ["man_remove", "ffill", "stamp", "outbound", "frozen", "outlier"]
    if "frozen_high" in df.columns:
        flag_cols.append("frozen_high")

    for col in flag_cols:
        df[col] = to_bool_int(df[col])

    # Add station/file id from filename
    station_id = os.path.splitext(os.path.basename(path))[0]
    df["station_id"] = station_id

    return df


def resample_station(df_station, signal_col):
    """
    Resample one station to fixed 5-minute intervals.
    Numeric values: mean
    Flags: max (if any bad flag in interval => 1)
    """
    df_station = df_station.sort_values("time").set_index("time")

    value_cols = ["raw_value", "value_no_err", "depth_s", "level"]
    flag_cols = ["man_remove", "ffill", "stamp", "outbound", "frozen", "outlier"]

    if "frozen_high" in df_station.columns:
        flag_cols.append("frozen_high")

    agg_map = {c: "mean" for c in value_cols}
    agg_map.update({c: "max" for c in flag_cols})

    out = df_station.resample(RESAMPLE_RULE).agg(agg_map)

    # Use time-based interpolation only on core signal columns
    for c in value_cols:
        out[c] = out[c].interpolate(method="time", limit_direction="both")

    # Fill flags safely
    for c in flag_cols:
        out[c] = out[c].fillna(0).astype(int)

    out["station_id"] = df_station["station_id"].iloc[0]
    out = out.reset_index()

    # Rename chosen training signal
    out["level_signal"] = out[signal_col]

    return out


def add_features(df):
    """
    Add lag/trend/rolling/time features per station.
    """
    df = df.sort_values(["station_id", "time"]).copy()

    grp = df.groupby("station_id", group_keys=False)

    # Basic lags
    df["level_lag_1"] = grp["level_signal"].shift(1)
    df["level_lag_3"] = grp["level_signal"].shift(3)
    df["level_lag_6"] = grp["level_signal"].shift(6)

    # Diffs
    df["level_diff_1"] = df["level_signal"] - df["level_lag_1"]
    df["level_diff_3"] = df["level_signal"] - df["level_lag_3"]
    df["level_diff_6"] = df["level_signal"] - df["level_lag_6"]

    # Rolling stats
    df["level_roll_mean_3"] = grp["level_signal"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["level_roll_mean_6"] = grp["level_signal"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    df["level_roll_max_6"] = grp["level_signal"].transform(lambda s: s.rolling(6, min_periods=1).max())
    df["level_roll_std_3"] = grp["level_signal"].transform(lambda s: s.rolling(3, min_periods=1).std())

    # Time features
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek

    # Future target support
    df["future_level"] = grp["level_signal"].shift(-PREDICTION_HORIZON_STEPS)

    return df


def time_split(df):
    """
    Global time-based split:
    first 70% train, next 15% valid, last 15% test
    """
    df = df.sort_values("time").copy()
    unique_times = df["time"].sort_values().unique()

    n = len(unique_times)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train_times = unique_times[:train_end]
    valid_times = unique_times[train_end:valid_end]
    test_times = unique_times[valid_end:]

    train_df = df[df["time"].isin(train_times)].copy()
    valid_df = df[df["time"].isin(valid_times)].copy()
    test_df = df[df["time"].isin(test_times)].copy()

    return train_df, valid_df, test_df


def evaluate_model(name, model, X, y):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "model": name,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else None,
    }
    return metrics, prob, pred


# =========================
# MAIN
# =========================
def main():
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    print(f"Found {len(csv_files)} CSV files")

    # Read all cleaned CSVs
    frames = []
    for path in csv_files:
        df = read_one_csv(path)
        frames.append(df)

    raw_all = pd.concat(frames, ignore_index=True)
    raw_all.to_csv(os.path.join(OUTPUT_DIR, "raw_combined.csv"), index=False)

    # Resample each station/file
    station_frames = []
    for station_id, df_station in raw_all.groupby("station_id"):
        rs = resample_station(df_station, signal_col=MAIN_SIGNAL)
        station_frames.append(rs)

    data = pd.concat(station_frames, ignore_index=True)

    # Add features
    data = add_features(data)

    # Drop rows that cannot produce future target
    data = data.dropna(subset=["level_signal", "future_level"]).copy()

    # Time split first
    train_df, valid_df, test_df = time_split(data)

    # Define hazard threshold from TRAINING distribution only
    hazard_threshold = train_df["level_signal"].quantile(HAZARD_PERCENTILE)
    print(f"Hazard threshold ({HAZARD_PERCENTILE*100:.0f}th percentile): {hazard_threshold:.4f}")

    # Create target: will future level be hazardous?
    for part in [train_df, valid_df, test_df]:
        part["target"] = (part["future_level"] >= hazard_threshold).astype(int)

    feature_cols = [
        "level_signal",
        "level_lag_1", "level_lag_3", "level_lag_6",
        "level_diff_1", "level_diff_3", "level_diff_6",
        "level_roll_mean_3", "level_roll_mean_6",
        "level_roll_max_6", "level_roll_std_3",
        "man_remove", "ffill", "stamp", "outbound", "frozen", "outlier",
        "hour", "day_of_week",
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df["target"]

    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    # Models
    logreg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    models = {
        "logistic_regression": logreg,
        "random_forest": rf
    }

    results = []

    best_model_name = None
    best_model = None
    best_valid_f1 = -1

    for name, model in models.items():
        model.fit(X_train, y_train)

        valid_metrics, _, _ = evaluate_model(name, model, X_valid, y_valid)
        test_metrics, _, _ = evaluate_model(name, model, X_test, y_test)

        row = {
            "model": name,
            "valid_precision": valid_metrics["precision"],
            "valid_recall": valid_metrics["recall"],
            "valid_f1": valid_metrics["f1"],
            "valid_roc_auc": valid_metrics["roc_auc"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
        }
        results.append(row)

        print(f"\nModel: {name}")
        print(json.dumps(row, indent=2))

        if valid_metrics["f1"] > best_valid_f1:
            best_valid_f1 = valid_metrics["f1"]
            best_model_name = name
            best_model = model

    results_df = pd.DataFrame(results).sort_values("valid_f1", ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_results.csv"), index=False)

    # Save best model + metadata
    joblib.dump(best_model, os.path.join(MODEL_DIR, "phase1_best_model.joblib"))

    metadata = {
        "main_signal": MAIN_SIGNAL,
        "resample_rule": RESAMPLE_RULE,
        "prediction_horizon_steps": PREDICTION_HORIZON_STEPS,
        "prediction_horizon_minutes": PREDICTION_HORIZON_STEPS * 5,
        "hazard_percentile": HAZARD_PERCENTILE,
        "hazard_threshold": float(hazard_threshold),
        "feature_columns": feature_cols,
        "best_model_name": best_model_name,
        "num_csv_files": len(csv_files),
    }

    with open(os.path.join(MODEL_DIR, "phase1_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save processed dataset
    all_parts = []
    for name, part in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        tmp = part.copy()
        tmp["split"] = name
        all_parts.append(tmp)

    final_dataset = pd.concat(all_parts, ignore_index=True)
    final_dataset.to_csv(os.path.join(OUTPUT_DIR, "phase1_training_dataset.csv"), index=False)

    print("\nDone.")
    print(f"Best model: {best_model_name}")
    print(f"Saved model to {os.path.join(MODEL_DIR, 'phase1_best_model.joblib')}")


if __name__ == "__main__":
    main()