import os
import json
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

MODEL_PATH = os.getenv("MODEL_PATH", "models/phase1_best_model.joblib")
META_PATH = os.getenv("META_PATH", "models/phase1_metadata.json")

DATABASE_URL = os.getenv("DATABASE_URL")

# Optional mapping file: station_id -> location_id
STATION_MAP_PATH = os.getenv("STATION_MAP_PATH", "station_map.csv")

# How much recent history to read from DB
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "6"))

# Prediction write settings
MODEL_VERSION = os.getenv("MODEL_VERSION", "phase1_v1")
PHASE_NAME = os.getenv("PHASE_NAME", "hardware_level_only")
MAIN_SENSOR_TYPE = os.getenv("MAIN_SENSOR_TYPE", "Ultrasonic")

# Risk thresholds from predicted probability
LOW_MAX = float(os.getenv("LOW_MAX", "0.35"))
MEDIUM_MAX = float(os.getenv("MEDIUM_MAX", "0.65"))


def load_model_and_meta():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta


def get_engine():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set")
    return create_engine(DATABASE_URL)


def risk_level_from_score(score: float) -> str:
    if score < LOW_MAX:
        return "low"
    elif score < MEDIUM_MAX:
        return "medium"
    return "high"


def load_station_map():
    if os.path.exists(STATION_MAP_PATH):
        df = pd.read_csv(STATION_MAP_PATH)
        expected = {"station_id", "location_id"}
        if not expected.issubset(df.columns):
            raise ValueError(f"{STATION_MAP_PATH} must contain columns: {expected}")
        return df
    return pd.DataFrame(columns=["station_id", "location_id"])


def fetch_recent_sensor_data(engine):
    """
    Assumes your DB has:
      sensor_reading(sensor_id, time_stamp, raw_value)
      sensor(sensor_id, sensor_type_id, node_id, ...)
      sensor_type(sensor_type_id, type_name, ...)
      sensor_node(node_id, location_id, ...)

    Adjust names if your schema differs slightly.
    """

    query = text("""
        SELECT
            sr.sensor_id,
            sr.time_stamp,
            sr.raw_value,
            s.node_id,
            sn.location_id,
            st.type_name
        FROM sensor_reading sr
        JOIN sensor s
          ON sr.sensor_id = s.sensor_id
        JOIN sensor_type st
          ON s.sensor_type_id = st.sensor_type_id
        JOIN sensor_node sn
          ON s.node_id = sn.node_id
        WHERE sr.time_stamp >= NOW() - (:lookback || ' hours')::interval
          AND LOWER(st.type_name) LIKE '%ultrasonic%'
        ORDER BY sr.time_stamp ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"lookback": LOOKBACK_HOURS})

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time_stamp"], utc=True, errors="coerce")
    df["level_signal"] = pd.to_numeric(df["raw_value"], errors="coerce")

    # Build a stable station identifier from sensor_id
    df["station_id"] = df["sensor_id"].astype(str)

    # Since DB raw data does not have these cleaning flags yet, default to 0
    for col in ["man_remove", "ffill", "stamp", "outbound", "frozen", "outlier"]:
        df[col] = 0

    return df[
        [
            "station_id", "sensor_id", "location_id", "time",
            "level_signal",
            "man_remove", "ffill", "stamp", "outbound", "frozen", "outlier"
        ]
    ].copy()


def prepare_latest_features(df, meta):
    """
    Build the same feature set expected by the trained phase-1 model.
    We resample to 5 minutes and then compute lag/rolling features.
    """
    if df.empty:
        return pd.DataFrame()

    frames = []

    for station_id, g in df.groupby("station_id"):
        g = g.sort_values("time").copy()
        location_id = g["location_id"].iloc[0]

        g = g.set_index("time")

        # Resample to 5 min to match training
        out = g.resample("5min").agg({
            "level_signal": "mean",
            "man_remove": "max",
            "ffill": "max",
            "stamp": "max",
            "outbound": "max",
            "frozen": "max",
            "outlier": "max",
        })

        out["level_signal"] = out["level_signal"].interpolate(method="time", limit_direction="both")

        for c in ["man_remove", "ffill", "stamp", "outbound", "frozen", "outlier"]:
            out[c] = out[c].fillna(0).astype(int)

        out = out.reset_index()
        out["station_id"] = station_id
        out["location_id"] = location_id
        frames.append(out)

    feat_df = pd.concat(frames, ignore_index=True)
    feat_df = feat_df.sort_values(["station_id", "time"]).copy()

    grp = feat_df.groupby("station_id", group_keys=False)

    feat_df["level_lag_1"] = grp["level_signal"].shift(1)
    feat_df["level_lag_3"] = grp["level_signal"].shift(3)
    feat_df["level_lag_6"] = grp["level_signal"].shift(6)

    feat_df["level_diff_1"] = feat_df["level_signal"] - feat_df["level_lag_1"]
    feat_df["level_diff_3"] = feat_df["level_signal"] - feat_df["level_lag_3"]
    feat_df["level_diff_6"] = feat_df["level_signal"] - feat_df["level_lag_6"]

    feat_df["level_roll_mean_3"] = grp["level_signal"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    feat_df["level_roll_mean_6"] = grp["level_signal"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    feat_df["level_roll_max_6"] = grp["level_signal"].transform(lambda s: s.rolling(6, min_periods=1).max())
    feat_df["level_roll_std_3"] = grp["level_signal"].transform(lambda s: s.rolling(3, min_periods=1).std())

    feat_df["hour"] = feat_df["time"].dt.hour
    feat_df["day_of_week"] = feat_df["time"].dt.dayofweek

    # Keep latest row per station, only if enough history exists
    latest_rows = []
    for station_id, g in feat_df.groupby("station_id"):
        g = g.sort_values("time").copy()
        row = g.iloc[-1]

        required_cols = [
            "level_signal",
            "level_lag_1", "level_lag_3", "level_lag_6",
            "level_diff_1", "level_diff_3", "level_diff_6",
            "level_roll_mean_3", "level_roll_mean_6",
            "level_roll_max_6", "level_roll_std_3",
            "man_remove", "ffill", "stamp", "outbound", "frozen", "outlier",
            "hour", "day_of_week",
        ]

        if row[required_cols].isna().any():
            continue

        latest_rows.append(row)

    if not latest_rows:
        return pd.DataFrame()

    latest = pd.DataFrame(latest_rows).reset_index(drop=True)
    return latest


def build_prediction_rows(features_df, model, meta):
    feature_cols = meta["feature_columns"]
    X = features_df[feature_cols].copy()

    probs = model.predict_proba(X)[:, 1]

    out = features_df[["station_id", "location_id", "time"]].copy()
    out["risk_score"] = probs
    out["risk_level"] = out["risk_score"].apply(risk_level_from_score)
    out["predicted_hazard_ts"] = out["time"] + pd.to_timedelta(meta["prediction_horizon_minutes"], unit="m")

    out["meta_json"] = out.apply(
        lambda r: json.dumps({
            "model_name": meta.get("best_model_name", "phase1_model"),
            "model_version": MODEL_VERSION,
            "phase": PHASE_NAME,
            "main_signal": meta.get("main_signal", "level"),
            "hazard_threshold": meta.get("hazard_threshold"),
            "prediction_horizon_minutes": meta.get("prediction_horizon_minutes"),
            "station_id": str(r["station_id"]),
            "sensor_type": MAIN_SENSOR_TYPE,
        }),
        axis=1
    )

    return out


def insert_predictions(engine, pred_df):
    """
    Assumes prediction table has at least:
      location_id, time_stamp, risk_score, risk_level, predicted_hazard_ts, meta_json

    Adjust column names if needed.
    """
    if pred_df.empty:
        print("No predictions to insert.")
        return

    insert_sql = text("""
        INSERT INTO prediction (
            location_id,
            time_stamp,
            risk_score,
            risk_level,
            predicted_hazard_ts,
            meta_json
        )
        VALUES (
            :location_id,
            :time_stamp,
            :risk_score,
            :risk_level,
            :predicted_hazard_ts,
            CAST(:meta_json AS jsonb)
        )
    """)

    rows = []
    for _, r in pred_df.iterrows():
        if pd.isna(r["location_id"]):
            continue

        rows.append({
            "location_id": int(r["location_id"]),
            "time_stamp": pd.Timestamp(r["time"]).to_pydatetime(),
            "risk_score": float(r["risk_score"]),
            "risk_level": str(r["risk_level"]),
            "predicted_hazard_ts": pd.Timestamp(r["predicted_hazard_ts"]).to_pydatetime(),
            "meta_json": str(r["meta_json"]),
        })

    if not rows:
        print("Predictions were created, but no valid location_id values were available.")
        return

    with engine.begin() as conn:
        conn.execute(insert_sql, rows)

    print(f"Inserted {len(rows)} predictions into prediction table.")


def main():
    model, meta = load_model_and_meta()
    engine = get_engine()

    raw_df = fetch_recent_sensor_data(engine)
    if raw_df.empty:
        print("No recent ultrasonic sensor data found.")
        return

    features_df = prepare_latest_features(raw_df, meta)
    if features_df.empty:
        print("Not enough history to build prediction features.")
        return

    pred_df = build_prediction_rows(features_df, model, meta)

    # Save a local copy for debugging
    os.makedirs("outputs", exist_ok=True)
    pred_df.to_csv("outputs/db_predictions_preview.csv", index=False)

    insert_predictions(engine, pred_df)


if __name__ == "__main__":
    main()