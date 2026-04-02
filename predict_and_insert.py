import os
import json
import logging
import joblib
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

MODEL_PATH = os.getenv("MODEL_PATH", "models/phase1_best_model.joblib")
META_PATH = os.getenv("META_PATH", "models/phase1_metadata.json")
DATABASE_URL = os.getenv("DATABASE_URL")

LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "6"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "phase1_v1")
PHASE_NAME = os.getenv("PHASE_NAME", "hardware_level_only")
MAIN_SENSOR_TYPE = os.getenv("MAIN_SENSOR_TYPE", "Ultrasonic")
LOW_MAX = float(os.getenv("LOW_MAX", "0.35"))
MEDIUM_MAX = float(os.getenv("MEDIUM_MAX", "0.65"))
DUPLICATE_WINDOW_MINUTES = int(os.getenv("DUPLICATE_WINDOW_MINUTES", "5"))
SAVE_PREVIEW = os.getenv("SAVE_PREVIEW", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("predict_and_insert")


def load_model_and_meta():
    logger.info("Loading model and metadata")
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta


def get_engine():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set")
    logger.info("Creating database engine")
    return create_engine(DATABASE_URL)


def risk_level_from_score(score: float) -> str:
    if score < LOW_MAX:
        return "low"
    elif score < MEDIUM_MAX:
        return "medium"
    return "high"


def fetch_recent_sensor_data(engine):
    logger.info("Fetching recent ultrasonic sensor readings from database")
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
        logger.warning("No recent ultrasonic sensor data found")
        return df

    df["time"] = pd.to_datetime(df["time_stamp"], utc=True, errors="coerce")
    df["level_signal"] = pd.to_numeric(df["raw_value"], errors="coerce")
    df["station_id"] = df["sensor_id"].astype(str)

    for col in ["man_remove", "ffill", "stamp", "outbound", "frozen", "outlier"]:
        df[col] = 0

    df = df[
        [
            "station_id", "sensor_id", "location_id", "time",
            "level_signal",
            "man_remove", "ffill", "stamp", "outbound", "frozen", "outlier"
        ]
    ].copy()

    logger.info(
        "Fetched %s rows for %s stations",
        len(df),
        df["station_id"].nunique()
    )
    return df


def prepare_latest_features(df):
    if df.empty:
        return pd.DataFrame()

    frames = []

    for station_id, g in df.groupby("station_id"):
        g = g.sort_values("time").copy()
        location_id = g["location_id"].iloc[0]

        g = g.set_index("time")

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

    latest_rows = []
    required_cols = [
        "level_signal",
        "level_lag_1", "level_lag_3", "level_lag_6",
        "level_diff_1", "level_diff_3", "level_diff_6",
        "level_roll_mean_3", "level_roll_mean_6",
        "level_roll_max_6", "level_roll_std_3",
        "man_remove", "ffill", "stamp", "outbound", "frozen", "outlier",
        "hour", "day_of_week",
    ]

    skipped = 0
    for station_id, g in feat_df.groupby("station_id"):
        g = g.sort_values("time").copy()
        row = g.iloc[-1]

        if row[required_cols].isna().any():
            skipped += 1
            continue

        latest_rows.append(row)

    if not latest_rows:
        logger.warning("No stations had enough history for feature generation")
        return pd.DataFrame()

    latest = pd.DataFrame(latest_rows).reset_index(drop=True)
    logger.info("Built feature rows for %s stations, skipped %s", len(latest), skipped)
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

    logger.info("Built %s prediction rows", len(out))
    return out


def filter_duplicates(engine, pred_df):
    if pred_df.empty:
        return pred_df

    query = text("""
        SELECT 1
        FROM prediction
        WHERE location_id = :location_id
          AND time_stamp >= NOW() - (:dup_window || ' minutes')::interval
        LIMIT 1
    """)

    kept = []
    skipped = 0

    with engine.connect() as conn:
        for _, r in pred_df.iterrows():
            location_id = r["location_id"]
            if pd.isna(location_id):
                skipped += 1
                continue

            exists = conn.execute(
                query,
                {
                    "location_id": int(location_id),
                    "dup_window": DUPLICATE_WINDOW_MINUTES
                }
            ).fetchone()

            if exists:
                skipped += 1
                continue

            kept.append(r)

    filtered = pd.DataFrame(kept) if kept else pd.DataFrame(columns=pred_df.columns)
    logger.info(
        "Duplicate filter kept %s rows and skipped %s rows",
        len(filtered),
        skipped
    )
    return filtered


def insert_predictions(engine, pred_df):
    if pred_df.empty:
        logger.warning("No predictions to insert")
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
        logger.warning("Predictions existed, but none had a valid location_id")
        return

    with engine.begin() as conn:
        conn.execute(insert_sql, rows)

    logger.info("Inserted %s predictions into prediction table", len(rows))


def main():
    logger.info("Prediction job started")
    model, meta = load_model_and_meta()
    engine = get_engine()

    raw_df = fetch_recent_sensor_data(engine)
    if raw_df.empty:
        logger.warning("Job finished: no source data")
        return

    features_df = prepare_latest_features(raw_df)
    if features_df.empty:
        logger.warning("Job finished: not enough history to compute features")
        return

    pred_df = build_prediction_rows(features_df, model, meta)

    if SAVE_PREVIEW:
        os.makedirs("outputs", exist_ok=True)
        pred_df.to_csv("outputs/db_predictions_preview.csv", index=False)
        logger.info("Saved preview to outputs/db_predictions_preview.csv")

    pred_df = filter_duplicates(engine, pred_df)
    insert_predictions(engine, pred_df)

    logger.info("Prediction job finished")


if __name__ == "__main__":
    main()