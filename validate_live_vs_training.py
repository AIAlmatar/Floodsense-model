import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

DATABASE_URL = os.getenv("DATABASE_URL")
TRAINING_DATASET_PATH = os.getenv("TRAINING_DATASET_PATH", "outputs/phase1_training_dataset.csv")
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "24"))
CHUNK_SIZE = int(os.getenv("VALIDATION_CHUNK_SIZE", "200000"))

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")

engine = create_engine(DATABASE_URL)


def load_training_signal():
    if not os.path.exists(TRAINING_DATASET_PATH):
        raise FileNotFoundError(f"Training dataset file not found: {TRAINING_DATASET_PATH}")

    pieces = []

    for chunk in pd.read_csv(TRAINING_DATASET_PATH, chunksize=CHUNK_SIZE):
        if "level_signal" in chunk.columns:
            s = pd.to_numeric(chunk["level_signal"], errors="coerce").dropna()
        elif "level" in chunk.columns:
            s = pd.to_numeric(chunk["level"], errors="coerce").dropna()
        else:
            raise ValueError("Training dataset does not contain 'level_signal' or 'level' column")

        pieces.append(s)

    if not pieces:
        return pd.Series(dtype=float)

    return pd.concat(pieces, ignore_index=True)


def load_live_ultrasonic_signal():
    query = text("""
        SELECT
            sr.sensor_id,
            sr.time_stamp,
            sr.raw_value,
            st.type_name
        FROM sensor_reading sr
        JOIN sensor s
          ON sr.sensor_id = s.sensor_id
        JOIN sensor_type st
          ON s.sensor_type_id = st.sensor_type_id
        WHERE sr.time_stamp >= NOW() - (:lookback || ' hours')::interval
          AND LOWER(st.type_name) LIKE '%ultrasonic%'
        ORDER BY sr.time_stamp ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"lookback": LOOKBACK_HOURS})

    if df.empty:
        return pd.Series(dtype=float)

    df["raw_value"] = pd.to_numeric(df["raw_value"], errors="coerce")
    return df["raw_value"].dropna()


def describe_series(name, s):
    if s.empty:
        return {"name": name, "count": 0}

    return {
        "name": name,
        "count": int(s.shape[0]),
        "min": float(s.min()),
        "p01": float(s.quantile(0.01)),
        "p10": float(s.quantile(0.10)),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std())
    }


def main():
    training = load_training_signal()
    live = load_live_ultrasonic_signal()

    report = {
        "training_signal": describe_series("training_signal", training),
        "live_raw_value": describe_series("live_raw_value", live),
    }

    print(json.dumps(report, indent=2))

    if not training.empty and not live.empty:
        train_p50 = training.quantile(0.50)
        live_p50 = live.quantile(0.50)

        ratio = None
        if train_p50 != 0:
            ratio = float(live_p50 / train_p50)

        print("\nQuick interpretation:")
        print(f"- training median: {train_p50:.4f}")
        print(f"- live median:     {live_p50:.4f}")
        if ratio is not None:
            print(f"- median ratio live/training: {ratio:.4f}")

        if ratio is not None and (ratio < 0.5 or ratio > 2.0):
            print("\nWarning: live DB raw_value scale looks meaningfully different from training signal scale.")
            print("You should calibrate raw_value or retrain using DB-shaped values.")
        else:
            print("\nThe live DB raw_value scale looks reasonably close to the training signal.")


if __name__ == "__main__":
    main()