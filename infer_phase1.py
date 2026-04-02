import os
import json
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/phase1_best_model.joblib"
META_PATH = "models/phase1_metadata.json"

# Example input file with latest recent readings
LATEST_FILE = "outputs/raw_combined.csv"


def load_model_and_meta():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta


def prepare_latest_features(df, meta):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values(["station_id", "time"])

    # Keep one level signal
    df["level_signal"] = pd.to_numeric(df[meta["main_signal"]], errors="coerce")

    # Boolean flags
    flag_cols = ["man_remove", "ffill", "stamp", "outbound", "frozen", "outlier"]
    for c in flag_cols:
        df[c] = (
            df[c].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0}).fillna(0).astype(int)
        )

    rows = []
    for station_id, g in df.groupby("station_id"):
        g = g.sort_values("time").tail(6).copy()  # last 6 rows = 30 minutes if already 5-min spaced
        if len(g) < 6:
            continue

        current = g.iloc[-1]

        row = {
            "station_id": station_id,
            "time": current["time"],
            "level_signal": current["level_signal"],
            "level_lag_1": g.iloc[-2]["level_signal"],
            "level_lag_3": g.iloc[-4]["level_signal"],
            "level_lag_6": g.iloc[-6]["level_signal"],
            "level_diff_1": current["level_signal"] - g.iloc[-2]["level_signal"],
            "level_diff_3": current["level_signal"] - g.iloc[-4]["level_signal"],
            "level_diff_6": current["level_signal"] - g.iloc[-6]["level_signal"],
            "level_roll_mean_3": g["level_signal"].tail(3).mean(),
            "level_roll_mean_6": g["level_signal"].tail(6).mean(),
            "level_roll_max_6": g["level_signal"].tail(6).max(),
            "level_roll_std_3": g["level_signal"].tail(3).std(),
            "man_remove": int(g["man_remove"].tail(6).max()),
            "ffill": int(g["ffill"].tail(6).max()),
            "stamp": int(g["stamp"].tail(6).max()),
            "outbound": int(g["outbound"].tail(6).max()),
            "frozen": int(g["frozen"].tail(6).max()),
            "outlier": int(g["outlier"].tail(6).max()),
            "hour": current["time"].hour,
            "day_of_week": current["time"].dayofweek,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def risk_level_from_score(score):
    if score < 0.35:
        return "low"
    elif score < 0.65:
        return "medium"
    return "high"


def main():
    model, meta = load_model_and_meta()

    df = pd.read_csv(LATEST_FILE)
    if "station_id" not in df.columns:
        # if station_id missing, assume one station
        df["station_id"] = "station_1"

    features_df = prepare_latest_features(df, meta)
    if features_df.empty:
        print("No enough recent rows to predict.")
        return

    X = features_df[meta["feature_columns"]]
    probs = model.predict_proba(X)[:, 1]

    output = features_df[["station_id", "time"]].copy()
    output["risk_score"] = probs
    output["risk_level"] = output["risk_score"].apply(risk_level_from_score)
    output["predicted_hazard_minutes"] = meta["prediction_horizon_minutes"]

    print(output)
    output.to_csv("outputs/latest_predictions.csv", index=False)


if __name__ == "__main__":
    main()