import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

DATASET_PATH = "outputs/phase1_training_dataset.csv"
MODEL_PATH = "models/phase1_best_model.joblib"
META_PATH = "models/phase1_metadata.json"
OUTPUT_JSON = "outputs/phase1_evidence_report.json"
OUTPUT_CSV = "outputs/phase1_evidence_predictions_test.csv"


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Missing {DATASET_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Missing {META_PATH}")

    df = pd.read_csv(DATASET_PATH)
    model = joblib.load(MODEL_PATH)

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    feature_cols = meta["feature_columns"]

    if "split" not in df.columns:
        raise ValueError("phase1_training_dataset.csv must contain a 'split' column")

    if "target" not in df.columns:
        raise ValueError("phase1_training_dataset.csv must contain a 'target' column")

    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        raise ValueError("No test rows found in phase1_training_dataset.csv")

    X_test = test_df[feature_cols]
    y_test = test_df["target"].astype(int)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else None

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else None
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else None

    report = {
        "model_name": meta.get("best_model_name"),
        "main_signal": meta.get("main_signal"),
        "prediction_horizon_minutes": meta.get("prediction_horizon_minutes"),
        "hazard_threshold": meta.get("hazard_threshold"),
        "test_rows": int(len(test_df)),
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc) if auc is not None else None,
            "false_alarm_rate": float(false_alarm_rate) if false_alarm_rate is not None else None,
            "miss_rate": float(miss_rate) if miss_rate is not None else None,
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": classification_report(
            y_test, preds, output_dict=True, zero_division=0
        ),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    out = test_df[["time", "station_id", "target"]].copy()
    out["predicted_probability"] = probs
    out["predicted_label"] = preds
    out.to_csv(OUTPUT_CSV, index=False)

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()