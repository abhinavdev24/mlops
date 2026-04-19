import argparse
import json
import os

import joblib
from sklearn.metrics import accuracy_score, f1_score

from data import load_data, split_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    timestamp = args.timestamp
    model_version = f"model_{timestamp}_dt_model"

    try:
        model = joblib.load(f"{model_version}.joblib")
    except Exception as exc:
        raise ValueError("Failed to load the timestamped model artifact") from exc

    try:
        X, y = load_data()
        _, X_test, _, y_test = split_data(X, y)
    except Exception as exc:
        raise ValueError("Failed to load Wine dataset for evaluation") from exc

    y_predict = model.predict(X_test)
    metrics = {
        "F1_Score": f1_score(y_test, y_predict, average="macro"),
        "Accuracy": accuracy_score(y_test, y_predict),
    }

    os.makedirs("metrics", exist_ok=True)
    with open(f"{timestamp}_metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
               
    
