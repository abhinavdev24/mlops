import argparse
import datetime
import os
import pickle

import mlflow
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import load_data, split_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    os.makedirs("data", exist_ok=True)
    with open("data/data.pickle", "wb") as data_file:
        pickle.dump(X, data_file)
    with open("data/target.pickle", "wb") as target_file:
        pickle.dump(y, target_file)

    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Wine"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=dataset_name):
        params = {
            "dataset_name": dataset_name,
            "number of datapoint": X.shape[0],
            "number of dimensions": X.shape[1],
            "model": "StandardScaler+SVC(rbf)",
        }
        mlflow.log_params(params)

        model = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)
        mlflow.log_metrics(
            {
                "Accuracy": accuracy_score(y_test, y_predict),
                "F1 Score": f1_score(y_test, y_predict, average="macro"),
            }
        )

        os.makedirs("models", exist_ok=True)

        model_version = f"model_{timestamp}"
        model_filename = f"{model_version}_dt_model.joblib"
        dump(model, model_filename)
                    

