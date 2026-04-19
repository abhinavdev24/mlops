import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import load_data, split_data


def test_load_data_returns_expected_wine_shape():
    X, y = load_data()

    assert X.shape == (178, 13)
    assert y.shape == (178,)
    assert set(np.unique(y)) == {0, 1, 2}


def test_split_data_is_deterministic_and_stratified():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    assert X_train.shape[0] == 124
    assert X_test.shape[0] == 54
    assert y_train.shape[0] == 124
    assert y_test.shape[0] == 54

    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)

    assert train_counts.tolist() == [41, 50, 33]
    assert test_counts.tolist() == [18, 21, 15]


def test_svc_pipeline_trains_and_predicts_on_wine_data():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert predictions.shape == y_test.shape
    assert (predictions == y_test).mean() >= 0.85
