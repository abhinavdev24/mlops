import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train a Support Vector Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    svc_classifier = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    svc_classifier.fit(X_train, y_train)
    joblib.dump(svc_classifier, "../model/wine_model.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)
