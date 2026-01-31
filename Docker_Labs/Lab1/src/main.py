# Import necessary libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

if __name__ == "__main__":
    # Load the Wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a Support Vector Classifier
    model = SVC(kernel="rbf", random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set and print accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"The model training was successful. Test accuracy: {accuracy:.2%}")

    # Save the model to a file
    joblib.dump(model, "wine_model.pkl")
